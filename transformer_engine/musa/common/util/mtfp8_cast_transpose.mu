#include "mtfp8_cast_transpose.h"

#include <musa_runtime.h>

#include "../util/string.h"
#include "../utils.muh"
#include "mtfp8_utils.muh"


namespace transformer_engine {

namespace mtfp8 {

using CType = float;
constexpr size_t warps_per_tile = 4;
constexpr size_t block_size = wrap_size * warps_per_tile;

namespace {

template <typename T>
__device__ __forceinline__ T warpReduceMax(T max_value) {
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 16));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 8));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 4));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 2));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 1));
  return max_value;
}

constexpr int max(int a, int b) {
  return a > b ? a : b;
}

}


template <
    typename IType,
    typename OType,
    size_t N_ELEMENTS_PER_THREAD_X = 4/* VLEN */,
    size_t N_ELEMENTS_PER_THREAD_Y = 4,
    size_t BLOCK_SIZE_X = 32,
    size_t BLOCK_SIZE_Y = 16,
    size_t GROUP_SIZE = 128
>
__global__ void  mtfp8_cast_transpose_general_kernel(
    const IType *__restrict__ const inp,
    const CType *__restrict__ const noop,
    OType *__restrict__ const out_c,
    OType *__restrict__ const out_t,
    CType *__restrict__ const scale_inv,
    CType *__restrict__ const columnwise_scale_inv,
    size_t ncols,
    size_t nrows) {
    // rowwise_group_size and columnwise_group_size should be equal

    using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
    using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
    using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;

    const uint32_t local_col_base_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
    const uint32_t global_col_base_id = blockIdx.x * GROUP_SIZE + local_col_base_id;
    const uint32_t local_row_base_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
    const uint32_t global_row_base_id = blockIdx.y * GROUP_SIZE;

    if (global_row_base_id >= nrows) {
      return;
    }

    const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
    OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
    CType* rowwise_scale_inv_ptr = scale_inv + global_row_base_id * (ncols / GROUP_SIZE) + blockIdx.x;
    OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
    CType* columnwise_scale_inv_ptr = columnwise_scale_inv + blockIdx.y * ncols + global_col_base_id;
    
    constexpr int REPEAT_Y = DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);
    constexpr int ELEMENTS_PER_BANK = 4 / sizeof(IType);  // dword of bank is 32 bits by default
    static_assert(ELEMENTS_PER_BANK != 0);
    constexpr int NDWORD = N_ELEMENTS_PER_THREAD_X / ELEMENTS_PER_BANK;
    static_assert(NDWORD != 0);

    using shm_vec_t = Vec<IType, ELEMENTS_PER_BANK>;  // elementwise also should be fine
  
    // 0, 1, 2, ..., 31
    // 128 * 128 * 2 / 1024 + 128 * BLOCK_SIZE_Y * 2 / 1024
    __shared__ IType shm[GROUP_SIZE][NDWORD][GROUP_SIZE / NDWORD];
    __shared__ IType shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE];

    float amax_rowwise[N_ELEMENTS_PER_THREAD_Y] = {(float)(-INFINITY)};
    float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {(float)(-INFINITY)};

    #pragma unroll
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      // assume no multiple loads along X dimension

      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      // load input values into shared memory
      #pragma unroll
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        input_vec_t tmp_reg = static_cast<IType>(0.f);
        if ((global_row_base_id + group_inner_y_id + ii_y) < nrows) {
          tmp_reg.load_from(inp_load_ptr + (group_inner_y_id + ii_y) * ncols, 0);
        }

        #pragma unroll
        for (int ndword = 0; ndword < NDWORD; ndword++) {
          // shm_vec_t* shm_vec_ptr = reinterpret_cast<shm_vec_t*>([group_inner_y_id][ndword]);
          #pragma unroll
          for (int elm_id = 0; elm_id < ELEMENTS_PER_BANK; elm_id++) {
            int offset = ndword * ELEMENTS_PER_BANK + elm_id;
            IType value = tmp_reg.data.elt[offset];
            shm[group_inner_y_id + ii_y][ndword][threadIdx.x * ELEMENTS_PER_BANK + elm_id] = value;
            amax_rowwise[ii_y] = fmaxf(amax_rowwise[ii_y], fabsf(value));
            amax_columnwise[offset] = fmaxf(amax_columnwise[offset], fabsf(value));
          }
        }

        amax_rowwise[ii_y] = warpReduceMax(amax_rowwise[ii_y]);
        amax_rowwise[ii_y] *= (float)(Quantized_Limits<OType>::max_norm_rcp);
      }

      // write back to scale_inv and out_c
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        out_vec_t out_tmp_reg;
        int group_inner_y_offset = group_inner_y_id + ii_y;
        if ((global_row_base_id + group_inner_y_offset) < nrows) {
          
          for (int ndword = 0; ndword < NDWORD; ndword++) {
            for (int elm_id = 0; elm_id < ELEMENTS_PER_BANK; elm_id++) {
              int offset = ndword * ELEMENTS_PER_BANK + elm_id;
              out_tmp_reg.data.elt[offset] = static_cast<OType>(float(shm[group_inner_y_offset][ndword][threadIdx.x * ELEMENTS_PER_BANK + elm_id]) / amax_rowwise[ii_y]);
            }
          }
          out_tmp_reg.store_to(out_c_store_ptr + group_inner_y_offset * ncols, 0);
          if (threadIdx.x == 0) {
            rowwise_scale_inv_ptr[group_inner_y_offset * (ncols / GROUP_SIZE)] = amax_rowwise[ii_y];
          }
        }
      }

      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = amax_columnwise[ii_x];
      }
    }

    // RUN COLUMNWISE

    __syncthreads();
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
      int col_offset = local_col_base_id + ii;
      if (threadIdx.y == 0) {
        amax_columnwise[ii] = shm_amax_columnwise[0][col_offset];
        for (int kk = 1; kk < blockDim.y; kk++) {
          amax_columnwise[ii] = 
              fmaxf(amax_columnwise[ii], shm_amax_columnwise[kk][col_offset]);
        }
        shm_amax_columnwise[0][col_offset] = amax_columnwise[ii];
      }
    }

    __syncthreads();
    #pragma unroll
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
      amax_columnwise[ii] = (float)shm_amax_columnwise[0][local_col_base_id + ii] * (float)(Quantized_Limits<OType>::max_norm_rcp);
    }

    // write back to columnwise_scale_inv and out_t
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        int group_inner_y_offset = group_inner_y_id + ii_y;
        if ((global_row_base_id + group_inner_y_offset) < nrows) {
          out_vec_t out_tmp_reg;
          scale_vec_t scale_tmp_reg;
          bool should_write_scale = threadIdx.y == 0 && ii_y == 0;
          for (int ndword = 0; ndword < NDWORD; ndword++) {
            for (int elm_id = 0; elm_id < ELEMENTS_PER_BANK; elm_id++) {
              int offset = ndword * ELEMENTS_PER_BANK + elm_id;
              float scale_inv = amax_columnwise[offset];
              float value = (float)shm[group_inner_y_offset][ndword][threadIdx.x * ELEMENTS_PER_BANK + elm_id] / scale_inv;
              out_tmp_reg.data.elt[offset] = static_cast<OType>(value);
              if (should_write_scale) {
                scale_tmp_reg.data.elt[offset] = scale_inv;
              }
            }
          }
          out_tmp_reg.store_to(out_t_store_ptr + group_inner_y_offset * ncols, 0);
          if (should_write_scale) {
            scale_tmp_reg.store_to(columnwise_scale_inv_ptr, 0);
          }
        }
      }
    }
}


// DELETE this kernel later
// template <size_t load_size, size_t store_size, typename IType, typename OType, bool Aligned = true>
// __global__ void  mtfp8_cast_transpose_general_kernel_deprecated(
//     const IType *__restrict__ const inp,
//     const CType *__restrict__ const noop,
//     OType *__restrict__ const out_c,
//     OType *__restrict__ const out_t,
//     CType *__restrict__ const scale_inv,
//     CType *__restrict__ const columnwise_scale_inv,
//     size_t ncols,
//     size_t nrows,
//     size_t rowwise_group_size) {
//   // if (noop != nullptr && noop[0] == 1.0f) return;
//   constexpr int N_WARPS_X_PER_BLOCK = 1;
//   constexpr int N_WARPS_Y_PER_BLOCK = 16;
//   constexpr int N_ELEMENTS_PER_THREAD_X = 4;
//   constexpr int N_ELEMENTS_PER_THREAD_Y = 4;
//   constexpr int GROUP_SIZE = 128;

//   // static_assert(N_ELEMENTS_PER_THREAD_X * 32 >= GROUP_SIZE);  // large group size may fail

//   using inp_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
//   using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
//   using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;

//   const uint32_t local_col_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;

//   const uint32_t col_id = blockIdx.x * GROUP_SIZE + local_col_id;
//   const uint32_t row_start_id = blockIdx.y * GROUP_SIZE;

//   IType input_regs[N_ELEMENTS_PER_THREAD_Y * N_ELEMENTS_PER_THREAD_X];
//   float maxval_or_scale_rowwise[N_ELEMENTS_PER_THREAD_Y] = {(float)(-INFINITY)};
//   float maxval_or_scale_columnwise[N_ELEMENTS_PER_THREAD_X] = {(float)(-INFINITY)};

//   const IType* inp_load_ptr = inp + row_start_id * ncols + col_id;
//   OType* out_c_store_ptr = out_c + row_start_id * ncols + col_id;
//   CType* scale_store_ptr = scale_inv + row_start_id * (ncols / GROUP_SIZE) + blockIdx.x;
//   CType* columnwise_scale_inv_ptr = columnwise_scale_inv + blockIdx.y * ncols + col_id;
//   OType* out_t_store_ptr = out_t + row_start_id * ncols + col_id;
//   inp_vec_t* input_regs_vec_ptr = reinterpret_cast<inp_vec_t*>(&input_regs);

//   //__shared__ float shm[GROUP_SIZE][GROUP_SIZE];
//   __shared__ IType shm[GROUP_SIZE][GROUP_SIZE];
//   //__shared__ float shm_maxval_columnwise[N_WARPS_Y_PER_BLOCK][GROUP_SIZE];
//   __shared__ IType shm_maxval_columnwise[N_WARPS_Y_PER_BLOCK][GROUP_SIZE];

//   using shm_vec_t = Vec<float, N_ELEMENTS_PER_THREAD_X>;
  
//   for (int group_inner_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y; group_inner_id < GROUP_SIZE; group_inner_id += (blockDim.y * N_ELEMENTS_PER_THREAD_Y)) {
//     #pragma unroll
//     for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
//       // un-aligned cases are not taken into account NOW !!!
//       int offset = group_inner_id + ii_y;
//       (input_regs_vec_ptr + ii_y)->load_from(inp_load_ptr + offset * ncols, 0);
//     }

//     // ROW-WISE blockwise quantization (along column)
//     #pragma unroll
//     for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
//       maxval_or_scale_rowwise[ii_y] = (float)(-INFINITY);
//       #pragma unroll
//       for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
//         maxval_or_scale_rowwise[ii_y] = fmaxf(maxval_or_scale_rowwise[ii_y], fabsf(input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x]));
//       }
//       maxval_or_scale_rowwise[ii_y] = warpReduceMax(maxval_or_scale_rowwise[ii_y]);
//       maxval_or_scale_rowwise[ii_y] *= (float)(Quantized_Limits<OType>::max_norm_rcp);  // E4M3
//     }

//     // write back to scale_inv and out_c
//     for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
//       int offset = group_inner_id + ii_y;
//       out_vec_t out_tmp_reg;
//       for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
//         out_tmp_reg.data.elt[ii_x] = static_cast<OType>((float)(input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x]) / maxval_or_scale_rowwise[ii_y]);
//       }

//       out_tmp_reg.store_to((out_c_store_ptr + offset * ncols), 0);
//       if (threadIdx.x == 0) {
//         *(scale_store_ptr + offset * ncols / GROUP_SIZE) = maxval_or_scale_rowwise[ii_y];
//       }
//     }

//     for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
//       int offset = group_inner_id + ii_y;
//       //reinterpret_cast<shm_vec_t*>(shm[offset] + local_col_id)->load_from(input_regs + ii_y * N_ELEMENTS_PER_THREAD_X, 0);
//       for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
//         shm[offset][local_col_id + ii_x] = input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x];
//         maxval_or_scale_columnwise[ii_x] = fmaxf(maxval_or_scale_columnwise[ii_x], fabsf(input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x]));
//       }
//     }

//     // write local max value into shared memory
//     // shm_maxval_columnwise
//     for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
//       shm_maxval_columnwise[threadIdx.y][local_col_id + ii] = maxval_or_scale_columnwise[ii];
//     }
//   }

//   __syncthreads();

//   // COLUMN-WISE blockwise quantization (along row)
//   for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
//     for (int stride = blockDim.y >> 1; stride > 0; stride >>= 1) {
//       if (threadIdx.y < stride) {
//         maxval_or_scale_columnwise[ii] = fmaxf(shm_maxval_columnwise[threadIdx.y][local_col_id + ii], shm_maxval_columnwise[threadIdx.y + stride][threadIdx.x * N_ELEMENTS_PER_THREAD_X + ii]);
//         shm_maxval_columnwise[threadIdx.y][local_col_id + ii] = maxval_or_scale_columnwise[ii];
//       }
//       __syncthreads();
//     }
//     maxval_or_scale_columnwise[ii] = (float)shm_maxval_columnwise[0][local_col_id + ii];
//   }

//   // write back to columnwise_scale_inv and out_t
//   for (int group_inner_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y; group_inner_id < GROUP_SIZE; group_inner_id += (blockDim.y * N_ELEMENTS_PER_THREAD_Y)) {
//     for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
//       out_vec_t out_tmp_reg;
//       scale_vec_t scale_tmp_reg;
//       bool should_write_scale = threadIdx.y == 0 && ii_y == 0;
//       for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
//         float value = shm[group_inner_id + ii_y][local_col_id + ii_x];
//         float scale_inv = maxval_or_scale_columnwise[ii_x] * (float)(Quantized_Limits<OType>::max_norm_rcp);
//         out_tmp_reg.data.elt[ii_x] = static_cast<OType>(value / scale_inv);
//         if (should_write_scale) {
//           scale_tmp_reg.data.elt[ii_x] = scale_inv;
//         }
//       }
//       out_tmp_reg.store_to(out_t_store_ptr + (group_inner_id + ii_y) * ncols, 0);
//       if (should_write_scale) {
//         scale_tmp_reg.store_to(columnwise_scale_inv_ptr, 0);
//       }
//     }
//   }
// }


template <size_t load_size, size_t store_size, typename IType, typename OType>
__global__ void  mtfp8_cast_transpose_general_kernel_aligned(
    const IType *__restrict__ const inp,
    const CType *__restrict__ const noop,
    OType *__restrict__ const out_c,
    OType *__restrict__ const out_t,
    CType *__restrict__ const scale_inv,
    CType *__restrict__ const columnwise_scale_inv,
    size_t ncols,
    size_t nrows,
    size_t rowwise_group_size) {
  // if (noop != nullptr && noop[0] == 1.0f) return;
  constexpr int N_WARPS_X_PER_BLOCK = 1;
  constexpr int N_WARPS_Y_PER_BLOCK = 16;
  constexpr int N_ELEMENTS_PER_THREAD_X = 4;
  constexpr int N_ELEMENTS_PER_THREAD_Y = 4;
  constexpr int GROUP_SIZE = 128;

  // static_assert(N_ELEMENTS_PER_THREAD_X * 32 >= GROUP_SIZE);  // large group size may fail

  using inp_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
  using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
  using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;

  const uint32_t local_col_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;

  const uint32_t col_id = blockIdx.x * GROUP_SIZE + local_col_id;
  const uint32_t row_start_id = blockIdx.y * GROUP_SIZE;

  IType input_regs[N_ELEMENTS_PER_THREAD_Y * N_ELEMENTS_PER_THREAD_X];
  float maxval_or_scale_rowwise[N_ELEMENTS_PER_THREAD_Y] = {(float)(-INFINITY)};
  float maxval_or_scale_columnwise[N_ELEMENTS_PER_THREAD_X] = {(float)(-INFINITY)};

  const IType* inp_load_ptr = inp + row_start_id * ncols + col_id;
  OType* out_c_store_ptr = out_c + row_start_id * ncols + col_id;
  CType* scale_store_ptr = scale_inv + row_start_id * (ncols / GROUP_SIZE) + blockIdx.x;
  CType* columnwise_scale_inv_ptr = columnwise_scale_inv + blockIdx.y * ncols + col_id;
  OType* out_t_store_ptr = out_t + row_start_id * ncols + col_id;
  inp_vec_t* input_regs_vec_ptr = reinterpret_cast<inp_vec_t*>(&input_regs);

  //__shared__ float shm[GROUP_SIZE][GROUP_SIZE];
  constexpr int shm_size1 = GROUP_SIZE / (128 / sizeof(IType));
  constexpr int shm_size2 = 128 / sizeof(IType);
  // __shared__ IType shm[GROUP_SIZE][GROUP_SIZE];
  __shared__ IType shm[GROUP_SIZE][shm_size1][shm_size2];
  //__shared__ float shm_maxval_columnwise[N_WARPS_Y_PER_BLOCK][GROUP_SIZE];
  __shared__ IType shm_maxval_columnwise[N_WARPS_Y_PER_BLOCK][GROUP_SIZE];

  using shm_vec_t = Vec<float, N_ELEMENTS_PER_THREAD_X>;
  
  for (int group_inner_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y; group_inner_id < GROUP_SIZE; group_inner_id += (blockDim.y * N_ELEMENTS_PER_THREAD_Y)) {
    #pragma unroll
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      // un-aligned cases are not taken into account NOW !!!
      int offset = group_inner_id + ii_y;
      (input_regs_vec_ptr + ii_y)->load_from(inp_load_ptr + offset * ncols, 0);
    }

    // ROW-WISE blockwise quantization (along column)
    #pragma unroll
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      maxval_or_scale_rowwise[ii_y] = (float)(-INFINITY);
      #pragma unroll
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        maxval_or_scale_rowwise[ii_y] = fmaxf(maxval_or_scale_rowwise[ii_y], fabsf(input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x]));
      }
      maxval_or_scale_rowwise[ii_y] = warpReduceMax(maxval_or_scale_rowwise[ii_y]);
      maxval_or_scale_rowwise[ii_y] *= (float)(Quantized_Limits<OType>::max_norm_rcp);  // E4M3
    }

    // write back to scale_inv and out_c
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int offset = group_inner_id + ii_y;
      out_vec_t out_tmp_reg;
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        out_tmp_reg.data.elt[ii_x] = static_cast<OType>((float)(input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x]) / maxval_or_scale_rowwise[ii_y]);
      }

      out_tmp_reg.store_to((out_c_store_ptr + offset * ncols), 0);
      if (threadIdx.x == 0) {
        *(scale_store_ptr + offset * ncols / GROUP_SIZE) = maxval_or_scale_rowwise[ii_y];
      }
    }

    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int offset = group_inner_id + ii_y;
      //reinterpret_cast<shm_vec_t*>(shm[offset] + local_col_id)->load_from(input_regs + ii_y * N_ELEMENTS_PER_THREAD_X, 0);
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        // shm[offset][local_col_id + ii_x] = input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x];
        shm[offset][ii_x / 2][ii_x % 2 + threadIdx.x * 2] = input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x];
        // shm[offset][local_col_id + ii_x] = input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x];
        maxval_or_scale_columnwise[ii_x] = fmaxf(maxval_or_scale_columnwise[ii_x], fabsf(input_regs[ii_y * N_ELEMENTS_PER_THREAD_X + ii_x]));
      }
    }

    // write local max value into shared memory
    // shm_maxval_columnwise
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
      //shm_maxval_columnwise[threadIdx.y][local_col_id + ii] = (float)maxval_or_scale_columnwise[ii];
      shm_maxval_columnwise[threadIdx.y][local_col_id + ii] = maxval_or_scale_columnwise[ii];
      // reset for next loop
      //maxval_or_scale_columnwise[ii] = (float)(-INFINITY);
    }
  }

  __syncthreads();

  // COLUMN-WISE blockwise quantization (along row)
  for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
    // for (int stride = blockDim.y >> 1; stride > 0; stride >>= 1) {
    //   if (threadIdx.y < stride) {
    //     maxval_or_scale_columnwise[ii] = fmaxf(shm_maxval_columnwise[threadIdx.y][local_col_id + ii], shm_maxval_columnwise[threadIdx.y + stride][threadIdx.x * N_ELEMENTS_PER_THREAD_X + ii]);
    //     shm_maxval_columnwise[threadIdx.y][local_col_id + ii] = maxval_or_scale_columnwise[ii];
    //   }
    //   __syncthreads();
    // }
    // maxval_or_scale_columnwise[ii] = (float)shm_maxval_columnwise[0][local_col_id + ii];

    if (threadIdx.y == 0) {
      maxval_or_scale_columnwise[ii] = shm_maxval_columnwise[0][local_col_id + ii];
      for (int kk = 1; kk < blockDim.y; kk++) {
        maxval_or_scale_columnwise[ii] = fmaxf(maxval_or_scale_columnwise[ii], shm_maxval_columnwise[kk][local_col_id + ii]);
      }
      shm_maxval_columnwise[0][local_col_id + ii] = maxval_or_scale_columnwise[ii];
    }
  }

  __syncthreads();
  for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
    maxval_or_scale_columnwise[ii] = (float)shm_maxval_columnwise[0][local_col_id + ii];
  }

  // write back to columnwise_scale_inv and out_t
  for (int group_inner_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y; group_inner_id < GROUP_SIZE; group_inner_id += (blockDim.y * N_ELEMENTS_PER_THREAD_Y)) {
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      out_vec_t out_tmp_reg;
      scale_vec_t scale_tmp_reg;
      bool should_write_scale = threadIdx.y == 0 && ii_y == 0;
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        // float value = shm[group_inner_id + ii_y][local_col_id + ii_x];
        float value = shm[group_inner_id + ii_y][ii_x / 2][ii_x % 2 + threadIdx.x * 2];
        float scale_inv = maxval_or_scale_columnwise[ii_x] * (float)(Quantized_Limits<OType>::max_norm_rcp);
        out_tmp_reg.data.elt[ii_x] = static_cast<OType>(value / scale_inv);
        if (should_write_scale) {
          scale_tmp_reg.data.elt[ii_x] = scale_inv;
        }
      }
      out_tmp_reg.store_to(out_t_store_ptr + (group_inner_id + ii_y) * ncols, 0);
      if (should_write_scale) {
        scale_tmp_reg.store_to(columnwise_scale_inv_ptr, 0);
      }
    }
  }
}

} // namespace mtfp8

void mtfp8_cast_transpose(const Tensor* input, const Tensor* noop, Tensor* output, musaStream_t stream) {
  using namespace mtfp8;
  CheckNoopTensor(*noop, "mtfp8_cast_transpose_noop");
  CheckInputTensor(*input, "mtfp8_cast_transpose_input");
  CheckOutputTensor(*output, "mtfp8_cast_transpose_output");

  // Check that inputs and outputs are available
  NVTE_CHECK(input->has_data(), "Input is not allocated");
  NVTE_CHECK(output->has_data(), "Output rowwise data is not allocated");
  NVTE_CHECK(output->has_columnwise_data(), "Output columnwise is not allocated");

  // Flatten tensor to 2D
  NVTE_CHECK(input->data.shape == output->data.shape,
             "Input and output shapes do not match (input=", input->data.shape,
             ", output=", output->data.shape);
  const size_t row_length = input->flat_last_dim();
  const size_t num_rows = input->flat_first_dim();
  NVTE_CHECK(output->flat_first_dim() == num_rows && output->flat_last_dim() == row_length,
             "Invalid output dimensions (expected ", std::vector<size_t>{num_rows, row_length},
             ", got ", std::vector<size_t>{output->flat_first_dim(), output->flat_last_dim()}, ")");
  
  const auto rowwise_sinv_m = output->scale_inv.shape[0];
  const auto rowwise_sinv_n = output->scale_inv.shape[1];
  const auto columnwise_sinv_m = output->columnwise_scale_inv.shape[0];
  const auto columnwise_sinv_n = output->columnwise_scale_inv.shape[1];

  // Assume block size is [1, N] padded.
  NVTE_CHECK((rowwise_sinv_m == num_rows) && (row_length % rowwise_sinv_n == 0));
  NVTE_CHECK(columnwise_sinv_n == row_length);  // allow unaligned case in batch dimension
  const size_t group_size = (row_length / rowwise_sinv_n);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->data.dtype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OutputType,
        
        constexpr int GROUP_SIZE = 128;  // TODO: extend other group_size
        NVTE_CHECK(group_size == GROUP_SIZE);
        constexpr int BLOCK_SIZE_Y = 16;
        constexpr int BLOCK_SIZE_X = 32;
        constexpr int N_ELEMENTS_PER_THREAD_X = std::min(GROUP_SIZE / BLOCK_SIZE_X, 8);
        constexpr int N_ELEMENTS_PER_THREAD_Y = 4;

        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 grid(DIVUP(row_length, group_size), DIVUP(num_rows, group_size));

          mtfp8_cast_transpose_general_kernel<InputType, OutputType, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<const InputType*>(input->data.dptr),
                reinterpret_cast<const CType*>(noop->data.dptr),
                reinterpret_cast<OutputType*>(output->data.dptr),
                reinterpret_cast<OutputType*>(output->columnwise_data.dptr),
                reinterpret_cast<CType*>(output->scale_inv.dptr),
                reinterpret_cast<CType*>(output->columnwise_scale_inv.dptr),
                row_length,
                num_rows);
      );
  );
}

}  // namespace transformer_engine
