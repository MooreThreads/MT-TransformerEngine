/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <musa_runtime.h>
#include <transformer_engine/transpose.h>

#include <cfloat>
#include <iostream>
#include <vector>

#include "../common.h"
#include "../utils.muh"
#include "../util/mtfp8_utils.muh"

namespace transformer_engine {

namespace {

// Parameters to tune
constexpr int BLOCK_SIZE_Y = 32;
constexpr int BLOCK_SIZE_X = 16;
constexpr int n_warps_per_tile = 4;
constexpr int threads_per_block = THREADS_PER_WARP * n_warps_per_tile;
constexpr int desired_load_size = 8;
constexpr int desired_store_size = 8;
constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB
struct MultiCastTransposeArgs {
  // (input) Data buffers for input tensors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for cast output tensors
  void* output_c_list[kMaxTensorsPerKernel];
  // (output) Data buffers for transpose output tensors
  void* output_t_list[kMaxTensorsPerKernel];
  // (input) Scaling factor for output tensors
  // void* scale_list[kMaxTensorsPerKernel];
  // (output) AMAX's of input tensors
  // void* amax_list[kMaxTensorsPerKernel];
  // (output) Inverse of scaling factor for output tensors
  void* scale_inv_list[kMaxTensorsPerKernel];
  // (output) Inverse of columnwise scaling factor for output tensors
  void* columnwise_scale_inv_list[kMaxTensorsPerKernel];
  // Input matrix heights
  int num_rows_list[kMaxTensorsPerKernel];
  // Input matrix widths
  int row_length_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

// template <int nvec_in, int nvec_out, bool aligned, typename CType, typename IType, typename OType>
// __global__ void __launch_bounds__(threads_per_block)
//     multi_cast_transpose_kernel(MultiCastTransposeArgs args) {
//   using IVec = Vec<IType, nvec_in>;
//   using OVecC = Vec<OType, nvec_in>;
//   using OVecT = Vec<OType, nvec_out>;

//   // Thread indices
//   // Note: Block is interpreted as a warp_size x num_warps grid
//   constexpr int bdimx = THREADS_PER_WARP;
//   constexpr int bdimy = n_warps_per_tile;
//   const int tid = threadIdx.x;
//   const int tidx = tid % bdimx;
//   const int tidy = tid / bdimx;
//   const int bid = blockIdx.x;

//   // Input tensors are divided into tiles
//   // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
//   constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
//   constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

//   // Number of nvec_out x nvec_in subtiles for each thread to
//   // load/store
//   constexpr int n_iterations = THREADS_PER_WARP / n_warps_per_tile;

//   // Find tensor corresponding to block
//   int tensor_id = 0;
//   while (args.block_range[tensor_id + 1] <= bid) {
//     ++tensor_id;
//   }
//   const IType* input = reinterpret_cast<const IType*>(args.input_list[tensor_id]);
//   OType* output_c = reinterpret_cast<OType*>(args.output_c_list[tensor_id]);
//   OType* output_t = reinterpret_cast<OType*>(args.output_t_list[tensor_id]);
//   const CType* scale_ptr = reinterpret_cast<CType*>(args.scale_list[tensor_id]);
//   const CType scale = scale_ptr == nullptr ? 1 : *scale_ptr;
//   // CType* amax_ptr = reinterpret_cast<CType*>(args.amax_list[tensor_id]);
//   CType* scale_inv_ptr = reinterpret_cast<CType*>(args.scale_inv_list[tensor_id]);
//   const int num_rows = args.num_rows_list[tensor_id];
//   const int row_length = args.row_length_list[tensor_id];

//   // Find position of tile within tensor
//   const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
//   const int tile_id = bid - args.block_range[tensor_id];
//   const int tile_id_m = tile_id / num_tiles_n;
//   const int tile_id_n = tile_id % num_tiles_n;
//   const int tile_row = tile_id_m * tile_dim_m;
//   const int tile_col = tile_id_n * tile_dim_n;

//   // Load input and store to registers
//   // Note: Each thread loads n_iterations subtiles, casts to output
//   // type, and transposes in registers.
//   OVecT local_output_t[nvec_in][n_iterations];
//   CType local_amax = 0;
// #pragma unroll
//   for (int iter = 0; iter < n_iterations; ++iter) {
//     const int i1 = tidy + iter * bdimy;
//     const int j1 = tidx;
// #pragma unroll
//     for (int i2 = 0; i2 < nvec_out; ++i2) {
//       const int row = tile_row + i1 * nvec_out + i2;
//       const int col = tile_col + j1 * nvec_in;
//       IVec local_input;
//       OVecC local_output_c;
//       if constexpr (aligned) {
//         local_input.load_from(&input[row * row_length + col]);
//       } else {
//         local_input.clear();
//         if (row < num_rows) {
// #pragma unroll
//           for (int j2 = 0; j2 < nvec_in; ++j2) {
//             if (col + j2 < row_length) {
//               local_input.data.elt[j2] = input[row * row_length + col + j2];
//             }
//           }
//         }
//       }
// #pragma unroll
//       for (int j2 = 0; j2 < nvec_in; ++j2) {
//         const CType x = CType(local_input.data.elt[j2]);
//         const OType y = OType(scale * x);
//         local_output_c.data.elt[j2] = y;
//         local_output_t[j2][iter].data.elt[i2] = y;
//         __builtin_assume(local_amax >= 0);
//         local_amax = fmaxf(fabsf(x), local_amax);
//       }
//       if constexpr (aligned) {
//         local_output_c.store_to(&output_c[row * row_length + col]);
//       } else {
//         if (row < num_rows) {
// #pragma unroll
//           for (int j2 = 0; j2 < nvec_in; ++j2) {
//             if (col + j2 < row_length) {
//               output_c[row * row_length + col + j2] = local_output_c.data.elt[j2];
//             }
//           }
//         }
//       }
//     }
//   }

//   // Copy transposed output from registers to global memory
//   __shared__ OVecT shared_output_t[THREADS_PER_WARP][THREADS_PER_WARP + 1];
// #pragma unroll
//   for (int j2 = 0; j2 < nvec_in; ++j2) {
// #pragma unroll
//     for (int iter = 0; iter < n_iterations; ++iter) {
//       const int i1 = tidy + iter * bdimy;
//       const int j1 = tidx;
//       shared_output_t[j1][i1] = local_output_t[j2][iter];
//     }
//     __syncthreads();
// #pragma unroll
//     for (int iter = 0; iter < n_iterations; ++iter) {
//       const int i1 = tidx;
//       const int j1 = tidy + iter * bdimy;
//       const int row = tile_row + i1 * nvec_out;
//       const int col = tile_col + j1 * nvec_in + j2;
//       if constexpr (aligned) {
//         shared_output_t[j1][i1].store_to(&output_t[col * num_rows + row]);
//       } else {
//         if (col < row_length) {
// #pragma unroll
//           for (int i2 = 0; i2 < nvec_out; ++i2) {
//             if (row + i2 < num_rows) {
//               output_t[col * num_rows + row + i2] = shared_output_t[j1][i1].data.elt[i2];
//             }
//           }
//         }
//       }
//     }
//     __syncthreads();
//   }

//   // Finalize fp8 factors
//   local_amax = reduce_max<n_warps_per_tile>(local_amax, tidy);
//   if (tid == 0) {
//     static_assert(std::is_same<CType, float>::value);
//     if (amax_ptr != nullptr) atomicMaxFloat(amax_ptr, local_amax);
//   }
//   if (tile_id == 0 && tid == 0 && scale_inv_ptr != nullptr) {
//     reciprocal<CType>(scale_inv_ptr, scale);
//   }
// }

}  // namespace

template <typename T, int WIDTH = 32>
__device__ __forceinline__ T warpReduceMax(T v, unsigned mask = 0xffffffffu) {
  // static_assert((WIDTH & (WIDTH - 1)) == 0, "WIDTH must be power of 2");
  // static_assert(WIDTH <= 32, "WIDTH must be <= warpSize");

  #pragma unroll
  for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
    T other = __shfl_xor_sync(mask, v, offset, WIDTH);
    v = fmaxf(v, other);
  }
  return v;
}

constexpr int max(int a, int b) {
  return a > b ? a : b;
}

template <typename IType, typename OType,
          typename CType,
          size_t N_ELEMENTS_PER_THREAD_X,
          size_t N_ELEMENTS_PER_THREAD_Y,
          size_t BLOCK_SIZE_X,
          size_t BLOCK_SIZE_Y,
          size_t GROUP_SIZE>
__device__ void
mtfp8_cast_transpose_impl(
    int bx, int by,
    const IType* __restrict__ const inp,
    const CType* __restrict__ const noop,
    OType* __restrict__ const out_c,
    OType* __restrict__ const out_t,
    CType* __restrict__ const scale_inv,
    CType* __restrict__ const columnwise_scale_inv,
    size_t ncols, size_t nrows) {

  using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
  using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
  using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;
  using in_vec2 = Vec<IType, 2>;
  using f32_vec2 = Vec<float, 2>;
  using f32_vec_t = Vec<float, N_ELEMENTS_PER_THREAD_X>;

  const uint32_t local_col_base_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
  const uint32_t global_col_base_id = bx * GROUP_SIZE + local_col_base_id;
  const uint32_t local_row_base_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
  const uint32_t global_row_base_id = by * GROUP_SIZE;

  const uint32_t rowwise_scale_inv_stride = ncols / GROUP_SIZE;

  const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
  OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
  CType* rowwise_scale_inv_ptr =
      scale_inv + global_row_base_id * rowwise_scale_inv_stride + bx;
  OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
  CType* columnwise_scale_inv_ptr = columnwise_scale_inv + by * ncols + global_col_base_id;

  const uint32_t rows_in_this_block = min(GROUP_SIZE, nrows - global_row_base_id);
  constexpr int REPEAT_Y =
      DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);  // 128 / (16 * 2) = 4

  __shared__ __align__(128) IType shm[GROUP_SIZE][GROUP_SIZE];
  __shared__ IType shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE + 2];
  __shared__ float shm_rcp_amax_col_final[GROUP_SIZE];

  int local_tidx = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = local_tidx >> 5;
  int lane_id = local_tidx & 31;

  float amax_rowwise;
  float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {0.f};

  #pragma unroll
  for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ++ii_x) {
    shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = 0.0f;
    amax_columnwise[ii_x] = 0.0f;
  }
  __syncthreads_lm();

  input_vec_t tmp_load_reg;
  out_vec_t tmp_store_reg;
  scale_vec_t scale_store_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    // TODO: try prefetch

    int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
// load input values into shared memory
#pragma unroll
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      amax_rowwise = 0.f;
      int ld_st_offset = group_inner_y_id + ii_y;
      if (ld_st_offset >= rows_in_this_block) {
         break;
      }
      *reinterpret_cast<input_vec_t*>(shm[group_inner_y_id + ii_y] + local_col_base_id) =
          *reinterpret_cast<const input_vec_t*>(inp_load_ptr + ld_st_offset * ncols);
      tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);

#pragma unroll
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        float abs_val = fabsf(tmp_load_reg.data.elt[ii_x]);
        float val_with_min = fmaxf(abs_val, global_amax_min);
        amax_rowwise = fmaxf(amax_rowwise, val_with_min);
        amax_columnwise[ii_x] = fmaxf(amax_columnwise[ii_x], val_with_min);
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = amax_columnwise[ii_x];
      }

      amax_rowwise = warpReduceMax<float, 16>(amax_rowwise) * (float)(Quantized_Limits<fp8e4m3>::max_norm_rcp);
      const float rcp_amax_rowwise =
          1.0f / amax_rowwise;  //(amax_rowwise > 0.f) ? (1.0f / amax_rowwise) : 0.f;
                                //// write back to scale_inv and out_c [rowwise result]
#pragma unroll
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        tmp_store_reg.data.elt[ii_x] =
            (OType)(float(tmp_load_reg.data.elt[ii_x]) * rcp_amax_rowwise);
      }
      tmp_store_reg.store_to(out_c_store_ptr + ld_st_offset * ncols, 0);
      if (threadIdx.x == 0) {
        rowwise_scale_inv_ptr[ld_st_offset * rowwise_scale_inv_stride] = amax_rowwise;
      }
    }
  }

  // RUN COLUMNWISE
  __syncthreads_lm();

  in_vec2 tid_amax_vec2;
  in_vec2 amax_col_vec2;
  int base_col = warp_id << 3;
  int col_pair = lane_id >> 3;
  int row_base = lane_id & 7;
  int col_idx = base_col + col_pair * 2;
  amax_col_vec2.load_from(&(shm_amax_columnwise[row_base][col_idx]));
  tid_amax_vec2 = amax_col_vec2;
#pragma unroll
  for (int k = 1; k < 4; k++) {
    int row_idx = row_base + k * 8;
    amax_col_vec2.load_from(&(shm_amax_columnwise[row_idx][col_idx]));
    tid_amax_vec2.data.elt[0] = fmaxf(tid_amax_vec2.data.elt[0], amax_col_vec2.data.elt[0]);
    tid_amax_vec2.data.elt[1] = fmaxf(tid_amax_vec2.data.elt[1], amax_col_vec2.data.elt[1]);
  }

  in_vec2 reduce_amax_col = tid_amax_vec2;
  for (int offset = 4; offset >= 1; offset /= 2) {
    reduce_amax_col.data.elt[0] =
        fmaxf(reduce_amax_col.data.elt[0],
              __shfl_down_sync(0xffffffff, reduce_amax_col.data.elt[0], offset));
    reduce_amax_col.data.elt[1] =
        fmaxf(reduce_amax_col.data.elt[1],
              __shfl_down_sync(0xffffffff, reduce_amax_col.data.elt[1], offset));
  }

  if (row_base == 0) {
    reduce_amax_col.store_to(&(shm_amax_columnwise[0][col_idx]));
  }
  __syncthreads_lm();



  input_vec_t amax_columnwise_vec_reg;
  amax_columnwise_vec_reg.load_from(&(shm_amax_columnwise[0][local_col_base_id]));
  f32_vec_t rcp_amax_col_final_vec_reg;
  if (threadIdx.y == 0) {
#pragma unroll
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ++ii) {
      amax_columnwise[ii] = (float)(amax_columnwise_vec_reg.data.elt[ii]) *
                            (Quantized_Limits<fp8e4m3>::max_norm_rcp);
      shm_rcp_amax_col_final[local_col_base_id + ii] = 1.0f / amax_columnwise[ii];
      columnwise_scale_inv_ptr[ii] = amax_columnwise[ii];
    }
  }
  __syncthreads_lm();

  // write back to columnwise_scale_inv and out_t
  scale_vec_t rcp_amax_columnwise;
  rcp_amax_columnwise.load_from(&(shm_rcp_amax_col_final[local_col_base_id]));

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
#pragma unroll
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int group_inner_y_offset = group_inner_y_id + ii_y;
      //   int store_offset =
    //       (global_row_base_id + group_inner_y_offset) < nrows ? group_inner_y_offset : 0;
      if (group_inner_y_offset < rows_in_this_block) {
         tmp_load_reg.load_from(&(shm[group_inner_y_offset][local_col_base_id]));
#pragma unroll
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        
        float value = (float)(tmp_load_reg.data.elt[ii_x]) * rcp_amax_columnwise.data.elt[ii_x];
        tmp_store_reg.data.elt[ii_x] = (OType)(value);
      }
      tmp_store_reg.store_to(out_t_store_ptr + group_inner_y_offset * ncols, 0);
      }
    }
  }
}

template <typename IType, typename OType, typename CType, size_t N_ELEMENTS_PER_THREAD_X = 8 /* VLEN */,
          size_t N_ELEMENTS_PER_THREAD_Y = 4, size_t BLOCK_SIZE_X = 16, size_t BLOCK_SIZE_Y = 32,
          size_t GROUP_SIZE = 128>
__device__ void mtfp8_rowwise_cast_impl(
    int bx, int by,
    const IType* __restrict__ const inp,
    const CType* __restrict__ const noop,
    OType* __restrict__ const out_c,  
    CType* __restrict__ const scale_inv, 
    size_t ncols, size_t nrows) {
  // if (noop != nullptr && noop[0] == 1.0f) return;

  using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
  using out_vec_t   = Vec<OType, N_ELEMENTS_PER_THREAD_X>;

  const uint32_t local_col_base_id  = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
  const uint32_t global_col_base_id = bx * GROUP_SIZE + local_col_base_id;

  const uint32_t local_row_base_id  = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
  const uint32_t global_row_base_id = by * GROUP_SIZE;

  const uint32_t rowwise_scale_inv_stride = ncols / GROUP_SIZE;
  
  const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
  OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
  CType* rowwise_scale_inv_ptr =
      scale_inv + global_row_base_id * rowwise_scale_inv_stride + bx;

  const uint32_t rows_in_this_block = min((uint32_t)GROUP_SIZE, (uint32_t)(nrows - global_row_base_id));
  constexpr int REPEAT_Y =
      DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);

  // Input tile staged into shared memory via TME
  __shared__ __align__(128) IType shm[GROUP_SIZE][GROUP_SIZE];

  int local_tidx = threadIdx.y * bx + threadIdx.x;
  int warp_id    = local_tidx >> 5;

  input_vec_t tmp_load_reg;
  out_vec_t   tmp_store_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id =
        loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;

#pragma unroll
    for (int ii_y = 0; ii_y < (int)N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      float amax_rowwise = 0.f;

      int ld_st_offset = group_inner_y_id + ii_y;
      if (ld_st_offset >= (int)rows_in_this_block) {
        break;
      }

      *reinterpret_cast<input_vec_t*>(shm[group_inner_y_id + ii_y] + local_col_base_id) =
          *reinterpret_cast<const input_vec_t*>(inp_load_ptr + ld_st_offset * ncols);
      tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);

#pragma unroll
      for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
        float abs_val     = fabsf((float)tmp_load_reg.data.elt[ii_x]);
        float val_with_min = fmaxf(abs_val, global_amax_min);
        amax_rowwise = fmaxf(amax_rowwise, val_with_min);
      }

      // warp reduce (assuming warpReduceMax<float, 16> matches your original implementation)
      amax_rowwise =
          warpReduceMax<float, 16>(amax_rowwise) *
          (float)(Quantized_Limits<fp8e4m3>::max_norm_rcp);

      const float rcp_amax_rowwise = 1.0f / amax_rowwise;

#pragma unroll
      for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
        tmp_store_reg.data.elt[ii_x] =
            (OType)((float)tmp_load_reg.data.elt[ii_x] * rcp_amax_rowwise);
      }
      tmp_store_reg.store_to(out_c_store_ptr + ld_st_offset * ncols, 0);

      if (threadIdx.x == 0) {
        rowwise_scale_inv_ptr[ld_st_offset * rowwise_scale_inv_stride] = (CType)amax_rowwise;
      }
    }
  }
}

template <typename IType, typename OType, typename CType, size_t N_ELEMENTS_PER_THREAD_X = 8 /* VLEN */,
          size_t N_ELEMENTS_PER_THREAD_Y = 4, size_t BLOCK_SIZE_X = 16, size_t BLOCK_SIZE_Y = 32,
          size_t GROUP_SIZE = 128>
__device__ void mtfp8_columnwise_cast_impl(
    int bx, int by,
    const IType* __restrict__ const inp,
    const CType* __restrict__ const noop,
    OType* __restrict__ const out_t,   // [nrows, ncols]
    CType* __restrict__ const columnwise_scale_inv,  // [nrows / GROUP_SIZE, ncols]
    size_t ncols, size_t nrows) {
  // if (noop != nullptr && noop[0] == 1.0f) return;

  using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
  using out_vec_t   = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
  using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;
  using in_vec2     = Vec<IType, 2>;
  using f32_vec_t   = Vec<float, N_ELEMENTS_PER_THREAD_X>;

  const uint32_t local_col_base_id  = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
  const uint32_t global_col_base_id = bx * GROUP_SIZE + local_col_base_id;

  const uint32_t local_row_base_id  = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
  const uint32_t global_row_base_id = by * GROUP_SIZE;
  
  const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
  OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
  CType* columnwise_scale_inv_ptr =
      columnwise_scale_inv + by * ncols + global_col_base_id;

  const uint32_t rows_in_this_block = min((uint32_t)GROUP_SIZE, (uint32_t)(nrows - global_row_base_id));
  constexpr int REPEAT_Y =
      DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);

  __shared__ __align__(128) IType shm[GROUP_SIZE][GROUP_SIZE];
  __shared__ IType  shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE + 2];
  __shared__ float  shm_rcp_amax_col_final[GROUP_SIZE];

  int local_tidx = threadIdx.y * bx + threadIdx.x;
  int warp_id    = local_tidx >> 5;
  int lane_id    = local_tidx & 31;

  // Build per-(threadIdx.y, col) partial columnwise max
  float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {0.f};

#pragma unroll
  for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ++ii_x) {
    shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = (IType)0;
    amax_columnwise[ii_x] = 0.0f;
  }
  __syncthreads_lm();

  input_vec_t tmp_load_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id =
        loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;

#pragma unroll
    for (int ii_y = 0; ii_y < (int)N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int ld_st_offset = group_inner_y_id + ii_y;
      if (ld_st_offset >= (int)rows_in_this_block) {
        break;
      }

      *reinterpret_cast<input_vec_t*>(shm[group_inner_y_id + ii_y] + local_col_base_id) =
          *reinterpret_cast<const input_vec_t*>(inp_load_ptr + ld_st_offset * ncols);
      tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);

#pragma unroll
      for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
        float abs_val      = fabsf((float)tmp_load_reg.data.elt[ii_x]);
        float val_with_min = fmaxf(abs_val, global_amax_min);
        amax_columnwise[ii_x] = fmaxf(amax_columnwise[ii_x], val_with_min);
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = (IType)amax_columnwise[ii_x];
      }
    }
  }
  // Reduce shm_amax_columnwise over "rows" dimension (32) to get final per-column amax
  __syncthreads_lm();

  in_vec2 tid_amax_vec2;
  in_vec2 amax_col_vec2;
  int base_col = warp_id << 3;      // warp_id * 8
  int col_pair = lane_id >> 3;      // / 8
  int row_base = lane_id & 7;       // % 8
  int col_idx  = base_col + col_pair * 2;

  amax_col_vec2.load_from(&(shm_amax_columnwise[row_base][col_idx]));
  tid_amax_vec2 = amax_col_vec2;

#pragma unroll
  for (int k = 1; k < 4; k++) {
    int row_idx = row_base + k * 8;
    amax_col_vec2.load_from(&(shm_amax_columnwise[row_idx][col_idx]));
    tid_amax_vec2.data.elt[0] = fmaxf((float)tid_amax_vec2.data.elt[0], (float)amax_col_vec2.data.elt[0]);
    tid_amax_vec2.data.elt[1] = fmaxf((float)tid_amax_vec2.data.elt[1], (float)amax_col_vec2.data.elt[1]);
  }

  in_vec2 reduce_amax_col = tid_amax_vec2;
  for (int offset = 4; offset >= 1; offset >>= 1) {
    reduce_amax_col.data.elt[0] =
        fmaxf((float)reduce_amax_col.data.elt[0],
              __shfl_down_sync(0xffffffff, (float)reduce_amax_col.data.elt[0], offset));
    reduce_amax_col.data.elt[1] =
        fmaxf((float)reduce_amax_col.data.elt[1],
              __shfl_down_sync(0xffffffff, (float)reduce_amax_col.data.elt[1], offset));
  }

  if (row_base == 0) {
    reduce_amax_col.store_to(&(shm_amax_columnwise[0][col_idx]));
  }
  __syncthreads_lm();

  // Compute (amax * max_norm_rcp), store columnwise_scale_inv, and cache rcp for output cast
  input_vec_t amax_columnwise_vec_reg;
  amax_columnwise_vec_reg.load_from(&(shm_amax_columnwise[0][local_col_base_id]));

  if (threadIdx.y == 0) {
#pragma unroll
    for (int ii = 0; ii < (int)N_ELEMENTS_PER_THREAD_X; ++ii) {
      float amax =
          (float)(amax_columnwise_vec_reg.data.elt[ii]) *
          (float)(Quantized_Limits<fp8e4m3>::max_norm_rcp);

      shm_rcp_amax_col_final[local_col_base_id + ii] = 1.0f / amax;
      columnwise_scale_inv_ptr[ii] = (CType)amax;
    }
  }
  __syncthreads_lm();

  // Cast to FP8 using columnwise scale, write out_t
  scale_vec_t rcp_amax_columnwise;
  rcp_amax_columnwise.load_from(&(shm_rcp_amax_col_final[local_col_base_id]));

  out_vec_t tmp_store_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id =
        loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;

#pragma unroll
    for (int ii_y = 0; ii_y < (int)N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int group_inner_y_offset = group_inner_y_id + ii_y;
      if (group_inner_y_offset < (int)rows_in_this_block) {
        tmp_load_reg.load_from(&(shm[group_inner_y_offset][local_col_base_id]));

#pragma unroll
        for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
          float value =
              (float)(tmp_load_reg.data.elt[ii_x]) * (float)rcp_amax_columnwise.data.elt[ii_x];
          tmp_store_reg.data.elt[ii_x] = (OType)value;
        }
        tmp_store_reg.store_to(out_t_store_ptr + group_inner_y_offset * ncols, 0);
      }
    }
  }
}

template <typename IType, 
          typename OType,
          typename CType,
          size_t N_ELEMENTS_PER_THREAD_X,
          size_t N_ELEMENTS_PER_THREAD_Y,
          size_t BLOCK_SIZE_X,
          size_t BLOCK_SIZE_Y,
          size_t GROUP_SIZE>
__global__ void group_mtfp8_cast_kernel(
    MultiCastTransposeArgs args,
    const CType* __restrict__ const noop) {

  const int bid = (int)blockIdx.x;

  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) { 
    ++tensor_id; 
  }
  const int local_bid = bid - args.block_range[tensor_id];

  const int num_rows   = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  const int grid_x = DIVUP(row_length, (int)GROUP_SIZE);
  const int bx = local_bid % grid_x;
  const int by = local_bid / grid_x;

  const IType* inp = reinterpret_cast<const IType*>(args.input_list[tensor_id]);

  OType* out_c     = reinterpret_cast<OType*>(args.output_c_list[tensor_id]);
  CType* scale_inv = reinterpret_cast<CType*>(args.scale_inv_list[tensor_id]);
  
  OType* out_t     = reinterpret_cast<OType*>(args.output_t_list[tensor_id]);
  CType* col_scale_inv = reinterpret_cast<CType*>(args.columnwise_scale_inv_list[tensor_id]);

  
  mtfp8_cast_transpose_impl<
    IType, OType, CType,
    N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y,
    BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>(
    bx, by,
    inp, noop, out_c, out_t, scale_inv, col_scale_inv,
    (size_t)row_length, (size_t)num_rows);
  
}

template <typename IType, 
          typename OType,
          typename CType,
          size_t N_ELEMENTS_PER_THREAD_X,
          size_t N_ELEMENTS_PER_THREAD_Y,
          size_t BLOCK_SIZE_X,
          size_t BLOCK_SIZE_Y,
          size_t GROUP_SIZE>
__global__ void group_mtfp8_rowwise_cast_kernel(
    MultiCastTransposeArgs args,
    const CType* __restrict__ const noop) {

  const int bid = (int)blockIdx.x;

  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) { 
    ++tensor_id; 
  }
  const int local_bid = bid - args.block_range[tensor_id];

  const int num_rows   = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  const int grid_x = DIVUP(row_length, (int)GROUP_SIZE);
  const int bx = local_bid % grid_x;
  const int by = local_bid / grid_x;

  const IType* inp = reinterpret_cast<const IType*>(args.input_list[tensor_id]);

  OType* out_c     = reinterpret_cast<OType*>(args.output_c_list[tensor_id]);
  CType* scale_inv = reinterpret_cast<CType*>(args.scale_inv_list[tensor_id]);

  mtfp8_rowwise_cast_impl<
    IType, OType, CType,
    N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y,
    BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>(
    bx, by,
    inp, noop, out_c, scale_inv,
    (size_t)row_length, (size_t)num_rows);
  
}

template <typename IType, 
          typename OType,
          typename CType,
          size_t N_ELEMENTS_PER_THREAD_X,
          size_t N_ELEMENTS_PER_THREAD_Y,
          size_t BLOCK_SIZE_X,
          size_t BLOCK_SIZE_Y,
          size_t GROUP_SIZE>
__global__ void group_mtfp8_columnwise_cast_kernel(
    MultiCastTransposeArgs args,
    const CType* __restrict__ const noop) {

  const int bid = (int)blockIdx.x;

  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) { 
    ++tensor_id; 
  }
  const int local_bid = bid - args.block_range[tensor_id];

  const int num_rows   = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  const int grid_x = DIVUP(row_length, (int)GROUP_SIZE);
  const int bx = local_bid % grid_x;
  const int by = local_bid / grid_x;

  const IType* inp = reinterpret_cast<const IType*>(args.input_list[tensor_id]);
  
  OType* out_t     = reinterpret_cast<OType*>(args.output_t_list[tensor_id]);
  CType* col_scale_inv = reinterpret_cast<CType*>(args.columnwise_scale_inv_list[tensor_id]);

  
  mtfp8_columnwise_cast_impl<
    IType, OType, CType,
    N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y,
    BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>(
    bx, by,
    inp, noop, out_t, col_scale_inv,
    (size_t)row_length, (size_t)num_rows);
  
}

void multi_cast_transpose(const std::vector<Tensor*> input_list, std::vector<Tensor*> output_list,
                          musaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_list.size() == input_list.size(),
             "Number of input and output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType itype = input_list[0]->data.dtype;
  DType otype = output_list[0]->dtype();
  bool return_rowwise = output_list[0]->has_data();
  bool return_columnwise = output_list[0]->has_columnwise_data();
  
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& output = *output_list[tensor_id];
    CheckInputTensor(input, "multi_cast_transpose_input_" + std::to_string(tensor_id));
    CheckInputTensor(output, "multi_cast_transpose_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == itype, "Input tensor types do not match.");
    NVTE_CHECK(output.data.dtype == otype, "C output tensor types do not match.");
    NVTE_CHECK(output.data.dtype == otype, "T output tensor types do not match.");
    
    if (return_rowwise) {
      NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions, but shape is ",
               input.data.shape);
      NVTE_CHECK(output.data.shape == input.data.shape, "C output tensor shape ", output.data.shape,
                  "does not match input tensor shape ", input.data.shape);
    }
    if (return_columnwise) {
      NVTE_CHECK(output.columnwise_data.shape.size() == 2, "T output tensor shape ",
               output.columnwise_data.shape, "does not match input tensor shape ",
               input.data.shape);
      NVTE_CHECK(output.columnwise_data.shape[0] == input.data.shape[0], "T output tensor shape ",
                output.columnwise_data.shape, "does not match input tensor shape ",
                input.data.shape);
      NVTE_CHECK(output.columnwise_data.shape[1] == input.data.shape[1], "T output tensor shape ",
                output.columnwise_data.shape, "does not match input tensor shape ",
                input.data.shape);
    }
  }

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int GROUP_SIZE = 128; 
  const int tile_dim_m = GROUP_SIZE; //THREADS_PER_WARP * desired_store_size / typeToSize(otype);
  const int tile_dim_n = GROUP_SIZE; //THREADS_PER_WARP * desired_load_size / typeToSize(itype);

  // Add tensors to kernel argument struct
  MultiCastTransposeArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  constexpr int N_ELEMENTS_PER_THREAD_X = std::min(GROUP_SIZE / BLOCK_SIZE_X, 8);
  constexpr int N_ELEMENTS_PER_THREAD_Y = 4;
  dim3 threads_per_block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  Tensor* noop = nullptr;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args_unaligned.num_tensors == kMaxTensorsPerKernel) {
      TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
        itype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
            otype, OutputType,
            const int n_blocks = kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors];
            if (return_rowwise && return_columnwise) {
              group_mtfp8_cast_kernel<InputType, OutputType, fp32, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
              <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args_unaligned, reinterpret_cast<const fp32*>(noop));
            } else if (return_rowwise) {
              group_mtfp8_rowwise_cast_kernel<InputType, OutputType, fp32, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
              <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args_unaligned, reinterpret_cast<const fp32*>(noop));
            } else {
              group_mtfp8_columnwise_cast_kernel<InputType, OutputType, fp32, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
              <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args_unaligned, reinterpret_cast<const fp32*>(noop));
            }
        );  // NOLINT(*)        
      );             
      kernel_args_unaligned.num_tensors = 0;
    }

    // Calculate number of thread blocks needed for tensor
    const uint64_t num_rows = input_list[tensor_id]->data.shape[0];
    const uint64_t row_length = input_list[tensor_id]->data.shape[1];
    const uint64_t num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
    const uint64_t num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const uint64_t num_tiles = num_tiles_m * num_tiles_n;

    // Figure out whether to use aligned or unaligned kernel
    // const bool aligned =
    //     ((num_tiles_m * tile_dim_m == num_rows) && (num_tiles_n * tile_dim_n == row_length));
    auto& kernel_args = kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;

    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id]->data.dptr);
    kernel_args.output_c_list[pos] = output_list[tensor_id]->data.dptr;
    kernel_args.output_t_list[pos] = output_list[tensor_id]->columnwise_data.dptr;
    kernel_args.scale_inv_list[pos] = output_list[tensor_id]->scale_inv.dptr;
    kernel_args.columnwise_scale_inv_list[pos] = output_list[tensor_id]->columnwise_scale_inv.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }
  // Launch kernel
  if (kernel_args_unaligned.num_tensors > 0) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
        itype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
            otype, OutputType,
            const int n_blocks = kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors];
            if (return_rowwise && return_columnwise) {
              group_mtfp8_cast_kernel<InputType, OutputType, fp32, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
              <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args_unaligned, reinterpret_cast<const fp32*>(noop));
            } else if (return_rowwise) {
              group_mtfp8_rowwise_cast_kernel<InputType, OutputType, fp32, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
              <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args_unaligned, reinterpret_cast<const fp32*>(noop));
            } else {
              group_mtfp8_columnwise_cast_kernel<InputType, OutputType, fp32, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
              <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args_unaligned, reinterpret_cast<const fp32*>(noop));
            }
        );  // NOLINT(*)        
    );                                // NOLINT(*)
  }
}

}  // namespace transformer_engine

void nvte_multi_cast_transpose(size_t num_tensors, const NVTETensor* input_list,
                               NVTETensor* output_list, musaStream_t stream) {
  NVTE_API_CALL(nvte_multi_cast_transpose);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, output_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(reinterpret_cast<Tensor*>(const_cast<NVTETensor&>(input_list[i])));
    output_list_.push_back(reinterpret_cast<Tensor*>(output_list[i]));
  }
  multi_cast_transpose(input_list_, output_list_, stream);
}
