#include "mtfp8_cast_transpose.h"

#include <musa_runtime.h>

#include "../util/string.h"
#include "../utils.muh"
#include "mtfp8_utils.muh"


namespace transformer_engine {

namespace mtfp8 {

using CType = float;
constexpr size_t warps_per_tile = 4;
constexpr size_t block_size = warp_size * warps_per_tile;

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
__global__ void  mtfp8_cast_transpose_general_kernel_column_aligned(
    const IType *__restrict__ const inp,
    const CType *__restrict__ const noop,
    OType *__restrict__ const out_c,
    OType *__restrict__ const out_t,
    CType *__restrict__ const scale_inv,
    CType *__restrict__ const columnwise_scale_inv,
    size_t ncols,
    size_t nrows) {
    // rowwise_group_size and columnwise_group_size should be equal

    if (noop != nullptr && noop[0] == 1.0f) return;

    using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
    using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
    using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;

    const uint32_t local_col_base_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
    const uint32_t global_col_base_id = blockIdx.x * GROUP_SIZE + local_col_base_id;
    const uint32_t local_row_base_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
    const uint32_t global_row_base_id = blockIdx.y * GROUP_SIZE;

    const uint32_t rowwise_scale_inv_stride = ncols / GROUP_SIZE;

    // if ((global_row_base_id + local_row_base_id) >= nrows) {
    //   return;
    // }

    const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
    OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
    CType* rowwise_scale_inv_ptr = scale_inv + global_row_base_id * rowwise_scale_inv_stride + blockIdx.x;
    OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
    CType* columnwise_scale_inv_ptr = columnwise_scale_inv + blockIdx.y * ncols + global_col_base_id;
    
    constexpr int REPEAT_Y = DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);
    constexpr int ELEMENTS_PER_BANK = 4 / sizeof(IType);  // dword of bank is 32 bits by default
    static_assert(ELEMENTS_PER_BANK != 0);
    constexpr int NDWORD = N_ELEMENTS_PER_THREAD_X / ELEMENTS_PER_BANK;
    static_assert(NDWORD != 0);

    // 0, 1, 2, ..., 31
    // 128 * 128 * 2 / 1024 + 128 * BLOCK_SIZE_Y * 2 / 1024
    // __shared__ IType shm[GROUP_SIZE][NDWORD][GROUP_SIZE / NDWORD];
    __shared__ IType shm[GROUP_SIZE][GROUP_SIZE];
    __shared__ IType shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE + 2];

    float amax_rowwise;
    float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {0.f};

    input_vec_t tmp_load_reg;
    out_vec_t tmp_store_reg;
    scale_vec_t scale_store_reg;

    #pragma unroll
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      // assume no multiple loads along X dimension

      // TODO: try prefetch

      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      // load input values into shared memory
      #pragma unroll
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        amax_rowwise = 0.f;
        int ld_st_offset = global_row_base_id + group_inner_y_id + ii_y < nrows ?
                           group_inner_y_id + ii_y:
                           0;
        *reinterpret_cast<input_vec_t*>(shm[group_inner_y_id + ii_y] + local_col_base_id) = *reinterpret_cast<const input_vec_t*>(inp_load_ptr + ld_st_offset * ncols);
        tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);
        
        #pragma unroll
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
          amax_rowwise = fmaxf(fmaxf(amax_rowwise, fabsf(tmp_load_reg.data.elt[ii_x])), global_amax_min);
          amax_columnwise[ii_x] = fmaxf(fmaxf(amax_columnwise[ii_x], fabsf(tmp_load_reg.data.elt[ii_x])), global_amax_min);
        }

        amax_rowwise = warpReduceMax(amax_rowwise) * (float)(Quantized_Limits<OType>::max_norm_rcp);

        //// write back to scale_inv and out_c [rowwise result]
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
            tmp_store_reg.data.elt[ii_x] = static_cast<OType>(float(tmp_load_reg.data.elt[ii_x]) / amax_rowwise);
        }
        tmp_store_reg.store_to(out_c_store_ptr + ld_st_offset * ncols, 0);
        if (threadIdx.x == 0) {
          rowwise_scale_inv_ptr[ld_st_offset * rowwise_scale_inv_stride] = amax_rowwise;
        }
      }

      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = amax_columnwise[ii_x];
      }
    }

    // RUN COLUMNWISE

    __syncthreads_lm();

    for (int i = threadIdx.y; i < GROUP_SIZE; i += blockDim.y) {
        IType amax = threadIdx.x < blockDim.y ? 
                     shm_amax_columnwise[threadIdx.x][i] :
                     (IType)0.f;
        amax = warpReduceMax((float)amax);
        if (threadIdx.x == 0) {
          shm_amax_columnwise[0][i] = amax;
        }
    }

    __syncthreads_lm();
    #pragma unroll
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
      amax_columnwise[ii] = (float)shm_amax_columnwise[0][local_col_base_id + ii] * (float)(Quantized_Limits<OType>::max_norm_rcp);
    }

    // write back to columnwise_scale_inv and out_t
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        int group_inner_y_offset = group_inner_y_id + ii_y;
        int store_offset = (global_row_base_id + group_inner_y_offset) < nrows ?
                            group_inner_y_offset :
                            0;
        
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
          float value = (float)shm[group_inner_y_offset][local_col_base_id + ii_x] / amax_columnwise[ii_x];
          tmp_store_reg.data.elt[ii_x] = static_cast<OType>(value);
        }
        tmp_store_reg.store_to(out_t_store_ptr + store_offset * ncols, 0);
      }
    }
    if (threadIdx.y == 0) {
      #pragma unroll
      for (int i = 0; i < N_ELEMENTS_PER_THREAD_X; i++) {
        scale_store_reg.data.elt[i] = amax_columnwise[i];
      }
      scale_store_reg.store_to(columnwise_scale_inv_ptr, 0);
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
__global__ void  mtfp8_cast_transpose_general_kernel_column_unaligned(
    const IType *__restrict__ const inp,
    const CType *__restrict__ const noop,
    OType *__restrict__ const out_c,
    OType *__restrict__ const out_t,
    CType *__restrict__ const scale_inv,
    CType *__restrict__ const columnwise_scale_inv,
    size_t ncols,
    size_t nrows) {
    // rowwise_group_size and columnwise_group_size should be equal

    // if (noop != nullptr && noop[0] == 1.0f) return;

    using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
    using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
    using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;

    const uint32_t local_col_base_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
    uint32_t global_col_base_id = blockIdx.x * GROUP_SIZE;
    global_col_base_id += ((global_col_base_id + local_col_base_id) < ncols ? local_col_base_id : 0);
    const uint32_t local_row_base_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
    const uint32_t global_row_base_id = blockIdx.y * GROUP_SIZE;

    const uint32_t rowwise_scale_inv_stride = (ncols + GROUP_SIZE - 1) / GROUP_SIZE;

    const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
    OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
    CType* rowwise_scale_inv_ptr = scale_inv + global_row_base_id * rowwise_scale_inv_stride + blockIdx.x;
    OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
    CType* columnwise_scale_inv_ptr = columnwise_scale_inv + blockIdx.y * ncols + global_col_base_id;
    
    constexpr int REPEAT_Y = DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);
    constexpr int ELEMENTS_PER_BANK = 4 / sizeof(IType);  // dword of bank is 32 bits by default
    static_assert(ELEMENTS_PER_BANK != 0);
    constexpr int NDWORD = N_ELEMENTS_PER_THREAD_X / ELEMENTS_PER_BANK;
    static_assert(NDWORD != 0);

    // 0, 1, 2, ..., 31
    // 128 * 128 * 2 / 1024 + 128 * BLOCK_SIZE_Y * 2 / 1024
    // __shared__ IType shm[GROUP_SIZE][NDWORD][GROUP_SIZE / NDWORD];
    __shared__ IType shm[GROUP_SIZE][GROUP_SIZE];
    __shared__ IType shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE + 2];

    float amax_rowwise;
    float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {0.f};

    input_vec_t tmp_load_reg;
    out_vec_t tmp_store_reg;
    scale_vec_t scale_store_reg;

    #pragma unroll
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      // assume no multiple loads along X dimension

      // TODO: try prefetch

      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      // load input values into shared memory
      #pragma unroll
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        amax_rowwise = 0.f;
        int ld_st_offset = global_row_base_id + group_inner_y_id + ii_y < nrows ?
                           group_inner_y_id + ii_y:
                           0;
        *reinterpret_cast<input_vec_t*>(shm[group_inner_y_id + ii_y] + local_col_base_id) = *reinterpret_cast<const input_vec_t*>(inp_load_ptr + ld_st_offset * ncols);
        tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);
        
        #pragma unroll
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
          amax_rowwise = fmaxf(fmaxf(amax_rowwise, fabsf(tmp_load_reg.data.elt[ii_x])), global_amax_min);
          amax_columnwise[ii_x] = fmaxf(fmaxf(amax_columnwise[ii_x], fabsf(tmp_load_reg.data.elt[ii_x])), global_amax_min);
        }

        amax_rowwise = warpReduceMax(amax_rowwise) * (float)(Quantized_Limits<OType>::max_norm_rcp);

        //// write back to scale_inv and out_c [rowwise result]
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
            tmp_store_reg.data.elt[ii_x] = static_cast<OType>(float(tmp_load_reg.data.elt[ii_x]) / amax_rowwise);
        }
        tmp_store_reg.store_to(out_c_store_ptr + ld_st_offset * ncols, 0);
        if (threadIdx.x == 0) {
          rowwise_scale_inv_ptr[ld_st_offset * rowwise_scale_inv_stride] = amax_rowwise;
        }
      }

      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = amax_columnwise[ii_x];
      }
    }

    // RUN COLUMNWISE

    __syncthreads_lm();

    for (int i = threadIdx.y; i < GROUP_SIZE; i += blockDim.y) {
        IType amax = threadIdx.x < blockDim.y ? 
                     shm_amax_columnwise[threadIdx.x][i] :
                     (IType)0.f;
        amax = warpReduceMax((float)amax);
        if (threadIdx.x == 0) {
          shm_amax_columnwise[0][i] = amax;
        }
    }

    __syncthreads_lm();
    #pragma unroll
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
      amax_columnwise[ii] = (float)shm_amax_columnwise[0][local_col_base_id + ii] * (float)(Quantized_Limits<OType>::max_norm_rcp);
    }

    // write back to columnwise_scale_inv and out_t
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        int group_inner_y_offset = group_inner_y_id + ii_y;
        int store_offset = (global_row_base_id + group_inner_y_offset) < nrows ?
                            group_inner_y_offset :
                            0;
        
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
          float value = (float)shm[group_inner_y_offset][local_col_base_id + ii_x] / amax_columnwise[ii_x];
          tmp_store_reg.data.elt[ii_x] = static_cast<OType>(value);
        }
        tmp_store_reg.store_to(out_t_store_ptr + store_offset * ncols, 0);
      }
    }
    if (threadIdx.y == 0) {
      #pragma unroll
      for (int i = 0; i < N_ELEMENTS_PER_THREAD_X; i++) {
        scale_store_reg.data.elt[i] = amax_columnwise[i];
      }
      scale_store_reg.store_to(columnwise_scale_inv_ptr, 0);
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

  const size_t group_size = next_power_of_2(row_length / rowwise_sinv_n);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->data.dtype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OutputType,
        
        constexpr int GROUP_SIZE = 128;  // TODO: extend other group_size
        NVTE_CHECK(group_size == GROUP_SIZE);
        constexpr int BLOCK_SIZE_Y = 16;
        constexpr int BLOCK_SIZE_X = 32;
        constexpr int N_ELEMENTS_PER_THREAD_X = std::min(GROUP_SIZE / BLOCK_SIZE_X, 8);
        constexpr int N_ELEMENTS_PER_THREAD_Y = 2;

        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 grid(DIVUP(row_length, group_size), DIVUP(num_rows, group_size));
        
        // std::cout << "output->scale_inv.dptr: " << reinterpret_cast<std::uintptr_t>((void*)(output->scale_inv.dptr)) << std::endl;
        if ((row_length % GROUP_SIZE) != 0) {
          mtfp8_cast_transpose_general_kernel_column_unaligned<InputType, OutputType, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<const InputType*>(input->data.dptr),
                reinterpret_cast<const CType*>(noop->data.dptr),
                reinterpret_cast<OutputType*>(output->data.dptr),
                reinterpret_cast<OutputType*>(output->columnwise_data.dptr),
                reinterpret_cast<CType*>(output->scale_inv.dptr),
                reinterpret_cast<CType*>(output->columnwise_scale_inv.dptr),
                row_length,
                num_rows);
        } else {
          mtfp8_cast_transpose_general_kernel_column_aligned<InputType, OutputType, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<const InputType*>(input->data.dptr),
                reinterpret_cast<const CType*>(noop->data.dptr),
                reinterpret_cast<OutputType*>(output->data.dptr),
                reinterpret_cast<OutputType*>(output->columnwise_data.dptr),
                reinterpret_cast<CType*>(output->scale_inv.dptr),
                reinterpret_cast<CType*>(output->columnwise_scale_inv.dptr),
                row_length,
                num_rows);
        }
      );
  );
}

}  // namespace transformer_engine
