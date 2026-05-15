/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <musa_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "../util/musa_runtime.h"
#include "../util/system.h"
#include "../utils.muh"
#include "common/util/musa_runtime.h"
#include "utils.h"

namespace transformer_engine {

using CompType = float;

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_forward_kernel(const DataType* probs,
                                                  const IndexType* tokens_per_expert,
                                                  int num_rows, int num_cols, float* Const_buf) {
  // Multi-block reduction to improve device utilization.
  int warp_num = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  __shared__ CompType warp_sums[kThreadsPerWarp];

  int total_elements = num_rows * num_cols;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  CompType thread_sum = 0.0f;
  for (int i = idx; i < total_elements; i += stride) {
    int col = i % num_cols;
    thread_sum +=
        static_cast<CompType>(probs[i]) * static_cast<CompType>(tokens_per_expert[col]);
  }

  for (int s = 16; s > 0; s /= 2) {
    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, s);
  }
  if (lane_id == 0) {
    warp_sums[warp_id] = thread_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    CompType block_sum = (lane_id < warp_num) ? warp_sums[lane_id] : 0.0f;
    for (int s = 16; s > 0; s /= 2) {
      block_sum += __shfl_xor_sync(0xffffffff, block_sum, s);
    }
    if (lane_id == 0) {
      atomicAdd(Const_buf, static_cast<float>(block_sum));
    }
  }
}

template <typename DataType>
__global__ void fused_moe_aux_loss_forward_finalize_kernel(int total_num_tokens, int num_experts,
                                                           int topk, float coeff, DataType* aux_loss,
                                                           float* Const_buf) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float C_coeff = (num_experts * coeff) / topk / total_num_tokens / total_num_tokens;
    float sum = Const_buf[0];
    aux_loss[0] = static_cast<DataType>(sum * C_coeff);
    // Keep backward behavior unchanged: Const_buf stores C_coeff.
    Const_buf[0] = C_coeff;
  }
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_forward_kernel_launcher(const DataType* probs,
                                                const IndexType* tokens_per_expert,
                                                int total_num_tokens, int num_experts, int num_rows,
                                                int num_cols, int topk, float coeff,
                                                DataType* aux_loss, float* Const_buf,
                                                musaStream_t stream) {
  int block_size = transformer_engine::getenv<int>("TE_ROUTER_MOE_AUX_LOSS_FWD_BLOCK_SIZE", 512);
  // block_size must be warp-aligned and within MUSA launch constraints.
  if (block_size <= 0) block_size = 256;
  block_size = (block_size / kThreadsPerWarp) * kThreadsPerWarp;
  if (block_size < kThreadsPerWarp) block_size = kThreadsPerWarp;
  if (block_size > 1024) block_size = 1024;

  int total_elements = num_rows * num_cols;
  int grid_size_needed = (total_elements + block_size - 1) / block_size;

  int waves = transformer_engine::getenv<int>("TE_ROUTER_MOE_AUX_LOSS_FWD_WAVES", 8);
  if (waves <= 0) waves = 8;
  int sm_count = 64;
  int max_grid_size = sm_count * waves;
  if (max_grid_size <= 0) max_grid_size = 1;

  int grid_size = (grid_size_needed < max_grid_size) ? grid_size_needed : max_grid_size;
  grid_size = grid_size > 0 ? grid_size : 1;

  NVTE_CHECK_CUDA(musaMemsetAsync(Const_buf, 0, sizeof(float), stream));
  fused_moe_aux_loss_forward_kernel<DataType, IndexType>
      <<<grid_size, block_size, 0, stream>>>(probs, tokens_per_expert, num_rows, num_cols,
                                             Const_buf);
  fused_moe_aux_loss_forward_finalize_kernel<DataType>
      <<<1, 1, 0, stream>>>(total_num_tokens, num_experts, topk, coeff, aux_loss, Const_buf);
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_moe_aux_loss_forward(const Tensor& probs, const Tensor& tokens_per_expert,
                                int total_num_tokens, int num_experts, int num_rows, int num_cols,
                                int topk, float coeff, Tensor& aux_loss, Tensor& Const_buf,
                                musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      probs.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_forward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<DataType*>(probs.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), total_num_tokens,
              num_experts, num_rows, num_cols, topk, coeff,
              reinterpret_cast<DataType*>(aux_loss.data.dptr),
              reinterpret_cast<float*>(Const_buf.data.dptr), stream);););
}

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_backward_kernel(const float* Const_buf,
                                                   const IndexType* tokens_per_expert, int num_rows,
                                                   int num_cols, DataType* grad_aux_loss,
                                                   DataType* grad_probs) {
  int col_start = blockIdx.x * blockDim.x + threadIdx.x;
  int col_stride = blockDim.x * gridDim.x;
  float scale = Const_buf[0] * static_cast<float>(grad_aux_loss[0]);

  for (int col = col_start; col < num_cols; col += col_stride) {
    float value = scale * static_cast<float>(tokens_per_expert[col]);
    int row_offset = blockIdx.y * num_cols + col;
    int row_stride = gridDim.y * num_cols;
    for (int row = blockIdx.y; row < num_rows; row += gridDim.y) {
      grad_probs[row_offset] = static_cast<DataType>(value);
      row_offset += row_stride;
    }
  }
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_backward_kernel_launcher(const float* Const_buf,
                                                 const IndexType* tokens_per_expert, int num_rows,
                                                 int num_cols, DataType* grad_aux_loss,
                                                 DataType* grad_probs, musaStream_t stream) {
  int block_size = transformer_engine::getenv<int>("TE_ROUTER_MOE_AUX_LOSS_BWD_BLOCK_SIZE", 512);
  if (block_size <= 0) block_size = 256;
  block_size = (block_size / kThreadsPerWarp) * kThreadsPerWarp;
  if (block_size < kThreadsPerWarp) block_size = kThreadsPerWarp;
  if (block_size > 1024) block_size = 1024;

  int grid_x_needed = (num_cols + block_size - 1) / block_size;
  if (grid_x_needed <= 0) grid_x_needed = 1;

  int waves = transformer_engine::getenv<int>("TE_ROUTER_MOE_AUX_LOSS_BWD_WAVES", 8);
  if (waves <= 0) waves = 8;
  int sm_count = 64;
  int target_blocks = sm_count * waves;
  if (target_blocks <= 0) target_blocks = 1;

  int grid_x = (grid_x_needed < target_blocks) ? grid_x_needed : target_blocks;
  if (grid_x <= 0) grid_x = 1;
  if (grid_x > 65535) grid_x = 65535;

  int grid_y = (target_blocks + grid_x - 1) / grid_x;
  if (grid_y <= 0) grid_y = 1;
  if (grid_y > num_rows) grid_y = num_rows;
  if (grid_y > 65535) grid_y = 65535;

  dim3 grid(grid_x, grid_y, 1);
  fused_moe_aux_loss_backward_kernel<DataType, IndexType><<<grid, block_size, 0, stream>>>(
      Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss, grad_probs);
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_moe_aux_loss_backward(const Tensor& Const_buf, const Tensor& tokens_per_expert,
                                 int num_rows, int num_cols, Tensor& grad_aux_loss,
                                 Tensor& grad_probs, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_aux_loss.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_backward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<float*>(Const_buf.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), num_rows, num_cols,
              reinterpret_cast<DataType*>(grad_aux_loss.data.dptr),
              reinterpret_cast<DataType*>(grad_probs.data.dptr), stream);););
}

}  // namespace transformer_engine

void nvte_fused_moe_aux_loss_forward(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                     int total_num_tokens, int num_experts, int num_rows,
                                     int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                     NVTETensor Const_buf, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_moe_aux_loss_forward(
      *reinterpret_cast<Tensor*>(probs), *reinterpret_cast<Tensor*>(tokens_per_expert), total_num_tokens,
      num_experts, num_rows, num_cols, topk, coeff, *reinterpret_cast<Tensor*>(aux_loss),
      *reinterpret_cast<Tensor*>(Const_buf), stream);
}

void nvte_fused_moe_aux_loss_backward(const NVTETensor Const_buf,
                                      const NVTETensor tokens_per_expert, int num_rows,
                                      int num_cols, NVTETensor grad_aux_loss, NVTETensor grad_probs,
                                      musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_moe_aux_loss_backward(*reinterpret_cast<Tensor*>(Const_buf),
                              *reinterpret_cast<Tensor*>(tokens_per_expert), num_rows, num_cols,
                              *reinterpret_cast<Tensor*>(grad_aux_loss),
                              *reinterpret_cast<Tensor*>(grad_probs), stream);
}
