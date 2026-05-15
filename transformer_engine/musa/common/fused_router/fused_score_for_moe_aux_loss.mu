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
#include "../utils.muh"
#include "utils.h"

namespace transformer_engine {

template <typename DataType, int kScoreFunction>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(const DataType *logits, int num_tokens,
                                                            int num_experts, int topk, DataType *scores,
                                                            bool *routing_map,
                                                            DataType *intermediate_output) {
  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     *   So DataType address is assigned firstly to avoid the alignment issue
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_scores_for_aux_loss[];
  DataType *logits_buf = reinterpret_cast<DataType *>(shmem_scores_for_aux_loss);
  int *topk_indices_buf =
      reinterpret_cast<int *>(logits_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_logits = logits_buf + warp_id * num_experts;
  int *topk_indices = topk_indices_buf + warp_id * topk;
  const bool separate_intermediate_output = intermediate_output != scores;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the logits to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // Load the logits to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_logits[i] = logits[pos_offset + i];
    }
    __syncwarp();

    /***
         * Section: Preprocess
         * Possible preprocess the scores before the topk operation
         * - Pre-softmax
         * - Sigmoid
         * - Sigmoid post-processing when topk > 1
         * This is in-place scores update
         */
    if constexpr (kScoreFunction == 1) {
      // score_function == 1 means softmax
      // Apply softmax to the logits before the topk
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      // For softmax, backward can reuse scores directly when the two outputs alias.
      if (separate_intermediate_output) {
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          intermediate_output[pos_offset + i] = local_logits[i];
        }
      }
    } else {
      // score_function == 0 means sigmoid
      // Apply sigmoid to the logits
      apply_sigmoid_on_float(local_logits, num_experts, lane_id);
      // When topk == 1, scores and intermediate_output are identical and can share storage.
      if (separate_intermediate_output) {
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          intermediate_output[pos_offset + i] = local_logits[i];
        }
      }
      if (topk > 1) {
        __syncwarp();
        float sum_logits = static_cast<float>(
            warp_reduce_on_shmem(local_logits, num_experts, ReduceFuncType::SUM, lane_id));
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          local_logits[i] = static_cast<DataType>(static_cast<float>(local_logits[i]) /
                                                  (sum_logits + epsilon));
        }
      }
    }

    // Write the scores to the output tensor before topk modifies local_logits.
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      scores[pos_offset + i] = local_logits[i];
    }

    /***
         * Section: Topk
         * Get the topk indices
         */
    naive_topk_indices_inplace(local_logits, num_experts, topk, topk_indices, lane_id);

    // Write the routing_map to the output tensor
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
    }
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    DataType *scores, bool *routing_map, DataType *intermediate_output, musaStream_t stream) {
  const size_t output_elements = static_cast<size_t>(num_tokens) * num_experts;
  NVTE_CHECK_CUDA(musaMemsetAsync(routing_map, 0, output_elements * sizeof(bool), stream));

  // Meta data for the kernel
  const int threads_per_block = get_router_threads_per_block();
  size_t num_token_per_block = threads_per_block / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // logits
                              + topk * num_token_per_block * sizeof(int);           // topk_indices
  if (score_function == 1) {
    fused_score_for_moe_aux_loss_forward_kernel<DataType, 1>
        <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
            logits, num_tokens, num_experts, topk, scores, routing_map, intermediate_output);
  } else {
    fused_score_for_moe_aux_loss_forward_kernel<DataType, 0>
        <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
            logits, num_tokens, num_experts, topk, scores, routing_map, intermediate_output);
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_score_for_moe_aux_loss_forward(const Tensor &logits, int num_tokens, int num_experts,
                                          int topk, int score_function, Tensor &scores,
                                          Tensor &routing_map, Tensor &intermediate_output,
                                          musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      fused_score_for_moe_aux_loss_forward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(scores.data.dptr),
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream););
}

template <typename DataType, int kScoreFunction>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(
    const DataType *__restrict__ intermediate_output, const DataType *__restrict__ grad_scores,
    int num_tokens, int num_experts, int topk, DataType *__restrict__ grad_logits) {
  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  DataType *grad_scores_buf = reinterpret_cast<DataType *>(shmem);
  // To store the output of softmax/sigmoid from the fwd
  DataType *act_from_fwd_buf =
      reinterpret_cast<DataType *>(grad_scores_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_grad = grad_scores_buf + warp_id * num_experts;
  DataType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the dgrad/output_from_fwd to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // Load the dgrad/output_from_fwd to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_scores[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
    }
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid Post-processing bwd when topk > 1
         * - Sigmoid bwd
         * - Write the grad_logits to the global mem
         */
    if constexpr (kScoreFunction == 0) {
      // Sigmoid post-processing bwd when topk > 1.
      if (topk > 1) {
        float local_sum_fwd_input = 0.0f;
        float local_sum_output_x_grad = 0.0f;
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          float act = static_cast<float>(local_act_from_fwd[i]);
          local_sum_fwd_input += act;
          local_sum_output_x_grad += static_cast<float>(local_grad[i]) * act;
        }
        for (int s = 16; s > 0; s /= 2) {
          local_sum_fwd_input += __shfl_xor_sync(0xffffffff, local_sum_fwd_input, s);
          local_sum_output_x_grad += __shfl_xor_sync(0xffffffff, local_sum_output_x_grad, s);
        }
        float norm_inv = 1.0f / (local_sum_fwd_input + epsilon);
        float corr = local_sum_output_x_grad * norm_inv * norm_inv;
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          local_grad[i] = static_cast<DataType>(static_cast<float>(local_grad[i]) * norm_inv - corr);
        }
      }
      // Sigmoid bwd.
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
    } else {
      // Pre-softmax bwd.
      float local_sum_output_x_grad = 0.0f;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_sum_output_x_grad +=
            static_cast<float>(local_grad[i]) * static_cast<float>(local_act_from_fwd[i]);
      }
      for (int s = 16; s > 0; s /= 2) {
        local_sum_output_x_grad += __shfl_xor_sync(0xffffffff, local_sum_output_x_grad, s);
      }
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float act = static_cast<float>(local_act_from_fwd[i]);
        float grad = static_cast<float>(local_grad[i]);
        local_grad[i] = static_cast<DataType>(act * (grad - local_sum_output_x_grad));
      }
    }
    // Write the grad_logits to the global mem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = local_grad[i];
    }
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const DataType *intermediate_output, const DataType *grad_scores, int num_tokens,
    int num_experts, int topk, int score_function, DataType *grad_logits, musaStream_t stream) {
  // Meta data for the kernel
  const int threads_per_block = get_router_threads_per_block();
  size_t num_token_per_block = threads_per_block / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // grad_scores
                              +
                              num_experts * num_token_per_block * sizeof(DataType);  // act_from_fwd
  if (score_function == 1) {
    fused_score_for_moe_aux_loss_backward_kernel<DataType, 1>
        <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
            intermediate_output, grad_scores, num_tokens, num_experts, topk, grad_logits);
  } else {
    fused_score_for_moe_aux_loss_backward_kernel<DataType, 0>
        <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
            intermediate_output, grad_scores, num_tokens, num_experts, topk, grad_logits);
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_score_for_moe_aux_loss_backward(const Tensor &intermediate_output,
                                           const Tensor &grad_scores, int num_tokens,
                                           int num_experts, int topk, int score_function,
                                           Tensor &grad_logits, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_scores.data.dtype, DataType,
      fused_score_for_moe_aux_loss_backward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_scores.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace transformer_engine

void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, const NVTETensor routing_map,
                                               const NVTETensor intermediate_output,
                                               musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_score_for_moe_aux_loss_forward(*reinterpret_cast<Tensor*>(logits), num_tokens, num_experts,
                                       topk, score_function, *reinterpret_cast<Tensor*>(scores),
                                       *reinterpret_cast<Tensor*>(routing_map),
                                       *reinterpret_cast<Tensor*>(intermediate_output), stream);
}

void nvte_fused_score_for_moe_aux_loss_backward(const NVTETensor intermediate_output,
                                                const NVTETensor grad_scores, int num_tokens,
                                                int num_experts, int topk, int score_function,
                                                NVTETensor grad_logits, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_score_for_moe_aux_loss_backward(
      *reinterpret_cast<Tensor*>(intermediate_output), *reinterpret_cast<Tensor*>(grad_scores),
      num_tokens, num_experts, topk, score_function, *reinterpret_cast<Tensor*>(grad_logits), stream);
}
