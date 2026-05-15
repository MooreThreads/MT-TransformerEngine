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

template <typename DataType, typename BiasType, int kScoreFunction, bool kHasGroupTopk,
          bool kHasExpertBias, bool kUsePreSoftmax>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, int num_groups,
    int group_topk, float scaling_factor, const BiasType *expert_bias, DataType *probs,
    bool *routing_map, DataType *intermediate_output) {
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
  extern __shared__ float shmem[];
  int scores_stride = num_experts * num_token_per_block;
  DataType *scores_buf0 = reinterpret_cast<DataType *>(shmem);
  DataType *topk_scores_buf = scores_buf0 + scores_stride;
  DataType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if constexpr (kHasGroupTopk) {
    masked_scores_buf = reinterpret_cast<DataType *>(topk_scores_buf + topk * num_token_per_block);
    group_scores_buf =
        reinterpret_cast<DataType *>(masked_scores_buf + num_experts * num_token_per_block);
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk * num_token_per_block);
  }
  // The address of buffers on the current warp
  DataType *scores = scores_buf0 + warp_id * num_experts;
  DataType *topk_scores = topk_scores_buf + warp_id * topk;
  DataType *masked_scores = nullptr;
  DataType *group_scores = nullptr;
  if constexpr (kHasGroupTopk) {
    masked_scores = masked_scores_buf + warp_id * num_experts;
    group_scores = group_scores_buf + warp_id * num_groups;
  }
  int *topk_indices = topk_indices_buf + warp_id * topk;

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
    // Clear the output buffers for the current token in-kernel to match the official CUDA path.
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      probs[pos_offset + i] = static_cast<DataType>(0.0f);
      routing_map[pos_offset + i] = false;
      if constexpr (kScoreFunction == 1) {
        intermediate_output[pos_offset + i] = -std::numeric_limits<DataType>::infinity();
      }
    }
    // Load the logits to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      scores[i] = logits[pos_offset + i];
    }
    // If group_topk > 0, init the masked_scores to -inf
    if constexpr (kHasGroupTopk) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<DataType>::infinity();
      }
    }
    __syncwarp();

    /***
         * Section: Preprocess
         * Possible preprocess the scores before the topk operation
         * - Pre-softmax
         * - Sigmoid
         * - Expert bias
         * This is in-place scores update
         */
    if constexpr (kScoreFunction == 1) {
      // score_function == 1 means softmax
      if constexpr (kUsePreSoftmax) {
        // Apply softmax and write intermediate_output in one normalization pass.
        float max_val = static_cast<float>(
            warp_reduce_on_shmem(scores, num_experts, ReduceFuncType::MAX, lane_id));
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          scores[i] = static_cast<DataType>(expf(static_cast<float>(scores[i]) - max_val));
        }
        __syncwarp();
        float sum_val = static_cast<float>(
            warp_reduce_on_shmem(scores, num_experts, ReduceFuncType::SUM, lane_id));
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          float softmax_val = static_cast<float>(scores[i]) / sum_val;
          scores[i] = static_cast<DataType>(softmax_val);
          intermediate_output[pos_offset + i] = static_cast<DataType>(softmax_val);
        }
        __syncwarp();
      }
    } else {
      // score_function == 0 means sigmoid
      // Sigmoid + save intermediate + optional bias add in one pass.
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float sigmoid_val = 1.0f / (1.0f + expf(-static_cast<float>(scores[i])));
        intermediate_output[pos_offset + i] = static_cast<DataType>(sigmoid_val);
        if constexpr (kHasExpertBias) {
          sigmoid_val += static_cast<float>(expert_bias[i]);
        }
        scores[i] = static_cast<DataType>(sigmoid_val);
      }
    }

    __syncwarp();  //Confirm the scores is written to the softmax/sigmoid output

    /***
         * Section: Topk
         * Get the topk indices
         * - group_topk
         * - naive topk
         * - topk with expert bias
         */
    // Topk on the scores
    // The bias is not empty only happens at the sigmod case
    if constexpr (kHasGroupTopk) {
      int group_size = num_experts / num_groups;
      // Top2
      for (int i = 0; i < num_groups; i++) {
        naive_topk_and_mask(
            /*scores ptr = */ scores + i * group_size,
            /*data size = */ group_size,
            /*topk = */ topk / group_topk,
            /*topk indices ptr = */ topk_indices,
            /*topk scores ptr = */ topk_scores,
            /*lane id = */ lane_id);
        // Compute the group score
        if (lane_id == 0) {
          DataType tmp = 0.0f;
          for (int j = 0; j < topk / group_topk; j++) {
            tmp = tmp + topk_scores[j];
          }
          group_scores[i] = tmp;
        }
      }
      __syncwarp();

      // select the topk groups
      naive_topk_and_mask_inplace(
          /*scores ptr = */ group_scores,
          /*data size = */ num_groups,
          /*topk = */ group_topk,
          /*topk indices ptr = */ topk_indices,
          /*topk scores ptr = */ topk_scores,
          /*lane id = */ lane_id);
      // Copy the unmasked scores to the buffer
      for (int i = 0; i < group_topk; i++) {
        int st = topk_indices[i] * group_size;
        int ed = st + group_size;
        for (int j = st + lane_id; j < ed; j += kThreadsPerWarp) {
          masked_scores[j] = scores[j];
        }
      }
      __syncwarp();
      naive_topk_and_mask_inplace(masked_scores, num_experts, topk, topk_indices, topk_scores,
                                  lane_id);

    } else {
      naive_topk_and_mask_inplace(scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    }

    /***
         * Section: Postprocess
         * Possible postprocess the scores after the topk operation
         * - Revert Expert bias
         * - Softmax
         * - Sigmoid post-processing when topk > 1
         * - Write the result with scaling_factor
         */
    // Revert Expert bias from the topk scores
    if constexpr (kScoreFunction == 0 && kHasExpertBias) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        topk_scores[i] = static_cast<DataType>(static_cast<float>(topk_scores[i]) -
                                               static_cast<float>(expert_bias[topk_indices[i]]));
      }
      if (topk > 1) __syncwarp();
    }

    if constexpr (kScoreFunction == 1) {
      // score_function == 1 means softmax
      if constexpr (!kUsePreSoftmax) {
        // Apply softmax to the topk logits
        apply_softmax_on_float(topk_scores, topk, lane_id);
        // Save the softmax output for backward
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
        }
      }
    } else {
      // score_function == 0 means sigmoid
      if (topk > 1) {
        float sum_scores = static_cast<float>(
            warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id));
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = static_cast<DataType>(static_cast<float>(topk_scores[i]) /
                                                 (sum_scores + epsilon));
        }
      }
    }

    // Write the probs/routing_map to the output tensor
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
      probs[pos_offset + topk_indices[i]] =
          static_cast<DataType>(scaling_factor * static_cast<float>(topk_scores[i]));
    }
  }
}

template <typename DataType, typename BiasType>
void fused_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    bool use_double_buffer, const BiasType *expert_bias, DataType *probs, bool *routing_map,
    DataType *intermediate_output, musaStream_t stream) {


  const int threads_per_block = get_router_threads_per_block();
  size_t num_token_per_block = threads_per_block / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  (void)use_double_buffer;  // Keep API compatible; double buffer path is disabled.
  size_t shared_memory_size =
      num_experts * num_token_per_block * sizeof(DataType)  // scores
      + topk * num_token_per_block * sizeof(DataType)                        // topk_scores
      + topk * num_token_per_block * sizeof(int);                            // topk_indices
  bool has_group_topk = group_topk > 0;
  if (has_group_topk) {
    shared_memory_size += num_groups * num_token_per_block * sizeof(DataType);   // group_scores
    shared_memory_size += num_experts * num_token_per_block * sizeof(DataType);  // maksed_scores
  }
  bool has_expert_bias = expert_bias != nullptr;
  if (score_function == 1) {
    if (has_group_topk) {
      if (has_expert_bias) {
        if (use_pre_softmax) {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, true, true, true>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        } else {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, true, true, false>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        }
      } else {
        if (use_pre_softmax) {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, true, false, true>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        } else {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, true, false, false>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        }
      }
    } else {
      if (has_expert_bias) {
        if (use_pre_softmax) {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, false, true, true>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        } else {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, false, true, false>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        }
      } else {
        if (use_pre_softmax) {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, false, false, true>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        } else {
          fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, false, false, false>
              <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                  logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                  expert_bias, probs, routing_map, intermediate_output);
        }
      }
    }
  } else {
    if (has_group_topk) {
      if (has_expert_bias) {
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, 0, true, true, false>
            <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                expert_bias, probs, routing_map, intermediate_output);
      } else {
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, 0, true, false, false>
            <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                expert_bias, probs, routing_map, intermediate_output);
      }
    } else {
      if (has_expert_bias) {
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, 0, false, true, false>
            <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                expert_bias, probs, routing_map, intermediate_output);
      } else {
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, 0, false, false, false>
            <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
                logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
                expert_bias, probs, routing_map, intermediate_output);
      }
    }
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_topk_with_score_function_forward(const Tensor logits, int num_tokens, int num_experts,
                                            int topk, bool use_pre_softmax, int num_groups,
                                            int group_topk, float scaling_factor,
                                            int score_function, const Tensor expert_bias,
                                            bool use_double_buffer, Tensor probs, Tensor routing_map,
                                            Tensor intermediate_output, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
          expert_bias.data.dtype, BiasType,
          fused_topk_with_score_function_forward_kernel_launcher<DataType, BiasType>(
              reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
              use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
              use_double_buffer, reinterpret_cast<BiasType *>(expert_bias.data.dptr),
              reinterpret_cast<DataType *>(probs.data.dptr),
              reinterpret_cast<bool *>(routing_map.data.dptr),
              reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream);););
}

template <typename DataType, int kScoreFunction, bool kUsePreSoftmax>
__global__ void fused_topk_with_score_function_backward_kernel(
    // Inputs tensor
    const bool *__restrict__ routing_map, const DataType *__restrict__ intermediate_output,
    const DataType *__restrict__ grad_probs,
    // Other parameters
    int num_tokens, int num_experts, int topk, float scaling_factor,
    // Output tensor
    DataType *__restrict__ grad_logits) {
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
  DataType *grad_probs_buf = reinterpret_cast<DataType *>(shmem);
  // To store the output of softmax/sigmoid from the fwd
  DataType *act_from_fwd_buf =
      reinterpret_cast<DataType *>(grad_probs_buf + num_experts * num_token_per_block);
  // To store the routing_map from the fwd
  bool *routing_map_buf =
      reinterpret_cast<bool *>(act_from_fwd_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_grad = grad_probs_buf + warp_id * num_experts;
  DataType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  bool *local_routing_map = routing_map_buf + warp_id * num_experts;

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
      local_grad[i] = grad_probs[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
      local_routing_map[i] = routing_map[pos_offset + i];
    }

    /***
         * Section: Backward of ops after the topk
         * - Backward of the used scaling_factor
         * - Sigmoid Post-processing bwd when topk > 1
         * - Softmax bwd if use_pre_softmax is false
         */
    // Backward of scaling_factor.
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (local_routing_map[i]) {
        local_grad[i] = static_cast<DataType>(static_cast<float>(local_grad[i]) * scaling_factor);
      }
    }

    if constexpr (kScoreFunction == 1) {
      if constexpr (!kUsePreSoftmax) {
        // Softmax bwd on top-k entries only (masked).
        float local_sum_output_x_grad = 0.0f;
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          if (local_routing_map[i]) {
            local_sum_output_x_grad +=
                static_cast<float>(local_grad[i]) * static_cast<float>(local_act_from_fwd[i]);
          }
        }
        for (int s = 16; s > 0; s /= 2) {
          local_sum_output_x_grad += __shfl_xor_sync(0xffffffff, local_sum_output_x_grad, s);
        }
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          if (local_routing_map[i]) {
            float act = static_cast<float>(local_act_from_fwd[i]);
            float grad = static_cast<float>(local_grad[i]);
            local_grad[i] = static_cast<DataType>(act * (grad - local_sum_output_x_grad));
          } else {
            local_grad[i] = 0.0f;
          }
        }
      } else {
        // First mask non-topk grads, then apply pre-softmax bwd on full vector.
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          if (!local_routing_map[i]) {
            local_grad[i] = 0.0f;
          }
        }
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
    } else {
      // Sigmoid post-processing bwd when topk > 1.
      if (topk > 1) {
        float local_sum_fwd_input = 0.0f;
        float local_sum_output_x_grad = 0.0f;
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          if (local_routing_map[i]) {
            float act = static_cast<float>(local_act_from_fwd[i]);
            local_sum_fwd_input += act;
            local_sum_output_x_grad += static_cast<float>(local_grad[i]) * act;
          }
        }
        for (int s = 16; s > 0; s /= 2) {
          local_sum_fwd_input += __shfl_xor_sync(0xffffffff, local_sum_fwd_input, s);
          local_sum_output_x_grad += __shfl_xor_sync(0xffffffff, local_sum_output_x_grad, s);
        }
        float norm_inv = 1.0f / (local_sum_fwd_input + epsilon);
        float corr = local_sum_output_x_grad * norm_inv * norm_inv;
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          if (local_routing_map[i]) {
            local_grad[i] = static_cast<DataType>(static_cast<float>(local_grad[i]) * norm_inv - corr);
          } else {
            local_grad[i] = 0.0f;
          }
        }
      } else {
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          if (!local_routing_map[i]) {
            local_grad[i] = 0.0f;
          }
        }
      }
      // Sigmoid bwd.
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
    }

    // Write the grad_logits to the global mem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = local_grad[i];
    }
  }
}

template <typename DataType>
void fused_topk_with_score_function_backward_kernel_launcher(
    const bool *routing_map, const DataType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, musaStream_t stream) {
  // Meta data for the kernel
  const int threads_per_block = get_router_threads_per_block();
  size_t num_token_per_block = threads_per_block / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // grad_probs
                              +
                              num_experts * num_token_per_block * sizeof(DataType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(bool);    // routing_map
  if (score_function == 1) {
    if (use_pre_softmax) {
      fused_topk_with_score_function_backward_kernel<DataType, 1, true>
          <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
              routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
              scaling_factor, grad_logits);
    } else {
      fused_topk_with_score_function_backward_kernel<DataType, 1, false>
          <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
              routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
              scaling_factor, grad_logits);
    }
  } else {
    fused_topk_with_score_function_backward_kernel<DataType, 0, false>
        <<<grid_size, threads_per_block, shared_memory_size, stream>>>(
            routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
            scaling_factor, grad_logits);
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_topk_with_score_function_backward(const Tensor &routing_map,
                                             const Tensor &intermediate_output,
                                             const Tensor &grad_probs, int num_tokens,
                                             int num_experts, int topk, bool use_pre_softmax,
                                             float scaling_factor, int score_function,
                                             Tensor &grad_logits, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      fused_topk_with_score_function_backward_kernel_launcher<DataType>(
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<DataType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function,
          reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace transformer_engine

void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    int use_double_buffer,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_forward);
  using namespace transformer_engine;
  fused_topk_with_score_function_forward(
      *reinterpret_cast<Tensor*>(logits), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), num_groups, group_topk, scaling_factor, score_function,
      *reinterpret_cast<Tensor*>(expert_bias), static_cast<bool>(use_double_buffer),
      *reinterpret_cast<Tensor*>(probs),
      *reinterpret_cast<Tensor*>(routing_map), *reinterpret_cast<Tensor*>(intermediate_output), stream);
}

void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward);
  using namespace transformer_engine;
  fused_topk_with_score_function_backward(
      *reinterpret_cast<Tensor*>(routing_map), *reinterpret_cast<Tensor*>(intermediate_output),
      *reinterpret_cast<Tensor*>(grad_probs), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *reinterpret_cast<Tensor*>(grad_logits), stream);
}
