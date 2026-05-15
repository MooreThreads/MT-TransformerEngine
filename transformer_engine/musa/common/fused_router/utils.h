/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_

#include <cstdlib>
#include <limits>
#include <math.h>

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {

constexpr size_t kThreadsPerWarp = 32;
constexpr int kThreadsPerBlock =
    128;  // Default: 4 warps in 1 CTA, each warp is responsible for 1 token.
inline int get_router_threads_per_block() {
  static const int threads_per_block = []() {
    int value = kThreadsPerBlock;
    const char *env_value = std::getenv("TE_ROUTER_THREADS_PER_BLOCK");
    if (env_value == nullptr || env_value[0] == '\0') {
      return value;
    }

    char *parse_end = nullptr;
    long parsed_value = std::strtol(env_value, &parse_end, 10);
    if (parse_end == env_value || *parse_end != '\0') {
      return value;
    }
    if (parsed_value <= 0 || parsed_value > 1024 || parsed_value % kThreadsPerWarp != 0) {
      return value;
    }
    return static_cast<int>(parsed_value);
  }();
  return threads_per_block;
}
constexpr float epsilon = 1e-20;

template <typename T>
__device__ inline T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ inline T sum(T a, T b) {
  return a + b;
}

enum ReduceFuncType {
  SUM,
  MAX,
};

template <typename T>
__device__ inline T warp_reduce_on_shmem(T *data_ptr, int data_size, ReduceFuncType type,
                                         int lane_id) {
  const float default_val = (type == ReduceFuncType::SUM) ? 0.0f
                                                           : -std::numeric_limits<float>::infinity();

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  float val = lane_id < data_size ? static_cast<float>(data_ptr[lane_id]) : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    float cur = static_cast<float>(data_ptr[i]);
    val = (type == ReduceFuncType::SUM) ? (val + cur) : fmaxf(val, cur);
  }

  // Warp shuffle between threads
  for (int s = 16; s > 0; s /= 2) {
    float shuffled = __shfl_xor_sync(0xffffffff, val, s);
    val = (type == ReduceFuncType::SUM) ? (val + shuffled) : fmaxf(val, shuffled);
  }
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_on_float(DataType *scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    float score = static_cast<float>(scores[i]);
    scores[i] = static_cast<float>(1.0f / (1.0f + expf(-score)));
  }
}

template <typename T>
__device__ inline T masked_warp_reduce_on_shmem(T *data_ptr, bool *mask, int data_size,
                                                ReduceFuncType type, int lane_id) {
  const float default_val = (type == ReduceFuncType::SUM) ? 0.0f
                                                           : -std::numeric_limits<float>::infinity();

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  float val =
      lane_id < data_size && mask[lane_id] ? static_cast<float>(data_ptr[lane_id]) : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    if (mask[i]) {
      float cur = static_cast<float>(data_ptr[i]);
      val = (type == ReduceFuncType::SUM) ? (val + cur) : fmaxf(val, cur);
    }
  }

  // Warp shuffle between threads
  for (int s = 16; s > 0; s /= 2) {
    float shuffled = __shfl_xor_sync(0xffffffff, val, s);
    val = (type == ReduceFuncType::SUM) ? (val + shuffled) : fmaxf(val, shuffled);
  }
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    float fwd_out = static_cast<float>(fwd_output[i]);
    grad[i] = static_cast<float>(grad[i]) * fwd_out * (1.0f - fwd_out);
  }
}

template <typename DataType>
__device__ inline void apply_softmax_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  DataType *comp_buf, bool *mask, int data_size,
                                                  int lane_id) {
  // Put the result of output * grad to the comp_buf
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
      else
        comp_buf[i] = 0.0f;
    } else {
      comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
    }
  }
  __syncwarp();
  float sum_Output_x_Grad = warp_reduce_on_shmem(
      /*data ptr = */ comp_buf,
      /*data size = */ data_size,
      /*reduce func = */ ReduceFuncType::SUM, lane_id);
  // In-place update
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        grad[i] =
            static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
      else
        grad[i] = 0.0f;
    } else {
      grad[i] =
          static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
    }
  }
}

template <typename DataType>
__device__ inline void apply_softmax_on_float(DataType *scores, int data_size, int lane_id) {
  // 1. compute the max of value
  float max_val =
      static_cast<float>(warp_reduce_on_shmem(scores, data_size, ReduceFuncType::MAX, lane_id));
  // 2. value -> exp_value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(expf(static_cast<float>(scores[i]) - max_val));
  }
  __syncwarp();
  // 3. compute the sum of exp_value
  float sum_val =
      static_cast<float>(warp_reduce_on_shmem(scores, data_size, ReduceFuncType::SUM, lane_id));
  // 4. update the softmax value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(scores[i]) / sum_val;
  }
  __syncwarp();
}

template <typename T>
__device__ inline void naive_topk_and_mask(T *scores, int data_size, int topk, int *topk_indices,
                                           T *topk_scores, int lane_id) {
  // Check if the index is masked by the later iteration
  auto is_masked = [&topk_indices](int k, int index) {
    if (k == 0) return false;
    for (int i = 0; i < k; i++) {
      if (topk_indices[i] == index) return true;
    }
    return false;
  };
  // Topk Times: Find the max value and its index
  // Then mask it, and record the index in the topk_indices
  // After looping topk times, the topk_indices will be the topk indices
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int k = 0; k < topk; k++) {
    // Find the max value and its index
    float val = (lane_id < data_size && !is_masked(k, lane_id))
                    ? static_cast<float>(scores[lane_id])
                    : neg_inf;
    int index = (lane_id < data_size) ? lane_id : 0;
    // Some value is hanlded in local thread
    // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
    // Reduce the value in local thread
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      float cur_val = (is_masked(k, i)) ? neg_inf : static_cast<float>(scores[i]);
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    // Warp shuffle between threads
    for (int s = 16; s > 0; s /= 2) {
      float shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      int shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      topk_scores[k] = static_cast<T>(val);
    }
    __syncwarp();
  }
}

template <typename T>
__device__ inline void naive_topk_and_mask_inplace(T *scores, int data_size, int topk,
                                                    int *topk_indices, T *topk_scores,
                                                    int lane_id) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int k = 0; k < topk; k++) {
    float val = lane_id < data_size ? static_cast<float>(scores[lane_id]) : neg_inf;
    int index = lane_id < data_size ? lane_id : 0;
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      float cur_val = static_cast<float>(scores[i]);
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    for (int s = 16; s > 0; s /= 2) {
      float shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      int shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      topk_scores[k] = static_cast<T>(val);
      scores[index] = static_cast<T>(neg_inf);
    }
    __syncwarp();
  }
}

template <typename T>
__device__ inline void naive_topk_indices_inplace(T *scores, int data_size, int topk,
                                                   int *topk_indices, int lane_id) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int k = 0; k < topk; k++) {
    float val = lane_id < data_size ? static_cast<float>(scores[lane_id]) : neg_inf;
    int index = lane_id < data_size ? lane_id : 0;
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      float cur_val = static_cast<float>(scores[i]);
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    for (int s = 16; s > 0; s /= 2) {
      float shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      int shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      scores[index] = static_cast<T>(neg_inf);
    }
    __syncwarp();
  }
}

// Current TE only support float32/bf16/fp16, float64 probs should be considered in the future
#define TE_ROUTER_PROBS_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kFloat32: {                               \
      using type = float;                                 \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kFloat16: {                               \
      using type = fp16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kBFloat16: {                              \
      using type = bf16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }

#define TE_ROUTER_INDEX_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kInt32: {                                 \
      using type = int32_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kInt64: {                                 \
      using type = int64_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kBFloat16: {                              \
      using type = bf16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kFloat32: {                               \
      using type = float;                                 \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }
}  // namespace transformer_engine
#endif
