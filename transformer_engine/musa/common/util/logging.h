/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_

#include <mublas.h>
#include <musa_runtime_api.h>
#include <mudnn.h>
// #include <nvrtc.h>

#include <stdexcept>

#include "../util/string.h"

#define NVTE_ERROR(...)                                              \
  do {                                                               \
    throw ::std::runtime_error(::transformer_engine::concat_strings( \
        __FILE__ ":", __LINE__, " in function ", __func__, ": ",     \
        ::transformer_engine::concat_strings(__VA_ARGS__)));         \
  } while (false)

#define NVTE_CHECK(expr, ...)                                        \
  do {                                                               \
    if (!(expr)) {                                                   \
      NVTE_ERROR("Assertion failed: " #expr ". ",                    \
                 ::transformer_engine::concat_strings(__VA_ARGS__)); \
    }                                                                \
  } while (false)

#define NVTE_CHECK_CUDA(expr)                                                 \
  do {                                                                        \
    const musaError_t status_NVTE_CHECK_MUSA = (expr);                        \
    if (status_NVTE_CHECK_MUSA != musaSuccess) {                              \
      NVTE_ERROR("MUSA Error: ", musaGetErrorString(status_NVTE_CHECK_MUSA)); \
    }                                                                         \
  } while (false)

#define NVTE_CHECK_CUBLAS(expr)                                                      \
  do {                                                                               \
    const mublasStatus_t status_NVTE_CHECK_MUBLAS = (expr);                          \
    if (status_NVTE_CHECK_MUBLAS != MUBLAS_STATUS_SUCCESS) {                         \
      NVTE_ERROR("muBLAS Error: ", mublasStatus_to_string(status_NVTE_CHECK_MUBLAS));\
    }                                                                                \
  } while (false)

#define NVTE_CHECK_CUDNN(expr)                                                  \
  do {                                                                          \
    const ::musa::dnn::Status status_NVTE_CHECK_MUDNN = (expr);                 \
    if (status_NVTE_CHECK_MUDNN != ::musa::dnn::Status::SUCCESS) {              \
      NVTE_ERROR("muDNN Runtime Error(",                                        \
                 static_cast<int>(status_NVTE_CHECK_MUDNN),                     \
                 "). For more information, enable muDNN logging "               \
                 "by setting MUDNN_LOG_LEVEL=INFO in the environment.");        \
    }                                                                           \
  } while (false)

#define NVTE_CHECK_CUDNN_FE(expr)                                    \
  do {                                                               \
  } while (false)

#define NVTE_CHECK_NVRTC(expr)                                                   \
  do {                                                                           \
  } while (false)

#define NVTE_CHECK_MU(expr)                       \
  do {                                            \
    MUresult status = (expr);                     \
    if (status != MUSA_SUCCESS) {                 \
      const char* err_str;                        \
      muGetErrorString(status, &err_str);         \
      NVTE_ERROR("musa driver Error: ", err_str); \
    }                                             \
  } while (false)

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_
