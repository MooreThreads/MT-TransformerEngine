#ifndef TRANSFORMER_ENGINE_MUSA_COMMON_UTIL_MTFP8_CAST_TRANSPOSE_H_
#define TRANSFORMER_ENGINE_MUSA_COMMON_UTIL_MTFP8_CAST_TRANSPOSE_H_

#include "../common.h"

namespace transformer_engine {

void mtfp8_cast_transpose(const Tensor* input, const Tensor* noop, Tensor* output_, musaStream_t stream);

} // namespace transformer_engine

#endif // TRANSFORMER_ENGINE_MUSA_COMMON_UTIL_MTFP8_CAST_TRANSPOSE_MUH_
