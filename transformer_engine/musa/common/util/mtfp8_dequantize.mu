#include "mtfp8_cast.muh"

#include <musa_runtime.h>

#include "../util/string.h"
#include "../utils.muh"
#include "mtfp8_utils.muh"

namespace transformer_engine {

namespace mtfp8 {

template <
    typename IType,
    typename OType,
    typename CType,
    size_t VLEN>
__global__ void fp8_general_dequantize(
    const IType* inp,
    OType* out,
    const CType* sinv,
    size_t M,
    size_t N,
    size_t sinv_m,
    size_t sinv_n,
    size_t block_m,
    size_t block_n) {
  using IVecT = Vec<IType, VLEN>;
  using OVecT = Vec<OType, VLEN>;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t offset = tid * VLEN;

  const size_t dimx = offset / N;
  const size_t dimy = offset % N;
  const bool valid = dimx < M;

  const size_t groupx = dimx / block_m;
  const size_t groupy = dimy / block_n;

  IVecT vec_in;
  OVecT vec_out;
  if (valid) {
    const CType s_inv = sinv[groupx * sinv_n + groupy];
    vec_in.load_from(inp + offset, 0);
#pragma unroll
    for (size_t j = 0; j < VLEN; ++j) {
      vec_out.data.elt[j] = (OType)((CType)(vec_in.data.elt[j]) * s_inv);
    }
    vec_out.store_to(out + offset, 0);
  }
}

} // namespace mtfp8

void mtfp8_dequantize(const Tensor* input, Tensor* output, musaStream_t stream) {
  NVTE_CHECK(is_fp8_dtype(input->data.dtype), "Input must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input->data.shape, "Input and output shapes need to match.");

  using namespace mtfp8;

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input->data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->data.dtype, OType,

        NVTE_CHECK(input->scale_inv.dtype == DType::kFloat32);
        using CType = float;

        const auto M = input->flat_first_dim();
        const auto N = input->flat_last_dim();
        const auto sinv_m = input->scale_inv.shape[0];
        const auto sinv_n = input->scale_inv.shape[1];

        size_t block_m = 1;
        const size_t block_n = 128; //(N / sinv_n);
        if (M != sinv_m) {
          block_m = block_n;
        }

        constexpr size_t VLEN = io_bytes / sizeof(OType);
        const size_t threads = max_threads_per_block;
        const size_t blocks = ceil_div(M * N, threads * VLEN);

        fp8_general_dequantize<IType, OType, CType, VLEN>
            <<<(int)blocks, (int)threads, 0, stream>>>(
                reinterpret_cast<const IType*>(input->data.dptr),
                reinterpret_cast<OType*>(output->data.dptr),
                reinterpret_cast<const CType*>(input->scale_inv.dptr),
                M,
                N,
                sinv_m,
                sinv_n,
                block_m,
                block_n);
        NVTE_CHECK_CUDA(musaGetLastError());
      );
  );
}

} // namespace transformer_engine
