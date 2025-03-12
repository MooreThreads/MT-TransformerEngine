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

template <size_t load_size, size_t store_size, typename IType, typename OType>
__global__ void __launch_bounds__(block_size) mtfp8_cast_transpose_general_kernel(
    const IType *__restrict__ const inp,
    const CType *__restrict__ const noop,
    OType *__restrict__ const out_c,
    OType *__restrict__ const out_t,
    CType *__restrict__ const scale_inv,
    CType *__restrict__ const columnwise_scale_inv,
    size_t row_length,
    size_t num_rows,
    size_t rowwise_group_size) {
  /* do something */
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
  
  const auto sinv_m = output->scale_inv.shape[0];
  const auto sinv_n = output->scale_inv.shape[1];

  // Assume block size is [1, N] padded.
  NVTE_CHECK((sinv_m == num_rows) && (row_length % sinv_n == 0) && (num_rows % sinv_n == 0));
  const size_t group_size = (row_length / sinv_n);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->data.dtype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OutputType,
        constexpr size_t itype_size = sizeof(InputType);
        constexpr size_t otype_size = sizeof(OutputType);

        constexpr size_t load_size = 4;
        constexpr size_t store_size = 4;
        constexpr size_t row_tile_size = load_size / itype_size * THREADS_PER_WARP;
        constexpr size_t col_tile_size = store_size / otype_size * THREADS_PER_WARP;
        const int num_blocks =
            (DIVUP(row_length, row_tile_size) * DIVUP(num_rows, col_tile_size));

          mtfp8_cast_transpose_general_kernel<load_size, store_size, InputType, OutputType>
            <<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const InputType*>(input->data.dptr),
                reinterpret_cast<const CType*>(noop->data.dptr),
                reinterpret_cast<OutputType*>(output->data.dptr),
                reinterpret_cast<OutputType*>(output->columnwise_data.dptr),
                reinterpret_cast<CType*>(output->scale_inv.dptr),
                reinterpret_cast<CType*>(output->columnwise_scale_inv.dptr),
                row_length,
                num_rows,
                group_size);
      );
  );
}

}  // namespace transformer_engine
