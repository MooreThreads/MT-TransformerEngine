/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "ATen/core/TensorBody.h"
#include "extensions.h"
#include "pybind.h"
#include "util.h"
#include "common.h"
namespace transformer_engine::pytorch {

void _batch_init_alloc_outputs(size_t hidden_dim, std::vector<int>& m_splits,
                             std::vector<std::unique_ptr<Quantizer>>& quantizers,
                             std::vector<py::handle> quantizer_list,
                             std::vector<TensorWrapper>& output_list,
                             std::vector<py::object>& output_list_py,
                             transformer_engine::DType otype) {
    using namespace py::literals;
    int num_splits = m_splits.size();

    // Validate all quantizers are consistent
    bool rowwise_usage = quantizers[0]->rowwise_usage;
    bool columnwise_usage = quantizers[0]->columnwise_usage;
    transformer_engine::DType fp8_dtype = static_cast<Float8Quantizer*>(quantizers[0].get())->dtype;
    NVTEScalingMode scaling_mode = static_cast<Float8Quantizer*>(quantizers[0].get())->get_scaling_mode();

    for (size_t i = 1; i < quantizers.size(); i++) {
        NVTE_CHECK(rowwise_usage == quantizers[i]->rowwise_usage, 
                 "All quantizers must have same rowwise usage");
        NVTE_CHECK(columnwise_usage == quantizers[i]->columnwise_usage,
                 "All quantizers must have same columnwise usage");
        NVTE_CHECK(fp8_dtype == static_cast<Float8Quantizer*>(quantizers[i].get())->dtype,
                 "All quantizers must have same dtype");
    }
    bool create_transpose = columnwise_usage && !non_tn_fp8_gemm_supported();

    // size_t hidden_dim = input_view.size(1);
    size_t fp8_elem_size = 1; // FP8 uses 1 byte per element

    // Precompute all shapes and sizes
    std::vector<std::vector<size_t>> rowwise_shapes;
    std::vector<std::vector<size_t>> columnwise_shapes;
    std::vector<size_t> rowwise_sizes;
    std::vector<size_t> columnwise_sizes;

    for (int i = 0; i < num_splits; i++) {
        // Rowwise shape is [m_splits[i], hidden_dim]
        std::vector<size_t> r_shape = {(size_t)m_splits[i], hidden_dim};
        rowwise_shapes.push_back(std::move(r_shape));
        rowwise_sizes.push_back(m_splits[i] * hidden_dim * fp8_elem_size);

        // Columnwise shape is [hidden_dim, m_splits[i]] (transposed)
        std::vector<size_t> c_shape = {hidden_dim, (size_t)m_splits[i]};
        columnwise_shapes.push_back(std::move(c_shape));
        columnwise_sizes.push_back(hidden_dim * m_splits[i] * fp8_elem_size);
    }

    // Compute total sizes for bulk allocation
    size_t total_rowwise = std::accumulate(rowwise_sizes.begin(), rowwise_sizes.end(), 0);
    size_t total_columnwise = std::accumulate(columnwise_sizes.begin(), columnwise_sizes.end(), 0);

    // Allocate memory in bulk
    at::TensorOptions opts = at::TensorOptions()
                            .dtype(torch::kUInt8)
                            .device(torch::kMUSA);

    // Create scale inverse tensors (batched)
    std::vector<at::Tensor> scale_tensors;
    for (auto& quantizer : quantizers) {
        Float8Quantizer* fq = static_cast<Float8Quantizer*>(quantizer.get());
        scale_tensors.push_back(fq->scale);
    }

    at::Tensor all_scales = torch::stack(scale_tensors);
    at::Tensor all_scale_invs = at::reciprocal(all_scales);

    at::Tensor rowwise_full_tensor;
    at::Tensor columnwise_full_tensor;
    // each from_blob will hold a reference to the full tensor, since we need to keep the full tensor alive
    // when all the views are gone, the full tensor will be garbage collected
    std::shared_ptr<at::Tensor> rowwise_full_tensor_holder;
    std::shared_ptr<at::Tensor> columnwise_full_tensor_holder;

    // Allocate and split rowwise data
    std::vector<at::Tensor> rowwise_data_list;
    if (rowwise_usage > 0) {
        rowwise_full_tensor = at::empty({(int64_t)total_rowwise}, opts);
        rowwise_full_tensor_holder = std::make_shared<at::Tensor>(rowwise_full_tensor);
        uint8_t* rowwise_ptr = rowwise_full_tensor.data_ptr<uint8_t>();

        for (int i = 0; i < num_splits; i++) {
            if (rowwise_sizes[i] == 0) {
                rowwise_data_list.emplace_back(at::empty({static_cast<int64_t>(rowwise_shapes[i][0]),
                                    static_cast<int64_t>(rowwise_shapes[i][1])},
                                    opts
                                    ));
            } else {
                rowwise_data_list.emplace_back(at::from_blob(
                    rowwise_ptr,
                    {static_cast<int64_t>(rowwise_shapes[i][0]),static_cast<int64_t>(rowwise_shapes[i][1])},
                    [rowwise_full_tensor_holder](void*) {},  // Keep buffer alive
                    opts
                    ));
            }
            // rowwise_data_list.push_back(tensor);
            rowwise_ptr += rowwise_sizes[i];
        }
    }

    // Allocate and split columnwise data
    std::vector<at::Tensor> columnwise_data_list;
    if (create_transpose > 0) {
        columnwise_full_tensor = at::empty({(int64_t)total_columnwise}, opts);
        columnwise_full_tensor_holder = std::make_shared<at::Tensor>(columnwise_full_tensor);
        uint8_t* columnwise_ptr = columnwise_full_tensor.data_ptr<uint8_t>();

        for (int i = 0; i < num_splits; i++) {
            if (columnwise_sizes[i] == 0) {
                columnwise_data_list.emplace_back(at::empty({static_cast<int64_t>(columnwise_shapes[i][0]),
                                    static_cast<int64_t>(columnwise_shapes[i][1])},
                                    opts
                                    ));
            } else {
                columnwise_data_list.emplace_back(at::from_blob(
                    columnwise_ptr,
                    {static_cast<int64_t>(columnwise_shapes[i][0]),static_cast<int64_t>(columnwise_shapes[i][1])},
                    [columnwise_full_tensor_holder](void*) {},  // Keep buffer alive
                    opts
                    ));
            }
            columnwise_ptr += columnwise_sizes[i];
        }
    }

    float* scale_invs_ptr = all_scale_invs.data_ptr<float>();

    // Create output tensors and Python objects
    for (int i = 0; i < num_splits; i++) {

        // Create Python Float8Tensor object
        py::object rowwise_py = rowwise_usage ? py::cast(rowwise_data_list[i]) : py::none();
        py::object columnwise_py = create_transpose ? py::cast(columnwise_data_list[i]) : py::none();
        py::object scale_inv_py = py::cast(all_scale_invs[i]);

        py::object py_tensor;
        if (quantizers[i]->internal) {
            py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
            py_tensor = Float8TensorClass(
                "data"_a = rowwise_py,
                "fp8_scale_inv"_a = scale_inv_py,
                "fp8_dtype"_a = fp8_dtype,
                "data_transpose"_a = columnwise_py,
                "quantizer"_a = quantizer_list[i]
                );
        } else {
            py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
            std::vector<int64_t> rowwise_torch_shape = {
                static_cast<int64_t>(rowwise_shapes[i][0]),
                static_cast<int64_t>(rowwise_shapes[i][1])
            };
            py_tensor = Float8TensorClass(
                "shape"_a = rowwise_torch_shape,
                "dtype"_a = GetATenDType(otype),
                "data"_a = rowwise_py,
                "fp8_scale_inv"_a = scale_inv_py,
                "fp8_dtype"_a = fp8_dtype,
                "data_transpose"_a = columnwise_py,
                "quantizer"_a = quantizer_list[i]
                );
        }
        output_list_py.emplace_back(std::move(py_tensor));

        // as for tensor wrappers, these tensor wrappers are going to be quantized, so no need to insert empty tensors here
        // even if m_split[i]==0 we also need to perform the operation below, 
        // otherwise will meet "Unable to cast Python instance of type <class 'NoneType'> to C++ type 'at::Tensor'" before following gemm 
        
        // Create TensorWrapper
        TensorWrapper tensor(scaling_mode);

        if (rowwise_usage) {
            tensor.set_rowwise_data(
                rowwise_data_list[i].data_ptr(),
                fp8_dtype,
                rowwise_shapes[i]
            );
            // Explicitly specify the shape type as std::vector<size_t>
            tensor.set_rowwise_scale_inv<std::vector<size_t>>(
                scale_invs_ptr + i,
                DType::kFloat32,
                {1}  // Scale shape is always [1]
            );
            
        }

        if (create_transpose) {
            tensor.set_columnwise_data(
                columnwise_data_list[i].data_ptr(),
                fp8_dtype,
                columnwise_shapes[i]
            );
            // Explicitly specify the shape type as std::vector<size_t>
            tensor.set_columnwise_scale_inv<std::vector<size_t>>(
                scale_invs_ptr + i,
                DType::kFloat32,
                {1}  // Scale shape is always [1]
            );
            
        }

        // Set quantization parameters
        static_cast<Float8Quantizer*>(quantizers[i].get())->set_quantization_params(&tensor);
        if (m_splits[i] == 0) {
            continue;
        }
        output_list.emplace_back(std::move(tensor));
    }
}

std::vector<py::object> fused_multi_quantize_batch_init(std::vector<py::handle> input_list,
                                             size_t hidden_dim,
                                             std::vector<int> m_splits,
                                             std::vector<py::handle> quantizer_list, 
                                             transformer_engine::DType otype) {
    init_extension();
    std::vector<NVTETensor> nvte_inputs;
    std::vector<NVTETensor> nvte_outputs;
    std::vector<py::object> py_outputs;
    std::vector<TensorWrapper> input_wrappers;
    std::vector<TensorWrapper> output_wrappers;
    std::vector<TensorWrapper> tensor_wrappers;
    auto none = py::none();

    // Validate inputs
    NVTE_CHECK(input_list.size() == quantizer_list.size(),
              "Input list and quantizer list must have same size");
    NVTE_CHECK(input_list.size() == m_splits.size(),
              "Input list and m_splits must have same size");

    // Convert quantizers
    std::vector<std::unique_ptr<Quantizer>> quantizers;
    for (auto& q : quantizer_list) {
        quantizers.push_back(convert_quantizer(q));
    }

    // Check if we can use bulk allocation (all Float8 quantizers with same config)
    bool use_batch_init = true;
    if (!detail::IsFloat8Quantizers(quantizer_list[0].ptr())) {
        use_batch_init = false;
    } else {
        auto* first_q = static_cast<Float8Quantizer*>(quantizers[0].get());
        for (size_t i = 1; i < quantizers.size(); i++) {
            auto* q = static_cast<Float8Quantizer*>(quantizers[i].get());
            if (q->rowwise_usage != first_q->rowwise_usage ||
                q->columnwise_usage != first_q->columnwise_usage ||
                q->dtype != first_q->dtype) {
                use_batch_init = false;
                break;
            }
        }
    }

    // Process inputs
    if (use_batch_init) {
        // Create input tensor wrappers
        for (size_t i = 0; i < input_list.size(); i++) {
            if (m_splits[i] == 0){
              continue;
            }            
            auto input_tensor = makeTransformerEngineTensor(input_list[i], none);
            nvte_inputs.emplace_back(input_tensor.data());
            input_wrappers.emplace_back(std::move(input_tensor));
        }

        // Bulk allocate outputs
        _batch_init_alloc_outputs(hidden_dim, m_splits, quantizers, quantizer_list,
                                output_wrappers, py_outputs, otype);

        // Prepare output tensor list
        for (auto& wrapper : output_wrappers) {
            if (wrapper.data()) {  // Skip empty tensors
                nvte_outputs.emplace_back(wrapper.data());
            }
        }
    } else {
        // Fallback to original per-tensor allocation
        for (size_t i = 0; i < input_list.size(); i++) {
            auto input_tensor = makeTransformerEngineTensor(input_list[i], none);
            const NVTEShape input_shape = input_tensor.shape();

            TensorWrapper output_tensor;

            std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
            py::object o;
            std::tie(output_tensor, o) = 
                quantizers[i]->create_tensor(output_shape, otype);
            py_outputs.push_back(o);
            if (input_tensor.numel() == 0) continue;

            nvte_inputs.emplace_back(input_tensor.data());
            nvte_outputs.emplace_back(output_tensor.data());
            tensor_wrappers.emplace_back(std::move(input_tensor));
            tensor_wrappers.emplace_back(std::move(output_tensor));
        }
    }

    // Validate tensor lists
    NVTE_CHECK(nvte_outputs.size() == nvte_inputs.size(),
              "Input/output tensor count mismatch");

    // Check if we can use fused kernel
    bool with_fused_kernel = true;
    for (auto& tensor : nvte_outputs) {
        if (nvte_tensor_scaling_mode(tensor) != NVTE_DELAYED_TENSOR_SCALING ||
            nvte_tensor_columnwise_data(tensor) == nullptr) {
            with_fused_kernel = false;
            break;
        }
    }

    // Launch TE kernel
    if (with_fused_kernel) {
        nvte_multi_cast_transpose(nvte_inputs.size(), nvte_inputs.data(),
                                 nvte_outputs.data(), at::musa::getCurrentMUSAStream());
    } else {
        for (size_t i = 0; i < nvte_outputs.size(); i++) {
            nvte_quantize(nvte_inputs[i], nvte_outputs[i],
                         at::musa::getCurrentMUSAStream());
        }
    }

    return py_outputs;
}
}

std::vector<py::object> fused_multi_quantize(std::vector<py::handle> input_list,
                                             std::optional<std::vector<py::handle>> output_list,
                                             std::vector<py::handle> quantizer_list,
                                             transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;
  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  std::vector<py::object> py_output_objects_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto none = py::none();

  // create TE tensors from input
  for (int i = 0; i < input_list.size(); i++) {
    auto input_tensor = makeTransformerEngineTensor(input_list[i], none);
    const NVTEShape input_shape = input_tensor.shape();

    transformer_engine::TensorWrapper output_tensor;

    if (output_list == std::nullopt) {
      std::unique_ptr<Quantizer> quantizer = convert_quantizer(quantizer_list[i]);
      std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
      py::object o;
      std::tie(output_tensor, o) = quantizer->create_tensor(output_shape, otype);
      py_output_objects_list.push_back(o);
    } else {
      output_tensor = makeTransformerEngineTensor((*output_list)[i], quantizer_list[i]);
    }
    if (input_tensor.numel() == 0) continue;

    nvte_tensor_output_list.emplace_back(output_tensor.data());
    nvte_tensor_input_list.emplace_back(input_tensor.data());
    tensor_wrappers.emplace_back(std::move(input_tensor));
    tensor_wrappers.emplace_back(std::move(output_tensor));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_tensor_output_list.size() == nvte_tensor_input_list.size(),
             "Number of input and output tensors must match");

  // Choose implementation
  // Note: Currently only have fused kernel for FP8 cast-transpose
  bool with_fused_kernel = true;
  for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
    const auto& tensor = nvte_tensor_output_list[i];
    if (nvte_tensor_scaling_mode(tensor) != NVTE_DELAYED_TENSOR_SCALING) {
      with_fused_kernel = false;
      break;
    }
    if (nvte_tensor_columnwise_data(tensor) == nullptr) {
      with_fused_kernel = false;
      break;
    }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    nvte_multi_cast_transpose(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                              nvte_tensor_output_list.data(), at::musa::getCurrentMUSAStream());
  } else {
    for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
      nvte_quantize(nvte_tensor_input_list[i], nvte_tensor_output_list[i],
                    at::musa::getCurrentMUSAStream());
    }
  }
  return py_output_objects_list;
}

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype,
                         std::optional<at::Tensor> output) {
  using namespace transformer_engine::pytorch;

  const auto dim = input.dim();
  NVTE_CHECK(dim >= 2, "Need at least 2D tensor to transpose.");

  if (input.dim() > 2) {
    input = input.view({-1, input.size(dim - 1)});
  }

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    out = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  }
  if (M == 0 || N == 0) return out;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::musa::getCurrentMUSAStream());

  return out;
}
