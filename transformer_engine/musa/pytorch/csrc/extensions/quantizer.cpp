/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include "common.h"
#include "pybind.h"
#include "torch/torch.h"
#include "util.h"

namespace transformer_engine::pytorch {

constexpr size_t MXFP8_BLOCK_SIZE = 32;

Quantizer::Quantizer(const py::handle& quantizer) {
  if (quantizer.is_none()) {
    this->rowwise_usage = true;
    this->columnwise_usage = true;
    this->internal = false;
  } else {
    this->rowwise_usage = quantizer.attr("rowwise_usage").cast<bool>();
    this->columnwise_usage = quantizer.attr("columnwise_usage").cast<bool>();
    this->internal = quantizer.attr("internal").cast<bool>();
    this->quantizer = quantizer;
  }
}

Float8Quantizer::Float8Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  const at::Tensor& scale = quantizer.attr("scale").cast<at::Tensor>();
  const at::Tensor& amax = quantizer.attr("amax").cast<at::Tensor>();
  const DType type = quantizer.attr("dtype").cast<DType>();

  this->amax = amax;
  this->scale = scale;
  this->dtype = type;
}

std::pair<TensorWrapper, py::object> NoneQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  at::TensorOptions opts;
  opts = opts.dtype(GetATenDType(dtype)).device(torch::kPrivateUse1);
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }
  at::Tensor ret;
  if (rowwise_data.has_value()) {
    ret = std::move(*rowwise_data);
  } else {
    ret = at::empty(torch_shape, opts);
  }

  TensorWrapper tensor;
  tensor.set_rowwise_data(ret.data_ptr(), dtype, shape);
  return {std::move(tensor), py::cast(ret)};
}

void Float8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  at::TensorOptions opts;
  opts = opts.dtype(torch::kFloat32).device(torch::kPrivateUse1);
  tensor->set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> Float8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> rowwise_torch_shape;
  std::vector<int64_t> columnwise_torch_shape;

  if (!shape.empty()) {
    columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape.back()));
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i < shape.size() - 1) {
      columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
    }
    rowwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
  }
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kPrivateUse1);
  at::Tensor data;
  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = at::empty(rowwise_torch_shape, opts);
    }
  }
  const py::object py_data = rowwise_usage ? py::cast(data) : py::none();
  at::Tensor columnwise_data;
  bool create_transpose = columnwise_usage && !non_tn_fp8_gemm_supported();
  if (create_transpose) {
    columnwise_data = at::empty(columnwise_torch_shape, opts);
  }
  const py::object py_columnwise_data = create_transpose ? py::cast(columnwise_data) : py::none();
  opts = opts.dtype(torch::kFloat32);
  at::Tensor scale_inv = at::reciprocal(scale);
  py::object ret;
  if (internal) {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
    ret = Float8TensorClass("data"_a = py_data, "fp8_scale_inv"_a = scale_inv,
                            "fp8_dtype"_a = this->dtype, "data_transpose"_a = py_columnwise_data,
                            "quantizer"_a = this->quantizer);
  } else {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
    ret = Float8TensorClass("shape"_a = rowwise_torch_shape, "dtype"_a = GetATenDType(dtype),
                            "data"_a = py_data, "fp8_scale_inv"_a = scale_inv,
                            "fp8_dtype"_a = this->dtype, "data_transpose"_a = py_columnwise_data,
                            "quantizer"_a = this->quantizer);
  }
  TensorWrapper tensor(this->get_scaling_mode());
  if (rowwise_usage) {
    tensor.set_rowwise_data(data.data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  if (create_transpose) {
    std::vector<size_t> transposed_shape;
    for (auto s : columnwise_torch_shape) {
      transposed_shape.emplace_back(static_cast<size_t>(s));
    }
    tensor.set_columnwise_data(columnwise_data.data_ptr(), this->dtype, transposed_shape);
    tensor.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  this->set_quantization_params(&tensor);
  return {std::move(tensor), std::move(ret)};
}

MXFP8Quantizer::MXFP8Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
}

void MXFP8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> MXFP8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  size_t numel = 1;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
    numel *= s;
  }

  TensorWrapper tensor(NVTE_MXFP8_1D_SCALING);
  at::TensorOptions opts;
  at::Tensor rowwise_data1, columnwise_data, rowwise_scale_inv,
      columnwise_scale_inv;  // TODO(pgadzinski) - change
  opts = opts.dtype(torch::kUInt8).device(torch::kPrivateUse1);
  auto last_dim = static_cast<size_t>(torch_shape.back());

  NVTE_CHECK(last_dim % MXFP8_BLOCK_SIZE == 0 && (numel / last_dim) % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisble by ", MXFP8_BLOCK_SIZE,
             " (got shape=", torch_shape, ")");

  at::Tensor data;
  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = at::empty(torch_shape, opts);
    }
    auto sinv0 = roundup(numel / last_dim, 128);
    auto sinv1 = roundup(last_dim / MXFP8_BLOCK_SIZE, 4);
    rowwise_scale_inv = at::zeros({sinv0, sinv1}, opts);
    tensor.set_rowwise_data(data.data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(rowwise_scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 std::vector<size_t>{static_cast<size_t>(sinv0), static_cast<size_t>(sinv1)});
  }

  if (columnwise_usage) {
    auto sinv0 = roundup(numel / (last_dim * MXFP8_BLOCK_SIZE), 4);
    auto sinv1 = roundup(last_dim, 128);
    columnwise_data = at::empty(torch_shape, opts);
    columnwise_scale_inv = at::zeros({sinv0, sinv1}, opts);

    tensor.set_columnwise_data(columnwise_data.data_ptr(), this->dtype, shape);
    tensor.set_columnwise_scale_inv(columnwise_scale_inv.data_ptr(), DType::kFloat8E8M0,
                                    std::vector<size_t>{static_cast<size_t>(sinv0), static_cast<size_t>(sinv1)});
  }
  this->set_quantization_params(&tensor);

  py::object ret;
  if (internal) {
    py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorBasePythonClass));
    ret = MXFP8TensorClass("rowwise_data"_a = data, "columnwise_data"_a = columnwise_data,
                           "rowwise_scale_inv"_a = rowwise_scale_inv,
                           "columnwise_scale_inv"_a = columnwise_scale_inv,
                           "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
  } else {
    py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorPythonClass));
    ret = MXFP8TensorClass("shape"_a = torch_shape, "dtype"_a = GetATenDType(dtype),
                           "rowwise_data"_a = data, "columnwise_data"_a = columnwise_data,
                           "rowwise_scale_inv"_a = rowwise_scale_inv,
                           "columnwise_scale_inv"_a = columnwise_scale_inv,
                           "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
  }

  return {std::move(tensor), std::move(ret)};
}

MTFP8Quantizer::MTFP8Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  dtype = quantizer.attr("dtype").cast<DType>();
  block_m = quantizer.attr("block_m").cast<int64_t>();
  block_n = quantizer.attr("block_n").cast<int64_t>();

  NVTE_CHECK(block_m > 0);
  if (block_m == 1) {
    NVTE_CHECK(block_n > 0 && block_n % 16 == 0);
  } else {
    NVTE_CHECK((block_m % 16 == 0) && (block_m == block_n));
  }
}

void MTFP8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(
      rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype), rowwise_data.shape);
  tensor->set_columnwise_data(
      columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype), columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> MTFP8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType fake_dtype, std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  int64_t numel = 1;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
    numel *= static_cast<int64_t>(s);
  }
  const auto dim_n = torch_shape.back();
  const auto dim_m = numel / dim_n;

  TensorWrapper tensor(NVTE_MTFP8_BLOCK_SCALING);
  auto opt = at::TensorOptions().device(torch::kPrivateUse1);

  at::Tensor data, rowwise_scale_inv;
  at::Tensor columnwise_data, columnwise_scale_inv;
  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = at::empty(torch_shape, opt.dtype(torch::kUInt8));
    }

    const auto sinv0 = (dim_m + block_m - 1) / block_m;
    const auto sinv1 = (dim_n + block_n - 1) / block_n;

    rowwise_scale_inv = at::empty({sinv0, sinv1}, opt.dtype(torch::kFloat));
    tensor.set_rowwise_data(data.data_ptr(), dtype, shape);
    tensor.set_rowwise_scale_inv(rowwise_scale_inv.data_ptr(), DType::kFloat32,
                                 std::vector<size_t>{static_cast<size_t>(sinv0), static_cast<size_t>(sinv1)});
  }

  const bool can_not_share = (block_m != block_n);
  if (columnwise_usage && can_not_share) {
    const auto sinv0 = (dim_m + block_n - 1) / block_n;
    const auto sinv1 = (dim_n + block_m - 1) / block_m;

    columnwise_data = at::empty(torch_shape, opt.dtype(torch::kUInt8));
    columnwise_scale_inv = at::empty({sinv0, sinv1}, opt.dtype(torch::kFloat));
    tensor.set_columnwise_data(columnwise_data.data_ptr(), dtype, shape);
    tensor.set_columnwise_scale_inv(columnwise_scale_inv.data_ptr(), DType::kFloat32,
                                    std::vector<size_t>{static_cast<size_t>(sinv0), static_cast<size_t>(sinv1)});
  }
  this->set_quantization_params(&tensor);

  py::object ret;
  if (internal) {
    py::handle MTFP8TensorClass(reinterpret_cast<PyObject*>(MTFP8TensorBasePythonClass));
    ret = MTFP8TensorClass("rowwise_data"_a = data,
                           "columnwise_data"_a = columnwise_data,
                           "rowwise_scale_inv"_a = rowwise_scale_inv,
                           "columnwise_scale_inv"_a = columnwise_scale_inv,
                           "fp8_dtype"_a = dtype,
                           "quantizer"_a = quantizer);
  } else {
    py::handle MTFP8TensorClass(reinterpret_cast<PyObject*>(MTFP8TensorPythonClass));
    ret = MTFP8TensorClass("shape"_a = torch_shape,
                           "dtype"_a = GetATenDType(fake_dtype),
                           "rowwise_data"_a = data,
                           "columnwise_data"_a = columnwise_data,
                           "rowwise_scale_inv"_a = rowwise_scale_inv,
                           "columnwise_scale_inv"_a = columnwise_scale_inv,
                           "fp8_dtype"_a = dtype,
                           "quantizer"_a = quantizer);
  }

  return {std::move(tensor), std::move(ret)};
}

}  // namespace transformer_engine::pytorch
