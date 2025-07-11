/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#define PYBIND11_DETAILED_ERROR_MESSAGES  // TODO remove

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "common.h"
#include "transformer_engine/transformer_engine.h"

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)
namespace pybind11::detail {

template <>
struct type_caster<at::ScalarType> {
 public:
  PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));

  type_caster() : value(at::kFloat) {}

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPDtype_Check(obj)) {
      value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
      return true;
    }
    return false;
  }

  static handle cast(
      const at::ScalarType& src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return Py_NewRef(torch::getTHPDtype(src));
  }
};

} // namespace pybind11::detail
#endif

namespace transformer_engine::pytorch {

extern PyTypeObject *Float8TensorPythonClass;
extern PyTypeObject *Float8TensorBasePythonClass;
extern PyTypeObject *Float8QuantizerClass;
extern PyTypeObject *MXFP8TensorPythonClass;
extern PyTypeObject *MXFP8TensorBasePythonClass;
extern PyTypeObject *MXFP8QuantizerClass;

extern PyTypeObject *MTFP8TensorPythonClass;
extern PyTypeObject *MTFP8TensorBasePythonClass;
extern PyTypeObject *MTFP8QuantizerClass;

void init_extension();

void init_float8_extension();

void init_mxfp8_extension();

namespace detail {
inline bool IsFloat8Quantizers(PyObject *obj) { return Py_TYPE(obj) == Float8QuantizerClass; }
inline bool IsFloat8QParams(PyObject *obj) { return Py_TYPE(obj) == Float8QuantizerClass; }

inline bool IsFloat8Tensor(PyObject *obj) {
  return Py_TYPE(obj) == Float8TensorPythonClass || Py_TYPE(obj) == Float8TensorBasePythonClass;
}

inline bool IsMXFP8QParams(PyObject *obj) { return Py_TYPE(obj) == MXFP8QuantizerClass; }

inline bool IsMXFP8Tensor(PyObject *obj) {
  return Py_TYPE(obj) == MXFP8TensorPythonClass || Py_TYPE(obj) == MXFP8TensorBasePythonClass;
}

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, Quantizer *quantizer);

template <typename T>
std::unique_ptr<Quantizer> CreateQuantizer(const py::handle quantizer) {
  return std::make_unique<T>(quantizer);
}

TensorWrapper NVTETensorFromMXFP8Tensor(py::handle tensor, Quantizer *quantization_params);

std::unique_ptr<Quantizer> CreateMXFP8Params(const py::handle params);

inline bool IsFloatingPointType(at::ScalarType type) {
  return type == at::kFloat || type == at::kHalf || type == at::kBFloat16;
}

inline bool IsMTFP8Tensor(PyObject *obj) {
  return Py_TYPE(obj) == MTFP8TensorPythonClass || Py_TYPE(obj) == MTFP8TensorBasePythonClass;
}

inline bool IsMTFP8QParams(PyObject *obj) { return Py_TYPE(obj) == MTFP8QuantizerClass; }

TensorWrapper NVTETensorFromMTFP8Tensor(py::handle tensor, Quantizer *quantization_params);

std::unique_ptr<Quantizer> CreateMTFP8Params(const py::handle params);

constexpr std::array custom_types_converters = {
    std::make_tuple(IsFloat8Tensor, IsFloat8QParams, NVTETensorFromFloat8Tensor,
                    CreateQuantizer<Float8Quantizer>),
    std::make_tuple(IsMTFP8Tensor, IsMTFP8QParams, NVTETensorFromMTFP8Tensor,
                    CreateQuantizer<MTFP8Quantizer>),
    std::make_tuple(IsMXFP8Tensor, IsMXFP8QParams, NVTETensorFromMXFP8Tensor,
                    CreateQuantizer<MXFP8Quantizer>)};

}  // namespace detail

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
