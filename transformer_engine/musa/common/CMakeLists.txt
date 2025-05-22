cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MUSA_DIR "/usr/local/musa")
set(MUSA_ARCH "31")

# Transformer Engine library
project(transformer_engine LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${MUSA_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${MUSA_DIR}/lib/cmake/mudnn")

find_package(MUSA REQUIRED)
string(APPEND MUSA_MCC_FLAGS " -std=c++${CMAKE_CXX_STANDARD}")
string(APPEND MUSA_MCC_FLAGS " --offload-arch=mp_${MUSA_ARCH}")
# -mllvm -mtgpu-tempint-prealloc=1 just work for MUSA_ARCH=31
if (MUSA_ARCH STREQUAL "31")
  string(APPEND MUSA_MCC_FLAGS " -mllvm -mtgpu-tempint-prealloc=1")
endif()
set(MUSA_VERBOSE_BUILD ON)
set(MUSA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  string(APPEND MUSA_MCC_FLAGS " -g")
endif()

set(DEPENDENT_TARGETS)

find_package(MUSAToolkit REQUIRED)
list(APPEND DEPENDENT_TARGETS MUSA::toolkit)

include(mudnnTargets)
list(APPEND DEPENDENT_TARGETS mudnn)

find_package(MCCL REQUIRED)
add_library(MUSA::mccl SHARED IMPORTED)
set_target_properties(MUSA::mccl PROPERTIES
  IMPORTED_LOCATION ${MCCL_LIBRARIES}
  INTERFACE_INCLUDE_DIRECTORIES ${MCCL_INCLUDE_DIRS}
)
list(APPEND DEPENDENT_TARGETS MUSA::mccl)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
list(APPEND DEPENDENT_TARGETS Python::Module)

execute_process(
  COMMAND
  ${Python_EXECUTABLE} -c "import os, torch_musa;print(os.path.dirname(torch_musa.__file__))"
  ERROR_QUIET
  OUTPUT_VARIABLE TORCH_MUSA_PYTHONPATH
)
string(REGEX REPLACE "^(.+)\n$" "\\1" TORCH_MUSA_PYTHONPATH ${TORCH_MUSA_PYTHONPATH})

add_library(torch_musa_python SHARED IMPORTED)
set_target_properties(torch_musa_python PROPERTIES
  IMPORTED_LOCATION "${TORCH_MUSA_PYTHONPATH}/lib/libmusa_python.so"
)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/.."
)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/share/torch_musa_codegen"
)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/share/generated_cuda_compatible/include"
)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/share/generated_cuda_compatible/include/torch/csrc/api/include"
)
list(APPEND DEPENDENT_TARGETS torch_musa_python)

execute_process(
  COMMAND
  ${Python_EXECUTABLE} -c "import os, torch;print(os.path.dirname(torch.__file__))"
  ERROR_QUIET
  OUTPUT_VARIABLE TORCH_PYTHONPATH
)
string(REGEX REPLACE "^(.+)\n$" "\\1" TORCH_PYTHONPATH ${TORCH_PYTHONPATH})

add_library(torch_python SHARED IMPORTED)
set_target_properties(torch_python PROPERTIES
  IMPORTED_LOCATION "${TORCH_PYTHONPATH}/lib/libtorch_python.so"
)
list(APPEND DEPENDENT_TARGETS torch_python)

# Configure Transformer Engine library
set(transformer_engine_SOURCES)
set(PLUGIN_NAME "transformer_engine")
list(APPEND transformer_engine_SOURCES
  common.mu
  transformer_engine.cpp
  activation/gelu.mu
  activation/relu.mu
  activation/swiglu.mu
  comm_gemm_overlap/comm_gemm_overlap.cpp
  comm_gemm_overlap/userbuffers/ipcsocket.cc
  comm_gemm_overlap/userbuffers/userbuffers-host.cpp
  comm_gemm_overlap/userbuffers/userbuffers.mu
  fused_attn/fused_attn.cpp
  fused_attn/thd_utils.mu
  fused_rope/fused_rope.mu
  fused_softmax/scaled_aligned_causal_masked_softmax.mu
  fused_softmax/scaled_masked_softmax.mu
  fused_softmax/scaled_upper_triang_masked_softmax.mu
  gemm/mudnn_gemm.cpp
  permutation/permutation.mu
  permutation/permutation_mask.mu
  recipe/delayed_scaling.mu
  swizzle/swizzle.mu
  transpose/multi_cast_transpose.mu
  transpose/cast_transpose_fusion.mu
  transpose/transpose_fusion.mu
  transpose/transpose.mu
  transpose/cast_transpose.mu
  util/cast.mu
  util/musa_driver.cpp
  util/musa_runtime.cpp
  util/padding.mu
  util/rtc.cpp
  util/system.cpp
  util/mtfp8_cast_transpose.mu
  util/mtfp8_dequantize.mu
)
set_source_files_properties(${transformer_engine_SOURCES}
  PROPERTIES
  MUSA_SOURCE_PROPERTY_FORMAT OBJ
)

musa_add_library(${PLUGIN_NAME} SHARED ${transformer_engine_SOURCES})
target_include_directories(${PLUGIN_NAME} PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${DEPENDENT_TARGETS})

# Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
if (NVTE_UB_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(${PLUGIN_NAME} PUBLIC MPI::MPI_CXX)
    target_compile_definitions(${PLUGIN_NAME} PUBLIC NVTE_UB_WITH_MPI)
endif()

# Helper functions to make header files with C++ strings
function(make_string_header STRING STRING_NAME)
  configure_file(
    "util/string_header.h.in"
    "string_headers/${STRING_NAME}.h"
    @ONLY
  )
endfunction()
function(make_string_header_from_file file_ STRING_NAME)
  file(READ "${file_}" STRING)
  configure_file(
    util/string_header.h.in
    "string_headers/${STRING_NAME}.h"
    @ONLY
  )
endfunction()

# Header files with C++ strings
make_string_header(
  "${MUSA_DIR}/include"
  string_path_musa_include
)
make_string_header_from_file(
  transpose/rtc/cast_transpose_fusion.mu
  string_code_transpose_rtc_cast_transpose_fusion_mu
)
make_string_header_from_file(
  transpose/rtc/cast_transpose.mu
  string_code_transpose_rtc_cast_transpose_mu
)
make_string_header_from_file(
  transpose/rtc/transpose.mu
  string_code_transpose_rtc_transpose_mu
)
make_string_header_from_file(
  utils.muh
  string_code_utils_muh
)
make_string_header_from_file(
  util/math.h
  string_code_util_math_h
)
target_include_directories(${PLUGIN_NAME} PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/string_headers"
)

set_target_properties(${PLUGIN_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

# Install library
install(TARGETS ${PLUGIN_NAME} DESTINATION .)
