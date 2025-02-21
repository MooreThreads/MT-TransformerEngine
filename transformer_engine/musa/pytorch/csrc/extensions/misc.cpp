/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

size_t get_mublas_version() { return MUBLAS_VERSION_MAJOR * 10000ul + MUBLAS_VERSION_MINOR * 100ul + MUBLAS_VERSION_PATCH; }

size_t get_mudnn_version() { return ::musa::dnn::GetVersion(); }
