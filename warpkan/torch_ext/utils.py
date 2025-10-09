# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List
import warp as wp
import torch

MODULES = {}


def get_module(name: str, dtype: List[str]):
    """
    Get the module for the given name and dtype
    """
    full_name = name + "_" + "_".join(get_dtype(d) for d in dtype)
    if full_name not in MODULES:
        raise ValueError(f"Module {full_name} not found")
    return MODULES[full_name]


def add_module(name: str, dtype: List[str], kernel: wp.Kernel):
    """
    Add the module for the given name and dtype
    """
    full_name = name + "_" + "_".join(get_dtype(d) for d in dtype)
    if full_name not in MODULES:
        MODULES[full_name] = kernel
    return MODULES[full_name]


def get_dtype(dtype: str):
    """
    Get the dtype for the given dtype
    WIP
    """
    if dtype.endswith("16"):
        return "fp16"
    elif dtype.endswith("32"):
        return "fp32"
    elif dtype.endswith("64"):
        return "fp64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_wp_fp_dtype(dtype: str):
    """
    Get the warp dtype for the given dtype
    WIP
    """
    if dtype.endswith("16"):
        return wp.float16
    elif dtype.endswith("32"):
        return wp.float32
    elif dtype.endswith("64"):
        return wp.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_stream(device: torch.device):
    """
    Get the stream for the given device
    """
    if device.type == "cuda":
        return wp.stream_from_torch(torch.cuda.current_stream(device))
    else:
        return None
