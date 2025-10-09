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


import torch
from torch import nn, Tensor

from warpkan.torch_ext.fn_ubspline import fn_ubspline


class UBSpline(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 order: int,
                 grid_delta: float,
                 enable_standalone_scale_spline: bool = True,
                 ):
        super(UBSpline, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.order = order
        self.grid_delta = grid_delta
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                0.001*torch.randn(1, out_dim * in_dim, 1)
            )

    def forward(self, x: Tensor, coeff: Tensor, grid_ids: Tensor):
        if self.enable_standalone_scale_spline:
            coeff = coeff * self.spline_scaler

        spline_output = fn_ubspline(
            x, coeff, grid_ids, self.grid_delta, self.order, self.in_dim, self.out_dim)

        return spline_output
