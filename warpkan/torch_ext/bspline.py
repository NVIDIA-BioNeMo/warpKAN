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
from torch import nn, Tensor
import torch.nn.functional as F


from warpkan.torch_ext.fn_bspline import fn_bspline


class BSpline(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 order: int,
                 grid_num: int,
                 grid_range: List[float],
                 scale_noise=0.1,
                 scale_spline=1.0,
                 enable_standalone_scale_spline=False
                 ):
        super(BSpline, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.order = order

        h = (grid_range[1] - grid_range[0]) / grid_num
        grid = (
            (
                torch.arange(-order-1, grid_num + order + 2, 1) * h
                + grid_range[0]
            )
            .expand(in_dim, -1)
            .contiguous()
        )
        self.grid_num = grid_num + 2 * (order + 1) + 1
        self.register_buffer("grid", grid)

        self.coeff = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim, self.grid_num - order - 1)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_dim, in_dim)
            )

        self.scale_noise = scale_noise
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        self.reset_parameters()

    @property
    def scaled_coeff(self):
        return self.coeff * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def reset_parameters(self):

        self.coeff.data.copy_(
            (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) *
            0.001*torch.randn_like(self.coeff)
        )

        if self.enable_standalone_scale_spline:

            torch.nn.init.kaiming_uniform_(
                self.spline_scaler, a=5**0.5 * self.scale_spline)

    def forward(self, x: Tensor):
        input_shape = x.shape
        x = x.reshape((-1,) + (input_shape[-1],))
        spline_output = fn_bspline(
            x, self.scaled_coeff, self.grid, self.order, self.in_dim, self.out_dim, self.grid_num)

        return spline_output.reshape(input_shape[:-1] + (-1, ))
