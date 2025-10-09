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

import torch
from torch import nn, Tensor

import torch.nn.functional as F
from warpkan.torch_ext.bspline import BSpline


class KANLinear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 order: int,
                 grid_num: int,
                 grid_range: List[float],
                 scale_noise=0.1,
                 scale_base=1.0,
                 scale_spline=1.0,
                 enable_standalone_scale_spline=False,
                 base_activation=torch.nn.SiLU,
                 has_mlp: bool = False):
        super(KANLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.order = order
        self.has_mlp = has_mlp

        if self.has_mlp:
            self.base_weight = torch.nn.Parameter(
                torch.Tensor(out_dim, in_dim))

            self.scale_base = scale_base
            self.base_activation = base_activation()

        self.bspline = BSpline(in_dim,
                               out_dim,
                               order,
                               grid_num,
                               grid_range,
                               scale_noise,
                               scale_spline,
                               enable_standalone_scale_spline
                               )

        self.reset_parameters()

    @property
    def scaled_coeff(self):
        return self.coeff * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def reset_parameters(self):
        if self.has_mlp:
            torch.nn.init.kaiming_uniform_(
                self.base_weight, a=5 ** 0.5 * self.scale_base)
        self.bspline.reset_parameters()

    def forward(self, x: Tensor):

        # shape = x.shape
        spline_output = self.bspline(x)  # .view(shape[:-1] + (self.out_dim,))

        if self.has_mlp:
            base_output = F.linear(self.base_activation(x), self.base_weight)
            return spline_output + base_output

        return spline_output
