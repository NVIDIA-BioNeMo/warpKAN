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

from warpkan.torch_ext.kanlinear import KANLinear


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_num: int = 5,
        order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=torch.nn.SiLU,
        grid_range: List[float] = [-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_num = grid_num
        self.order = order

        self.layers = torch.nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_dim,
                    out_dim,
                    order,
                    grid_num,
                    grid_range,
                    scale_noise,
                    scale_base,
                    scale_spline,
                    enable_standalone_scale_spline,
                    base_activation
                )
            )

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
