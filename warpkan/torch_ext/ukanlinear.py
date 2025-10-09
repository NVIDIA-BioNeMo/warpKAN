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


import math

import torch
from torch import nn, Tensor

import torch.nn.functional as F
from warpkan.torch_ext.ubspline import UBSpline


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, order: int):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.order = order
        self.d_model = d_model
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x: Tensor):
        y = torch.empty(
            (x.shape[0], x.shape[1], self.d_model), device=x.device)
        y[:, :, 0::2] = torch.sin(x.unsqueeze(-1) * self.div_term)
        y[:, :, 1::2] = torch.cos(x.unsqueeze(-1) * self.div_term)

        return y.detach()  # TODO: check if detach is right


class UKANLinear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 order: int,
                 grid_delta: float,
                 embedding_dim: int,
                 scale_base=1.0,
                 base_activation=torch.nn.SiLU,
                 enable_standalone_scale_spline: bool = True,
                 has_mlp: bool = True):
        super(UKANLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.order = order
        self.grid_delta = grid_delta
        self.embedding_dim = embedding_dim
        self.base_activation = base_activation()

        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        # coefficient generator
        self.coeff_gen = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim),
                                       self.base_activation,
                                       nn.Linear(
                                           embedding_dim, self.out_dim * (self.order + 1))
                                       )

        # UBSpline module
        self.ubspline: UBSpline = UBSpline(
            in_dim, out_dim, order, grid_delta, enable_standalone_scale_spline)

        # grid index embedding
        self.embed = SinusoidalPositionalEncoding(embedding_dim, self.order)

        # feature index embedding
        self.embed_loc = nn.Embedding(in_dim, embedding_dim)
        loc_embed = self.embed_loc(torch.arange(in_dim)).reshape(1, -1)
        loc_embed = torch.nn.Parameter(loc_embed, requires_grad=False)
        self.register_buffer("loc_embed", loc_embed)

        # mlp component of KANLinear
        self.has_mlp = has_mlp
        if has_mlp:
            self.base_weight = torch.nn.Parameter(
                torch.Tensor(out_dim, in_dim))

            self.scale_base = scale_base

        self.reset_parameters()

    def reset_parameters(self):
        if self.has_mlp:
            torch.nn.init.kaiming_uniform_(
                self.base_weight, a=5 ** 0.5 * self.scale_base)
        # TODO: reset self.coeff_gen

    def forward(self, x: Tensor):
        input_shape = x.shape
        x = x.reshape((-1, x.shape[-1]))

        with torch.no_grad():
            id_grids = torch.floor_divide(x, self.grid_delta)
            id_grids_group = torch.floor_divide(id_grids, self.order + 1)

            group_embeds = self.embed(id_grids_group)
            group_embeds_prev = self.embed(id_grids_group - 1)

        loc_embed = self.loc_embed.view(-1, self.in_dim, self.embedding_dim).repeat(
            x.shape[0], 1, 1)

        coeff = self.coeff_gen(torch.cat([loc_embed,
                                          group_embeds],
                                         dim=-1)
                               ).reshape(-1, self.in_dim * self.out_dim, self.order+1)

        coeff_prev = self.coeff_gen(torch.cat([loc_embed,
                                               group_embeds_prev],
                                              dim=-1)
                                    ).reshape(-1, self.in_dim * self.out_dim, self.order+1)

        id_grids = id_grids.int()
        coeff = torch.cat([coeff_prev, coeff], dim=-1)

        spline_output = self.ubspline(x, coeff, id_grids.int())

        if self.has_mlp:
            base_output = F.linear(self.base_activation(x), self.base_weight)

            return (spline_output + base_output).reshape(input_shape[:-1] + (-1, ))

        return spline_output.reshape(input_shape[:-1] + (-1, ))
