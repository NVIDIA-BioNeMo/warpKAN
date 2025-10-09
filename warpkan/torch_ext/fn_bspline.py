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
from torch import Tensor

from warpkan.kernel.bspline import make_forward_closure_bspline, make_backward_closure_bspline
from warpkan.torch_ext.utils import get_stream, get_module, add_module


@torch.library.custom_op(
    "warpkan::bspline_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"]
)
def _(x: Tensor, coeff: Tensor, grid: Tensor, order: int, in_dim: int, out_dim: int, grid_num: int) -> Tensor:
    model_name = f"bspline_fwd_{order}_{in_dim}_{out_dim}_{grid_num}_{str(x.dtype).replace('.', '')}"
    try:
        model = get_module(model_name, [str(x.dtype)])
    except ValueError:
        # Module not found, create and add it
        model = add_module(model_name, [str(x.dtype)], make_forward_closure_bspline(
            order, in_dim, out_dim, grid_num, str(x.dtype)))

    device = wp.device_from_torch(x.device)
    stream = get_stream(x.device)
    bs = x.shape[0]
    y = torch.empty((bs, out_dim), device=x.device, dtype=x.dtype)
    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    coeff_wp = wp.from_torch(coeff.detach(), return_ctype=True)
    grid_wp = wp.from_torch(grid.detach(), return_ctype=True)
    y_wp = wp.from_torch(y, return_ctype=True)
    wp.launch(model,
              (bs, out_dim),
              [x_wp, coeff_wp, grid_wp, y_wp],
              device=device,
              stream=stream,
              record_tape=False)
    return y


@torch.library.register_fake("warpkan::bspline_fwd_primitive")
def _(x: Tensor, coeff: Tensor, grid: Tensor, order: int, in_dim: int, out_dim: int, grid_num: int) -> Tensor:
    bs = x.shape[0]
    y = torch.empty((bs, out_dim), device=x.device, dtype=x.dtype)
    return y


@torch.library.custom_op(
    "warpkan::bspline_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"]
)
def _(grad_outputs: Tensor, x: Tensor, coeff: Tensor, grid: Tensor, order: int, in_dim: int, out_dim: int, grid_num: int) -> List[Tensor]:

    model_name = f"bspline_bwd_{order}_{in_dim}_{out_dim}_{grid_num}_{str(x.dtype).replace('.', '')}"
    try:
        model = get_module(model_name, [str(x.dtype)])
    except ValueError:
        model = add_module(model_name, [str(x.dtype)], make_backward_closure_bspline(
            order, in_dim, out_dim, grid_num, str(x.dtype)))

    device = wp.device_from_torch(x.device)
    stream = get_stream(x.device)
    bs = x.shape[0]
    dx = torch.empty_like(x)
    dcoeff = torch.zeros_like(coeff)

    grad_outputs_wp = wp.from_torch(grad_outputs.detach(), return_ctype=True)
    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    coeff_wp = wp.from_torch(coeff.detach(), return_ctype=True)
    grid_wp = wp.from_torch(grid.detach(), return_ctype=True)
    dx_wp = wp.from_torch(dx, return_ctype=True)
    dcoeff_wp = wp.from_torch(dcoeff, return_ctype=True)

    wp.launch(model,
              (bs, in_dim),
              [x_wp, coeff_wp, grid_wp, grad_outputs_wp, dx_wp, dcoeff_wp],
              device=device,
              stream=stream,
              record_tape=False)
    return [dx, dcoeff]


@torch.library.register_fake("warpkan::bspline_bwd_primitive")
def _(grad_outputs: Tensor, x: Tensor, coeff: Tensor, grid: Tensor, order: int, in_dim: int, out_dim: int, grid_num: int) -> List[Tensor]:
    bs = x.shape[0]
    dx = torch.empty_like(x)
    dcoeff = torch.zeros_like(coeff)
    return [dx, dcoeff]


@torch.compiler.allow_in_graph
def bspline_fwd(*args) -> Tensor:
    return torch.ops.warpkan.bspline_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def bspline_bwd(ctx, grad_output) -> List[Tensor]:
    x, coeff, grid = ctx.saved_tensors
    order = ctx.order
    in_dim = ctx.in_dim
    out_dim = ctx.out_dim
    grid_num = ctx.grid_num
    outputs = torch.ops.warpkan.bspline_bwd_primitive(
        grad_output, x, coeff, grid, order, in_dim, out_dim, grid_num)
    return outputs[0], outputs[1], None, None, None, None, None, None


def bspline_setup_fwd_context(ctx, inputs, output):
    x, coeff, grid, order, in_dim, out_dim, grid_num = inputs
    ctx.save_for_backward(x, coeff, grid)
    ctx.order = order
    ctx.in_dim = in_dim
    ctx.out_dim = out_dim
    ctx.grid_num = grid_num


torch.library.register_autograd(
    "warpkan::bspline_fwd_primitive",
    bspline_bwd,
    setup_context=bspline_setup_fwd_context,
)


def fn_bspline(x: Tensor, coeff: Tensor, grid:
               Tensor, order: int, in_dim: int, out_dim: int, grid_num: int) -> Tensor:
    output = torch.ops.warpkan.bspline_fwd_primitive(
        x, coeff, grid, order, in_dim, out_dim, grid_num)
    return output
