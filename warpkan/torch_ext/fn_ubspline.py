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

from warpkan.kernel.ubspline import make_forward_closure_ubspline, make_backward_closure_ubspline
from warpkan.torch_ext.utils import get_stream, get_module, add_module


@torch.library.custom_op(
    "warpkan::ubspline_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"]
)
def _(x: Tensor, coeff: Tensor, grid_ids: Tensor, grid_delta: float, order: int, in_dim: int, out_dim: int) -> Tensor:
    model_name = f"ubspline_fwd_{order}_{in_dim}_{out_dim}_{str(x.dtype).replace('.', '')}"
    try:
        model = get_module(model_name, [str(x.dtype)])
    except ValueError:
        # Module not found, create and add it
        model = add_module(model_name, [str(x.dtype)], make_forward_closure_ubspline(
            order, in_dim, out_dim, str(x.dtype)))

    device = wp.device_from_torch(x.device)
    stream = get_stream(x.device)
    bs = x.shape[0]
    y = torch.empty((bs, out_dim), device=x.device, dtype=x.dtype)
    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    coeff_wp = wp.from_torch(coeff.detach(), return_ctype=True)
    grid_ids_wp = wp.from_torch(grid_ids.detach(), return_ctype=True)
    y_wp = wp.from_torch(y, return_ctype=True)
    wp.launch(model,
              (bs, out_dim),
              [x_wp, coeff_wp, grid_ids_wp, grid_delta, y_wp],
              device=device,
              stream=stream,
              record_tape=False)
    return y


@torch.library.register_fake("warpkan::ubspline_fwd_primitive")
def _(x: Tensor, coeff: Tensor, grid_ids: Tensor, grid_delta: float, order: int, in_dim: int, out_dim: int) -> Tensor:
    bs = x.shape[0]
    y = torch.empty((bs, out_dim), device=x.device, dtype=x.dtype)
    return y


@torch.library.custom_op(
    "warpkan::ubspline_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"]
)
def _(grad_outputs: Tensor, x: Tensor, coeff: Tensor, grid_ids: Tensor, grid_delta: float, order: int, in_dim: int, out_dim: int) -> List[Tensor]:
    model_name = f"ubspline_bwd_{order}_{in_dim}_{out_dim}_{str(x.dtype).replace('.', '')}"
    try:
        model = get_module(model_name, [str(x.dtype)])
    except ValueError:
        model = add_module(model_name, [str(x.dtype)], make_backward_closure_ubspline(
            order, in_dim, out_dim, str(x.dtype)))

    device = wp.device_from_torch(x.device)
    stream = get_stream(x.device)
    bs = x.shape[0]
    dx = torch.empty((bs, in_dim), device=x.device, dtype=x.dtype)
    dcoeff = torch.zeros((bs, in_dim * out_dim, (order + 1)
                         * 2), device=x.device, dtype=x.dtype)

    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    coeff_wp = wp.from_torch(coeff.detach(), return_ctype=True)
    grid_ids_wp = wp.from_torch(grid_ids.detach(), return_ctype=True)
    grad_outputs_wp = wp.from_torch(
        grad_outputs.detach().contiguous(), return_ctype=True)
    dx_wp = wp.from_torch(dx, return_ctype=True)
    dcoeff_wp = wp.from_torch(dcoeff, return_ctype=True)

    wp.launch(model,
              (bs, in_dim),
              [x_wp, coeff_wp, grid_ids_wp, grid_delta,
                  grad_outputs_wp, dx_wp, dcoeff_wp],
              device=device,
              stream=stream,
              record_tape=False)
    return [dx, dcoeff]


@torch.library.register_fake("warpkan::ubspline_bwd_primitive")
def _(grad_outputs: Tensor, x: Tensor, coeff: Tensor, grid_ids: Tensor, grid_delta: float, order: int, in_dim: int, out_dim: int) -> List[Tensor]:
    bs = x.shape[0]
    dx = torch.empty((bs, in_dim), device=x.device, dtype=x.dtype)
    dcoeff = torch.empty((bs, in_dim * out_dim, (order + 1)
                         * 2), device=x.device, dtype=x.dtype)
    return [dx, dcoeff]


@torch.compiler.allow_in_graph
def ubspline_fwd(*args) -> Tensor:
    return torch.ops.warpkan.ubspline_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def ubspline_bwd(ctx, grad_output) -> List[Tensor]:
    x, coeff, grid_ids = ctx.saved_tensors
    order = ctx.order
    in_dim = ctx.in_dim
    out_dim = ctx.out_dim
    grid_delta = ctx.grid_delta
    outputs = torch.ops.warpkan.ubspline_bwd_primitive(
        grad_output, x, coeff, grid_ids, grid_delta, order, in_dim, out_dim)
    return outputs[0], outputs[1], None, None, None, None, None, None


def ubspline_setup_fwd_context(ctx, inputs, output):
    x, coeff, grid_ids, grid_delta, order, in_dim, out_dim = inputs
    ctx.order = order
    ctx.in_dim = in_dim
    ctx.out_dim = out_dim
    ctx.grid_delta = grid_delta
    ctx.save_for_backward(x, coeff, grid_ids)


torch.library.register_autograd(
    "warpkan::ubspline_fwd_primitive",
    ubspline_bwd,
    setup_context=ubspline_setup_fwd_context,
)


def fn_ubspline(x: Tensor, coeff: Tensor, grid_ids: Tensor, grid_delta: float, order: int, in_dim: int, out_dim: int) -> Tensor:
    output = torch.ops.warpkan.ubspline_fwd_primitive(
        x, coeff, grid_ids, grid_delta, order, in_dim, out_dim)
    return output

# if __name__ == "__main__":
#     wp.init()

#     order = 3
#     in_dim = 2
#     out_dim = 3
#     x = torch.randn(1, in_dim, device="cuda").double()
#     x.requires_grad = True
#     coeff = torch.randn(1, in_dim * out_dim, (order + 1)*2, device="cuda").double()
#     coeff.requires_grad = True
#     grid_ids = torch.randint(0, order + 1, (1, in_dim), device="cuda")
#     grid_delta = 1.0

#     torch.autograd.gradcheck(fn_ubspline, (x, coeff, grid_ids, grid_delta, order, in_dim, out_dim))
#     y = fn_ubspline(x, coeff, grid_ids, grid_delta, order, in_dim, out_dim)
#     print(y)
#     dy = torch.randn_like(y)
#     dx, dcoeff = torch.autograd.grad(y, (x, coeff), dy)
#     print(dx)
#     print(dcoeff)
