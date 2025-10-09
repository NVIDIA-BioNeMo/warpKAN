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

import pytest

import warp as wp
import torch
import torch.nn.functional as F

from warpkan.torch_ext.bspline import BSpline

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
torch.backends.cudnn.enabled = False
wp.build.clear_kernel_cache()

wp.config.cache_kernel = False
torch.manual_seed(1848943)


def bspline(x, grid, coeff, spline_order: int, out_dim: int):
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = (
                (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1:] - x)
            / (grid[:, k + 1:] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )

    return F.linear(bases.contiguous().view(x.size(0), -1),
                    coeff.view(out_dim, -1))


@pytest.mark.warpkan
@pytest.mark.needs_gpu
@pytest.mark.parametrize("in_dim", (1, 6, 26, 50))
@pytest.mark.parametrize("out_dim", (1, 5, 25))
@pytest.mark.parametrize("order", (1, 2, 3, 4,))
@pytest.mark.parametrize("grid_num", (6, 11, 21, 50, 51))
@pytest.mark.parametrize("batch_size", (32, 2**10, 2**12))
def test_bspline(in_dim, out_dim, grid_num, order, batch_size):
    x = 2 * torch.randn(batch_size, in_dim).cuda()
    x.requires_grad = True

    wpbspline = BSpline(in_dim, out_dim, order, grid_num,
                        [-1, 1], enable_standalone_scale_spline=False).cuda()

    grid = wpbspline.grid
    wpbspline.coeff.data = torch.randn_like(wpbspline.coeff.data)
    coeff = torch.clone(wpbspline.coeff.data)
    coeff.requires_grad = True

    ywp = wpbspline(x)
    torch.cuda.synchronize()  # Caching illegal memory access
    ytr = bspline(x, grid, coeff, order, out_dim)
    torch.testing.assert_close(ywp, ytr, atol=1e-4, rtol=1e-4)

    dy = torch.rand_like(ywp)
    dxwp, dcwp = torch.autograd.grad(ywp, (x, wpbspline.coeff), dy)

    dxtr, dctr = torch.autograd.grad(ytr, (x, coeff), dy)

    min_dxtr = dxtr.min()
    max_dxtr = dxtr.max()

    torch.testing.assert_close((dxwp - min_dxtr) / max(1, max_dxtr - min_dxtr),
                               (dxtr - min_dxtr) / max(1, max_dxtr - min_dxtr), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dcwp, dctr, atol=1e-4, rtol=1e-4)

    torch.cuda.synchronize()
