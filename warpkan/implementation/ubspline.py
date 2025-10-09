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


from typing import Optional

import warp as wp
from warpkan.kernel.ubspline import make_forward_closure_ubspline, make_backward_closure_ubspline


class UBSpline:
    def __init__(self, in_dim: int,  out_dim: int, order: int = 3,  grid_delta: float = 1.0) -> None:
        """


        Args:
            in_dim (int): _description_
            out_dim (int): _description_
            order (int, optional): _description_. Defaults to 3.
            grid_delta (float): _description_. Defaults to 1.0
        """
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_delta = grid_delta

        self.forward = make_forward_closure_ubspline(order, in_dim, out_dim)

        self.backward = make_backward_closure_ubspline(order, in_dim, out_dim)

    def __repr__(self) -> str:
        return f"WpUBSpline {self.order}, {self.in_dim}, {self.out_dim}, {self.grid_delta}"

    def adj_launch(self,
                   x: wp.array2d(dtype=float),
                   coeff: wp.array3d(dtype=float),
                   grid: wp.array2d(dtype=float),
                   idg: wp.array2d(dtype=int),
                   dy: wp.array2d(dtype=float),
                   dx: wp.array2d(dtype=float),
                   dcoeff: wp.array3d(dtype=float),
                   stream=None,
                   ):

        bs = x.shape[0]
        wp.launch(self.backward, (bs, self.in_dim), [
                  x, coeff, grid, idg, self.grid_delta, dy, dx, dcoeff], record_tape=False, device=x.device,
                  )

    def launch(self,
               x: wp.array2d(dtype=float),
               coeff: wp.array3d(dtype=float),
               grid: wp.array2d(dtype=float),
               idg: wp.array2d(dtype=int),
               y: wp.array2d(dtype=float),
               stream=None):

        bs = x.shape[0]
        wp.launch(self.forward, (bs, self.out_dim), [
                  x, coeff, grid, idg, self.grid_delta, y], record_tape=False, device=x.device)
