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
from warpkan.kernel.bspline import make_forward_closure_bspline, make_backward_closure_bspline


class BSpline:
    def __init__(self, in_dim: int,  out_dim: int, order: int = 3,  grid_num: int = 11, grid: Optional[wp.array3d] = None, internal_params: bool = False, coeff: Optional[wp.array3d] = None) -> None:
        """


        Args:
            in_dim (int): _description_
            out_dim (int): _description_
            order (int, optional): _description_. Defaults to 3.
            grid_num (int, optional): _description_. Defaults to 11.
            grid (Optional[wp.array3d], optional): _description_. Defaults to None.
            internal_params (bool, optional): _description_. Defaults to False.
            coeff (Optional[wp.array3d], optional): _description_. Defaults to None.
        """
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.internal_params = internal_params
        self.grid_num = grid_num
        self.grid = grid

        if internal_params == True:
            assert self.grid.shape[-1] == self.grid_num
            self.coeff = coeff
            # TODO: better errmsg
            assert self.coeff.shape[-1] + self.order + \
                1 == self.grid.shape[-1], "Coefficients and grid shape does not match!"

        # TODO: Add it
        assert internal_params == False, "Not supported"

        self.forward = make_forward_closure_bspline(
            order, in_dim, out_dim, self.grid_num)
        self.backward = make_backward_closure_bspline(
            order, in_dim, out_dim, self.grid_num)

    def __repr__(self) -> str:
        return f"WpBSpline {self.order}, {self.in_dim}, {self.out_dim}, {self.grid_num}"

    def set_params(self, coeff: wp.array3d, grid: wp.array2d):
        assert self.internal_params == True
        self.coeff = coeff
        self.grid = grid

    def adj_launch(self,
                   x: wp.array2d(dtype=float),
                   coeff: wp.array3d(dtype=float),
                   grid: wp.array2d(dtype=float),
                   dy: wp.array2d(dtype=float),
                   dx: wp.array2d(dtype=float),
                   dcoeff: wp.array3d(dtype=float),
                   stream=None,

                   ):
        bs = x.shape[0]
        wp.launch(self.backward, (bs, self.in_dim), [
                  x, coeff, grid, dy, dx, dcoeff], record_tape=False, device=x.device,
                  )

    def launch(self,
               x: wp.array2d(dtype=float),
               coeff: wp.array3d(dtype=float),
               grid: wp.array2d(dtype=float),
               y: wp.array2d(dtype=float),
               stream=None):

        bs = x.shape[0]
        wp.launch(self.forward, (bs, self.out_dim), [
                  x, coeff, grid, y], record_tape=False, device=x.device)
