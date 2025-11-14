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

import warp as wp
from warpkan.kernel.utils import get_wp_fp_dtype

wp.config.enable_backward = False


def make_forward_closure_bspline(order: int, in_dim: int, out_dim: int, grid_num: int, dtype: str = "float32"):
    dtype_wp = get_wp_fp_dtype(dtype)

    @wp.func
    def binary_search(grid: wp.array2d(dtype=dtype_wp), x: dtype_wp, i: int,  num_grid: int):
        """

        Args:
            x (float): _description_
            i (int): _description_
            num_grid (int): _description_
            grid (wp.array2d, optional): _description_. Defaults to float).

        Returns:
            int: index of x in grid
        """
        if x < grid[i, 0] or x >= grid[i, num_grid-1]:
            return -1
        lo = int(0)
        hi = num_grid
        ans = int(0)
        while lo < hi:
            mid = (lo + hi) // 2
            if grid[i, mid] <= x:
                lo = mid + 1
                ans = mid
            else:
                hi = mid
        return ans

    class vec2(wp.types.vector(length=2, dtype=dtype_wp)):
        pass

    class vec3(wp.types.vector(length=3, dtype=dtype_wp)):
        pass

    class vec4(wp.types.vector(length=4, dtype=dtype_wp)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype_wp)):
        pass
    if order == 0:
        def bspline(x: wp.array2d(dtype=dtype_wp),
                    coeff: wp.array3d(dtype=dtype_wp),
                    grid: wp.array2d(dtype=dtype_wp),
                    y: wp.array2d(dtype=dtype_wp)
                    ):

            b, j = wp.tid()

            val = dtype_wp(0.0)

            for i in range(in_dim):
                cvec = dtype_wp(0.0)
                if x[b, i] >= grid[i, grid_num - 1] or grid[i, 0] > x[b, i]:
                    val = val
                else:
                    idg = binary_search(grid, x[b, i], i, grid_num)
                    cvec = wp.where(idg >= 0 and idg <
                                     grid_num - 1, coeff[j, i, idg], dtype_wp(0.0))
                    val = val + cvec

            y[b, j] = val

    elif order == 1:
        def bspline(x: wp.array2d(dtype=dtype_wp),
                    coeff: wp.array3d(dtype=dtype_wp),
                    grid: wp.array2d(dtype=dtype_wp),
                    y: wp.array2d(dtype=dtype_wp)
                    ):

            b, j = wp.tid()

            val = dtype_wp(0.0)
            tvec = vec2(dtype_wp(1.0))

            mat20 = vec2(dtype_wp(-1.), dtype_wp(1.))
            mat21 = vec2(dtype_wp(1.0), dtype_wp(0.))

            for i in range(in_dim):
                cvec = vec2(dtype_wp(0.0))
                dg = grid[i, 1] - grid[i, 0]

                if x[b, i] >= grid[i, grid_num - 1] or grid[i, 0] > x[b, i]:
                    val = val
                else:
                    idg = binary_search(grid, x[b, i], i, grid_num)
                    g = grid[i, idg]

                    t = (x[b, i] - g) / dg
                    tvec[0] = t

                    cnt = int(1)
                    for cidx in range(idg, idg - 2, -1):
                        cvec[cnt] = wp.where(
                            cidx >= 0 and cidx < grid_num - 2, coeff[j, i, cidx], dtype_wp(0.0))
                        cnt = cnt - int(1)

                    res_right = vec2(wp.dot(tvec, mat20),
                                     wp.dot(tvec, mat21)
                                     )

                    val = val + wp.dot(res_right, cvec)

            y[b, j] = val

    elif order == 2:
        def bspline(x: wp.array2d(dtype=dtype_wp),
                    coeff: wp.array3d(dtype=dtype_wp),
                    grid: wp.array2d(dtype=dtype_wp),
                    y: wp.array2d(dtype=dtype_wp)
                    ):

            b, j = wp.tid()

            val = dtype_wp(0.0)
            tvec = vec3(dtype_wp(1.0))

            mat30 = vec3(dtype_wp(1.0), dtype_wp(-2.), dtype_wp(1.))
            mat31 = vec3(dtype_wp(-2.), dtype_wp(2.0), dtype_wp(1.))
            mat32 = vec3(dtype_wp(1.0), dtype_wp(0.0), dtype_wp(0.))

            for i in range(in_dim):
                cvec = vec3(dtype_wp(0.0))
                dg = grid[i, 1] - grid[i, 0]

                if x[b, i] >= grid[i, grid_num - 1] or grid[i, 0] > x[b, i]:
                    val = val
                else:
                    idg = binary_search(grid, x[b, i], i, grid_num)
                    g = grid[i, idg]

                    t = (x[b, i] - g) / dg
                    tvec[1] = t
                    tvec[0] = t * t

                    cnt = int(2)
                    for cidx in range(idg, idg - 3, -1):
                        cvec[cnt] = wp.where(
                            cidx >= 0 and cidx < grid_num - 3, coeff[j, i, cidx], dtype_wp(0.0))
                        cnt = cnt - int(1)

                    res_right = vec3(wp.dot(tvec, mat30),
                                     wp.dot(tvec, mat31),
                                     wp.dot(tvec, mat32)
                                     ) / dtype_wp(2.0)

                    val = val + wp.dot(res_right, cvec)

            y[b, j] = val

    elif order == 3:

        def bspline(x: wp.array2d(dtype=dtype_wp),
                    coeff: wp.array3d(dtype=dtype_wp),
                    grid: wp.array2d(dtype=dtype_wp),
                    y: wp.array2d(dtype=dtype_wp)
                    ):

            b, j = wp.tid()

            mat43 = vec4(dtype_wp(1.), dtype_wp(
                4.), dtype_wp(1.0), dtype_wp(0.))
            mat42 = vec4(dtype_wp(-3.), dtype_wp(0.),
                         dtype_wp(3.), dtype_wp(0.))
            mat41 = vec4(dtype_wp(3.), dtype_wp(-6.),
                         dtype_wp(3.), dtype_wp(0.))
            mat40 = vec4(dtype_wp(-1.), dtype_wp(3.),
                         dtype_wp(-3.), dtype_wp(1.))

            val = dtype_wp(0.0)
            tvec = vec4(dtype_wp(1.0))

            for i in range(in_dim):
                cvec = vec4(dtype_wp(0.0))
                dg = grid[i, 1] - grid[i, 0]

                if x[b, i] >= grid[i, grid_num - 1] or grid[i, 0] > x[b, i]:
                    val = val
                else:
                    idg = binary_search(grid, x[b, i], i, grid_num)
                    g = grid[i, idg]

                    t = (x[b, i] - g) / dg
                    cnt = int(3)
                    for cidx in range(idg, idg - 4, -1):
                        cvec[cnt] = wp.where(
                            cidx >= 0 and cidx < grid_num - 4, coeff[j, i, cidx], dtype_wp(0.0))
                        cnt = cnt - int(1)

                    for w in range(2, -1, -1):
                        tvec[w] = tvec[w+1] * t

                    res_left = vec4(wp.dot(mat40, cvec),
                                    wp.dot(mat41, cvec),
                                    wp.dot(mat42, cvec),
                                    wp.dot(mat43, cvec),
                                    )
                    val = val + wp.dot(tvec, res_left)

            y[b, j] = val / dtype_wp(6.)

    elif order == 4:

        def bspline(x: wp.array2d(dtype=dtype_wp),
                    coeff: wp.array3d(dtype=dtype_wp),
                    grid: wp.array2d(dtype=dtype_wp),
                    y: wp.array2d(dtype=dtype_wp)
                    ):

            b, j = wp.tid()

            mat50 = vec5(dtype_wp(1.0), dtype_wp(-4.),
                         dtype_wp(6.0), dtype_wp(-4.), dtype_wp(1.))
            mat51 = vec5(dtype_wp(-4.), dtype_wp(12.),
                         dtype_wp(-6.), dtype_wp(-12.), dtype_wp(11.))
            mat52 = vec5(dtype_wp(6.), dtype_wp(-12.),
                         dtype_wp(-6.), dtype_wp(12.), dtype_wp(11.))
            mat53 = vec5(dtype_wp(-4.), dtype_wp(4.),
                         dtype_wp(6.), dtype_wp(4.), dtype_wp(1.))
            mat54 = vec5(dtype_wp(1.), dtype_wp(
                0.), dtype_wp(0.), dtype_wp(0.), dtype_wp(0.))

            val = dtype_wp(0.0)
            tvec = vec5(dtype_wp(1.0))

            for i in range(in_dim):
                cvec = vec5(dtype_wp(0.0))
                dg = grid[i, 1] - grid[i, 0]

                if x[b, i] >= grid[i, grid_num - 1] or grid[i, 0] > x[b, i]:
                    val = val
                else:
                    idg = binary_search(grid, x[b, i], i, grid_num)
                    g = grid[i, idg]

                    t = (x[b, i] - g) / dg
                    cnt = int(4)
                    for cidx in range(idg, idg - 5, -1):
                        cvec[cnt] = wp.where(
                            cidx >= 0 and cidx < grid_num - 5, coeff[j, i, cidx], dtype_wp(0.0))
                        cnt = cnt - int(1)

                    for w in range(3, -1, -1):
                        tvec[w] = tvec[w+1] * t

                    res_right = vec5(wp.dot(tvec, mat50),
                                     wp.dot(tvec, mat51),
                                     wp.dot(tvec, mat52),
                                     wp.dot(tvec, mat53),
                                     wp.dot(tvec, mat54),
                                     )
                    val = val + wp.dot(res_right, cvec)

            y[b, j] = val / dtype_wp(24.)

    key = f"bspline_fwd_{order}_{in_dim}_{out_dim}_{grid_num}_{dtype.replace('.', '')}"

    module = wp.context.Module(key, loader=None)

    return wp.Kernel(func=bspline, key=key, module=module)


def make_backward_closure_bspline(order: int, in_dim: int, out_dim: int, grid_num: int, dtype: str = "float32"):
    dtype_wp = get_wp_fp_dtype(dtype)

    class vec2(wp.types.vector(length=2, dtype=dtype_wp)):
        pass

    class vec3(wp.types.vector(length=3, dtype=dtype_wp)):
        pass

    class vec4(wp.types.vector(length=4, dtype=dtype_wp)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype_wp)):
        pass

    @wp.func
    def binary_search(grid: wp.array2d(dtype=dtype_wp), x: dtype_wp, i: int,  num_grid: int):
        """

        Args:
            x (float): _description_
            i (int): _description_
            num_grid (int): _description_
            grid (wp.array2d, optional): _description_. Defaults to float).

        Returns:
            int: index of x in grid
        """
        if x < grid[i, 0] or x >= grid[i, num_grid-1]:
            return -1
        lo = int(0)
        hi = num_grid
        ans = int(0)
        while lo < hi:
            mid = (lo + hi) // 2
            if grid[i, mid] <= x:
                lo = mid + 1
                ans = mid
            else:
                hi = mid
        return ans

    if order == 0:
        def adj_bspline(x: wp.array2d(dtype=dtype_wp),
                        coeff: wp.array3d(dtype=dtype_wp),
                        grid: wp.array2d(dtype=dtype_wp),
                        dy: wp.array2d(dtype=dtype_wp),
                        dx: wp.array2d(dtype=dtype_wp),
                        dcoeff: wp.array3d(dtype=dtype_wp)
                        ):
            b, i = wp.tid()
            dx_ = dtype_wp(0.0)
            idg = binary_search(grid, x[b, i], i, grid_num)

            if idg < 0:
                dx[b, i] = dx_
                return

            for j in range(out_dim):
                wp.atomic_add(dcoeff, j, i, idg,  dy[b, j])

            dx[b, i] = dx_

    elif order == 1:
        def adj_bspline(x: wp.array2d(dtype=dtype_wp),
                        coeff: wp.array3d(dtype=dtype_wp),
                        grid: wp.array2d(dtype=dtype_wp),
                        dy: wp.array2d(dtype=dtype_wp),
                        dx: wp.array2d(dtype=dtype_wp),
                        dcoeff: wp.array3d(dtype=dtype_wp)
                        ):
            b, i = wp.tid()
            dx_ = dtype_wp(0.0)
            idg = binary_search(grid, x[b, i], i, grid_num)

            if idg < 0:
                dx[b, i] = dx_
                return
            tvec = vec2(dtype_wp(1.0))
            dtvec = vec2(dtype_wp(0.0))
            dtvec[0] = dtype_wp(1.0)

            dg = grid[i, 1] - grid[i, 0]
            g = grid[i, idg]

            t = (x[b, i] - g) / dg
            tvec[0] = t

            mat20 = vec2(dtype_wp(-1.), dtype_wp(1.))
            mat21 = vec2(dtype_wp(1.0), dtype_wp(0.))

            for j in range(out_dim):
                cvec = vec2(dtype_wp(0.0))
                dy_ = dy[b, j]

                cnt = int(1)
                for cidx in range(idg, idg - 2, -1):
                    cvec[cnt] = wp.where(
                        cidx >= 0 and cidx < grid_num - 2, coeff[j, i, cidx], dtype_wp(0.0))
                    cnt = cnt - int(1)

                res_left = vec2(wp.dot(mat20, cvec),
                                   wp.dot(mat21, cvec)
                                   )

                res_right = vec2(wp.dot(mat20, tvec),
                                    wp.dot(mat21, tvec),
                                    )
                cnt = int(1)
                for cidx in range(idg, idg - 2, -1):
                    if cidx >= 0 and cidx < grid_num - 2:

                        wp.atomic_add(dcoeff, j, i, cidx,
                                      res_right[cnt] * dy_)
                    cnt = cnt - int(1)
                dx_ = dx_ + (wp.dot(dtvec, res_left) * dy_ / dg)

            dx[b, i] = dx_
    elif order == 2:
        def adj_bspline(x: wp.array2d(dtype=dtype_wp),
                        coeff: wp.array3d(dtype=dtype_wp),
                        grid: wp.array2d(dtype=dtype_wp),
                        dy: wp.array2d(dtype=dtype_wp),
                        dx: wp.array2d(dtype=dtype_wp),
                        dcoeff: wp.array3d(dtype=dtype_wp)
                        ):
            b, i = wp.tid()
            dx_ = dtype_wp(0.0)
            idg = binary_search(grid, x[b, i], i, grid_num)

            if idg < 0:
                dx[b, i] = dx_
                return
            tvec = vec3(dtype_wp(1.0))
            dtvec = vec3(dtype_wp(0.0))

            dg = grid[i, 1] - grid[i, 0]
            g = grid[i, idg]

            t = (x[b, i] - g) / dg
            tvec[1] = t
            tvec[0] = t * t

            dtvec[1] = dtype_wp(1.0)
            dtvec[0] = dtype_wp(2.0) * t

            mat30 = vec3(dtype_wp(1.0), dtype_wp(-2.), dtype_wp(1.))
            mat31 = vec3(dtype_wp(-2.), dtype_wp(2.0), dtype_wp(1.))
            mat32 = vec3(dtype_wp(1.0), dtype_wp(0.0), dtype_wp(0.))

            mat03 = vec3(dtype_wp(1.0), dtype_wp(-2.), dtype_wp(1.))
            mat13 = vec3(dtype_wp(-2.), dtype_wp(2.0), dtype_wp(0.))
            mat23 = vec3(dtype_wp(1.0), dtype_wp(1.0), dtype_wp(0.))

            for j in range(out_dim):
                cvec = vec3(dtype_wp(0.0))
                dy_ = dy[b, j] / dtype_wp(2.0)

                cnt = int(2)
                for cidx in range(idg, idg - 3, -1):
                    cvec[cnt] = wp.where(
                        cidx >= 0 and cidx < grid_num - 3, coeff[j, i, cidx], dtype_wp(0.0))
                    cnt = cnt - int(1)

                res_left = vec3(wp.dot(mat03, cvec),
                                wp.dot(mat13, cvec),
                                wp.dot(mat23, cvec)
                                )

                res_right = vec3(wp.dot(mat30, tvec),
                                 wp.dot(mat31, tvec),
                                 wp.dot(mat32, tvec),
                                 )
                cnt = int(2)
                for cidx in range(idg, idg - 3, -1):
                    if cidx >= 0 and cidx < grid_num - 3:

                        wp.atomic_add(dcoeff, j, i, cidx,
                                      res_right[cnt] * dy_)
                    cnt = cnt - int(1)
                dx_ = dx_ + (wp.dot(dtvec, res_left) * dy_ / dg)

            dx[b, i] = dx_

    elif order == 3:
        def adj_bspline(x: wp.array2d(dtype=dtype_wp),
                        coeff: wp.array3d(dtype=dtype_wp),
                        grid: wp.array2d(dtype=dtype_wp),
                        dy: wp.array2d(dtype=dtype_wp),
                        dx: wp.array2d(dtype=dtype_wp),
                        dcoeff: wp.array3d(dtype=dtype_wp)
                        ):
            b, i = wp.tid()

            mat43 = vec4(dtype_wp(1.), dtype_wp(
                4.), dtype_wp(1.), dtype_wp(0.))
            mat42 = vec4(dtype_wp(-3.), dtype_wp(0.),
                         dtype_wp(3.), dtype_wp(0.))
            mat41 = vec4(dtype_wp(3.), dtype_wp(-6.),
                         dtype_wp(3.), dtype_wp(0.))
            mat40 = vec4(dtype_wp(-1.), dtype_wp(3.),
                         dtype_wp(-3.), dtype_wp(1.))

            mat34 = vec4(dtype_wp(1.0), dtype_wp(
                0.), dtype_wp(0.), dtype_wp(0.))
            mat24 = vec4(dtype_wp(-3.), dtype_wp(3.),
                         dtype_wp(3.), dtype_wp(1.0))
            mat14 = vec4(dtype_wp(3.), dtype_wp(-6.),
                         dtype_wp(0.), dtype_wp(4.))
            mat04 = vec4(dtype_wp(-1.), dtype_wp(3.),
                         dtype_wp(-3.), dtype_wp(1.))

            dx_ = dtype_wp(0.0)

            if x[b, i] >= grid[i, grid_num - 1] or grid[i, 0] >= x[b, i]:
                dx[b, i] = dx_
                return
            idg = binary_search(grid, x[b, i], i, grid_num)

            g = grid[i, idg]
            dg = grid[i, 1] - grid[i, 0]

            tvec = vec4(dtype_wp(0.0))
            dtvec = vec4(dtype_wp(0.0))

            tvec[3] = dtype_wp(1.0)
            dtvec[2] = dtype_wp(1.0)
            t = (x[b, i] - g) / dg
            for w in range(2, -1, -1):
                tvec[w] = tvec[w+1] * t

            dtvec[1] = dtype_wp(2.0) * t
            dtvec[0] = dtype_wp(3.0) * t * t

            for j in range(out_dim):
                cvec = vec4(dtype_wp(0.0))
                dy_ = dy[b, j] / dtype_wp(6.0)

                cnt = int(3)
                for cidx in range(idg, idg - 4, -1):
                    cvec[cnt] = wp.where(
                        cidx >= 0 and cidx < grid_num - 4, coeff[j, i, cidx], dtype_wp(0.0))

                    cnt = cnt - int(1)

                res_left = vec4(wp.dot(mat40, cvec),
                                wp.dot(mat41, cvec),
                                wp.dot(mat42, cvec),
                                wp.dot(mat43, cvec),
                                )

                res_right = vec4(wp.dot(mat04, tvec),
                                 wp.dot(mat14, tvec),
                                 wp.dot(mat24, tvec),
                                 wp.dot(mat34, tvec),
                                 )
                cnt = int(3)

                for cidx in range(idg, idg - 4, -1):
                    if cidx >= 0 and cidx < grid_num - 4:

                        wp.atomic_add(dcoeff, j, i, cidx,
                                      res_right[cnt] * dy_)
                    cnt = cnt - int(1)
                dx_ = dx_ + (wp.dot(dtvec, res_left) * dy_ / dg)

            dx[b, i] = dx_
    elif order == 4:
        def adj_bspline(x: wp.array2d(dtype=dtype_wp),
                        coeff: wp.array3d(dtype=dtype_wp),
                        grid: wp.array2d(dtype=dtype_wp),
                        dy: wp.array2d(dtype=dtype_wp),
                        dx: wp.array2d(dtype=dtype_wp),
                        dcoeff: wp.array3d(dtype=dtype_wp)
                        ):
            b, i = wp.tid()

            mat50 = vec5(dtype_wp(1.0), dtype_wp(-4.),
                         dtype_wp(6.0), dtype_wp(-4.), dtype_wp(1.))
            mat51 = vec5(dtype_wp(-4.), dtype_wp(12.),
                         dtype_wp(-6.), dtype_wp(-12.), dtype_wp(11.))
            mat52 = vec5(dtype_wp(6.), dtype_wp(-12.),
                         dtype_wp(-6.), dtype_wp(12.), dtype_wp(11.))
            mat53 = vec5(dtype_wp(-4.), dtype_wp(4.),
                         dtype_wp(6.), dtype_wp(4.), dtype_wp(1.))
            mat54 = vec5(dtype_wp(1.), dtype_wp(
                0.), dtype_wp(0.), dtype_wp(0.), dtype_wp(0.))

            mat05 = vec5(dtype_wp(1.0), dtype_wp(-4.),
                         dtype_wp(6.0), dtype_wp(-4.), dtype_wp(1.))
            mat15 = vec5(dtype_wp(-4.), dtype_wp(12.),
                         dtype_wp(-12.), dtype_wp(4.), dtype_wp(0.))
            mat25 = vec5(dtype_wp(6.), dtype_wp(-6.),
                         dtype_wp(-6.), dtype_wp(6.), dtype_wp(0.))
            mat35 = vec5(dtype_wp(-4.), dtype_wp(-12.),
                         dtype_wp(12.), dtype_wp(4.), dtype_wp(0.))
            mat45 = vec5(dtype_wp(1.), dtype_wp(11.),
                         dtype_wp(11.), dtype_wp(1.), dtype_wp(0.))

            dx_ = dtype_wp(0.0)

            if x[b, i] >= grid[i, grid_num - 1] or grid[i, 0] >= x[b, i]:
                dx[b, i] = dx_
                return
            idg = binary_search(grid, x[b, i], i, grid_num)

            g = grid[i, idg]
            dg = grid[i, 1] - grid[i, 0]

            tvec = vec5(dtype_wp(0.0))
            dtvec = vec5(dtype_wp(0.0))

            tvec[4] = dtype_wp(1.0)
            dtvec[3] = dtype_wp(1.0)
            t = (x[b, i] - g) / dg
            for w in range(3, -1, -1):
                tvec[w] = tvec[w+1] * t

            dtvec[2] = dtype_wp(2.0) * t
            dtvec[1] = dtype_wp(3.0) * t * t
            dtvec[0] = dtype_wp(4.0) * t * t * t

            for j in range(out_dim):
                cvec = vec5(dtype_wp(0.0))
                dy_ = dy[b, j] / dtype_wp(24.0)

                cnt = int(4)
                for cidx in range(idg, idg - 5, -1):
                    cvec[cnt] = wp.where(
                        cidx >= 0 and cidx < grid_num - 5, coeff[j, i, cidx], dtype_wp(0.0))

                    cnt = cnt - int(1)

                res_left = vec5(wp.dot(mat05, cvec),
                                wp.dot(mat15, cvec),
                                wp.dot(mat25, cvec),
                                wp.dot(mat35, cvec),
                                wp.dot(mat45, cvec)
                                )

                res_right = vec5(wp.dot(mat50, tvec),
                                 wp.dot(mat51, tvec),
                                 wp.dot(mat52, tvec),
                                 wp.dot(mat53, tvec),
                                 wp.dot(mat54, tvec),
                                 )

                cnt = int(4)

                for cidx in range(idg, idg - 5, -1):
                    if cidx >= 0 and cidx < grid_num - 5:

                        wp.atomic_add(dcoeff, j, i, cidx,
                                      res_right[cnt] * dy_)
                    cnt = cnt - int(1)
                dx_ = dx_ + (wp.dot(dtvec, res_left) * dy_ / dg)

            dx[b, i] = dx_

    key = f"bspline_bwd_{order}_{in_dim}_{out_dim}_{grid_num}_{dtype.replace('.', '')}"
    module = wp.context.Module(key, loader=None)

    return wp.Kernel(func=adj_bspline, key=key, module=module)
