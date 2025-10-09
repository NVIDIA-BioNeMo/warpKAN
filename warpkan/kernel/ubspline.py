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


def make_forward_closure_ubspline(order: int, in_dim: int, out_dim: int, dtype: str = "float32"):
    dtype_wp = get_wp_fp_dtype(dtype)

    class vec2(wp.types.vector(length=2, dtype=dtype_wp)):
        pass

    class vec3(wp.types.vector(length=3, dtype=dtype_wp)):
        pass

    class vec4(wp.types.vector(length=4, dtype=dtype_wp)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype_wp)):
        pass

    if order == 3:

        def ubspline(x: wp.array2d(dtype=dtype_wp),
                     coeff: wp.array3d(dtype=dtype_wp),
                     idgs: wp.array2d(dtype=int),
                     grid_delta: dtype_wp,
                     y: wp.array2d(dtype=dtype_wp)):

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

                idg = idgs[b, i]
                t = x[b, i] / grid_delta - dtype_wp(idg)
                start = idg % 4
                start = wp.select(start < 0, start, start + 4)
                elem = int(3)
                for cnt in range(start+4, start, -1):
                    cvec[elem] = coeff[b, i + j * in_dim, cnt]
                    elem = elem - int(1)

                for w in range(2, -1, -1):
                    tvec[w] = tvec[w+1] * t

                res_left = vec4(wp.dot(mat40, cvec),
                                wp.dot(mat41, cvec),
                                wp.dot(mat42, cvec),
                                wp.dot(mat43, cvec))

                val = val + wp.dot(tvec, res_left) / dtype_wp(6.)

            y[b, j] = val
    elif order == 2:

        def ubspline(x: wp.array2d(dtype=dtype_wp),
                     coeff: wp.array3d(dtype=dtype_wp),
                     idgs: wp.array2d(dtype=int),
                     grid_delta: dtype_wp,
                     y: wp.array2d(dtype=dtype_wp)):

            b, j = wp.tid()

            val = dtype_wp(0.0)
            tvec = vec3(dtype_wp(1.0))

            mat30 = vec3(dtype_wp(1.0), dtype_wp(-2.), dtype_wp(1.))
            mat31 = vec3(dtype_wp(-2.), dtype_wp(2.0), dtype_wp(1.))
            mat32 = vec3(dtype_wp(1.0), dtype_wp(0.0), dtype_wp(0.))

            for i in range(in_dim):
                cvec = vec3(dtype_wp(0.0))

                idg = idgs[b, i]
                t = x[b, i] / grid_delta - dtype_wp(idg)
                start = idg % 3
                start = wp.select(start < 0, start, start + 3)
                elem = int(2)
                for cnt in range(start+3, start, -1):
                    cvec[elem] = coeff[b, i + j * in_dim, cnt]
                    elem = elem - int(1)

                for w in range(1, -1, -1):
                    tvec[w] = tvec[w+1] * t

                res_right = vec3(wp.dot(tvec, mat30),
                                 wp.dot(tvec, mat31),
                                 wp.dot(tvec, mat32)) / dtype_wp(2.0)

                val = val + wp.dot(res_right, cvec)

            y[b, j] = val

    elif order == 1:

        def ubspline(x: wp.array2d(dtype=dtype_wp),
                     coeff: wp.array3d(dtype=dtype_wp),
                     idgs: wp.array2d(dtype=int),
                     grid_delta: dtype_wp,
                     y: wp.array2d(dtype=dtype_wp)):

            b, j = wp.tid()

            val = dtype_wp(0.0)
            tvec = vec2(dtype_wp(1.0))

            mat20 = vec2(dtype_wp(-1.), dtype_wp(1.))
            mat21 = vec2(dtype_wp(1.0), dtype_wp(0.))

            for i in range(in_dim):
                cvec = vec2(dtype_wp(0.0))

                idg = idgs[b, i]
                t = x[b, i] / grid_delta - dtype_wp(idg)
                start = idg % 2
                start = wp.select(start < 0, start, start + 2)
                elem = int(1)
                for cnt in range(start+2, start, -1):
                    cvec[elem] = coeff[b, i + j * in_dim, cnt]
                    elem = elem - int(1)

                for w in range(0, -1, -1):
                    tvec[w] = tvec[w+1] * t

                res_right = vec2(wp.dot(tvec, mat20),
                                 wp.dot(tvec, mat21))

                val = val + wp.dot(res_right, cvec)

            y[b, j] = val

    elif order == 0:

        def ubspline(x: wp.array2d(dtype=dtype_wp),
                     coeff: wp.array3d(dtype=dtype_wp),
                     idgs: wp.array2d(dtype=int),
                     grid_delta: dtype_wp,
                     y: wp.array2d(dtype=dtype_wp)):

            b, j = wp.tid()

            val = dtype_wp(0.0)

            for i in range(in_dim):

                cvec = coeff[b, i + j * in_dim, 1]

                val = val + cvec

            y[b, j] = val

    key = f"ubspline_fwd_{order}_{in_dim}_{out_dim}_{dtype.replace('.', '')}"

    module = wp.context.Module(key, loader=None)

    return wp.Kernel(func=ubspline, key=key, module=module)


def make_backward_closure_ubspline(order: int, in_dim: int, out_dim: int, dtype: str = "float32"):
    dtype_wp = get_wp_fp_dtype(dtype)

    class vec2(wp.types.vector(length=2, dtype=dtype_wp)):
        pass

    class vec3(wp.types.vector(length=3, dtype=dtype_wp)):
        pass

    class vec4(wp.types.vector(length=4, dtype=dtype_wp)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype_wp)):
        pass

    if order == 3:
        def adj_ubspline(x: wp.array2d(dtype=dtype_wp),
                         coeff: wp.array3d(dtype=dtype_wp),
                         idgs: wp.array2d(dtype=int),
                         grid_delta: dtype_wp,
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

            transpose_mat43 = vec4(dtype_wp(1.0), dtype_wp(
                0.), dtype_wp(0.), dtype_wp(0.))
            transpose_mat42 = vec4(
                dtype_wp(-3.), dtype_wp(3.), dtype_wp(3.), dtype_wp(1.0))
            transpose_mat41 = vec4(
                dtype_wp(3.), dtype_wp(-6.), dtype_wp(0.), dtype_wp(4.))
            transpose_mat40 = vec4(
                dtype_wp(-1.), dtype_wp(3.), dtype_wp(-3.), dtype_wp(1.))

            dx_ = dtype_wp(0.0)

            idg = idgs[b, i]
            t = x[b, i] / grid_delta - dtype_wp(idg)

            tvec = vec4(dtype_wp(0.0))
            dtvec = vec4(dtype_wp(0.0))

            tvec[3] = dtype_wp(1.0)
            dtvec[2] = dtype_wp(1.0)
            for w in range(2, -1, -1):
                tvec[w] = tvec[w+1] * t

            dtvec[1] = dtype_wp(2.0) * t
            dtvec[0] = dtype_wp(3.0) * t * t

            cvec = vec4(dtype_wp(0.0))

            start = idg % 4
            start = wp.select(start < 0, start, start + 4)

            for j in range(out_dim):
                dy_ = dy[b, j] / dtype_wp(6.0)
                elem = int(3)
                for cnt in range(start+4, start, -1):
                    cvec[elem] = coeff[b, i + j * in_dim, cnt]
                    elem = elem - wp.int32(1)

                res_left = vec4(wp.dot(mat40, cvec),
                                wp.dot(mat41, cvec),
                                wp.dot(mat42, cvec),
                                wp.dot(mat43, cvec))

                res_right = vec4(wp.dot(transpose_mat40, tvec),
                                 wp.dot(transpose_mat41, tvec),
                                 wp.dot(transpose_mat42, tvec),
                                 wp.dot(transpose_mat43, tvec))
                elem = int(3)
                for cnt in range(start+4, start, -1):
                    wp.atomic_add(dcoeff, b, i + j * in_dim, cnt,
                                  res_right[elem] * dy_)
                    elem = elem - int(1)

                dx_ = dx_ + (wp.dot(dtvec, res_left) * dy_ / grid_delta)

            dx[b, i] = dx_
    elif order == 2:
        def adj_ubspline(x: wp.array2d(dtype=dtype_wp),
                         coeff: wp.array3d(dtype=dtype_wp),
                         idgs: wp.array2d(dtype=int),
                         grid_delta: dtype_wp,
                         dy: wp.array2d(dtype=dtype_wp),
                         dx: wp.array2d(dtype=dtype_wp),
                         dcoeff: wp.array3d(dtype=dtype_wp)
                         ):
            b, i = wp.tid()

            mat30 = vec3(dtype_wp(1.0), dtype_wp(-2.), dtype_wp(1.))
            mat31 = vec3(dtype_wp(-2.), dtype_wp(2.0), dtype_wp(1.))
            mat32 = vec3(dtype_wp(1.0), dtype_wp(0.0), dtype_wp(0.))

            mat03 = vec3(dtype_wp(1.0), dtype_wp(-2.), dtype_wp(1.))
            mat13 = vec3(dtype_wp(-2.), dtype_wp(2.0), dtype_wp(0.))
            mat23 = vec3(dtype_wp(1.0), dtype_wp(1.0), dtype_wp(0.))

            dx_ = dtype_wp(0.0)

            idg = idgs[b, i]
            t = x[b, i] / grid_delta - dtype_wp(idg)

            tvec = vec3(dtype_wp(0.0))
            dtvec = vec3(dtype_wp(0.0))

            tvec[2] = dtype_wp(1.0)
            dtvec[1] = dtype_wp(1.0)
            for w in range(1, -1, -1):
                tvec[w] = tvec[w+1] * t

            dtvec[0] = dtype_wp(2.0) * t

            cvec = vec3(dtype_wp(0.0))

            start = idg % 3
            start = wp.select(start < 0, start, start + 3)

            for j in range(out_dim):
                dy_ = dy[b, j] / dtype_wp(2.0)
                elem = int(2)
                for cnt in range(start+3, start, -1):
                    cvec[elem] = coeff[b, i + j * in_dim, cnt]
                    elem = elem - int(1)

                res_left = vec3(wp.dot(mat03, cvec),
                                wp.dot(mat13, cvec),
                                wp.dot(mat23, cvec))

                res_right = vec3(wp.dot(mat30, tvec),
                                 wp.dot(mat31, tvec),
                                 wp.dot(mat32, tvec))

                elem = int(2)
                for cnt in range(start+3, start, -1):
                    wp.atomic_add(dcoeff, b, i + j * in_dim, cnt,
                                  res_right[elem] * dy_)
                    elem = elem - int(1)

                dx_ = dx_ + (wp.dot(dtvec, res_left) * dy_ / grid_delta)

            dx[b, i] = dx_
    elif order == 1:
        def adj_ubspline(x: wp.array2d(dtype=dtype_wp),
                         coeff: wp.array3d(dtype=dtype_wp),
                         idgs: wp.array2d(dtype=int),
                         grid_delta: dtype_wp,
                         dy: wp.array2d(dtype=dtype_wp),
                         dx: wp.array2d(dtype=dtype_wp),
                         dcoeff: wp.array3d(dtype=dtype_wp)
                         ):
            b, i = wp.tid()

            mat20 = vec2(dtype_wp(-1.), dtype_wp(1.))
            mat21 = vec2(dtype_wp(1.0), dtype_wp(0.))

            dx_ = dtype_wp(0.0)

            idg = idgs[b, i]
            t = x[b, i] / grid_delta - dtype_wp(idg)

            tvec = vec2(dtype_wp(1.0))
            dtvec = vec2(dtype_wp(0.0))

            tvec[0] = t
            dtvec[0] = dtype_wp(1.0)

            cvec = vec2(dtype_wp(0.0))

            start = idg % 2
            start = wp.select(start < 0, start, start + 2)

            for j in range(out_dim):
                dy_ = dy[b, j]
                elem = int(1)
                for cnt in range(start+2, start, -1):
                    cvec[elem] = coeff[b, i + j * in_dim, cnt]
                    elem = elem - int(1)

                res_left = vec2(wp.dot(mat20, cvec),
                                wp.dot(mat21, cvec))

                res_right = vec2(wp.dot(mat20, tvec),
                                 wp.dot(mat21, tvec))

                elem = int(1)
                for cnt in range(start+2, start, -1):
                    wp.atomic_add(dcoeff, b, i + j * in_dim, cnt,
                                  res_right[elem] * dy_)
                    elem = elem - int(1)

                dx_ = dx_ + (wp.dot(dtvec, res_left) * dy_ / grid_delta)

            dx[b, i] = dx_
    elif order == 0:
        def adj_ubspline(x: wp.array2d(dtype=dtype_wp),
                         coeff: wp.array3d(dtype=dtype_wp),
                         idgs: wp.array2d(dtype=int),
                         grid_delta: dtype_wp,
                         dy: wp.array2d(dtype=dtype_wp),
                         dx: wp.array2d(dtype=dtype_wp),
                         dcoeff: wp.array3d(dtype=dtype_wp)
                         ):
            b, i = wp.tid()
            dx_ = dtype_wp(0.0)

            for j in range(out_dim):
                dy_ = dy[b, j]
                wp.atomic_add(dcoeff, b, i + j * in_dim, 1, dy_)

            dx[b, i] = dx_

    key = f"ubspline_bwd_{order}_{in_dim}_{out_dim}_{dtype.replace('.', '')}"
    module = wp.context.Module(key, loader=None)

    return wp.Kernel(func=adj_ubspline, key=key, module=module)
