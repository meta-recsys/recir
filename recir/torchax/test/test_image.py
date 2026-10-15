# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
from functools import partial

import jax
import recir.torchax
import recir.torchax.interop
import torch


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def upsample_jit(
    tensor: jax.Array,
    output_size: tuple[int, int],
    align_corners: bool,
    antialias: bool,
    method: str,
) -> jax.Array:
    tensor = recir.torchax.interop.torch_view(tensor)
    tensor = torch.nn.functional.interpolate(  # pyre-ignore[9]
        # pyrefly: ignore [bad-argument-type]
        tensor,
        size=output_size,
        mode=method,
        align_corners=align_corners,
        antialias=antialias,
    )
    return recir.torchax.interop.jax_view(tensor)


class TestResampling(unittest.TestCase):
    def _run_resampling(self, antialias: bool, align_corners: bool) -> None:
        method = "bicubic"
        input_tensor = torch.rand((1, 1, 256, 512), dtype=torch.float32)
        output_size = (128, 64)

        upsampled_tensor = torch.nn.functional.interpolate(
            input_tensor,
            size=output_size,
            mode=method,
            align_corners=align_corners,
            antialias=antialias,
        )

        env = recir.torchax.default_env()
        with env:
            input_tensor_xla = env.to_xla(input_tensor)
            input_tensor_xla = recir.torchax.interop.jax_view(input_tensor_xla)
            upsampled_tensor_xla = upsample_jit(
                input_tensor_xla,
                output_size,
                align_corners,
                antialias=antialias,
                method=method,
            )

        upsampled_tensor_xla = env.j2t_copy(upsampled_tensor_xla)
        abs_err = torch.abs(upsampled_tensor - upsampled_tensor_xla)

        assert torch.allclose(
            upsampled_tensor, upsampled_tensor_xla, atol=1e-4, rtol=1e-5
        ), f"{method} upsampling failed with error {abs_err.max()}"

    def test_bicubic_antialias_no_align(self) -> None:
        self._run_resampling(antialias=True, align_corners=False)

    def test_bicubic_antialias_align(self) -> None:
        self._run_resampling(antialias=True, align_corners=True)

    def test_bicubic_no_antialias_no_align(self) -> None:
        self._run_resampling(antialias=False, align_corners=False)

    def test_bicubic_no_antialias_align(self) -> None:
        self._run_resampling(antialias=False, align_corners=True)


if __name__ == "__main__":
    unittest.main()
