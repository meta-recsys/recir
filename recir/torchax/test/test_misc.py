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


"""If you don't know which file a test should go, and don't want to make a new file
for a small test. PUt it here
"""

import unittest
from typing import Any

import jax
import jax.numpy as jnp
import recir.torchax
import torch


class MiscTest(unittest.TestCase):
    def test_extract_jax_kwargs(self) -> None:
        class M(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.sin(a) + torch.cos(b)

        weights, func = recir.torchax.extract_jax(M())
        res = func(
            weights,
            args=(),
            kwargs={"a": jnp.array([1, 2, 3]), "b": jnp.array([3, 4, 5])},
        )
        self.assertTrue(
            jnp.allclose(
                res, jnp.sin(jnp.array([1, 2, 3])) + jnp.cos(jnp.array([3, 4, 5]))
            )
        )

    def test_to_device(self) -> None:
        env = recir.torchax.default_env()
        with env:
            step1 = torch.ones(
                100,
                100,
            )
            step2 = torch.triu(step1, diagonal=1)
            step3 = step2.to(dtype=torch.bool, device="jax")
            self.assertEqual(step3.device.type, "jax")

    def test_to_device_twice(self) -> None:
        env = recir.torchax.default_env()
        with env:
            step1 = torch.ones(
                100,
                100,
            )
            step2 = torch.triu(step1, diagonal=1)
            step3 = step2.to(dtype=torch.bool, device="jax")
            step3.to("jax")
            self.assertEqual(step3.device.type, "jax")

    def test_random_with_tensor_input(self) -> None:
        env = recir.torchax.default_env()
        with env:
            env.manual_seed(torch.tensor(2))
            x = torch.randn((2, 2), device="jax")

        with env:
            env.manual_seed(torch.tensor(2))
            y = torch.randn((2, 2), device="jax")

            self.assertTrue(torch.allclose(x, y))

    def test_random_with_tensor_input(self) -> None:
        env: Any = recir.torchax.default_env()

        with env:

            @recir.torchax.interop.jax_jit
            def rand_plus_one(rng: int) -> torch.Tensor:
                env.manual_seed(torch.tensor(2))
                x = torch.randn((2, 2), device="jax") + 1
                return x

            x = rand_plus_one(0)
            y = rand_plus_one(0)
            self.assertTrue(torch.allclose(x, y))

    def test_zeros_with_explicit_size(self) -> None:
        with recir.torchax.default_env():
            a = torch.zeros(size=(4,), device="jax")
            self.assertEqual(a.shape[0], 4)


class EmbeddingBagSegmentSumTest(unittest.TestCase):
    """``_aten__embedding_bag`` passes ``indices_are_sorted=True`` to segment_sum.

    That is a promise to the compiler, not a rewrite: if ``segment_ids`` were
    ever unsorted the reduction would silently scatter into the wrong bags
    rather than fail. The promise holds because ``searchsorted`` is monotone in
    its query and ``clip`` preserves that, so this asserts the property
    directly as well as checking the sums.
    """

    def test_segment_ids_are_sorted_and_sums_are_correct(self) -> None:
        import jax.numpy as jnp
        import numpy as np
        import torch
        from recir.torchax.ops.jaten import _aten__embedding_bag

        rng = np.random.default_rng(3)
        for n, num_bags, d in ((83200, 650, 16), (4096, 64, 8), (64, 8, 4)):
            # multinomial gives ragged bags including empty ones
            sizes = rng.multinomial(n, np.ones(num_bags) / num_bags)
            off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            weight = rng.standard_normal((500, d)).astype(np.float32)
            idx = rng.integers(0, 500, size=n).astype(np.int64)

            bag_offsets = jnp.asarray(off[:num_bags])
            positions = jnp.arange(n, dtype=bag_offsets.dtype)
            seg = jnp.clip(
                jnp.searchsorted(bag_offsets, positions, side="right") - 1,
                0,
                num_bags - 1,
            )
            self.assertTrue(bool(jnp.all(jnp.diff(seg) >= 0)))

            # All three modes carry indices_are_sorted=True, so all three are
            # checked; mode 2 (max) reduces differently and would not be caught
            # by the sum path.
            for mode_id, mode_name in ((0, "sum"), (1, "mean"), (2, "max")):
                out = np.asarray(
                    _aten__embedding_bag(
                        jnp.asarray(weight),
                        jnp.asarray(idx),
                        jnp.asarray(off[:num_bags]),
                        mode=mode_id,
                    )[0]
                )
                ref = np.asarray(
                    torch.nn.functional.embedding_bag(
                        torch.from_numpy(idx),
                        torch.from_numpy(weight),
                        torch.from_numpy(off[:num_bags]),
                        mode=mode_name,
                    )
                )
                # Empty bags are defined as 0 for sum/mean; max leaves them at
                # the identity, which torch reports as 0 too.
                np.testing.assert_allclose(
                    out, ref, rtol=1e-5, atol=1e-5, err_msg=f"mode={mode_name}"
                )


class ZZTwoLevelSearchsorted(unittest.TestCase):
    def test_matches_jnp_searchsorted(self) -> None:
        import jax.numpy as jnp
        import numpy as np
        from recir.torchax.ops.jaten import (
            _searchsorted,
            _SEARCHSORTED_VECTORIZE_MAX_BINS,
        )

        cap = _SEARCHSORTED_VECTORIZE_MAX_BINS
        rng = np.random.default_rng(17)
        cases = [
            ("prod shape", 32896, 313540),
            ("just over cap", cap + 1, 1000),
            ("exact power of two", 65536, 5000),
            ("odd length", 20011, 777),
        ]
        for name, L, N in cases:
            for dt in (np.int32, np.uint32, np.float32):
                for side in ("left", "right"):
                    base = np.sort(rng.integers(0, 4 * L, size=L)).astype(dt)
                    # queries: in-range, below all, above all, and exact hits
                    q = rng.integers(-5, 4 * L + 5, size=N).astype(dt)
                    q[:50] = base[rng.integers(0, L, size=50)]  # exact matches
                    if np.issubdtype(dt, np.signedinteger):
                        q[50:60] = -1
                    q[60:70] = 4 * L + 100
                    # A query equal to the dtype max collides with the padding
                    # sentinel; side="right" counted the pad before the clamp.
                    q[70:80] = (
                        np.iinfo(dt).max
                        if np.issubdtype(dt, np.integer)
                        else np.finfo(dt).max
                    )
                    a, v = jnp.asarray(base), jnp.asarray(q)
                    got = np.asarray(_searchsorted(a, v, side))
                    ref = np.asarray(jnp.searchsorted(a, v, side=side))
                    bad = int((got != ref).sum())
                    if bad:
                        i = int(np.flatnonzero(got != ref)[0])
                        print(
                            f"  MISMATCH {name} {dt.__name__} {side}: n={bad} first@{i} got={got[i]} ref={ref[i]}"
                        )
                    self.assertEqual(bad, 0, f"{name}/{dt.__name__}/{side}")
            print(f"  ok {name:20s} L={L:6d} N={N:6d}  (6 dtype x side combos)")

    def test_duplicate_heavy_and_degenerate(self) -> None:
        import jax.numpy as jnp
        import numpy as np
        from recir.torchax.ops.jaten import _searchsorted

        rng = np.random.default_rng(23)
        L, N = 32896, 4096
        for name, base in (
            ("all equal", np.full(L, 7, np.int32)),
            ("two values", np.sort(rng.integers(0, 2, size=L)).astype(np.int32)),
            ("heavy dupes", np.sort(rng.integers(0, 16, size=L)).astype(np.int32)),
        ):
            for side in ("left", "right"):
                q = rng.integers(-2, 20, size=N).astype(np.int32)
                a, v = jnp.asarray(base), jnp.asarray(q)
                got = np.asarray(_searchsorted(a, v, side))
                ref = np.asarray(jnp.searchsorted(a, v, side=side))
                self.assertEqual(int((got != ref).sum()), 0, f"{name}/{side}")
            print(f"  ok {name}")


if __name__ == "__main__":
    unittest.main()
