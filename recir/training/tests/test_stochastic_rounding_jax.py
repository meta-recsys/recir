# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import recir.torchax as torchax
from jax.sharding import Mesh, PartitionSpec as P
from recir.torchax.ops.jrandom import _jstochastic_round_fp16
from recir.training.optim import TensorLRRowWiseAdagrad


def _step(
    weight: jax.Array,
    gradient: jax.Array,
    row_sum: jax.Array,
    counter: jax.Array,
    seed: jax.Array,
    lr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    env = torchax.default_env()
    p, grad, state_sum, state_counter, state_seed, lr_tensor = env.j2t_iso(
        (weight, gradient, row_sum, counter, seed, lr)
    )
    with env:
        optimizer = TensorLRRowWiseAdagrad(
            [p], lr_tensor=lr_tensor, eps=0.0, weight_decay=0.0
        )
        p.grad = grad
        optimizer.state[p] = {
            "sum": state_sum,
            "rounding_counter": state_counter,
            "rounding_seed": state_seed,
        }
        optimizer.step()
    return env.t2j_iso((p, optimizer.state[p]["sum"], state_counter))


class StochasticRoundingJaxTest(unittest.TestCase):
    def test_compiled_optimizer_advances_counter_and_replays(self) -> None:
        step = jax.jit(_step)
        weight = jnp.ones((16384, 1), dtype=jnp.float16)
        args = (
            weight,
            jnp.ones_like(weight),
            jnp.zeros((weight.shape[0], 1), dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(37, dtype=jnp.int32),
            jnp.asarray(2**-12, dtype=jnp.float32),
        )
        rounded, row_sum, counter = step(*args)
        self.assertEqual(rounded.dtype, jnp.float16)
        self.assertEqual(int(counter), 1)
        np.testing.assert_array_equal(row_sum, np.ones(row_sum.shape))
        self.assertAlmostEqual(float(jnp.mean(rounded < 1)), 0.5, delta=0.02)
        np.testing.assert_array_equal(step(*args)[0], rounded)
        next_args = (*args[:3], counter, *args[4:])
        self.assertFalse(np.array_equal(step(*next_args)[0], rounded))

    def test_subnormals_exact_values_and_nonfinite(self) -> None:
        values = jnp.broadcast_to(
            jnp.asarray([2**-25, -(2**-25), 1, -0.0, jnp.inf, -jnp.inf, jnp.nan]),
            (16384, 7),
        )
        rounded = jax.jit(_jstochastic_round_fp16)(
            values, jnp.asarray(0), jnp.asarray(37)
        )
        for column in (0, 1):
            with self.subTest(column=column):
                self.assertAlmostEqual(
                    float(jnp.mean(rounded[:, column] != 0)), 0.5, delta=0.02
                )
                self.assertTrue(bool(jnp.all(jnp.abs(rounded[:, column]) <= 2**-24)))
        np.testing.assert_array_equal(rounded[:, 2], np.ones(16384))
        self.assertTrue(bool(jnp.all(jnp.signbit(rounded[:, 3]))))
        self.assertTrue(bool(jnp.all(jnp.isposinf(rounded[:, 4]))))
        self.assertTrue(bool(jnp.all(jnp.isneginf(rounded[:, 5]))))
        self.assertTrue(bool(jnp.all(jnp.isnan(rounded[:, 6]))))

    @unittest.skipUnless(len(jax.devices()) >= 2, "requires multiple devices")
    def test_manual_mesh_uses_distinct_rounding_streams(self) -> None:
        mesh = Mesh(np.asarray(jax.devices()[:2]), ("torch_dist",))
        round_shards = jax.jit(
            jax.shard_map(
                _jstochastic_round_fp16,
                mesh=mesh,
                in_specs=(P("torch_dist", None), P(), P()),
                out_specs=P("torch_dist", None),
                check_vma=False,
            )
        )
        values = jnp.full((2, 16384), 1 + 2**-11, dtype=jnp.float32)
        result = round_shards(values, jnp.asarray(0), jnp.asarray(37))
        self.assertFalse(np.array_equal(result[0], result[1]))
        for row in result:
            self.assertAlmostEqual(float(jnp.mean(row > 1)), 0.5, delta=0.02)


if __name__ == "__main__":
    unittest.main()
