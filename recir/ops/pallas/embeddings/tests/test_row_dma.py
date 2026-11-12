# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""``_row_dma`` picks the right row whichever static chunk holds it.

The 32-bit offset limit it works around exists only on TPU, so these run in
interpret mode with the limit lowered: a 64-row table then spans several chunks,
and every row must still come back, and be written, exactly.
"""

from __future__ import annotations

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from recir.ops.pallas.embeddings import ringshift

_ROWS, _WIDTH = 72, 128
# 16 * 128 elements per chunk: rows 0-15, 16-31, ..., 64-71 (a short last chunk).
_LIMIT = 16 * _WIDTH
_PROBE = (0, 15, 16, 17, 47, 48, 70, 71)


def _table() -> jax.Array:
    return jnp.broadcast_to(
        jnp.arange(_ROWS, dtype=jnp.int32)[:, None], (_ROWS, _WIDTH)
    )


def _copy(src, dst, sem) -> None:
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    cp.wait()


class RowDmaTest(unittest.TestCase):
    def test_reads_every_row_across_chunks(self) -> None:
        def kernel(idx_ref, table_ref, out_ref, buf, sem):
            def body(k, _):
                ringshift._row_dma(
                    table_ref,
                    idx_ref[k],
                    lambda src: _copy(src, buf.at[pl.ds(k, 1), :], sem.at[0]),
                )
                return None

            lax.fori_loop(0, len(_PROBE), body, None)
            out_ref[...] = buf[...]

        with mock.patch.object(ringshift, "_DMA_OFFSET_LIMIT", _LIMIT):
            out = pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((len(_PROBE), _WIDTH), jnp.int32),
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((len(_PROBE), _WIDTH), jnp.int32),
                    pltpu.SemaphoreType.DMA((1,)),
                ],
                interpret=True,
            )(jnp.array(_PROBE, jnp.int32), _table())
        np.testing.assert_array_equal(np.asarray(out[:, 0]), _PROBE)

    def test_writes_every_row_across_chunks(self) -> None:
        def kernel(idx_ref, table_ref, out_ref, buf, sem):
            buf[...] = -(lax.broadcasted_iota(jnp.int32, buf.shape, 0) + 1)

            def body(k, _):
                ringshift._row_dma(
                    out_ref,
                    idx_ref[k],
                    lambda dst: _copy(buf.at[pl.ds(k, 1), :], dst, sem.at[0]),
                )
                return None

            lax.fori_loop(0, len(_PROBE), body, None)

        with mock.patch.object(ringshift, "_DMA_OFFSET_LIMIT", _LIMIT):
            out = pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((_ROWS, _WIDTH), jnp.int32),
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=[
                    pltpu.VMEM((len(_PROBE), _WIDTH), jnp.int32),
                    pltpu.SemaphoreType.DMA((1,)),
                ],
                input_output_aliases={1: 0},
                interpret=True,
            )(jnp.array(_PROBE, jnp.int32), _table())
        want = np.arange(_ROWS)
        want[list(_PROBE)] = -(np.arange(len(_PROBE)) + 1)
        np.testing.assert_array_equal(np.asarray(out[:, 0]), want)


if __name__ == "__main__":
    unittest.main()
