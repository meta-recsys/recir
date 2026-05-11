# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Correctness test for the v_local-chunked ringshift wrapper.

Compares the chunked wrapper against the un-chunked fused ringshift kernel on
a small (W=4) config. The chunked wrapper splits the per-rank table along its
row axis, runs one fused kernel call per chunk, and scatters per-chunk
outputs back into a unified accumulator. For each input ID, exactly one
chunk owns the destination row, so the chunked output must equal the
un-chunked output bit-for-bit (modulo numerical accumulation order, which is
identical here because each output position is written by a single chunk).
"""

from __future__ import annotations

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from recir.ops.pallas.embeddings.jagged_a2a_lookup import (
    _TILE_INT32,
    jagged_a2a_embedding_lookup_f32_alltc_ringshift,
    jagged_a2a_embedding_lookup_f32_alltc_ringshift_chunked,
)
from recir.ops.pallas.embeddings.tests.test_jagged_a2a_lookup import (
    _build_local_table,
    _has_tpu,
    _make_skewed_inputs,
)


@unittest.skipUnless(_has_tpu(), "Requires real TPU")
class ChunkedVLocalCorrectnessTest(unittest.TestCase):
    """Validate the chunked wrapper against the un-chunked fused kernel."""

    def test_w4_chunked_matches_unchunked(self) -> None:
        world_size = 4
        v_local = 256
        embedding_dim = 128
        max_local_ids = 256  # ~80% valid -> ~204 IDs per rank
        v_local_chunk_size = 64  # -> 4 chunks per rank
        # Per-chunk safety factor 2x over uniform per-chunk count (~50 IDs).
        # Round up to TILE.
        chunk_max_local_ids = ((204 // 4 * 2 + _TILE_INT32 - 1) // _TILE_INT32) * _TILE_INT32

        recv_capacity = chunk_max_local_ids + world_size * _TILE_INT32
        recv_capacity = (
            (recv_capacity + _TILE_INT32 - 1) // _TILE_INT32
        ) * _TILE_INT32

        devices = jax.devices()[:world_size]
        if len(devices) < world_size:
            self.skipTest(f"Need {world_size} devices, have {len(devices)}")
        mesh = jax.sharding.Mesh(np.array(devices), ("x",))

        # Inputs: skewed traffic (75% to hot peer, ~80% valid).
        local_table_np = _build_local_table(
            world_size, v_local, embedding_dim, np.float32, seed=0
        )
        local_ids_np, local_id_count_np = _make_skewed_inputs(
            world_size, v_local, max_local_ids, seed=2
        )

        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        )
        local_ids = jax.device_put(jnp.array(local_ids_np), sharding)
        local_id_count = jax.device_put(jnp.array(local_id_count_np), sharding)
        local_table = jax.device_put(jnp.array(local_table_np), sharding)

        # Un-chunked reference.
        unchunked_recv_capacity = max_local_ids + world_size * _TILE_INT32
        unchunked_recv_capacity = (
            (unchunked_recv_capacity + _TILE_INT32 - 1) // _TILE_INT32
        ) * _TILE_INT32
        t0 = time.perf_counter()
        unchunked_out = jagged_a2a_embedding_lookup_f32_alltc_ringshift(
            local_ids, local_id_count, local_table,
            mesh=mesh, axis_name="x",
            recv_capacity=unchunked_recv_capacity,
            interpret=False, splitter="globalpad",
        )
        jax.block_until_ready(unchunked_out)
        unchunked_np = np.array(unchunked_out)
        print(
            f"[CHUNKED_TEST] unchunked out shape={unchunked_np.shape} "
            f"sum={unchunked_np.sum():.4f} "
            f"max={unchunked_np.max():.4f} min={unchunked_np.min():.4f} "
            f"({(time.perf_counter() - t0):.1f}s)",
            flush=True,
        )

        # Chunked wrapper, wrapped in jit so the 4 chunk dispatches compile
        # as one program (otherwise each chunk's pallas_call would re-trace).
        # _sync_payload="full" exercises the max-debuggable cross-rank
        # epoch barrier (full reduction over chunk_out per rank).
        print("[CHUNKED_TEST] starting chunked call (_sync_payload='full')...", flush=True)
        import functools as _ft
        chunked_jit = jax.jit(_ft.partial(
            jagged_a2a_embedding_lookup_f32_alltc_ringshift_chunked,
            mesh=mesh, axis_name="x",
            v_local_chunk_size=v_local_chunk_size,
            chunk_max_local_ids=chunk_max_local_ids,
            recv_capacity=recv_capacity,
            interpret=False, splitter="globalpad",
            _sync_payload="full",
        ))
        t0 = time.perf_counter()
        chunked_out = chunked_jit(local_ids, local_id_count, local_table)
        print(
            f"[CHUNKED_TEST] chunked traced ({(time.perf_counter() - t0):.1f}s); awaiting...",
            flush=True,
        )
        jax.block_until_ready(chunked_out)
        chunked_np = np.array(chunked_out)
        print(
            f"[CHUNKED_TEST] chunked done ({(time.perf_counter() - t0):.1f}s total)",
            flush=True,
        )
        print(
            f"[CHUNKED_TEST] chunked   out shape={chunked_np.shape} "
            f"sum={chunked_np.sum():.4f} "
            f"max={chunked_np.max():.4f} min={chunked_np.min():.4f}",
            flush=True,
        )

        max_diff = float(np.max(np.abs(chunked_np - unchunked_np)))
        print(
            f"[CHUNKED_TEST] max_diff={max_diff:.6e}",
            flush=True,
        )
        np.testing.assert_allclose(
            chunked_np, unchunked_np,
            atol=1e-5,
            err_msg=f"chunked vs un-chunked mismatch (max_diff={max_diff})",
        )
        print(
            f"[CHUNKED_TEST] PASS: max_diff={max_diff:.6e} <= 1e-5",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
