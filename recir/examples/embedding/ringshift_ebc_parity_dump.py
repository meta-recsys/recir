# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""TPU half of the torchrec EBC parity check: run the fused op, emit an artifact.

torchrec needs torch, which this image does not have (the Dockerfile installs
``jax[tpu]`` and nothing from the torch side), so the two cannot meet in one
process. This module implements a two-stage flow: run the fused kernel on TPU,
copy the output back to CPU, and compare it with a torchrec
``EmbeddingBagCollection`` run on the same inputs and weights.

The CPU half is
``recir/ops/pallas/embeddings/tests/test_embedding_bag_tpu_vs_torchrec_with_tpu_input.py``,
which reads the artifact this emits. The artifact is **self-describing**: it
carries the inputs as well as the outputs, so the two halves share no fixture
module and cannot drift apart. Adding a case here needs no change on the CPU
side.

Transport is base64 in the job log rather than a filesystem so the output works
with ephemeral containers and without object-store tooling. The payload is
approximately 170 KB. Run on a single host because the CPU half needs the whole
global table and a multi-host shard set is spread over processes.
"""

from __future__ import annotations

import base64
import io
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.embeddings.ringshift import (
    _POOLING_IMPL_ENV,
    ringshift_embedding_lookup,
)

# Same multi-host bootstrap as ringshift_fwd: jax.devices() only sees the full
# slice after initialize(), and it must run before the first devices() call.
if not jax.distributed.is_initialized() and (
    os.environ.get("TPU_WORKER_HOSTNAMES")
    or os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")
):
    jax.distributed.initialize()

WORLD_SIZE = 4
V_LOCAL = 64
DIM = 16
MAX_LOCAL_IDS = 256
NUM_BAGS = 32
SEED = 17

_MARKER = "[EBC_ARTIFACT]"


def _bag_lengths(counts: np.ndarray) -> np.ndarray:
    """Hand-built, mirroring RingshiftEbcCorrectnessTest._bag_lengths.

    A random split would only *probably* produce the interesting shapes. This
    makes the empty, singleton and hot cases certain, and keeps the layout
    identical across runs so a parity failure is reproducible.
    """
    out = np.zeros((WORLD_SIZE, NUM_BAGS), dtype=np.int32)
    for r in range(WORLD_SIZE):
        total = int(counts[r])
        lengths = np.zeros(NUM_BAGS, dtype=np.int32)
        lengths[0] = 0  # empty bag: mean's divisor clamp + trash segment
        lengths[1] = 1  # singleton: pooling must be a no-op on it
        spread = NUM_BAGS - 3
        base = (total - 1) // (spread + 4)  # leave a surplus for the hot bag
        lengths[2 : 2 + spread] = base
        lengths[-1] = total - int(lengths.sum())  # hot bag takes the rest
        assert lengths[-1] >= 0, (total, base)
        assert int(lengths.sum()) == total
        out[r] = lengths
    return out


def _build_case() -> dict[str, np.ndarray]:
    """Deterministic inputs, built on host so the artifact can carry them.

    Ids are MOD-consistent by construction (``id = owner + W * row``), which is
    the contract the fused kernel and the CPU-side shard assembly both assume.
    75% of each rank's ids go to one hot peer so the traffic is skewed, the same
    shape the correctness and perf harnesses use.
    """
    rng = np.random.default_rng(SEED)
    valid = (MAX_LOCAL_IDS * 4) // 5

    ids = np.zeros((WORLD_SIZE, MAX_LOCAL_IDS), dtype=np.int32)
    for r in range(WORLD_SIZE):
        hot_peer = (r + 1) % WORLD_SIZE
        is_hot = rng.random(valid) < 0.75
        owner = np.where(is_hot, hot_peer, rng.integers(0, WORLD_SIZE, valid))
        row = rng.integers(0, V_LOCAL, valid)
        ids[r, :valid] = (owner + WORLD_SIZE * row).astype(np.int32)

    counts = np.full((WORLD_SIZE,), valid, dtype=np.int32)
    full_table = rng.standard_normal((WORLD_SIZE * V_LOCAL, DIM)).astype(np.float32)
    # Non-uniform upstream cotangent: an all-ones seed makes every bag
    # contribute equally, which hides a fan-out that lands on the wrong bag.
    seed = rng.standard_normal((WORLD_SIZE, NUM_BAGS, DIM)).astype(np.float32)
    return {
        "ids": ids,
        "bag_lengths": _bag_lengths(counts),
        "full_table": full_table,
        "seed": seed,
    }


def _mod_shard(full_table: np.ndarray) -> np.ndarray:
    """``[V, D]`` -> ``[W, V_local, D]`` with global row ``i`` at ``[i % W, i // W]``."""
    return np.stack(
        [full_table[np.arange(V_LOCAL) * WORLD_SIZE + r] for r in range(WORLD_SIZE)]
    )


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(f"{_MARKER} {msg}", file=sys.stderr, flush=True)


def _emit(payload: dict[str, np.ndarray]) -> None:
    """Write the artifact to stdout as framed base64.

    stdout, not stderr: the log capture keeps both, but keeping the payload off
    the human-readable stream makes the progress lines still readable.
    """
    # Uncompressed on purpose: the payload is dominated by random f32, which
    # does not compress, so savez_compressed would only add a dependency on the
    # decompressor matching on the other side.
    buf = io.BytesIO()
    np.savez(buf, **payload)
    blob = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f"{_MARKER} BEGIN {len(blob)}", flush=True)
    for i in range(0, len(blob), 76):
        print(blob[i : i + 76], flush=True)
    print(f"{_MARKER} END", flush=True)


def main() -> int:
    devices = jax.devices()
    if len(devices) < WORLD_SIZE:
        _log(f"FAIL: need {WORLD_SIZE} devices, have {len(devices)}")
        return 1
    if jax.process_count() > 1:
        _log("FAIL: single-host only (the CPU half needs the whole global table)")
        return 1

    mesh = jax.sharding.Mesh(np.array(devices[:WORLD_SIZE]), ("x",))
    case = _build_case()
    sharded_table = _mod_shard(case["full_table"])
    recv_capacity = MAX_LOCAL_IDS + 128

    ids = jnp.asarray(case["ids"])
    bags = jnp.asarray(case["bag_lengths"])
    cot = jnp.asarray(case["seed"])
    table = jnp.asarray(sharded_table)

    _log(
        f"=== W={WORLD_SIZE} D={DIM} v_local={V_LOCAL} "
        f"max_local_ids={MAX_LOCAL_IDS} num_bags={NUM_BAGS} ==="
    )

    payload: dict[str, np.ndarray] = dict(case)
    payload["sharded_table"] = sharded_table

    for pooling in ("sum", "mean"):
        # Keep the existing artifact labels stable while using the explicit
        # environment value expected by the public lookup.
        for impl_label, pooling_impl in (
            ("xla", "xla"),
            ("kernel", "fused_kernel"),
        ):

            def arm(t, pooling=pooling, pooling_impl=pooling_impl):
                previous = os.environ.get(_POOLING_IMPL_ENV)
                os.environ[_POOLING_IMPL_ENV] = pooling_impl
                try:
                    return ringshift_embedding_lookup(
                        ids,
                        None,  # derived from bag_lengths when pooling is on
                        t,
                        mesh=mesh,
                        axis_name="x",
                        recv_capacity=recv_capacity,
                        splitter="globalpad",
                        pooling=pooling,
                        bag_lengths=bags,
                    )
                finally:
                    if previous is None:
                        os.environ.pop(_POOLING_IMPL_ENV, None)
                    else:
                        os.environ[_POOLING_IMPL_ENV] = previous

            fwd = jax.device_get(jax.jit(arm)(table))
            grad = jax.device_get(
                jax.jit(jax.grad(lambda t, a=arm: jnp.sum(a(t) * cot)))(table)
            )
            payload[f"fwd_{pooling}_{impl_label}"] = np.asarray(fwd)
            payload[f"grad_{pooling}_{impl_label}"] = np.asarray(grad)
            _log(f"captured {pooling}/{impl_label}")

    _emit(payload)
    _log("=== RESULT ebc_parity_dump: OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
