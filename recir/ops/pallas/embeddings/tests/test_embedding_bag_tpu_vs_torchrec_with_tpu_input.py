# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU half of the torchrec parity check for the *fused* pooled kernel.

``test_embedding_bag_torchrec`` pins torchrec down against a plain-JAX gather.
That covers the EBC layer but deliberately not the kernel, which cannot run on
CPU. This module closes the remaining gap: it compares the **real fused kernel's
TPU output** against torchrec, which is what deliverable 1.2 actually asks for.

Why two stages rather than one process: torchrec needs torch, and the TPU image
installs only ``jax[tpu]``. So the TPU side runs
``examples/embedding/ringshift_ebc_parity_dump`` and emits a self-describing
artifact; this side replays it against torchrec on CPU.

The artifact carries its own inputs, so the two halves share no fixture and
cannot drift. Point ``RECIR_EBC_TPU_ARTIFACT`` at either the raw job log (the
base64 block is extracted) or an already-extracted ``.npz``::

    buck2 test //recir/ops/pallas/embeddings/tests:test_embedding_bag_tpu_vs_torchrec_with_tpu_input \\
        -- --env RECIR_EBC_TPU_ARTIFACT=<path-to>/w4-ebc-parity-dump-latest.log

Skipped when the variable is unset, so it stays out of the way in CI until
someone has actually produced a TPU run. That is a deliberate trade: a test that
cannot run unattended is worth less than one that can, but the alternative here
is no coverage of the fused kernel against torchrec at all.
"""

from __future__ import annotations

import base64
import io
import os
import re
import unittest

import numpy as np
import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_ENV = "RECIR_EBC_TPU_ARTIFACT"
_MARKER = "[EBC_ARTIFACT]"
# Matches the tolerance the plain-JAX parity test uses. Not bitwise: the fused
# kernel accumulates in wave/sorted order and torch in caller order, and f32
# addition is not associative.
_TOL = 1e-5


def _extract_b64(text: str) -> bytes:
    """Pull the framed base64 block out of a job log."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(f"{_MARKER} BEGIN ")]
    if not starts:
        raise ValueError(f"no '{_MARKER} BEGIN' line found in artifact")
    # Take the LAST run: re-running the manifest appends to the same log.
    begin = starts[-1]
    try:
        end = next(
            i
            for i in range(begin + 1, len(lines))
            if lines[i].startswith(f"{_MARKER} END")
        )
    except StopIteration as exc:
        raise ValueError(f"'{_MARKER} BEGIN' with no matching END") from exc

    declared = int(lines[begin].split()[-1])
    payload_lines = [line.strip() for line in lines[begin + 1 : end]]
    invalid_lines = [
        line
        for line in payload_lines
        if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", line)
    ]
    if invalid_lines:
        raise ValueError("artifact contains a non-base64 line inside its frame")
    blob = "".join(payload_lines)
    if len(blob) != declared:
        raise ValueError(
            f"artifact truncated: header declares {declared} base64 chars, "
            f"found {len(blob)} (the log was probably cut off mid-block)"
        )
    return base64.b64decode(blob, validate=True)


def _load_artifact(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as artifact_file:
        raw = artifact_file.read()
    if not raw.startswith(b"PK"):  # npz is a zip; anything else is a log
        raw = _extract_b64(raw.decode("utf-8", errors="replace"))
    with np.load(io.BytesIO(raw)) as z:
        return {k: z[k] for k in z.files}


def _torchrec_forward_and_grad(
    full_table: np.ndarray,
    ids: np.ndarray,
    bag_lengths: np.ndarray,
    seed: np.ndarray,
    pooling: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run torchrec EBC per rank; return pooled outputs and the table gradient.

    All ranks' losses are summed before one ``backward()`` so the table gradient
    accumulates across ranks -- the same thing the distributed backward produces
    once every rank's contribution has landed on the owning shard.
    """
    world_size, num_bags = bag_lengths.shape
    vocab, dim = full_table.shape

    config = EmbeddingBagConfig(
        name="t",
        embedding_dim=dim,
        num_embeddings=vocab,
        feature_names=["f"],
        pooling=PoolingType.SUM if pooling == "sum" else PoolingType.MEAN,
    )
    ebc = EmbeddingBagCollection(tables=[config], device=torch.device("cpu"))
    weight = ebc.embedding_bags["t"].weight
    assert isinstance(weight, torch.Tensor)
    with torch.no_grad():
        weight.copy_(torch.from_numpy(full_table))

    outputs = []
    loss = torch.zeros((), dtype=torch.float32)
    for rank in range(world_size):
        lengths = bag_lengths[rank]
        total = int(lengths.sum())
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f"],
            # Only the first ``total`` slots are real; the rest is padding the
            # op routes to its trash bag and torchrec must never see.
            values=torch.tensor(ids[rank][:total].astype(np.int64)),
            lengths=torch.tensor(lengths.astype(np.int64)),
        )
        pooled = ebc(kjt)["f"]  # [num_bags, D]
        assert pooled.shape == (num_bags, dim), pooled.shape
        outputs.append(pooled)
        loss = loss + (pooled * torch.from_numpy(seed[rank])).sum()

    loss.backward()
    grad = weight.grad
    assert grad is not None
    return torch.stack(outputs).detach().numpy(), grad.detach().numpy()


def _mod_shard(full_table: np.ndarray, world_size: int) -> np.ndarray:
    """``[V, D]`` -> ``[W, V_local, D]`` with global row ``i`` at ``[i % W, i // W]``."""
    v_local = full_table.shape[0] // world_size
    return np.stack(
        [full_table[np.arange(v_local) * world_size + r] for r in range(world_size)]
    )


def _frame(payload: dict[str, np.ndarray]) -> str:
    """Build a job-log-shaped artifact block; mirrors the dumper's ``_emit``."""
    buf = io.BytesIO()
    np.savez(buf, **payload)
    blob = base64.b64encode(buf.getvalue()).decode("ascii")
    body = "\n".join(blob[i : i + 76] for i in range(0, len(blob), 76))
    return f"{_MARKER} BEGIN {len(blob)}\n{body}\n{_MARKER} END\n"


class ArtifactTransportTest(unittest.TestCase):
    """The log-parsing half, which needs no TPU and no env var.

    This is the fragile part of the two-stage scheme -- framing, truncation, a
    log holding more than one run -- and all of it is reachable on CPU, so it is
    covered here rather than discovered during a TPU turnaround.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.payload = {
            "ids": rng.integers(0, 16, (4, 8)).astype(np.int32),
            "full_table": rng.standard_normal((16, 3)).astype(np.float32),
        }

    def _round_trip(self, text: str) -> dict[str, np.ndarray]:
        with io.BytesIO(_extract_b64(text)) as raw, np.load(raw) as z:
            return {k: z[k] for k in z.files}

    def test_round_trips_through_a_log(self) -> None:
        text = f"noise before\n{_frame(self.payload)}noise after\n"
        got = self._round_trip(text)
        for k, v in self.payload.items():
            np.testing.assert_array_equal(got[k], v)

    def test_takes_the_last_run(self) -> None:
        """Re-running the manifest appends; the newest block is the real one."""
        stale = {k: np.zeros_like(v) for k, v in self.payload.items()}
        got = self._round_trip(_frame(stale) + _frame(self.payload))
        np.testing.assert_array_equal(got["full_table"], self.payload["full_table"])

    def test_truncated_block_raises(self) -> None:
        """A cut-off log must fail loudly, not decode into silent garbage."""
        lines = _frame(self.payload).splitlines()
        chopped = "\n".join(lines[:2] + lines[-1:])
        with self.assertRaisesRegex(ValueError, "truncated"):
            _extract_b64(chopped)

    def test_missing_end_marker_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no matching END"):
            _extract_b64(f"{_MARKER} BEGIN 8\nAAAA\n")

    def test_interleaved_log_line_raises(self) -> None:
        framed = _frame(self.payload)
        lines = framed.splitlines()
        lines.insert(2, "container restarted")
        declared = int(lines[0].split()[-1]) + len(lines[2])
        lines[0] = f"{_MARKER} BEGIN {declared}"
        with self.assertRaisesRegex(ValueError, "non-base64"):
            _extract_b64("\n".join(lines))

    def test_absent_block_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no .* BEGIN"):
            _extract_b64("job started\njob died\n")


@unittest.skipUnless(os.environ.get(_ENV), f"set {_ENV} to a TPU dump or job log")
class EmbeddingBagTpuVsTorchrecWithTpuInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.art = _load_artifact(os.environ[_ENV])

    def _check(self, pooling: str, impl: str) -> None:
        art = self.art
        want_fwd, want_grad_full = _torchrec_forward_and_grad(
            art["full_table"], art["ids"], art["bag_lengths"], art["seed"], pooling
        )
        np.testing.assert_allclose(
            art[f"fwd_{pooling}_{impl}"], want_fwd, atol=_TOL, rtol=_TOL
        )
        np.testing.assert_allclose(
            art[f"grad_{pooling}_{impl}"],
            _mod_shard(want_grad_full, art["bag_lengths"].shape[0]),
            atol=_TOL,
            rtol=_TOL,
        )

    def test_sum_kernel(self) -> None:
        """The fused Phase 4 reduction, forward and table gradient."""
        self._check("sum", "kernel")

    def test_mean_kernel(self) -> None:
        self._check("mean", "kernel")

    def test_sum_xla(self) -> None:
        """The XLA-reduction arm, so a failure can be attributed to Phase 4."""
        self._check("sum", "xla")

    def test_mean_xla(self) -> None:
        self._check("mean", "xla")

    def test_artifact_shard_assembly_round_trips(self) -> None:
        """Guard the mapping the whole comparison rests on."""
        art = self.art
        world_size = art["bag_lengths"].shape[0]
        expected = _mod_shard(art["full_table"], world_size)
        np.testing.assert_array_equal(art["sharded_table"], expected)
