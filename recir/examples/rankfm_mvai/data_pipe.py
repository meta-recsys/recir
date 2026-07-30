# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Data-loader plumbing for the rankfm TPU training example.

Loads preprocessed ``MTMLROOTrainBatch`` files from GCS or a local directory.
Per-step shape padding (RO/NRO/unary KJT lengths and values, NRO batch dim,
``num_candidates`` shape+sum) is handled entirely by
``recir.region_transforms.padding.PaddingRegionTransform`` (wired in
``train.py``) on the compiled path; the eager path (``train_cpu_eager.py``)
processes the real jagged batch directly. Either way the data layer yields the
REAL jagged batch untouched, only slicing to the trainer's per-rank
``local_batch_size`` and remapping IDs into the model's hash space.

Public surface:

    BatchReplayPipe              file/GCS reader, replicate-across-ranks toggle
    compute_kjt_key_to_size      EC config -> {feature_name: num_embeddings}
    remap_batch_ids              clamp loaded IDs into model's hash space
    select_rows_slice            slice loaded batch to RO rows [start, start+n)
    replay_batch_iter            yields ready-to-train batches from a
                                 BatchReplayPipe (slice + remap + per-step
                                 NRO-budget log)
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy third-party imports (soft deps so the module loads in any image)
# ---------------------------------------------------------------------------


def _import_gcs_storage() -> Any:
    try:
        from google.cloud import storage

        return storage
    except ImportError:
        return None


def _import_zstandard() -> Any:
    try:
        import zstandard

        return zstandard
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# BatchReplayPipe — load preproc batches from GCS or local fs
# ---------------------------------------------------------------------------


class BatchReplayPipe:
    """Load preprocessed training batches from GCS or a local directory.

    Expects a dataporter dump layout: a ``metadata.json`` file at the
    path root that records ``num_ranks`` and ``num_batches``, plus
    ``rank_X_batch_Y.pt.zst`` files. The filename list is synthesized
    from the metadata; the parent directory is never enumerated (which
    would take minutes on FUSE for a 100K-file dump).

    The path may be either a GCS URI (``gs://bucket/prefix``) or a local
    filesystem path. When ``replicate_across_ranks=True`` every rank yields
    the same files in the same order (matches synthetic-mode semantics:
    avoids torchrec ROW_WISE all-to-all "Split sizes doesn't match" failures
    from divergent KJT lengths). When False (default), files are sharded
    round-robin by ``i % world_size``.
    """

    def __init__(
        self,
        path: str,
        rank: int = 0,
        world_size: int = 1,
        replicate_across_ranks: bool = False,
        file_list: Optional[List[str]] = None,
    ) -> None:
        self._path: str = path.rstrip("/")
        self._rank = rank
        self._world_size = world_size
        self._replicate = replicate_across_ranks

        self._is_gcs: bool = self._path.startswith("gs://")
        if self._is_gcs:
            stripped = self._path[len("gs://") :]
            parts = stripped.split("/", 1)
            self._bucket_name: str = parts[0]
            self._prefix: str = parts[1] if len(parts) > 1 else ""
            storage_mod = _import_gcs_storage()
            self._client: Any = (
                storage_mod.Client() if storage_mod is not None else None
            )
        else:
            self._local_dir: str = os.path.abspath(os.path.expanduser(self._path))
            if not os.path.isdir(self._local_dir):
                raise FileNotFoundError(
                    f"data_path is not a directory: {self._local_dir}"
                )
            self._bucket_name = ""
            self._prefix = ""
            self._client = None

        self._file_list: List[str] = []
        if file_list is not None:
            # Explicit-list mode: skip directory enumeration entirely. This
            # is required when ``path`` is a FUSE mount with many files (e.g.
            # /mnt/barnfs over a 100K-file dataporter dump where os.listdir
            # would take 58+ min). Each filename is opened on-demand inside
            # ``_load_batch``, which is sub-second on FUSE for known names.
            if self._replicate:
                self._file_list = list(file_list)
            else:
                self._file_list = [
                    f
                    for i, f in enumerate(file_list)
                    if i % self._world_size == self._rank
                ]
            logger.info(
                f"[Replay] Explicit file list (replicate={self._replicate}): "
                f"rank {self._rank} sees {len(self._file_list)} files."
            )
        else:
            self._discover_files()

    def _discover_files(self) -> None:
        """Read ``metadata.json`` and synthesize the rank_*_batch_*.pt.zst
        filename list. Avoids listing the parent directory, which can hold
        100K+ files on a dataporter dump and takes minutes on FUSE.
        """
        prefix = f"{self._prefix}/" if (self._is_gcs and self._prefix) else ""
        metadata = self._load_json(f"{prefix}metadata.json")
        num_dump_ranks: int = metadata["num_ranks"]
        num_batches: int = metadata["num_batches"]

        if self._replicate:
            for r in range(num_dump_ranks):
                for b in range(num_batches):
                    self._file_list.append(f"{prefix}rank_{r}_batch_{b}.pt.zst")
            logger.info(
                f"[Replay] Dump (replicated): {num_dump_ranks} ranks x "
                f"{num_batches} batches = {len(self._file_list)} files per rank."
            )
            return

        total_files = num_dump_ranks * num_batches
        batches_per_rank = total_files // self._world_size
        dropped = total_files - batches_per_rank * self._world_size
        if dropped > 0:
            logger.warning(
                f"[Replay] Dropping {dropped}/{total_files} files "
                f"(not divisible by {self._world_size} ranks)"
            )
        for i in range(batches_per_rank):
            global_idx = self._rank + i * self._world_size
            dump_rank = global_idx % num_dump_ranks
            batch_idx = global_idx // num_dump_ranks
            self._file_list.append(f"{prefix}rank_{dump_rank}_batch_{batch_idx}.pt.zst")
        logger.info(
            f"[Replay] Dump: {num_dump_ranks} dump ranks x {num_batches} batches. "
            f"Rank {self._rank} assigned {batches_per_rank} batches."
        )

    def _download_bytes(self, name: str) -> bytes:
        if not self._is_gcs:
            with open(os.path.join(self._local_dir, name), "rb") as fh:
                return fh.read()
        if self._client is not None:
            bucket = self._client.bucket(self._bucket_name)
            return bucket.blob(name).download_as_bytes()
        with tempfile.NamedTemporaryFile() as tmp:
            subprocess.check_call(
                ["gsutil", "cp", f"gs://{self._bucket_name}/{name}", tmp.name]
            )
            return tmp.read()

    def _load_json(self, name: str) -> Dict[str, Any]:
        return json.loads(self._download_bytes(name))

    def _load_batch(self, name: str) -> Any:
        raw = self._download_bytes(name)
        if name.endswith(".zst"):
            zstd_mod = _import_zstandard()
            if zstd_mod is None:
                raise ImportError(
                    "zstandard package required for .pt.zst files: "
                    "pip install zstandard"
                )
            raw = zstd_mod.ZstdDecompressor().decompress(raw)
        return torch.load(io.BytesIO(raw), weights_only=False)

    def __iter__(self) -> Iterator[Any]:
        for i, name in enumerate(self._file_list):
            batch = self._load_batch(name)
            logger.info(
                f"[Replay] rank {self._rank} loaded {i + 1}/{len(self._file_list)}: "
                f"{name}"
            )
            yield batch

    def __len__(self) -> int:
        return len(self._file_list)


# ---------------------------------------------------------------------------
# Embedding-table introspection
# ---------------------------------------------------------------------------


def compute_kjt_key_to_size(model: nn.Module) -> Dict[str, int]:
    """Map KJT feature key -> the model's embedding-table ``num_embeddings``.

    Walks ``model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection``
    and builds ``{feature_name: num_embeddings}`` so callers can clamp loaded
    KJT values into the model's hash space (the dump was preprocessed against
    a different, usually larger, ``hash_size_cap``).

    Must be called BEFORE DMP — post-DMP the EC is wrapped in a
    ``ShardedEmbeddingCollection`` that doesn't expose ``embedding_configs()``.
    """
    ec = (
        model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection  # pyre-ignore[16]
    )
    result: Dict[str, int] = {}
    for table in ec.embedding_configs():
        for fname in table.feature_names:
            result[fname] = int(table.num_embeddings)
    return result


# ---------------------------------------------------------------------------
# ID remap (clamp loaded IDs into model's hash space)
# ---------------------------------------------------------------------------


def _remap_kjt_ids(
    kjt: KeyedJaggedTensor, key_to_size: Dict[str, int]
) -> KeyedJaggedTensor:
    """Modulo each per-key value slice into the model's hash space."""
    keys = kjt.keys()
    F = len(keys)
    lengths = kjt.lengths()
    B = lengths.shape[0] // F
    per_key_counts = lengths.view(F, B).sum(dim=1)
    offsets = torch.cat(
        [torch.zeros(1, dtype=per_key_counts.dtype), per_key_counts.cumsum(dim=0)]
    )
    values = kjt.values()
    chunks: List[torch.Tensor] = []
    for i, key in enumerate(keys):
        chunk = values[int(offsets[i]) : int(offsets[i + 1])]
        size = key_to_size.get(key)
        if size is not None and size > 0:
            chunk = chunk.remainder(size)
        chunks.append(chunk)
    return KeyedJaggedTensor(
        keys=keys,
        values=torch.cat(chunks).contiguous(),
        lengths=lengths,
        weights=kjt.weights_or_none(),
    )


def remap_batch_ids(batch: Any, key_to_size: Dict[str, int]) -> Any:
    """Clamp every KJT inside an ``MTMLROOTrainBatch`` into the model's hash space.

    The dump's IDs may exceed the model's embedding-table sizes (it was
    preprocessed under a different ``hash_size_cap``). ``torch.embedding``
    raises ``IndexError: index out of range`` if any ID >= ``num_embeddings``.
    Per-key modulo is correct because the dump already applied production-size
    hashing, so its IDs are uniformly distributed in ``[0, prod_size)``;
    modulo by our smaller capped size preserves coverage.
    """
    fields: Dict[str, Any] = {
        "ro_id_list_features": _remap_kjt_ids(batch.ro_id_list_features, key_to_size),
        "nro_id_list_features": _remap_kjt_ids(batch.nro_id_list_features, key_to_size),
    }
    if batch.unary_features is not None:
        fields["unary_features"] = _remap_kjt_ids(batch.unary_features, key_to_size)
    return dataclasses.replace(batch, **fields)


# ---------------------------------------------------------------------------
# Row slicing (loaded batch -> first n RO rows)
# ---------------------------------------------------------------------------


def _select_kjt_range(
    kjt: KeyedJaggedTensor, start: int, end: int
) -> KeyedJaggedTensor:
    """Keep rows ``[start, end)`` of a KJT (per key)."""
    keys = kjt.keys()
    F = len(keys)
    full_lengths = kjt.lengths()
    B = full_lengths.shape[0] // F
    if start == 0 and end == B:
        return kjt
    full_per_key = full_lengths.view(F, B)
    new_lens_per_key = full_per_key[:, start:end].contiguous()
    values = kjt.values()
    weights = kjt.weights_or_none()
    per_key_total = full_per_key.sum(dim=1)
    key_offsets = torch.cat(
        [torch.zeros(1, dtype=per_key_total.dtype), per_key_total.cumsum(0)]
    )
    new_values_chunks: List[torch.Tensor] = []
    new_weights_chunks: Optional[List[torch.Tensor]] = (
        [] if weights is not None else None
    )
    for ki in range(F):
        ko_start = int(key_offsets[ki])
        ko_end = int(key_offsets[ki + 1])
        key_vals = values[ko_start:ko_end]
        intra_start = int(full_per_key[ki, :start].sum()) if start > 0 else 0
        intra_end = int(full_per_key[ki, :end].sum())
        new_values_chunks.append(key_vals[intra_start:intra_end])
        if new_weights_chunks is not None and weights is not None:
            key_w = weights[ko_start:ko_end]
            new_weights_chunks.append(key_w[intra_start:intra_end])
    new_weights = (
        torch.cat(new_weights_chunks).contiguous()
        if new_weights_chunks is not None
        else None
    )
    return KeyedJaggedTensor(
        keys=keys,
        values=torch.cat(new_values_chunks).contiguous(),
        lengths=new_lens_per_key.flatten().contiguous(),
        weights=new_weights,
    )


def select_rows_slice(batch: Any, start: int, n: int) -> Any:
    """Slice the loaded batch to RO rows ``[start, start+n)`` (and matching NRO).

    Loaded preproc files have B_loaded ~hundreds-to-thousands of rows; the
    trainer's per-rank ``local_batch_size`` is much smaller, and
    PaddingRegionTransform's NRO budget is fixed at trace time as
    ``local_batch_size * cfg.num_candidates``, so we must shrink B_RO before
    handing the batch to the compiled train_step. The ``start`` offset lets
    callers rotate the window across steps so different files / steps see
    different rows of the dump (otherwise every step reuses rows 0..n-1
    only).
    """
    nc = batch.num_candidates
    B_orig = nc.shape[0]
    end = start + n
    if start < 0 or end > B_orig:
        raise ValueError(
            f"Requested rows [{start}, {end}) but loaded batch has B={B_orig}; "
            f"reduce --local_batch_size or use a larger preproc batch."
        )
    if start == 0 and n == B_orig:
        return batch
    nro_start = int(nc[:start].sum().item()) if start > 0 else 0
    nro_end = int(nc[:end].sum().item())
    fields: Dict[str, Any] = {
        "ro_float_features": batch.ro_float_features[start:end].contiguous(),
        "nro_float_features": batch.nro_float_features[nro_start:nro_end].contiguous(),
        "num_candidates": nc[start:end].contiguous(),
        "supervision_label": batch.supervision_label[:, nro_start:nro_end].contiguous(),
        "supervision_weight": batch.supervision_weight[
            :, nro_start:nro_end
        ].contiguous(),
        "ro_id_list_features": _select_kjt_range(batch.ro_id_list_features, start, end),
        "nro_id_list_features": _select_kjt_range(
            batch.nro_id_list_features, nro_start, nro_end
        ),
    }
    if batch.unary_features is not None:
        fields["unary_features"] = _select_kjt_range(
            batch.unary_features, nro_start, nro_end
        )
    if batch.all_labels is not None:
        fields["all_labels"] = {
            kk: v[nro_start:nro_end].contiguous() for kk, v in batch.all_labels.items()
        }
    return dataclasses.replace(batch, **fields)


# ---------------------------------------------------------------------------
# Top-level orchestrator (the public entry point trainers should use)
# ---------------------------------------------------------------------------


def replay_batch_iter(
    data_pipe: BatchReplayPipe,
    key_to_size: Dict[str, int],
    *,
    max_steps: int,
    local_batch_size: int,
    num_candidates: int,
) -> Iterator[Any]:
    """Yield up to ``max_steps`` ready-to-train batches from ``data_pipe``.

    Composes the standard real-data preprocessing chain in order:
      1. ``select_rows_slice`` — slice the loaded batch to ``local_batch_size``
         RO rows. Loaded files have B_loaded ~hundreds-thousands; the trace
         expects per-rank ``local_batch_size`` rows. The window rotates
         across steps (``yielded % n_windows``) so each step sees different
         rows of the loaded file rather than reusing rows 0..n-1.
      2. ``remap_batch_ids`` — modulo loaded KJT IDs into the model's hash
         space (the dump was preprocessed under a different ``hash_size_cap``).

    Always yields the REAL jagged batch untouched. On the compiled path RecIR
    handles the padding. Per-row over-budget jagged lengths are clipped inside the model's own
    ``truncate_ro_features`` (mot_preprocessor.py), so no pre-truncation
    is needed here.

    Cycles through the pipe indefinitely; ``max_steps`` is the only
    terminator.
    """
    yielded = 0
    while True:
        for batch in data_pipe:
            if yielded >= max_steps:
                return
            B_loaded = batch.num_candidates.shape[0]
            n_windows = max(1, B_loaded // local_batch_size)
            window = yielded % n_windows
            start = window * local_batch_size
            batch = select_rows_slice(batch, start, local_batch_size)
            batch = remap_batch_ids(batch, key_to_size)
            nc = batch.num_candidates
            real_total = int(nc.sum())
            budget = nc.shape[0] * num_candidates
            logger.info(
                f"[Replay] step {yielded + 1}: sum(num_candidates)={real_total}, "
                f"NRO budget={budget} (fill={real_total / budget:.2%}, "
                f"max nc[i]={int(nc.max())}/{num_candidates})"
            )
            yield batch
            yielded += 1
