# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SparseCore embedding backend for DLRMv3: torchax + SC backward via shard_map.

Per-step data flow (no D2H for embeddings):
  CPU:  raw IDs → KJT → numpy → preprocess_csr() → globally-sharded PreprocessedInput
  CPU:  raw batch → preprocess_payload() via _ZeroEC (avoids paying for CPU embedding lookup)
  TPU:  tpu_sparse_dense_matmul(csr, emb_vars) → sharded activations (stay on device)
  TPU:  torchax DlrmV3Dense + jax.value_and_grad → loss + (dense_grads, act_grads)
  TPU:  make_array_from_single_device_arrays lifts sharded act_grads to global 256-device array
  TPU:  shard_map tpu_sparse_dense_matmul_grad(act_grads_global, csr, emb_vars) → new emb_vars

Embedding activations and dense weight updates never touch the CPU: SC forward output
(sharded JAX arrays) flows directly into the dense jit, which fuses forward, backward,
and SGD update in one compiled unit. act_grads are lifted from local 8-device sharding to
global 256-device sharding via make_array_from_single_device_arrays, taking per-device
jax.Array objects (no np.asarray conversion).

Batch-size constraint: local_batch_size % (local_device_count × num_sc_per_device) == 0.
On TPU v7x with 8 JAX devices/pod and 4 SparseCores/device: multiples of 32.

Token-as-sample trick: sequence features are flattened (local_bs, seq_len) → (local_bs*seq_len, 1)
so each token is an independent 1-ID lookup; SparseCore returns [local_bs*seq_len, emb_dim].
"""

from __future__ import annotations

import collections
import logging
from contextlib import contextmanager
from typing import Any, Generator

import jax
import numpy as np
import torch
import torch.nn as nn
from recir.tests.xla.dlrmv3_test_utils import DlrmV3Preprocess
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

_NUM_SC_PER_DEVICE_FALLBACK: int = 4


def _lazy_jte() -> tuple[Any, Any, Any, Any]:
    """Lazily import jax_tpu_embedding (pip-only, not in Buck third-party)."""
    # pyre-fixme[21]: Could not find module `jax_tpu_embedding.sparsecore.lib.nn`.
    from jax_tpu_embedding.sparsecore.lib.nn import (
        embedding as jte_embedding,
        embedding_spec as jte_spec,
        table_stacking as jte_stacking,
    )

    # pyre-fixme[21]: Could not find module `jax_tpu_embedding.sparsecore.utils`.
    from jax_tpu_embedding.sparsecore.utils import utils as jte_utils

    return jte_embedding, jte_spec, jte_stacking, jte_utils


def _num_sc_per_device() -> int:
    _, _, _, jte_utils = _lazy_jte()
    try:
        return jte_utils.num_sparsecores_per_device()
    except ValueError as e:
        logger.warning(
            f"jte_utils.num_sparsecores_per_device() failed ({e}); "
            f"defaulting to {_NUM_SC_PER_DEVICE_FALLBACK}"
        )
        return _NUM_SC_PER_DEVICE_FALLBACK


def _patch_jte_utils_for_tpu7x() -> None:
    """Monkey-patch jte_utils.num_sparsecores_per_device to handle TPU7x.

    auto_stack_tables() calls num_sparsecores_per_device() internally and raises
    ValueError for unrecognised device kinds (e.g. TPU7x), ignoring the
    num_sc_per_device argument we pass. This patch makes it fall back to 4.
    Idempotent: a second call (e.g. in tests) does not double-wrap.
    """
    _, _, _, jte_utils = _lazy_jte()
    if getattr(jte_utils, "_patched_for_tpu7x", False):
        return
    _orig: Any = jte_utils.num_sparsecores_per_device

    def _safe_num_sc() -> int:
        try:
            return _orig()
        except ValueError:
            return _NUM_SC_PER_DEVICE_FALLBACK

    jte_utils.num_sparsecores_per_device = _safe_num_sc
    jte_utils._patched_for_tpu7x = True  # pyre-ignore[16]


# ---------------------------------------------------------------------------
# KJT → numpy (CSR preprocessing input)
# ---------------------------------------------------------------------------


def _kjt_to_numpy(
    features: KeyedJaggedTensor,
    seq_lens: dict[str, int],
) -> dict[str, np.ndarray]:
    """Convert KJT to per-feature 2D numpy arrays [B, max_seq_len].

    Right-padded with 0 for variable-length sequences.
    Required by preprocess_sparse_dense_matmul_input.
    """
    keys: list[str] = features.keys()
    all_ids: np.ndarray = features.values().cpu().numpy()
    all_lengths: np.ndarray = features.lengths().cpu().numpy()
    batch_size: int = len(all_lengths) // len(keys)

    result: dict[str, np.ndarray] = {}
    id_offset: int = 0
    for i, key in enumerate(keys):
        lengths_k = all_lengths[i * batch_size : (i + 1) * batch_size]
        total_ids_k = int(lengths_k.sum())
        ids_k = all_ids[id_offset : id_offset + total_ids_k]
        max_seq = seq_lens.get(key, 1)
        arr = np.zeros((batch_size, max_seq), dtype=np.int32)
        pos: int = 0
        for b, length in enumerate(lengths_k):
            take = min(int(length), max_seq)
            arr[b, :take] = ids_k[pos : pos + take]
            pos += int(length)
        result[key] = arr
        id_offset += total_ids_k
    return result


# ---------------------------------------------------------------------------
# _ZeroEC: dummy EmbeddingCollection for payload-only preprocessing
# ---------------------------------------------------------------------------


class _ZeroEC(nn.Module):
    """EmbeddingCollection replacement that returns zeros with correct shapes.

    Used by preprocess_payload() to run DlrmV3Preprocess.forward() without
    paying for the CPU embedding lookup. We only need payload_features
    (timestamps, offsets, supervision labels) from this call — the actual
    embedding values come from SparseCore.
    """

    def __init__(
        self,
        table_configs: list[EmbeddingConfig],
        seq_lens: dict[str, int],
        local_bs: int,
    ) -> None:
        super().__init__()
        self._table_configs = table_configs
        self._seq_lens = seq_lens
        self._local_bs = local_bs

    def forward(self, kjt: KeyedJaggedTensor) -> dict[str, JaggedTensor]:
        result: dict[str, JaggedTensor] = {}
        for tc in self._table_configs:
            for fname in tc.feature_names:
                seq_len = self._seq_lens.get(fname, 1)
                lengths = torch.full((self._local_bs,), seq_len, dtype=torch.long)
                values = torch.zeros(self._local_bs * seq_len, tc.embedding_dim)
                result[fname] = JaggedTensor(values=values, lengths=lengths)
        return result


@contextmanager
def _zero_ec_ctx(
    preprocess_mod: DlrmV3Preprocess,
    table_configs: list[EmbeddingConfig],
    seq_lens: dict[str, int],
    local_bs: int,
) -> Generator[None, None, None]:
    zero_ec = _ZeroEC(table_configs, seq_lens, local_bs)
    orig = preprocess_mod.model._embedding_collection
    # pyre-ignore[8]: intentional replacement
    preprocess_mod.model._embedding_collection = zero_ec
    try:
        yield
    finally:
        preprocess_mod.model._embedding_collection = orig


# ---------------------------------------------------------------------------
# Public preprocessing functions (run on CPU every step)
# ---------------------------------------------------------------------------


def preprocess_payload(
    preprocess_mod: DlrmV3Preprocess,
    table_configs: list[EmbeddingConfig],
    seq_lens: dict[str, int],
    local_bs: int,
    uih_values: torch.Tensor,
    uih_lengths: torch.Tensor,
    cand_values: torch.Tensor,
    cand_lengths: torch.Tensor,
) -> tuple[Any, ...]:
    """Run DlrmV3Preprocess with _ZeroEC to extract payload without embedding lookup.

    Returns the full DlrmV3Preprocess output tuple. The emb_values slots contain
    zeros — callers must replace them with SparseCore activations.
    """
    with _zero_ec_ctx(preprocess_mod, table_configs, seq_lens, local_bs):
        return preprocess_mod(uih_values, uih_lengths, cand_values, cand_lengths)


def preprocess_csr(
    uih_values: torch.Tensor,
    uih_lengths: torch.Tensor,
    cand_values: torch.Tensor,
    cand_lengths: torch.Tensor,
    uih_kjt_keys: list[str],
    cand_kjt_keys: list[str],
    feature_specs: dict[str, Any],
    seq_lens: dict[str, int],
    local_dev: int,
    global_dev: int,
    num_sc: int,
    sharding_strategy: str = "MOD",
    allow_id_dropping: bool = False,
) -> Any:
    """Convert raw batch tensors → CSR format for tpu_sparse_dense_matmul.

    Runs on CPU every step. The CSR result is passed into the jit'd train step
    where tpu_sparse_dense_matmul dispatches it to SparseCore hardware.
    """
    jte_embedding, _, _, _ = _lazy_jte()

    merged_kjt = KeyedJaggedTensor(
        keys=uih_kjt_keys + cand_kjt_keys,
        values=torch.cat([uih_values, cand_values]),
        lengths=torch.cat([uih_lengths, cand_lengths]),
    )
    ids_np = {
        k: v
        for k, v in _kjt_to_numpy(merged_kjt, seq_lens).items()
        if k in feature_specs
    }
    # Token-as-sample trick: flatten (local_bs, seq_len) → (local_bs*seq_len, 1)
    for k in list(ids_np.keys()):
        if seq_lens.get(k, 1) > 1:
            ids_np[k] = ids_np[k].reshape(-1, 1)

    # global_device_count for ID-to-SC-slot routing across all pods.
    csr_inputs, _ = jte_embedding.preprocess_sparse_dense_matmul_input(
        features=ids_np,
        features_weights=None,
        feature_specs=feature_specs,
        local_device_count=local_dev,
        global_device_count=global_dev,
        num_sc_per_device=num_sc,
        sharding_strategy=sharding_strategy,
        allow_id_dropping=allow_id_dropping,
    )
    return csr_inputs


# Global meshes/shardings — built once per process, reused every step.
_GLOBAL_MESH: jax.sharding.Mesh | None = None
_DATA_SHARDING: jax.sharding.NamedSharding | None = None
_REPL_SHARDING: jax.sharding.NamedSharding | None = None
# Local 8-device mesh for assembling SC forward per-device outputs into
# per-process (1600, 64) arrays (no D2H).
_LOCAL_MESH: jax.sharding.Mesh | None = None
_LOCAL_ACT_SHARDING: jax.sharding.NamedSharding | None = None


def _ensure_global_csr_mesh() -> None:
    global _GLOBAL_MESH, _DATA_SHARDING, _REPL_SHARDING
    if _GLOBAL_MESH is not None:
        return
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("dev",))
    _GLOBAL_MESH = mesh
    _DATA_SHARDING = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dev"))
    _REPL_SHARDING = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _ensure_local_act_mesh() -> None:
    global _LOCAL_MESH, _LOCAL_ACT_SHARDING
    if _LOCAL_MESH is not None:
        return
    # Use axis "lx" (local-x) to avoid name collision with the global 256-device
    # "x" mesh used for emb_vars (build_sparsecore_tables uses jax.devices()).
    _LOCAL_MESH = jax.sharding.Mesh(np.array(jax.local_devices()), ("lx",))
    _LOCAL_ACT_SHARDING = jax.sharding.NamedSharding(
        _LOCAL_MESH, jax.sharding.PartitionSpec("lx", None)
    )


def make_global_csr(
    csr_inputs: Any,
) -> tuple[Any, Any]:
    """Pre-shard a local CSR pytree for the global shard_map SC backward.

    Each of 32 processes contributes its per-process CSR rows. Across all
    processes, 256 devices each receive [4, ...] rows (32 processes × [32,...] /
    256 devices = [4,...] per device — divisible ✓).

    Returns:
      csr_global:   CSR with JAX arrays globally sharded via
                    make_array_from_process_local_data, scalars replicated.
      csr_in_specs: pytree of PartitionSpecs matching csr_global's leaf structure:
                    P("dev") for arrays (sharded), P() for scalars (replicated).
    """
    _ensure_global_csr_mesh()
    P: type[jax.sharding.PartitionSpec] = jax.sharding.PartitionSpec

    def _shard(x: Any) -> jax.Array:
        arr = np.asarray(x)
        sharding = _DATA_SHARDING if arr.ndim > 0 else _REPL_SHARDING
        return jax.make_array_from_process_local_data(sharding, arr)

    def _spec(x: Any) -> jax.sharding.PartitionSpec:
        return P("dev") if np.asarray(x).ndim > 0 else P()

    return jax.tree.map(_shard, csr_inputs), jax.tree.map(_spec, csr_inputs)


def activations_to_emb_values(
    activations: dict[str, jax.Array],
    emb_keys: list[str],
) -> list[jax.Array]:
    """Assemble per-device SC activations into per-process arrays without D2H.

    tpu_sparse_dense_matmul with global_device_count=8 returns per-device arrays:
    each of the 8 local devices holds shape (tokens/8, emb_dim), e.g. (200, 64).
    These are NOT logically a single (1600, 64) array — each device has different data.

    make_array_from_single_device_arrays reinterprets the 8 per-device jax.Array
    objects as shards of a (1600, 64) logical array with P("lx", None) sharding.
    shard.data is already on device — no np.asarray, no CPU roundtrip.
    """
    _ensure_local_act_mesh()
    # _LOCAL_ACT_SHARDING is set by _ensure_local_act_mesh() above; narrow the
    # module-global Optional[NamedSharding] for Pyre.
    sharding: jax.sharding.NamedSharding = _LOCAL_ACT_SHARDING  # type: ignore[assignment]
    assert sharding is not None
    result: list[jax.Array] = []
    for key in emb_keys:
        if key not in activations:
            raise KeyError(f"SparseCore activation missing feature '{key}'")
        act = activations[key]
        shards = act.addressable_shards
        if len(shards) <= 1:
            # Single-device (e.g. CPU backend fallback): use as-is.
            result.append(act)
            continue
        # shape: (per_dev_tokens, emb_dim), e.g. (200, 64) per device
        per_process_shape = (act.shape[0] * len(shards), act.shape[1])
        device_arrs = [s.data for s in shards]
        result.append(
            jax.make_array_from_single_device_arrays(
                per_process_shape,
                sharding,  # P("lx", None) on local 8-device mesh
                device_arrs,
            )
        )
    return result


# ---------------------------------------------------------------------------
# SparseCore table initialization
# ---------------------------------------------------------------------------


def build_sparsecore_tables(
    table_configs: list[EmbeddingConfig],
    seq_lens: dict[str, int],
    global_batch_size: int,
    world_size: int,
    embedding_lr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Initialize SparseCore EmbeddingVariables and FeatureSpecs.

    Returns:
      emb_vars:      stacked EmbeddingVariable dict, sharded across all SC units
      feature_specs: dict mapping feature name → FeatureSpec
    """
    _patch_jte_utils_for_tpu7x()
    jte_embedding, jte_spec, jte_stacking, _ = _lazy_jte()

    local_bs: int = global_batch_size // world_size
    num_sc = _num_sc_per_device()
    local_dev = jax.local_device_count()
    global_dev = jax.device_count()

    logger.info(
        f"build_sparsecore_tables: global_bs={global_batch_size}, world={world_size}, "
        f"local_bs={local_bs}, local_dev={local_dev}, global_dev={global_dev}, "
        f"num_sc_per_device={num_sc}"
    )

    if local_bs % (local_dev * num_sc) != 0:
        raise ValueError(
            f"local_batch_size ({local_bs}) must be divisible by "
            f"local_device_count ({local_dev}) × num_sc_per_device ({num_sc}) "
            f"= {local_dev * num_sc}"
        )

    feature_specs: dict[str, Any] = {}
    for tc in table_configs:
        # ONE TableSpec per table — shared across all features that use it.
        # Duplicate TableSpec objects with the same name corrupt auto_stack_tables.
        table_spec = jte_spec.TableSpec(
            vocabulary_size=tc.num_embeddings,
            embedding_dim=tc.embedding_dim,
            initializer=jax.nn.initializers.truncated_normal(
                stddev=1.0 / tc.embedding_dim**0.5
            ),
            optimizer=jte_spec.AdagradOptimizerSpec(
                learning_rate=embedding_lr,
                initial_accumulator_value=0.0,
            ),
            combiner="sum",  # no-op for 1-ID samples (token-as-sample trick)
            name=tc.name,
        )
        for fname in tc.feature_names:
            seq_len = seq_lens.get(fname, 1)
            if seq_len > 1:
                feat_input_shape: tuple[int, int] = (local_bs * seq_len, 1)
                feat_output_shape: tuple[int, int] = (
                    local_bs * seq_len,
                    tc.embedding_dim,
                )
            else:
                feat_input_shape = (local_bs, 1)
                feat_output_shape = (local_bs, tc.embedding_dim)
            feature_specs[fname] = jte_spec.FeatureSpec(
                name=fname,
                table_spec=table_spec,
                input_shape=feat_input_shape,
                output_shape=feat_output_shape,
            )
        logger.info(
            f"  table '{tc.name}': ({tc.num_embeddings}, {tc.embedding_dim}), "
            f"features={tc.feature_names}"
        )

    # Place each table in its own singleton stack instead of using
    # auto_stack_tables(). auto_stack_tables groups all tables into a single
    # StackedTableSpec, which at default hash_size=10M produces a 30 GB tensor
    # (3 tables × 10M × 256 × f32) that exceeds SC's 16 GiB per-tensor HBM
    # limit and fails tpu_sparse_dense_matmul with INVALID_ARGUMENT. Per-table
    # singleton stacks keep each table at ~10 GB, well within the limit.
    for tc in table_configs:
        jte_stacking.stack_tables(
            feature_specs,
            table_names=[tc.name],
            global_device_count=global_dev,
            num_sc_per_device=num_sc,
            stack_name=tc.name,
        )

    # Global 256-device mesh for emb_vars: the embedding table is distributed
    # across all SparseCores. SC forward uses global_device_count=8 (local) to
    # execute locally while PJRT coordinates all pods simultaneously.
    devices = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devices, ("x",))
    global_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None)
    )

    # Workaround: init_embedding_variables uses PartitionSpec != tuple comparison
    # which always returns True in newer JAX. Call _init_stacked_embedding_table
    # directly to skip the broken check.
    table_specs = jte_embedding.get_table_specs(feature_specs)
    stacks: dict[str, list[Any]] = collections.defaultdict(list)
    for tspec in jax.tree.leaves(table_specs):
        stacks[tspec.setting_in_stack.stack_name].append(tspec)
    for ts in stacks.values():
        ts.sort(key=lambda t: t.setting_in_stack.row_offset_in_shard)

    sharding_axis = next(s for s in global_sharding.spec if s is not None)
    per_table_width = global_sharding.mesh.size * num_sc
    rngs = jax.random.split(jax.random.PRNGKey(0), per_table_width * len(stacks))
    emb_vars: dict[str, Any] = {}
    for i, (stack_name, ts) in enumerate(stacks.items()):
        rngs_i = rngs[i * per_table_width : (i + 1) * per_table_width]
        emb_vars[stack_name] = jte_embedding._init_stacked_embedding_table(
            rng=rngs_i,
            stack_name=stack_name,
            table_specs=ts,
            global_sharding=global_sharding,
            sharding_axis=sharding_axis,
            num_sparsecore_per_device=num_sc,
        )

    logger.info(
        f"SparseCore: {len(stacks)} stacked tables initialized, "
        f"num_sc_per_device={num_sc}"
    )
    return emb_vars, feature_specs
