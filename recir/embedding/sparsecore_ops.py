# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""SparseCore compute primitives shared across recir models.

Provides:
  SparseCoreLookup      — jit-compiled tpu_sparse_dense_matmul wrapper.
  SparsecoreBackedEC    — drop-in torchrec EmbeddingCollection replacement.
  init_sc_embedding_variables — shared factory for JAX embedding init.
  _kjt_to_numpy / _lazy_jte / _num_sc_per_device / _swap_ec — helpers.

This module owns the JAX + jax_tpu_embedding dependency. The hardware-agnostic
backend interface lives in ``recir.embedding.backend``; import from there when
you only need the ABC, so CPU-only callers do not pay for the JAX stack.

Architecture:
  SparseCoreLookup wraps tpu_sparse_dense_matmul in jax.jit. Calling it
  inside jax.jit (rather than eager Python) is required for multi-host: with
  jax.distributed.initialize() in effect, PJRT coordinates all pods to enter
  the kernel simultaneously, satisfying the SparseCore hardware collective.

  SparsecoreBackedEC.forward() runs inside the recir graph break. It calls
  SparseCoreLookup.compute() which does CSR preprocessing + jit'd matmul,
  then packages the activations as JaggedTensors for the downstream model.

Batch-size constraint: local_batch_size must be a multiple of
  local_device_count × num_sc_per_device (e.g. multiples of 32 on TPU v7x
  with 8 JAX devices/pod and 4 SparseCores/device).
"""

from __future__ import annotations

import collections
import logging
from contextlib import contextmanager
from typing import Any, Generator

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# SparseCores per JAX device. jte_utils.num_sparsecores_per_device() raises
# ValueError for device kinds it doesn't recognise yet (e.g. "TPU7x"). Fall back
# to 4 — the value jte uses internally for TPU v5/v6/v7 chips.
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


def _kjt_to_numpy(features: KeyedJaggedTensor) -> dict[str, np.ndarray]:
    """Convert KJT to per-feature flat (N, 1) numpy arrays.

    PRT (via train.py's ``extra_alignment=sc_modulus × num_processes``) is
    configured to deliver per-host per-key totals that exactly match
    ``FeatureSpec.input_shape[0]``. This is a straight per-key slice +
    reshape — no per-sample reorganization, no alignment bridge.
    """
    keys: list[str] = features.keys()
    all_ids: np.ndarray = features.values().cpu().numpy().astype(np.int32, copy=False)
    all_lengths: np.ndarray = features.lengths().cpu().numpy()
    batch_size: int = len(all_lengths) // len(keys)

    result: dict[str, np.ndarray] = {}
    id_offset: int = 0
    for i, key in enumerate(keys):
        total = int(all_lengths[i * batch_size : (i + 1) * batch_size].sum())
        result[key] = all_ids[id_offset : id_offset + total].reshape(-1, 1)
        id_offset += total
    return result


@contextmanager
def _swap_ec(
    ec_holder: nn.Module,
    new_ec: nn.Module,
) -> Generator[None, None, None]:
    """Temporarily replace ec_holder._embedding_collection with new_ec.

    The caller navigates to the right parent module (the one that owns
    _embedding_collection as a direct attribute).

    DLRMv3 usage:  _swap_ec(orig_preprocess.model, jax_ec)
    xsurface usage: _swap_ec(real_sparse_arch, jax_ec)
    """
    orig_ec = ec_holder._embedding_collection  # pyre-ignore[16]
    # pyre-ignore[8,16]: Intentionally replacing EmbeddingCollection with Module
    ec_holder._embedding_collection = new_ec
    try:
        yield
    finally:
        ec_holder._embedding_collection = orig_ec  # pyre-ignore[16]


# ---------------------------------------------------------------------------
# SparseCoreLookup: jit-compiled SparseCore compute engine
# ---------------------------------------------------------------------------


class SparseCoreLookup:
    """jit-compiled SparseCore embedding lookup.

    Called from SparsecoreBackedEC.forward() inside the recir graph break.
    Wrapping tpu_sparse_dense_matmul in jax.jit (rather than calling it from
    eager Python) is required for multi-host: PJRT coordinates all pods to
    enter the kernel simultaneously, satisfying the SparseCore hardware
    collective requirement.
    """

    def __init__(
        self,
        emb_vars: Any,
        feature_specs: Any,
        global_device_count: int,
        local_device_count: int,
        num_sc_per_device: int,
        sharding_strategy: str = "MOD",
        allow_id_dropping: bool = False,
    ) -> None:
        jte_embedding, _, _, _ = _lazy_jte()
        self._jte_embedding: Any = jte_embedding
        self.emb_vars: Any = emb_vars
        self._feature_specs: Any = feature_specs
        self._global_device_count: int = global_device_count
        self._local_device_count: int = local_device_count
        self._num_sc: int = num_sc_per_device
        self._sharding_strategy: str = sharding_strategy
        self._allow_id_dropping: bool = allow_id_dropping
        self._last_csr: Any = None

        # Build jit-compiled wrappers once at init, closing over static args.
        # _kernel_num_devices: local_device_count (=8 per pod) so CSR and emb_vars
        # — both structured for the local 8 devices — match the kernel.
        # After jax.distributed.initialize(), jax.device_count() returns 256;
        # using that here would mismatch the locally-structured CSR and crash.
        _jte: Any = jte_embedding
        _fspecs: Any = feature_specs
        _kernel_num_devices: int = jax.local_device_count()
        _strat: str = sharding_strategy
        _nsc: int = num_sc_per_device

        @jax.jit
        def _matmul_jit(csr_inputs: Any, emb_vars: Any) -> dict[str, jax.Array]:
            return _jte.tpu_sparse_dense_matmul(
                csr_inputs,
                emb_vars,
                _fspecs,
                global_device_count=_kernel_num_devices,
                sharding_strategy=_strat,
                num_sc_per_device=_nsc,
            )

        @jax.jit
        def _grad_jit(
            grads: dict[str, jax.Array], csr_inputs: Any, emb_vars: Any
        ) -> Any:
            return _jte.tpu_sparse_dense_matmul_grad(
                grads,
                csr_inputs,
                emb_vars,
                _fspecs,
                global_device_count=_kernel_num_devices,
                sharding_strategy=_strat,
                num_sc_per_device=_nsc,
            )

        self._matmul_jit: Any = _matmul_jit
        self._grad_jit: Any = _grad_jit

    def compute(self, merged_kjt: KeyedJaggedTensor) -> dict[str, jax.Array]:
        """CSR preprocessing + jit'd SparseCore forward pass.

        ``merged_kjt`` is at the PRT-padded shape. PRT is configured with
        ``extra_alignment = sc_modulus × num_processes`` so each per-host
        per-key total already matches ``FeatureSpec.input_shape[0]``.
        ``_kjt_to_numpy`` is a straight per-key slice + reshape — no
        sentinel padding, no alignment bridge. The kernel processes each
        row as one independent lookup and the activation flows downstream
        at the same PRT-padded shape the dense graph was traced against.
        """
        logger.info("[SC] compute: start CSR preprocessing")
        ids_np = {
            k: v
            for k, v in _kjt_to_numpy(merged_kjt).items()
            if k in self._feature_specs
        }
        # Preprocessing uses jax.device_count() (global) for ID-to-SC-slot routing.
        # The kernel (_kernel_num_devices) uses local_device_count for execution;
        # PJRT via jax.distributed.initialize() ensures all pods execute the jit'd
        # kernel simultaneously so the hardware collective completes. This asymmetry
        # between preprocessing and kernel device counts is intentional.
        csr_inputs, _ = self._jte_embedding.preprocess_sparse_dense_matmul_input(
            features=ids_np,
            features_weights=None,
            feature_specs=self._feature_specs,
            local_device_count=self._local_device_count,
            global_device_count=jax.device_count(),
            num_sc_per_device=self._num_sc,
            sharding_strategy=self._sharding_strategy,
            allow_id_dropping=self._allow_id_dropping,
        )
        logger.info("[SC] compute: CSR done, calling _matmul_jit")
        self._last_csr = csr_inputs
        activations = self._matmul_jit(csr_inputs, self.emb_vars)
        jax.block_until_ready(activations)
        return activations

    def update_embeddings(self, grads: dict[str, jax.Array]) -> None:
        """Apply fused Adagrad update on SparseCore."""
        self.emb_vars = self._grad_jit(grads, self._last_csr, self.emb_vars)


# ---------------------------------------------------------------------------
# SparsecoreBackedEC: drop-in for torchrec EmbeddingCollection
# ---------------------------------------------------------------------------


class SparsecoreBackedEC(nn.Module):
    """EmbeddingCollection replacement using SparseCoreLookup.

    forward(KeyedJaggedTensor) -> Dict[str, JaggedTensor] — same interface as
    torchrec EmbeddingCollection, so it swaps into any model transparently.

    Shape contract: input KJT is at the PRT-padded shape (real IDs +
    sentinel-0 deficit absorbers, contiguously packed per key); the SC
    kernel's ``FeatureSpec.input_shape`` was configured to match the same
    per-key total at init time. Output is therefore already at the cached
    shape the dense graph was traced/compiled against — no strip on input,
    no zero-fill on output. Dummy padding-slot embeddings (from sentinel-0
    IDs) flow downstream alongside real embeddings, the same way PRT's
    deficit-absorbing dummy lengths flow.

    Gradient wiring: register_hook on each torch activation captures the
    upstream grad and forwards it (unchanged shape) to
    SparseCoreLookup.update_embeddings() for on-chip Adagrad update once
    all pending feature keys have received gradients.
    """

    def __init__(self, lookup: SparseCoreLookup) -> None:
        super().__init__()
        self._lookup: SparseCoreLookup = lookup
        self._pending: set[str] = set()
        self._grads: dict[str, jax.Array] = {}

    def forward(self, features: KeyedJaggedTensor) -> dict[str, JaggedTensor]:
        activations = self._lookup.compute(features)

        self._pending = set(activations.keys())
        self._grads = {}
        result: dict[str, JaggedTensor] = {}
        keys: list[str] = features.keys()
        all_lengths: torch.Tensor = features.lengths()
        batch_size: int = len(all_lengths) // len(keys)
        for i, key in enumerate(keys):
            if key not in activations:
                continue
            # SparseCore returns a sharded JAX array split across local devices.
            # np.asarray() only captures one shard, so we must explicitly gather
            # all addressable shards along the batch axis.
            act_raw = activations[key]
            if (
                hasattr(act_raw, "addressable_shards")
                and len(act_raw.addressable_shards) > 1
            ):
                act_np = np.concatenate(
                    [np.asarray(s.data) for s in act_raw.addressable_shards], axis=0
                )
            else:
                act_np = np.asarray(act_raw)
            logger.debug(f"[SC] gathered key={key}, shape={act_np.shape}")
            act_torch = torch.from_numpy(act_np).clone()
            act_torch.requires_grad_(True)
            act_torch.register_hook(lambda g, k=key: self._capture_grad(k, g))
            lengths_k = all_lengths[i * batch_size : (i + 1) * batch_size]
            result[key] = JaggedTensor(values=act_torch, lengths=lengths_k)
        return result

    def _capture_grad(self, key: str, grad: torch.Tensor) -> None:
        # Grad shape == FeatureSpec.input_shape (matches forward output).
        # SC grad kernel consumes it directly — no reshape, no pad.
        self._grads[key] = jnp.array(grad.detach().cpu().numpy())
        self._pending.discard(key)
        if not self._pending:
            self._lookup.update_embeddings(self._grads)


# ---------------------------------------------------------------------------
# Shared factory: EmbeddingVariables init + SparseCoreLookup + SparsecoreBackedEC
# ---------------------------------------------------------------------------


def init_sc_embedding_variables(
    table_configs: list[EmbeddingConfig],
    feature_padded_input_sizes: dict[str, int],
    world_size: int,
    embedding_lr: float,
) -> tuple[Any, Any, SparseCoreLookup, SparsecoreBackedEC]:
    """Init SparseCore EmbeddingVariables; return (emb_vars, feature_specs, lookup, ec).

    ``feature_padded_input_sizes`` maps each feature name → the per-host
    PRT-padded total values count (derived from the trace-time
    ``cached_input_kjt`` lengths). ``FeatureSpec.input_shape`` is set to
    ``(size, 1)`` so the SC kernel processes the same rows PRT produces at
    runtime — no per-sample reshape, no strip/zero-fill bridge needed.
    """
    jte_embedding, jte_spec, jte_stacking, _ = _lazy_jte()
    num_sc = _num_sc_per_device()
    local_dev = jax.local_device_count()
    global_dev = jax.device_count()
    sc_modulus = local_dev * num_sc

    logger.info(
        f"init_sc_embedding_variables: world_size={world_size}, "
        f"jax.device_count()={global_dev}, "
        f"jax.local_device_count()={local_dev}, "
        f"jax.process_count()={jax.process_count()}, "
        f"num_sc_per_device={num_sc}, "
        f"feature_padded_input_sizes={feature_padded_input_sizes}"
    )
    for fname, fsz in feature_padded_input_sizes.items():
        if fsz % sc_modulus != 0:
            raise ValueError(
                f"feature_padded_input_size for '{fname}' ({fsz}) must be a "
                f"multiple of local_device_count ({local_dev}) × "
                f"num_sc_per_device ({num_sc}) = {sc_modulus}. PRT's padded "
                f"per-host per-key total must be aligned to SC's shard width."
            )

    feature_specs: dict[str, Any] = {}
    for tc in table_configs:
        # ONE TableSpec per table, shared by all features on it.
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
            combiner="sum",
            name=tc.name,
        )
        for fname in tc.feature_names:
            if fname not in feature_padded_input_sizes:
                raise KeyError(
                    f"feature_padded_input_sizes is missing key '{fname}' "
                    f"(table '{tc.name}'). All features on every table "
                    f"need an entry derived from the cached_input_kjt."
                )
            input_sz = feature_padded_input_sizes[fname]
            feature_specs[fname] = jte_spec.FeatureSpec(
                name=fname,
                table_spec=table_spec,
                input_shape=(input_sz, 1),
                output_shape=(input_sz, tc.embedding_dim),
            )
        logger.info(
            f"SparseCore table '{tc.name}': "
            f"({tc.num_embeddings}, {tc.embedding_dim}), "
            f"features={tc.feature_names}"
        )

    # Place each table in its own singleton stack instead of using
    # auto_stack_tables(). auto_stack_tables groups all tables into a single
    # StackedTableSpec, which at large hash_size produces a tensor that exceeds
    # SC's 16 GiB per-tensor HBM limit. Per-table singleton stacks stay bounded.
    for tc in table_configs:
        jte_stacking.stack_tables(
            feature_specs,
            table_names=[tc.name],
            global_device_count=global_dev,
            num_sc_per_device=num_sc,
            stack_name=tc.name,
        )

    devices = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devices, ("x",))
    global_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None)
    )

    # Workaround: init_embedding_variables compares PartitionSpec with plain tuples
    # via !=, which always returns True in newer JAX where PartitionSpec is no longer
    # a tuple subclass. Call _init_stacked_embedding_table directly to skip the broken
    # validation, replicating init_embedding_variables logic.
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
        rngs_for_table = rngs[i * per_table_width : (i + 1) * per_table_width]
        emb_vars[stack_name] = jte_embedding._init_stacked_embedding_table(
            rng=rngs_for_table,
            stack_name=stack_name,
            table_specs=ts,
            global_sharding=global_sharding,
            sharding_axis=sharding_axis,
            num_sparsecore_per_device=num_sc,
        )
    logger.info(
        f"SparseCore: initialized {len(stacks)} stacked tables, "
        f"num_sc_per_device={num_sc}"
    )

    lookup = SparseCoreLookup(
        emb_vars=emb_vars,
        feature_specs=feature_specs,
        global_device_count=global_dev,
        local_device_count=local_dev,
        num_sc_per_device=num_sc,
    )
    ec = SparsecoreBackedEC(lookup)
    return emb_vars, feature_specs, lookup, ec
