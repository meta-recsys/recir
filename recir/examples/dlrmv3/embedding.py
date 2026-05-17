# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pluggable embedding backends for DLRMv3: CPU/torchrec or JAX/SparseCore.

Two backends share the same EmbeddingBackend ABC:

  CPUTorchRecBackend  — torchrec DMP ROW_WISE on CPU via gloo PG.

  JAXTPUBackend — jax_tpu_embedding SparseCore: tables as EmbeddingVariables
                  sharded on SparseCore HW, lookup via tpu_sparse_dense_matmul,
                  Adagrad fused on-chip. No gloo PG required.

Architecture for JAXTPUBackend:
  - SparseCoreLookup wraps tpu_sparse_dense_matmul in jax.jit.  Calling it
    inside jax.jit (rather than eager Python) is required for multi-host: with
    jax.distributed.initialize() in effect, PJRT coordinates all pods to enter
    the kernel simultaneously, satisfying the SparseCore hardware collective.
  - SparsecoreBackedEC.forward() runs inside the recir graph break.  It calls
    SparseCoreLookup.compute() which does CSR preprocessing + jit'd matmul,
    then packages the activations as JaggedTensors for the HSTU dense model.

Batch-size constraint (local_batch_size must be a multiple of
local_device_count × num_sc_per_device; e.g. multiples of 32 on TPU v7x
with 8 JAX devices/pod and 4 SparseCores/device).

Selected at runtime via DLRMv3Config.embedding_backend ("cpu" | "tpu").
"""

from __future__ import annotations

import collections
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from recir.api import Override
from recir.override_ops import default_override_patch_graph
from recir.tests.xla.dlrmv3_test_utils import (
    DlrmV3Preprocess,
    DlrmV3SparseOverride,
    DummyPreprocess,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
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


# ---------------------------------------------------------------------------
# KJT → numpy helper for jte preprocessing
# ---------------------------------------------------------------------------


def _kjt_to_numpy(
    features: KeyedJaggedTensor,
    seq_lens: dict[str, int],
) -> dict[str, np.ndarray]:
    """Convert KJT to per-feature 2D numpy arrays [B, max_seq_len].

    Required input format for jte_embedding.preprocess_sparse_dense_matmul_input.
    Variable-length sequences are right-padded with 0.
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
# SparseCoreLookup: jit-compiled SparseCore compute called from inside the
# recir graph break via SparsecoreBackedEC.forward().
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
        seq_lens: dict[str, int],
        global_device_count: int,
        local_device_count: int,
        num_sc_per_device: int,
        sharding_strategy: str = "MOD",
        allow_id_dropping: bool = False,
    ) -> None:
        jte_embedding, _, _, _ = _lazy_jte()
        self._jte_embedding: Any = jte_embedding
        self._matmul_jit: Any = None
        self._grad_jit: Any = None
        self.emb_vars: Any = emb_vars
        self._feature_specs: Any = feature_specs
        self._seq_lens: dict[str, int] = seq_lens
        self._global_device_count: int = global_device_count
        self._local_device_count: int = local_device_count
        self._num_sc: int = num_sc_per_device
        self._sharding_strategy: str = sharding_strategy
        self._allow_id_dropping: bool = allow_id_dropping
        self._last_csr: Any = None

        # Build jit-compiled wrappers once at init, closing over static args.
        # Calling tpu_sparse_dense_matmul inside jax.jit (not eager Python) is
        # what makes multi-host work: with jax.distributed.initialize() in effect,
        # PJRT coordinates all pods to enter the compiled kernel simultaneously so
        # the SparseCore hardware collective completes without deadlock.
        _jte: Any = jte_embedding
        _fspecs: Any = feature_specs
        # _kernel_num_devices: local_device_count (=8 per pod) so CSR and emb_vars
        # — both structured for the local 8 devices — match the kernel.
        # After jax.distributed.initialize(), jax.device_count() returns 256;
        # using that here would mismatch the locally-structured CSR and crash.
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

        self._matmul_jit = _matmul_jit
        self._grad_jit = _grad_jit

    def compute(self, merged_kjt: KeyedJaggedTensor) -> dict[str, jax.Array]:
        """CSR preprocessing + jit'd SparseCore forward pass."""
        logger.info("[SC] compute: start CSR preprocessing")
        ids_np = {
            k: v
            for k, v in _kjt_to_numpy(merged_kjt, self._seq_lens).items()
            if k in self._feature_specs
        }
        # Sequence features: flatten (local_bs, seq_len) → (local_bs*seq_len, 1)
        # so each token is an independent 1-ID sample (RecML token-as-sample trick).
        for k in list(ids_np.keys()):
            if self._seq_lens.get(k, 1) > 1:
                ids_np[k] = ids_np[k].reshape(-1, 1)
        # Preprocessing uses jax.device_count() (=256 in multi-node) for global
        # ID-to-SC-slot routing across all pods.  The kernel (_kernel_num_devices)
        # uses local_device_count (=8) for execution; PJRT via jax.distributed.initialize()
        # ensures all pods execute the jit'd kernel simultaneously so the hardware
        # collective completes.  This asymmetry between preprocessing and kernel
        # device counts is intentional and required for correct multi-host operation.
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
# SparsecoreBackedEC: drop-in for torchrec EmbeddingCollection.
# Packages pre-computed SparseCore activations into JaggedTensors.
# ---------------------------------------------------------------------------


class SparsecoreBackedEC(nn.Module):
    """EmbeddingCollection replacement using SparseCoreLookup.

    forward(KeyedJaggedTensor) -> Dict[str, JaggedTensor] — same interface as
    torchrec EmbeddingCollection, so it swaps into DlrmHSTU transparently.

    Calls SparseCoreLookup.compute() which must run inside the recir graph break
    (tpu_sparse_dense_matmul requires the XLA GSPMD execution context).

    Gradient wiring: register_hook on each torch activation captures the upstream
    grad, which is passed to SparseCoreLookup.update_embeddings() for on-chip
    Adagrad update.
    """

    def __init__(self, lookup: SparseCoreLookup) -> None:
        super().__init__()
        self._lookup: SparseCoreLookup = lookup
        self._pending: set[str] = set()
        self._grads: dict[str, jax.Array] = {}
        # Per-step metadata for gradient re-padding (3D → padded for tpu_sdm_grad)
        self._act_shapes: dict[str, tuple[int, ...]] = {}
        self._act_lengths: dict[str, np.ndarray] = {}

    def forward(self, features: KeyedJaggedTensor) -> dict[str, JaggedTensor]:
        activations = self._lookup.compute(features)

        self._pending = set(activations.keys())
        self._grads = {}
        self._act_shapes = {}
        self._act_lengths = {}
        result: dict[str, JaggedTensor] = {}
        keys: list[str] = features.keys()
        all_lengths: torch.Tensor = features.lengths()
        batch_size: int = len(all_lengths) // len(keys)
        for i, key in enumerate(keys):
            if key not in activations:
                continue
            # SparseCore always returns 2D: (N, emb_dim) where
            # N = local_bs*seq_len for sequence features, local_bs for pooled.
            # tpu_sparse_dense_matmul returns a sharded JAX array split across
            # local devices. np.asarray() only captures one shard, so we must
            # explicitly gather all addressable shards along the batch axis.
            act_raw = activations[key]
            logger.debug(f"[SC] gathering shards for key={key}")
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
            lengths_k = all_lengths[i * batch_size : (i + 1) * batch_size]
            lengths_k_np = lengths_k.cpu().numpy()
            self._act_lengths[key] = lengths_k_np
            seq_len = self._lookup._seq_lens.get(key, 1)

            if seq_len > 1:
                # Derive actual_batch from SparseCore output — robust to KJT batch_size
                # discrepancy in multi-process scenarios where the merged KJT structure
                # makes len(all_lengths) // len(keys) differ from the actual per-process
                # batch that SparseCore processed.
                actual_batch = act_np.shape[0] // seq_len
                assert actual_batch <= batch_size, (
                    f"actual_batch ({actual_batch}) exceeds batch_size ({batch_size}) "
                    f"for sequence feature '{key}'"
                )
                lengths_k = all_lengths[i * batch_size : i * batch_size + actual_batch]
                lengths_k = lengths_k.clamp(max=seq_len)
                lengths_k_np = lengths_k.cpu().numpy()
                self._act_lengths[key] = lengths_k_np
                act_3d = act_np.reshape(actual_batch, seq_len, act_np.shape[-1])
                self._act_shapes[key] = act_3d.shape  # (actual_batch, seq_len, emb_dim)
                rows = [
                    act_3d[b, : int(lengths_k_np[b]), :] for b in range(actual_batch)
                ]
                act_flat = (
                    np.concatenate(rows, axis=0)
                    if rows
                    else np.zeros((0, act_np.shape[-1]), dtype=act_np.dtype)
                )
            else:
                # Pooled: (actual_batch, emb_dim), one embedding per sample.
                # Re-slice lengths_k by actual activation size to match seq branch.
                actual_batch = act_np.shape[0]
                assert actual_batch <= batch_size, (
                    f"actual_batch ({actual_batch}) exceeds batch_size ({batch_size}) "
                    f"for pooled feature '{key}'"
                )
                lengths_k = all_lengths[i * batch_size : i * batch_size + actual_batch]
                lengths_k_np = lengths_k.cpu().numpy()
                self._act_lengths[key] = lengths_k_np
                self._act_shapes[key] = act_np.shape  # (actual_batch, emb_dim) — 2D
                act_flat = act_np
            act_torch = torch.from_numpy(act_flat).clone()

            act_torch.requires_grad_(True)
            act_torch.register_hook(lambda g, k=key: self._capture_grad(k, g))
            result[key] = JaggedTensor(values=act_torch, lengths=lengths_k)
        return result

    def _capture_grad(self, key: str, grad: torch.Tensor) -> None:
        g_np = grad.detach().cpu().numpy()
        act_shape = self._act_shapes[key]
        if len(act_shape) == 3:
            # Sequence feature: re-pad (total_len, emb_dim) → (local_bs, seq_len, emb_dim)
            # then flatten to (local_bs*seq_len, emb_dim) for tpu_sparse_dense_matmul_grad.
            local_bs, max_seq_len, emb_dim = act_shape
            lengths = self._act_lengths[key]
            g_3d = np.zeros(act_shape, dtype=g_np.dtype)
            pos = 0
            for b in range(local_bs):
                length = int(lengths[b])
                g_3d[b, :length, :] = g_np[pos : pos + length]
                pos += length
            self._grads[key] = jnp.array(g_3d.reshape(local_bs * max_seq_len, emb_dim))
        else:
            # Pooled feature: gradient is already (local_bs, emb_dim)
            self._grads[key] = jnp.array(g_np)
        self._pending.discard(key)
        if not self._pending:
            self._lookup.update_embeddings(self._grads)


# ---------------------------------------------------------------------------
# JAXPreprocess: runtime dispatch target for JAXEmbeddingOverride
# ---------------------------------------------------------------------------


@contextmanager
def _swap_ec(
    orig_preprocess: DlrmV3Preprocess,
    jax_ec: nn.Module,
) -> Generator[None, None, None]:
    """Temporarily replace the model's EmbeddingCollection with a JAX-backed one."""
    orig_ec = orig_preprocess.model._embedding_collection
    # pyre-ignore[8]: Intentionally replacing EmbeddingCollection with Module
    orig_preprocess.model._embedding_collection = jax_ec
    try:
        yield
    finally:
        orig_preprocess.model._embedding_collection = orig_ec


class JAXPreprocess:
    """Callable that runs DlrmV3Preprocess with JAX/SparseCore embedding lookup.

    Passed to default_override_patch_graph as the runtime dispatch target.
    Runs as a recir graph break (outside XLA compilation).  Calls
    SparsecoreBackedEC.forward() which triggers the jit'd SparseCore lookup.
    """

    def __init__(
        self,
        orig_preprocess: DlrmV3Preprocess,
        jax_ec: nn.Module,
    ) -> None:
        self._orig = orig_preprocess
        self._jax_ec = jax_ec
        # KJT keys exposed for pre-compute in training loop
        self.uih_kjt_keys: list[str] = list(
            getattr(orig_preprocess, "uih_kjt_keys", [])
        )
        self.cand_kjt_keys: list[str] = list(
            getattr(orig_preprocess, "cand_kjt_keys", [])
        )

    def __call__(
        self,
        uih_values: torch.Tensor,
        uih_lengths: torch.Tensor,
        cand_values: torch.Tensor,
        cand_lengths: torch.Tensor,
    ) -> Any:
        with _swap_ec(self._orig, self._jax_ec):
            return self._orig(uih_values, uih_lengths, cand_values, cand_lengths)


# ---------------------------------------------------------------------------
# JAXEmbeddingOverride: recir Override using JAX/SparseCore lookup
# ---------------------------------------------------------------------------


class JAXEmbeddingOverride(Override):
    """Mirrors DlrmV3SparseOverride but dispatches to SparseCore at runtime.

    replacement(): swaps DlrmV3Preprocess → DummyPreprocess for tracing.
    patch_graph(): wires JAXPreprocess (not orig DlrmV3Preprocess) as the
                   runtime graph-break dispatch target.
    """

    def __init__(
        self,
        dummy: DummyPreprocess,
        jax_ec: nn.Module,
        num_ranks: int,
    ) -> None:
        self._dummy: DummyPreprocess = dummy
        self._jax_ec: nn.Module = jax_ec
        self._num_ranks: int = num_ranks
        self._orig_preprocess: DlrmV3Preprocess | None = None
        self.jax_preprocess: JAXPreprocess | None = None

    def replacement(self, fqn: str, orig_submodule: nn.Module) -> nn.Module:
        if isinstance(orig_submodule, DlrmV3Preprocess):
            self._orig_preprocess = orig_submodule
            return self._dummy  # tracing shim with anchor nodes
        return orig_submodule

    def patch_graph(
        self,
        gm: fx.GraphModule,
        flat_state: list[torch.Tensor],
    ) -> fx.GraphModule:
        assert self._orig_preprocess is not None, (
            "replacement() must run before patch_graph()"
        )
        self.jax_preprocess = JAXPreprocess(self._orig_preprocess, self._jax_ec)
        return default_override_patch_graph(
            gm,
            flat_state,
            self.jax_preprocess,  # pyre-ignore[6]: JAXPreprocess wraps Module
            num_ranks=self._num_ranks,
        )


# ---------------------------------------------------------------------------
# EmbeddingBackend ABC
# ---------------------------------------------------------------------------


class EmbeddingBackend(ABC):
    """Backend-agnostic interface for embedding storage, lookup, and PG init."""

    @abstractmethod
    def init_pg(self) -> tuple[int, int]:
        """Initialize process group. Returns (rank, world_size)."""
        ...

    @abstractmethod
    def setup(
        self,
        wrapper: nn.Module,
        table_configs: list[EmbeddingConfig],
        embedding_lr: float,
        dense_lr: float,
        world_size: int,
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        """Set up embedding storage. Returns (model, optimizer)."""
        ...

    @abstractmethod
    def get_override(self, dummy: DummyPreprocess, num_ranks: int) -> Override:
        """Return the recir Override to pass to @compile."""
        ...


# ---------------------------------------------------------------------------
# CPUTorchRecBackend
# ---------------------------------------------------------------------------


class _TorchRecSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: dict[str, Any],
    ) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type
        fused_params.setdefault("learning_rate", 0.1)
        super().__init__(fused_params=fused_params)

    def sharding_types(self, compute_device_type: str) -> list[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> list[str]:
        return [self._kernel_type]


class CPUTorchRecBackend(EmbeddingBackend):
    """Original D101506286 approach: torchrec DMP ROW_WISE on CPU via gloo."""

    def init_pg(self) -> tuple[int, int]:
        jax_rank = jax.process_index()
        jax_world = jax.process_count()
        logger.info(
            f"JAX: process_index={jax_rank}, process_count={jax_world}, "
            f"devices={len(jax.devices())}, platform={jax.devices()[0].platform}"
        )
        tpu_addrs = os.environ.get("TPU_PROCESS_ADDRESSES", "")
        addr_list = [a.strip() for a in tpu_addrs.split(",") if a.strip()]
        if not addr_list or len(addr_list) <= 1:
            logger.info("Single-node mode: skipping gloo PG init")
            return 0, 1
        short_host = addr_list[0].split(":")[0]
        # MASTER_ADDR overrides TPU_PROCESS_ADDRESSES for gloo coordinator.
        # In GKE, TPU_PROCESS_ADDRESSES uses short-form hostnames that GKE's
        # JAX runtime resolves internally but gloo's TCP rendezvous cannot.
        # Setting MASTER_ADDR to the full cluster-local FQDN fixes this:
        #   e.g. <pod>.<headless-svc>.default.svc.cluster.local
        coordinator = os.environ.get("MASTER_ADDR", short_host)
        gloo_rank = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
        gloo_world = len(addr_list)
        gloo_port = int(os.environ.get("MASTER_PORT", "29500"))
        logger.info(
            f"Gloo init: rank={gloo_rank}, world={gloo_world}, "
            f"coordinator={coordinator}:{gloo_port}"
        )
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{coordinator}:{gloo_port}",
            world_size=gloo_world,
            rank=gloo_rank,
            timeout=torch.distributed.distributed_c10d.timedelta(seconds=300),
        )
        logger.info(
            f"Gloo PG initialized: rank={dist.get_rank()}, world={dist.get_world_size()}"
        )
        return gloo_rank, gloo_world

    def setup(
        self,
        wrapper: nn.Module,
        table_configs: list[EmbeddingConfig],
        embedding_lr: float,
        dense_lr: float,
        world_size: int,
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        if not dist.is_initialized():
            logger.info("Single-node: skipping DMP, using model directly")
            return wrapper, torch.optim.SGD(
                wrapper.parameters(), lr=dense_lr, foreach=False
            )

        pg = dist.distributed_c10d._get_default_group()
        sharders = [
            _TorchRecSharder(
                ShardingType.ROW_WISE.value,
                EmbeddingComputeKernel.FUSED.value,
                {"learning_rate": embedding_lr},
            )
        ]
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=world_size, compute_device="cpu"),
        )
        plan = planner.collective_plan(wrapper, sharders, pg)
        logger.info("DMP sharding plan:")
        for module_path, module_plan in plan.plan.items():
            for param_name, param_sharding in module_plan.items():
                logger.info(
                    f"  {module_path}.{param_name}: {param_sharding.sharding_type}"
                )
        dmp = DistributedModelParallel(
            wrapper,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            # pyre-ignore[6]: _TorchRecSharder is a valid ModuleSharder subclass
            sharders=sharders,
            device=torch.device("cpu"),
            init_data_parallel=False,
        )
        dense_opt: torch.optim.Optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(dmp.named_parameters())),
            lambda params: torch.optim.SGD(params, lr=dense_lr, foreach=False),
        )
        logger.info("CPUTorchRecBackend: DMP ROW_WISE initialized on CPU")
        return dmp, dense_opt

    def get_override(self, dummy: DummyPreprocess, num_ranks: int) -> Override:
        sparse_override = DlrmV3SparseOverride(dummy)
        sparse_override._num_ranks = num_ranks
        return sparse_override


# ---------------------------------------------------------------------------
# JAXTPUBackend
# ---------------------------------------------------------------------------


class JAXTPUBackend(EmbeddingBackend):
    """SparseCore embedding backend using jax_tpu_embedding.

    Tables stored as EmbeddingVariables sharded on SparseCore HW.
    Lookup: tpu_sparse_dense_matmul (SparseCore).
    Optimizer: Adagrad fused on-chip via tpu_sparse_dense_matmul_grad.
    No gloo PG required.

    Set seq_lens and global_batch_size before calling setup():
        backend.seq_lens = {"uih_post_id": cfg.uih_len, ...}
        backend.global_batch_size = cfg.local_batch_size * world_size
    """

    def __init__(self) -> None:
        self._jax_ec: SparsecoreBackedEC | None = None
        self.lookup: SparseCoreLookup | None = None  # exposed for training loop
        self.seq_lens: dict[str, int] = {}
        self.global_batch_size: int = 0

    def init_pg(self) -> tuple[int, int]:
        rank = jax.process_index()
        world = jax.process_count()
        logger.info(
            f"JAXTPUBackend: JAX PJRT PG (rank={rank}, world={world}), "
            f"no gloo init needed"
        )
        return rank, world

    def setup(
        self,
        wrapper: nn.Module,
        table_configs: list[EmbeddingConfig],
        embedding_lr: float,
        dense_lr: float,
        world_size: int,
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        if self.global_batch_size <= 0:
            raise RuntimeError("Set backend.global_batch_size before calling setup()")

        jte_embedding, jte_spec, jte_stacking, _ = _lazy_jte()

        # Build jte FeatureSpec / TableSpec from EmbeddingConfig list.
        # Use per-process batch size (not global) for input/output_shape so that
        # SparseCore buffers match what each process actually processes.
        # For sequence features (seq_len > 1) use a 3D output_shape to get
        # un-pooled per-token embeddings that HSTU's attention requires.
        local_bs: int = self.global_batch_size // world_size
        num_sc = _num_sc_per_device()
        local_dev = jax.local_device_count()
        global_dev = jax.device_count()
        logger.info(
            f"JAXTPUBackend setup: global_batch_size={self.global_batch_size}, "
            f"world_size={world_size}, local_bs={local_bs}, "
            f"jax.device_count()={global_dev}, "
            f"jax.local_device_count()={local_dev}, "
            f"jax.process_count()={jax.process_count()}, "
            f"num_sc_per_device={num_sc}"
        )
        # Batch-size constraint: local_bs must be divisible by
        # local_device_count × num_sc_per_device (= {local_dev * num_sc} here).
        # Violation surfaces as an opaque SparseCore firmware crash.
        if local_bs % (local_dev * num_sc) != 0:
            raise ValueError(
                f"local_batch_size ({local_bs}) must be a multiple of "
                f"local_device_count ({local_dev}) × num_sc_per_device ({num_sc})"
                f" = {local_dev * num_sc}"
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
                combiner="sum",  # no-op for 1-ID samples (sequence token trick)
                name=tc.name,
            )
            for fname in tc.feature_names:
                seq_len = self.seq_lens.get(fname, 1)
                if seq_len > 1:
                    # Sequence feature: each token = separate 1-ID sample (RecML trick).
                    # input  (local_bs * seq_len, 1)  — one row per token position
                    # output (local_bs * seq_len, emb_dim) — per-token embedding
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
                f"SparseCore table '{tc.name}': "
                f"({tc.num_embeddings}, {tc.embedding_dim}), "
                f"features={tc.feature_names}"
            )

        # Place each table in its own singleton stack instead of using
        # auto_stack_tables(). auto_stack_tables groups all tables into a single
        # StackedTableSpec, which at default hash_size=10M produces a 30 GB
        # tensor (3 tables × 10M × 256 × f32) that exceeds SC's 16 GiB
        # per-tensor HBM limit and fails tpu_sparse_dense_matmul with
        # INVALID_ARGUMENT. Per-table singleton stacks keep each at ~10 GB.
        for tc in table_configs:
            jte_stacking.stack_tables(
                feature_specs,
                table_names=[tc.name],
                global_device_count=global_dev,
                num_sc_per_device=num_sc,
                stack_name=tc.name,
            )

        # Mesh over all 256 JAX devices; each process holds its local 8 shards.
        devices = np.array(jax.devices())
        mesh = jax.sharding.Mesh(devices, ("x",))
        global_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x", None)
        )

        # Init EmbeddingVariables sharded on SparseCore.
        # Workaround: init_embedding_variables compares PartitionSpec with plain
        # tuples via !=, which always returns True in newer JAX where PartitionSpec
        # is no longer a tuple subclass. Call _init_stacked_embedding_table directly
        # to skip the broken validation, replicating init_embedding_variables logic.
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

        self.lookup = SparseCoreLookup(
            emb_vars=emb_vars,
            feature_specs=feature_specs,
            seq_lens=self.seq_lens,
            global_device_count=global_dev,
            local_device_count=local_dev,
            num_sc_per_device=num_sc,
        )
        self._jax_ec = SparsecoreBackedEC(self.lookup)

        # Dense params only — embedding state lives in emb_vars on SparseCore
        dense_params = [
            p
            for name, p in wrapper.named_parameters()
            if "_embedding_collection" not in name
        ]
        opt: torch.optim.Optimizer = torch.optim.SGD(
            dense_params, lr=dense_lr, foreach=False
        )
        logger.info("JAXTPUBackend: SparsecoreBackedEC initialized")
        return wrapper, opt

    def get_override(
        self, dummy: DummyPreprocess, num_ranks: int
    ) -> JAXEmbeddingOverride:
        if self._jax_ec is None:
            raise RuntimeError("call setup() before get_override()")
        return JAXEmbeddingOverride(dummy, self._jax_ec, num_ranks)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_embedding_backend(embedding_backend: str) -> EmbeddingBackend:
    if embedding_backend == "cpu":
        return CPUTorchRecBackend()
    if embedding_backend == "tpu":
        return JAXTPUBackend()
    raise ValueError(
        f"Unknown embedding_backend: {embedding_backend!r}. Use 'cpu' or 'tpu'."
    )
