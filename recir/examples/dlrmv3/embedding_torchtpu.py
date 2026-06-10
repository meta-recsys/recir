# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""torch_tpu SparseCore embedding backend.

Hosts the SparseCore fwd/bwd kernels on torch_tpu's PJRT via ``pallas.jax_op``
(JAX is trace/export-only), instead of the live ``jax.jit`` + GSPMD path in
``embedding.py``'s ``SparseCoreLookup`` / ``JAXTPUBackend``.

The fwd/bwd compute does NOT use jax_tpu_embedding's
``tpu_sparse_dense_matmul`` / ``_grad`` primitives. Instead ``sparsecore_ops``
emits the underlying ``stablehlo.custom_call`` targets directly
(``@SparseDenseMatmulOp`` / ``@SparseDenseMatmulGradOpWithOptimizerUpdate`` +
the fused-Adagrad subfunction; the kernels themselves live in libtpu), and the
per-feature unstack / per-feature-grad stack are reproduced as plain array ops
(jte's default, non-primitive path). jax_tpu_embedding is still used for the
host-side CSR preprocessing and the feature/table specs + SC-laid-out init.

Single-host / single-device first (global=local=1). The graph-break dispatch
(``JAXEmbeddingDispatcher`` -> ``JAXPreprocess`` -> ``SparsecoreBackedEC``) and
the ``SparsecoreBackedEC`` module itself are reused unchanged from
``embedding.py``: this module only provides a drop-in ``TorchTPUSparseCoreLookup``
whose ``compute()`` returns host arrays and whose ``update_embeddings()`` accepts
the captured grads, plus a backend that wires it up.

jax_op rejects ``*args`` / list / pytree args (only fixed-arity ``jax.Array`` /
``None``). The lookup wraps the FULL fused fwd/bwd kernels with a generated
flat tensor signature (``_make_flat_jax_op``); the fused call (not per-feature)
is required so the on-chip Adagrad updates each shared table exactly once.
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from recir.examples.dlrmv3.embedding import (
    _kjt_to_numpy,
    _lazy_jte,
    _num_sc_per_device,
    build_jte_sparsecore_feature_specs,
    EmbeddingBackend,
    init_jte_sparsecore_embedding_variables,
    SparsecoreBackedEC,
)
from recir.examples.dlrmv3.sparsecore_ops import (
    sc_matmul_fwd,
    sc_matmul_grad,
    stack_gradients,
    unstack_activation,
)
from recir.tests.xla.dlrmv3_test_utils import DummyPreprocess

# pyre-ignore[21]: torch_tpu is provided by the TPU runtime venv, not a buck
# dep, so it is unresolvable at build/type-check time.
from torch_tpu._internal.pallas import jax_op
from torchrec.modules.embedding_configs import EmbeddingConfig

logger: logging.Logger = logging.getLogger(__name__)

# jte sharding-strategy name -> the integer the SparseCore backend_config wants.
_SHARDING_STRATEGY_TO_INT = {"MOD": 1}


def _to_tpu_tensor(value: Any) -> torch.Tensor:
    """Create an owned contiguous torch TPU tensor from NumPy/JAX-like input."""
    # pyre-ignore[28]: `order` is a valid np.array kwarg; the bundled numpy
    # stub omits it from this overload.
    return torch.from_numpy(np.array(value, copy=True, order="C")).to("tpu")


def _feature_stack_layout(
    feature_specs: dict[str, Any],
) -> tuple[list[str], dict[str, str], list[str]]:
    """Extract fixed feature/stack ordering from the JTE layout metadata."""
    feature_names = list(feature_specs.keys())
    feature_to_stack = {
        f: feature_specs[f].table_spec.stacked_table_spec.stack_name
        for f in feature_names
    }
    seen: dict[str, None] = {}
    for f in feature_names:
        seen.setdefault(feature_to_stack[f], None)
    stack_names = list(seen.keys())
    return feature_names, feature_to_stack, stack_names


def _stack_kernel_params(
    jte: Any,
    feature_specs: dict[str, Any],
    stack_names: list[str],
    global_device_count: int,
) -> dict[str, tuple[int, int, int]]:
    """Read SparseCore custom-call static parameters from JTE metadata."""
    stacked_specs = jte.get_stacked_table_specs(feature_specs)
    return {
        s: (
            stacked_specs[s].total_sample_count // global_device_count,
            stacked_specs[s].max_ids_per_partition,
            stacked_specs[s].max_unique_ids_per_partition,
        )
        for s in stack_names
    }


def _copy_embedding_state_to_tpu(
    emb_vars_np: dict[str, tuple[np.ndarray, np.ndarray]],
    stack_names: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Move JTE-initialized table + Adagrad slot arrays into TorchTPU state."""
    tables = {s: _to_tpu_tensor(emb_vars_np[s][0]) for s in stack_names}
    slots = {s: _to_tpu_tensor(emb_vars_np[s][1]) for s in stack_names}
    return tables, slots


def _kjt_to_jte_feature_ids(
    merged_kjt: Any,
    feature_specs: dict[str, Any],
    seq_lens: dict[str, int],
) -> dict[str, np.ndarray]:
    """Convert TorchRec KJT input into the dense ID arrays JTE preprocesses."""
    ids_np = {
        k: v
        for k, v in _kjt_to_numpy(merged_kjt, seq_lens).items()
        if k in feature_specs
    }
    for k in list(ids_np.keys()):
        if seq_lens.get(k, 1) > 1:
            ids_np[k] = ids_np[k].reshape(-1, 1)
    return ids_np


def _csr_buffers_to_tpu_tensors(
    preprocessed: Any,
    stack_names: list[str],
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Move JTE host CSR buffers into TorchTPU tensors for SparseCore kernels."""
    return {
        s: (
            _to_tpu_tensor(preprocessed.lhs_row_pointers[s]),
            _to_tpu_tensor(preprocessed.lhs_embedding_ids[s]),
            _to_tpu_tensor(preprocessed.lhs_sample_ids[s]),
            _to_tpu_tensor(preprocessed.lhs_gains[s]),
        )
        for s in stack_names
    }


def _make_flat_jax_op(name: str, in_names: list[str], n_out: int, impl: Any) -> Any:
    """Wrap ``impl(list_of_arrays) -> array | list`` as a jax_op.

    jax_op infers its schema from the function signature and rejects varargs /
    non-Array args, so we codegen a fixed-arity, ``jax.Array``-annotated wrapper
    that forwards its positional args (by name) to ``impl``.
    """
    # Use positional a0..a{n-1} identifiers so feature/stack names (which
    # could contain non-identifier chars) can never break or inject into the
    # generated source.
    n = len(in_names)
    params = ", ".join(f"a{i}: jax.Array" for i in range(n))
    ret = (
        "jax.Array" if n_out == 1 else "tuple[" + ", ".join(["jax.Array"] * n_out) + "]"
    )
    body = ", ".join(f"a{i}" for i in range(n))
    src = f"def _g({params}) -> {ret}:\n    return _impl([{body}])\n"
    ns: dict[str, Any] = {"jax": jax, "_impl": impl}
    exec(src, ns)  # noqa: S102 - controlled fixed-arity codegen required by jax_op
    # pyre-ignore[16]: jax_op import is pyre-ignored above (runtime-only dep).
    return jax_op(name, ns["_g"])


class TorchTPUSparseCoreLookup:
    """SparseCore lookup hosting the fused kernels on torch_tpu via jax_op.

    Drop-in for ``embedding.SparseCoreLookup`` w.r.t. the interface
    ``SparsecoreBackedEC`` relies on: ``_seq_lens``, ``compute(kjt)`` (returns
    ``{feature: np.ndarray}``), and ``update_embeddings({feature: grad})``.
    Embedding state (table + Adagrad slot per stack) lives as torch tpu tensors.

    The phase split is:
      1. Reuse JTE processing for SparseCore layout and CSR buffers.
      2. Conversion boundary: copy JTE arrays into torch TPU tensors.
      3. TorchTPU execution: run SparseCore custom calls through jax_op.
    """

    def __init__(
        self,
        emb_vars_np: dict[str, tuple[np.ndarray, np.ndarray]],
        feature_specs: Any,
        seq_lens: dict[str, int],
        num_sc_per_device: int,
        embedding_lr: float,
        global_device_count: int = 1,
        local_device_count: int = 1,
        sharding_strategy: str = "MOD",
    ) -> None:
        jte, _, _, _ = _lazy_jte()
        self._jte: Any = jte
        self._feature_specs: Any = feature_specs
        self._seq_lens: dict[str, int] = seq_lens
        self._num_sc: int = num_sc_per_device
        self._global_dev: int = global_device_count
        self._local_dev: int = local_device_count
        self._strat: str = sharding_strategy
        self._strat_int: int = _SHARDING_STRATEGY_TO_INT[sharding_strategy]
        self._emb_lr: float = embedding_lr

        self._feats, self._feat_stack, self._stacks = _feature_stack_layout(
            feature_specs
        )
        self._stack_params = _stack_kernel_params(
            jte, feature_specs, self._stacks, global_device_count
        )
        self._tbl, self._slot = _copy_embedding_state_to_tpu(emb_vars_np, self._stacks)
        self._last_csr_t: dict[str, tuple[torch.Tensor, ...]] | None = None

        self._build_ops()

    # -- jax_op builders ----------------------------------------------------
    def _build_ops(self) -> None:
        stacks, feats, fs = self._stacks, self._feats, self._feature_specs
        gdev, nsc, strat_int = self._global_dev, self._num_sc, self._strat_int
        stack_params, emb_lr = self._stack_params, self._emb_lr
        feat_stack = self._feat_stack

        fwd_in = [f"{s}_{p}" for s in stacks for p in ("rp", "ei", "si", "gn")] + [
            f"{s}_tbl" for s in stacks
        ]

        def fwd_impl(xs: list[Any]) -> list[Any]:
            arrs = {s: tuple(xs[4 * i : 4 * i + 4]) for i, s in enumerate(stacks)}
            k = 4 * len(stacks)
            tables = {s: xs[k + i] for i, s in enumerate(stacks)}
            # Per-stack matmul -> one stacked activation per table.
            stacked_act: dict[str, Any] = {}
            for s in stacks:
                rp, ei, si, gn = arrs[s]
                dbs, mi, mui = stack_params[s]
                stacked_act[s] = sc_matmul_fwd(
                    rp,
                    ei,
                    si,
                    gn,
                    tables[s],
                    device_batch_size=dbs,
                    max_ids=mi,
                    max_unique_ids=mui,
                    sharding_strategy=strat_int,
                )
            # Split each stacked activation back into per-feature activations.
            return [
                unstack_activation(stacked_act[feat_stack[f]], fs[f], gdev, nsc)
                for f in feats
            ]

        self._fwd_op = _make_flat_jax_op("ttsc::fwd", fwd_in, len(feats), fwd_impl)

        bwd_in = (
            [f"{f}_g" for f in feats]
            + [f"{s}_{p}" for s in stacks for p in ("rp", "ei", "si", "gn")]
            + [f"{s}_{p}" for s in stacks for p in ("tbl", "slot")]
        )

        def bwd_impl(xs: list[Any]) -> list[Any]:
            nf = len(feats)
            grads = {f: xs[i] for i, f in enumerate(feats)}
            base = nf
            arrs = {
                s: tuple(xs[base + 4 * i : base + 4 * i + 4])
                for i, s in enumerate(stacks)
            }
            base2 = nf + 4 * len(stacks)
            tables = {s: xs[base2 + 2 * i] for i, s in enumerate(stacks)}
            slots = {s: xs[base2 + 2 * i + 1] for i, s in enumerate(stacks)}
            # Concatenate per-feature grads into one stacked grad per table.
            stacked_g = stack_gradients(grads, fs, nsc)
            out: list[Any] = []
            for i, s in enumerate(stacks):
                rp, ei, si, gn = arrs[s]
                dbs, mi, mui = stack_params[s]
                new_tbl, new_acc = sc_matmul_grad(
                    rp,
                    ei,
                    si,
                    gn,
                    tables[s],
                    slots[s],
                    stacked_g[s],
                    jnp.float32(emb_lr),
                    max_ids=mi,
                    max_unique_ids=mui,
                    sharding_strategy=strat_int,
                    # Unique per stack: each grad op gets its own optimizer FuncOp.
                    computation_name=f"adagrad_optimizer_update_{i}",
                )
                out += [new_tbl, new_acc]
            return out

        self._bwd_op = _make_flat_jax_op("ttsc::bwd", bwd_in, 2 * len(stacks), bwd_impl)

    # -- lookup interface (used by SparsecoreBackedEC) ----------------------
    def compute(self, merged_kjt: Any) -> dict[str, np.ndarray]:
        # 1. TorchRec KJT -> NumPy feature IDs in the format JTE expects.
        ids_np = _kjt_to_jte_feature_ids(
            merged_kjt, self._feature_specs, self._seq_lens
        )

        # 2. JTE host preprocessing -> SparseCore CSR buffers.
        pre, _ = self._jte.preprocess_sparse_dense_matmul_input(
            features=ids_np,
            features_weights=None,
            feature_specs=self._feature_specs,
            local_device_count=self._local_dev,
            global_device_count=self._global_dev,
            num_sc_per_device=self._num_sc,
            sharding_strategy=self._strat,
        )

        # 3. Conversion boundary: CSR buffers become torch TPU tensors.
        self._last_csr_t = _csr_buffers_to_tpu_tensors(pre, self._stacks)

        # 4. TorchTPU execution: SparseDenseMatmulOp custom call.
        return self._run_sparsecore_forward()

    def _run_sparsecore_forward(self) -> dict[str, np.ndarray]:
        csr_t = self._last_csr_t
        if csr_t is None:
            raise RuntimeError("forward called before CSR preprocessing")
        fwd_args = [csr_t[s][p] for s in self._stacks for p in range(4)]
        fwd_args += [self._tbl[s] for s in self._stacks]
        acts = self._fwd_op(*fwd_args)
        acts = list(acts) if isinstance(acts, (list, tuple)) else [acts]
        return {f: acts[i].detach().cpu().numpy() for i, f in enumerate(self._feats)}

    def update_embeddings(self, grads: dict[str, Any]) -> None:
        csr_t = self._last_csr_t
        if csr_t is None:
            raise RuntimeError("update_embeddings() called before compute()")

        # Captured activation grads follow the same conversion boundary: host
        # grad arrays -> torch TPU tensors -> SparseCore update custom call.
        grad_args = [_to_tpu_tensor(grads[f]) for f in self._feats]
        csr_args = [csr_t[s][p] for s in self._stacks for p in range(4)]
        state_args = [x for s in self._stacks for x in (self._tbl[s], self._slot[s])]

        # TorchTPU execution: SparseDenseMatmulGradOpWithOptimizerUpdate
        # returns the updated table and Adagrad slot tensors.
        outs = self._bwd_op(*(grad_args + csr_args + state_args))
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        for i, s in enumerate(self._stacks):
            self._tbl[s], self._slot[s] = outs[2 * i], outs[2 * i + 1]


def _init_jte_embedding_state_np(
    feature_specs: dict[str, Any], num_sc: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reuse JTE processing to initialize table + slot arrays in SC layout."""
    devices = np.array(jax.devices()[:1])
    mesh = jax.sharding.Mesh(devices, ("x",))
    global_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", None)
    )
    emb_vars = init_jte_sparsecore_embedding_variables(
        feature_specs, num_sc, global_sharding
    )
    return {
        stack_name: (np.asarray(ev.table), np.asarray(ev.slot[0]))
        for stack_name, ev in emb_vars.items()
    }


class TorchTPUEmbBackend(EmbeddingBackend):
    """SparseCore embedding backend on torch_tpu (single-host/device, jax_op).

    Set ``seq_lens`` and ``global_batch_size`` before ``setup()``.
    """

    def __init__(self) -> None:
        self._ec: SparsecoreBackedEC | None = None
        self.lookup: TorchTPUSparseCoreLookup | None = None
        self.seq_lens: dict[str, int] = {}
        self.global_batch_size: int = 0

    def init_pg(self) -> tuple[int, int]:
        if not dist.is_initialized():
            store = dist.TCPStore("127.0.0.1", 0, 1, is_master=True)
            dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
        return dist.get_rank(), dist.get_world_size()

    def setup(
        self,
        wrapper: nn.Module,
        table_configs: list[EmbeddingConfig],
        embedding_lr: float,
        dense_lr: float,
        world_size: int,
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        if self.global_batch_size <= 0:
            raise RuntimeError("set backend.global_batch_size before setup()")
        if world_size != 1:
            raise ValueError(
                "TorchTPUEmbBackend currently supports only the single-rank "
                f"train.py path; got world_size={world_size}"
            )
        global_dev = 1
        local_dev = 1
        num_sc = _num_sc_per_device()
        local_bs = self.global_batch_size // world_size
        if local_bs % (local_dev * num_sc) != 0:
            raise ValueError(
                f"local_batch_size ({local_bs}) must be a multiple of "
                f"local_device_count ({local_dev}) x num_sc_per_device ({num_sc})"
            )
        logger.info(
            "TorchTPUEmbBackend.setup: local_bs=%d num_sc=%d tables=%d",
            local_bs,
            num_sc,
            len(table_configs),
        )
        # Reuse JTE processing to define SparseCore table/feature layout.
        feature_specs = build_jte_sparsecore_feature_specs(
            table_configs, self.seq_lens, local_bs, num_sc, global_dev, embedding_lr
        )

        # Reuse JTE processing to initialize table + Adagrad slot arrays
        # in the physical SparseCore layout.
        emb_vars_np = _init_jte_embedding_state_np(feature_specs, num_sc)

        # TorchTPU phase: own TPU copies of state and run SparseCore kernels.
        self.lookup = TorchTPUSparseCoreLookup(
            emb_vars_np=emb_vars_np,
            feature_specs=feature_specs,
            seq_lens=self.seq_lens,
            num_sc_per_device=num_sc,
            embedding_lr=embedding_lr,
            global_device_count=global_dev,
            local_device_count=local_dev,
        )
        # TorchTPUSparseCoreLookup is a structural drop-in for SparseCoreLookup
        # (same compute()/update_embeddings()/_seq_lens surface) but not a
        # subclass, so SparsecoreBackedEC can be reused unchanged.
        # pyre-ignore[6]: intentional duck-typed lookup; see comment above.
        self._ec = SparsecoreBackedEC(self.lookup)
        dense_params = [
            p
            for name, p in wrapper.named_parameters()
            if "_embedding_collection" not in name
        ]
        opt: torch.optim.Optimizer = torch.optim.SGD(
            dense_params, lr=dense_lr, foreach=False
        )
        return wrapper, opt

    def get_dispatcher(self, dummy: DummyPreprocess, num_ranks: int) -> Any:
        if self._ec is None:
            raise RuntimeError("call setup() before get_dispatcher()")

        from recir.examples.dlrmv3.jax_embedding import JAXEmbeddingDispatcher

        return JAXEmbeddingDispatcher(dummy, self._ec, num_ranks)
