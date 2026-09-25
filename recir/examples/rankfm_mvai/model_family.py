#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""RankFM model family driven by the mvai_cloud RecTrainer, running on TPU.

``RankFMMvaiModelFamily`` subclasses the hammer udd_fm
``BlueReelsVDDModelFamily`` (itself a RecModelFamily). It inherits
``create_vanilla_train_module`` -- which builds the byte-identical
``BlueReelsVDDTrainV3`` -- and overrides:

* ``create_train_module_and_optimizer``: instead of the GPU TorchRec DMP +
  FBGEMM-fused-optimizer path in RecModelFamily, it wires the recir embedding
  backend (cpu | tpu | in_graph) before the recir compile: build model on CPU,
  mark params trainable, compute the KJT key->num_embeddings map, pull the
  trace/override batch, build overrides via ``backend.get_override``, run
  ``backend.setup`` (DMP for cpu / SparseCore for tpu / in-graph tables for
  in_graph), and ``model.set_recir_kjt_keys``. All the artifacts the
  RecirXLATrainPipeline needs to run the recir compile are stashed on the
  returned model wrapper.

* ``create_train_datapipe``: reuses the synthetic uniform / random batch and
  barnfs/GCS replay data path (instead of the base RandomDataPipe).

``model_type()`` is inherited from BlueReelsVDDModelFamily ('udd_fm').
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast, Dict, List, Optional, Tuple, TYPE_CHECKING

import jax

# Register the recir Helion HSTU attention op lowerings (torch.ops.recir.
# helion_ragged_attention_{fwd,bwd}) at load, before any compile builds a torchax
# Environment (all_aten_ops is snapshotted once). recir_core cannot import this
# (visibility), so the rankfm family owns the registration for the Helion path.
import recir.torchax.ops.jhstu_attention  # noqa: F401
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from recir.embedding.backend import SparseOptimizerRuntime

from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.parity.config_dump import config_dump
from hammer.v3.models.udd_fm_v3.parity.step_digest import (
    arm,
    compute_padding_spec,
    watch_params,
)
from hammer.v3.models.udd_fm_v3.recir_core import (
    compile_gspmd_step,
    LR_ITER_BUFFER,
    recir_train_step,
    RecirCompiledModuleBase,
    RecirCompileSpec,
    RecirMode,
    setup_dense_optimizer_and_warmup,
    WarmupController,
)
from hammer.v3.models.udd_fm_v3.recir_core.compile import (
    compile_step,
    IN_GRAPH_METRICS_ATTR,
    in_graph_metrics_enabled,
)
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.models.udd_fm_v3.udd_fm_family import BlueReelsVDDModelFamily
from minimal_viable_ai.core.dataloader.dataloader_configs import DataSourceConfig
from minimal_viable_ai.core.model_family_api.optimizer_configs import RecOptimizerConfig
from recir.api import (
    COMPILED_OBJECT_KEY,
    sync_device_state_to_torch,
    sync_named_state_to_torch,
)
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.examples.rankfm_mvai.budgeted_sampling import install_budgeted_sampling
from recir.examples.rankfm_mvai.checkpointable import HostStateBuffers
from recir.examples.rankfm_mvai.config import RankFMModelArgs, RankFMTrainerConfig
from recir.examples.rankfm_mvai.data_pipe import (
    BatchReplayPipe,
    compute_kjt_key_to_size,
    replay_batch_iter,
)
from recir.examples.rankfm_mvai.prefetch import PrefetchingIterator
from recir.examples.rankfm_mvai.rankfm_batch_utils import (
    create_random_train_batch,
    create_uniform_train_batch,
)
from recir.examples.rankfm_mvai.rankfm_embedding_backends import build_embedding_backend
from recir.examples.rankfm_mvai.rankfm_model_utils import (
    AttentionBackend,
    build_model_from_config,
    resolve_attention_backend,
)
from recir.examples.rankfm_mvai.xsurface_helpers import (
    _resolve_nro_per_sample,
    make_padding_region_transforms,
    prepare_override_build_sample,
)
from recir.region_transforms.dropout_prng import DropoutPRNGTransform
from recir.region_transforms.padding import compute_padded_shape
from torch.utils.data import IterableDataset

logger: logging.Logger = logging.getLogger(__name__)

# Per-step device->host state sync (sync_device_state_to_torch) skips state
# entries whose FQN contains any of these substrings. The on-device embedding
# tables (InGraphSparseArch's "_sharded_tables" / "_dp_tables" module dicts) are
# GBs each and are kept resident in the cached JAX device_state across steps, so
# mirroring them to host every step is O(table-size) waste (D2H copy + layout
# transpose) that dominates the step at large hash sizes. Dense params / LR /
# metric buffers don't match and still sync. Full sync (skip_fqns=None) is used
# before checkpointing/inspecting the tables (flush_host_state(full=True)).
_EMB_SYNC_SKIP_FQNS: Tuple[str, ...] = ("_sharded_tables", "_dp_tables")

# The only state the training loop needs mirrored host-ward between
# checkpoints: the warmup controller's LR buffers, which
# ``snapshot_metrics._lr_tensor`` reads for the learning_rate/* curves. Named
# after the module buffers (hammer recir_core/compile.py LR_*_BUFFER); the
# WarmupController holds the same tensors under its own attribute names.
_LR_SYNC_ONLY_FQNS: Tuple[str, ...] = (
    "_lr_dense_buffer",
    "_lr_sparse_buffer",
    "_lr_iter_buffer",
)


def _udd_fm_v3_train_step(
    mod: torch.nn.Module,
    opt: torch.optim.Optimizer,
    ro_float_features: torch.Tensor,
    nro_float_features: torch.Tensor,
    ro_id_list_values: torch.Tensor,
    ro_id_list_lengths: torch.Tensor,
    nro_id_list_values: torch.Tensor,
    nro_id_list_lengths: torch.Tensor,
    unary_values: torch.Tensor,
    unary_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    supervision_label_t: torch.Tensor,
    supervision_weight_t: torch.Tensor,
    all_labels: Dict[str, torch.Tensor],
    loss_denom_source: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Named-signature single-optimizer train step for the GSPMD/TPU compile.

    PaddingRegionTransform.bind_signature resolves batch_size_arg / jagged-pair
    arg names by inspecting the step function's parameter names. The shared
    recir_train_step is variadic (mod, *opts_and_batch), so those names are not
    visible and bind_signature raises e.g. "'ro_float_features' is not in list".
    This thin wrapper exposes the udd_fm_v3 forward arg names and delegates to
    recir_train_step so the dense-opt step / LR-rebind / detached named-tensors
    behavior is identical. Used by the cpu / tpu backends (single dense opt).
    """
    return recir_train_step(
        mod,
        opt,
        ro_float_features,
        nro_float_features,
        ro_id_list_values,
        ro_id_list_lengths,
        nro_id_list_values,
        nro_id_list_lengths,
        unary_values,
        unary_lengths,
        num_candidates,
        supervision_label_t,
        supervision_weight_t,
        all_labels,
        loss_denom_source,
    )


def _udd_fm_v3_train_step_two_opt(
    mod: torch.nn.Module,
    opt: torch.optim.Optimizer,
    emb_opt: torch.optim.Optimizer,
    ro_float_features: torch.Tensor,
    nro_float_features: torch.Tensor,
    ro_id_list_values: torch.Tensor,
    ro_id_list_lengths: torch.Tensor,
    nro_id_list_values: torch.Tensor,
    nro_id_list_lengths: torch.Tensor,
    unary_values: torch.Tensor,
    unary_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    supervision_label_t: torch.Tensor,
    supervision_weight_t: torch.Tensor,
    all_labels: Dict[str, torch.Tensor],
    loss_denom_source: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Named-signature two-optimizer train step for the in_graph backend.

    The in_graph embedding tables are plain FSDP nn.Parameters looked up + grad'd
    inside the compiled graph, so they get their OWN optimizer (``emb_opt``,
    SGD @ embedding_lr) alongside the dense optimizer. recir_train_step steps
    EVERY optimizer it is handed, so both ``opt.step()`` and ``emb_opt.step()``
    are traced into the single GSPMD program. emb_opt carries no LR buffer, so
    recir_train_step's LR rebind only touches the dense opt.
    """
    return recir_train_step(
        mod,
        opt,
        emb_opt,
        ro_float_features,
        nro_float_features,
        ro_id_list_values,
        ro_id_list_lengths,
        nro_id_list_values,
        nro_id_list_lengths,
        unary_values,
        unary_lengths,
        num_candidates,
        supervision_label_t,
        supervision_weight_t,
        all_labels,
        loss_denom_source,
    )


def _load_parity_checkpoint(target: nn.Module, ckpt_path: str) -> None:
    from hammer.v3.models.udd_fm_v3.parity.checkpoint import (
        content_hash,
        load_init_checkpoint,
    )

    logger.info(f"[PARITY] loading init weights from {ckpt_path}")
    logger.info(f"[PARITY] init ckpt sha256={content_hash(ckpt_path)}")
    load_init_checkpoint(target, ckpt_path)


def _in_graph_metric_states(
    prediction: torch.Tensor, label: torch.Tensor, weight: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """The in-graph ranking-metric reduction, as recir_train_step's hook."""
    from recir.examples.rankfm_mvai.metric_states import (
        compute_metric_states,
        emit_keys,
    )

    return emit_keys(compute_metric_states(prediction, label, weight))


class RecirXLACompiledModule(RecirCompiledModuleBase):
    """Self-compiling GSPMD/TPU recir train module (built by RankFMMvaiModelFamily).

    Owns the recir construction for the TPU cell, so the family stays focused on
    model + data wiring. In ``__init__`` it: expands the override sample from the
    trace batch, builds the preproc / sparse-arch dispatchers
    (``backend.get_override``), sets up embedding storage (``backend.setup``),
    builds the UNIFIED dense optimizer (``setup_dense_optimizer_and_warmup`` --
    the same ``TensorLRAdamW`` + 0-d LR buffer + ``WarmupController`` mechanism as
    the GPU cell), and -- for the in_graph backend -- a dedicated embedding
    optimizer (row-wise Adagrad) over the in-graph table params, wires
    ``set_recir_kjt_keys``, and builds the ``PaddingRegionTransform``. On the
    first forward it lazily compiles (``compile_gspmd_step`` for the single dense
    opt, or ``compile_step`` with (dense_opt, emb_opt) for the in_graph two-opt
    program) and runs each step through the runner.

    Exposes ``dense_opt`` so the family can return it to the trainer (recir owns
    the step; the trainer never calls opt.step()), avoiding a NoOpOptimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: Any,
        trace_batch: Any,
        module_config: BlueReelsVDDModuleConfig,
        optimizer_config: RecOptimizerConfig,
        trainer_cfg: RankFMTrainerConfig,
        num_ranks: int,
        num_processes: int,
        device: torch.device,
        attention_backend: AttentionBackend,
    ) -> None:
        super().__init__()
        num_candidates_per_sample: int = _resolve_nro_per_sample(
            module_config, trainer_cfg.enable_padding
        )
        global_bs = trainer_cfg.local_batch_size * num_processes
        padded_num_devices: int = math.lcm(
            num_ranks, backend.sc_modulus() * num_processes
        )
        # Padded RO sample count the PRT will produce -- the SAME formula
        # _discover uses (padding.py), which compute_padded_shape is documented to
        # centralize for compile-time consumers. Feeds the jagged HSTU pad-row
        # count; note padded_num_devices is the LCM modulus, NOT world_size: the
        # SparseCore backend makes it world_size * num_sc, so the pad is num_sc
        # rows per device rather than one.
        padded_ro_bs: int = (
            compute_padded_shape(global_bs, padded_num_devices)
            if trainer_cfg.enable_padding
            else global_bs
        )

        if trainer_cfg.budgeted_seq_len_sampling:
            # Before the override build, which runs the preprocessor with
            # sampling disabled: the swap has to be on the live modules by the
            # time fwd_fn calls them.
            install_budgeted_sampling(model, module_config)

        # Expanded (global / padded) override-build sample from the trace batch.
        # get_override only inspects structure (keys + budget-set lengths), so
        # the sample's values don't matter -- the trace batch is a valid source.
        batch_sample, override_bs = prepare_override_build_sample(
            trace_batch,
            module_config,
            global_bs,
            num_candidates_per_sample,
            padded_num_devices,
            trainer_cfg.enable_padding,
        )

        ro_kjt_keys: List[str] = list(trace_batch.ro_id_list_features.keys())
        nro_kjt_keys: List[str] = list(trace_batch.nro_id_list_features.keys())
        _unary = trace_batch.unary_features
        assert _unary is not None
        unary_kjt_keys: List[str] = list(_unary.keys())

        # Table configs from the EC BEFORE backend.setup mutates the model.
        # pyre-ignore[16]: dynamic submodule traversal through hammer v3 union
        ec = model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection
        table_configs = ec.embedding_configs()

        # Dispatchers BEFORE backend.setup (override builder runs the sparse arch
        # eagerly against the unmodified model).
        dispatchers = backend.get_override(
            model.recblock,
            batch_sample,
            device,
            override_bs,
            num_ranks,
            num_processes,
        )

        embedding_setup = backend.setup(
            model,
            table_configs,
            optimizer_config.sparse.default,
            device,
            num_ranks,
        )
        model = embedding_setup.model
        model.set_recir_kjt_keys(ro_kjt_keys, nro_kjt_keys, unary_kjt_keys)

        sparse_runtime = embedding_setup.sparse_runtime
        dense_params = list(embedding_setup.dense_named_parameters.values())
        # ONE warmup controller for both sides. The compiled path hands it the
        # buffer its in-graph sparse optimizer already reads; the backend-fused
        # path has no such buffer, so make a plain one here and let whoever runs
        # the step copy it into the fused optimizer's param_group -- the same
        # shape as what this leg already does for the dense LR.
        sparse_lr_buffer = sparse_runtime.lr_buffer
        if sparse_lr_buffer is None and sparse_runtime.fused_optimizer is not None:
            sparse_lr_buffer = torch.zeros((), dtype=torch.float32, device=device)
        dense_opt, warmup = setup_dense_optimizer_and_warmup(
            model,
            dense_params,
            optimizer_config.dense.default,
            device,
            sparse_cfg=(
                sparse_runtime.config if sparse_lr_buffer is not None else None
            ),
            sparse_lr_buffer=sparse_lr_buffer,
        )
        emb_opt = sparse_runtime.optimizer
        # Hand recir_train_step the reduction itself rather than letting it
        # import this package: recir_core is linked into every hammer
        # udd_fm_v3 binary, and the torchrec metrics chain behind
        # metric_states does not fit in those link trees.
        setattr(
            model,
            IN_GRAPH_METRICS_ATTR,
            _in_graph_metric_states if trainer_cfg.in_graph_metrics else None,
        )

        region_transforms: List[Any] = []
        if trainer_cfg.enable_padding:
            region_transforms = make_padding_region_transforms(
                num_candidates_per_sample, padded_num_devices, num_processes
            )
        # Dropout's Bernoulli draw keys off the LR warmup counter, which is a
        # registered buffer and therefore real graph state. Unconditional: it is
        # a no-op when the traced graph has no dropout
        region_transforms.append(DropoutPRNGTransform(counter_fqn=LR_ITER_BUFFER))

        self._model = model
        # Padded sizes for the parity digest's activation trim, and the hooks
        # that read the parameters either side of the step. Both inert unless
        # PARITY_DIGEST_KINDS is set.
        self._padding_spec: Tuple[int, int, int, int] = compute_padding_spec(
            model,
            global_bs,
            padded_num_devices,
            num_processes,
            num_candidates_per_sample,
            int(module_config.hstu_max_seq_len),
        )
        watch_params(self)
        self._dense_opt: torch.optim.Optimizer = dense_opt
        self._emb_opt: Optional[torch.optim.Optimizer] = emb_opt
        self._host_state_buffers: Any = None
        self._warmup: WarmupController = warmup
        self._sparse_runtime: SparseOptimizerRuntime = sparse_runtime
        self._sparse_lr_buffer: Optional[torch.Tensor] = sparse_lr_buffer
        self._dispatchers: List[Any] = dispatchers
        self._trace_batch = trace_batch
        self._num_ranks = num_ranks
        self._num_processes = num_processes
        self._global_bs: int = global_bs
        self._padded_ro_bs: int = padded_ro_bs
        self._attention_backend = attention_backend
        self._spec = RecirCompileSpec(
            mode=RecirMode.GSPMD_TPU,
            device=device,
            num_ranks=num_ranks,
            num_processes=num_processes,
            # RankFM losses use global candidate-count normalization, so GSPMD must sum
            # per-rank gradient contributions without another world-size division.
            loss_reduction="mean",
            region_transforms=region_transforms,
        )
        self._model.train(True)

    @property
    def dense_opt(self) -> torch.optim.Optimizer:
        return self._dense_opt

    def _build_compiled_step(self, first_batch: Any) -> Any:
        # Trace with the family-provided synthetic/replay batch, not the incoming
        # runtime batch. PaddingRegionTransform applies the configured static
        # shapes at trace entry when padding is enabled.
        #
        # The compile runs recir's eager make_fx trace synchronously (during which
        # the HSTU ops' register_fake kernels record their attention shapes) but the
        # first XLA lower is lazy, so stale sigs are cleared BEFORE the build and the
        # kernels warmed AFTER it -- Helion compile cannot nest inside the lower's
        # tracer.
        if self._attention_backend == "helion":
            from recir.ops.helion.jagged_hstu_attention import clear_traced_sigs

            clear_traced_sigs()

        if self._emb_opt is None:
            # Single dense optimizer (cpu / tpu backends).
            compiled_step = compile_gspmd_step(
                self._model,
                self._dense_opt,
                self._trace_batch,
                self._dispatchers,
                self._spec,
                train_step_fn=arm(_udd_fm_v3_train_step, self._padding_spec),
            )
        else:
            # in_graph backend: trace the dense optimizer AND the dedicated
            # embedding optimizer into the single GSPMD program. compile_gspmd_step
            # hard-wires a single optimizer, so use the shared multi-optimizer
            # driver compile_step directly, with a GSPMDParallel that knows the
            # in-graph FSDP table prefix.
            from recir.embedding.in_graph import SHARDED_TABLE_PREFIX

            emb_opt = self._emb_opt
            assert emb_opt is not None
            parallel_mode = GSPMDParallel(
                "replicate",
                loss_reduction=self._spec.loss_reduction,
                num_ranks=self._num_ranks,
                num_processes=self._num_processes,
                fsdp_table_substr=SHARDED_TABLE_PREFIX,
            )
            compiled_step = compile_step(
                arm(_udd_fm_v3_train_step_two_opt, self._padding_spec),
                self._model,
                (self._dense_opt, emb_opt),
                self._trace_batch,
                parallel_mode=parallel_mode,
                backend=XLAGSPMDBackend(),
                dispatchers=self._dispatchers,
                region_transforms=self._spec.region_transforms,
            )

        if self._attention_backend == "helion":
            # Warm the fwd + bwd kernels for the per-device shapes the sample-sharded
            # lowering will run, after the trace and before the first lower.
            # set_pad_seqs_per_device must run first: a wrong pad-row count faults the
            # TPU core or drops real sequences.
            from recir.ops.helion.jagged_hstu_attention import (
                set_pad_seqs_per_device,
                warm_traced_hstu_kernels,
            )

            world_size = len(jax.devices())
            set_pad_seqs_per_device(
                real_bs=self._global_bs,
                padded_bs=self._padded_ro_bs,
                world_size=world_size,
            )
            warm_traced_hstu_kernels(world_size=world_size)
        elif self._attention_backend == "pallas_jagged":
            from recir.ops.pallas.jagged_hstu_attention import set_pad_seqs_per_device

            set_pad_seqs_per_device(
                real_batch_size=self._global_bs,
                padded_batch_size=self._padded_ro_bs,
                world_size=len(jax.devices()),
            )

        return compiled_step

    def _step_args(self, batch: Any) -> Tuple[Any, ...]:
        if self._emb_opt is None:
            return (
                self._model,
                self._dense_opt,
                *flatten_batch(batch, global_loss_denom=True),
            )
        return (self._model, self._dense_opt, self._emb_opt, *flatten_batch(batch))

    def flat_step_args(self, batch: Any) -> List[Any]:
        return pytree.arg_tree_leaves(*self._step_args(batch))

    def _run_step(
        self, compiled_step: Any, batch: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        result: Tuple[torch.Tensor, Dict[str, torch.Tensor]] = compiled_step(
            *self._step_args(batch)
        )
        if self._emb_opt is None:
            sync_device_state_to_torch(compiled_step)
        return result

    @property
    def in_graph_metrics(self) -> bool:
        return in_graph_metrics_enabled(self._model)

    def flush_host_state(self, full: bool = False) -> None:
        """Mirror device-resident state back to torch for host consumers (e.g.
        before checkpointing).

        By default the GB-scale embedding tables are skipped (kept resident on
        device -- see _EMB_SYNC_SKIP_FQNS); pass full=True before checkpointing
        to also mirror them (skip_fqns=None). The table host params must already
        have real storage (see checkpointable.HostStateBuffers) or the
        copy_ is a silent no-op.
        """
        compiled_step = self._compiled_step
        if compiled_step is not None:
            skip_fqns = None if full else _EMB_SYNC_SKIP_FQNS
            sync_device_state_to_torch(compiled_step, skip_fqns=skip_fqns)

    def sync_lr_to_host(self) -> None:
        """Refresh only the LR buffers ``snapshot_metrics._lr_tensor`` reads."""
        compiled_step = self._compiled_step
        if compiled_step is not None:
            sync_named_state_to_torch(compiled_step, _LR_SYNC_ONLY_FQNS)

    def enable_output_pipelining(self) -> bool:
        manager = self._state_manager()
        if manager is None:
            return False
        manager.keep_output_jax = True
        return True

    def materialize_output(self, output: Any) -> Any:
        manager = self._state_manager()
        if manager is None:
            return output
        return manager.materialize_output(output)

    def _state_manager(self) -> Any:
        compiled_step = self._compiled_step
        if compiled_step is None:
            return None
        compiled_obj = cast(Any, compiled_step).__dict__.get(COMPILED_OBJECT_KEY)
        return None if compiled_obj is None else compiled_obj.state_manager

    def prepare_for_restore(self) -> None:
        """Compile, then build the run's host buffers so dcp.load can write into them."""
        if self._compiled_step is None:
            self._compiled_step = self._build_compiled_step(self._trace_batch)
        self._host_buffers()

    def _host_buffers(self) -> Any:
        buffers = self._host_state_buffers
        if buffers is None:
            compiled_obj = cast(Any, self._compiled_step).__dict__.get(
                COMPILED_OBJECT_KEY
            )
            if compiled_obj is None:
                raise RuntimeError(
                    "host state buffers need a compiled step; call "
                    "prepare_for_restore after the force-compile"
                )
            buffers = HostStateBuffers(
                compiled_obj.flat_state,
                compiled_obj.state_manager.state_fqns or [],
                self._model,
                [self._dense_opt, self._emb_opt],
                _EMB_SYNC_SKIP_FQNS,
            )
            buffers.build()
            self._host_state_buffers = buffers
        return buffers

    def finalize_restore(self) -> None:
        """Seed device state from the freshly loaded host buffers."""
        compiled_step = self._compiled_step
        if compiled_step is None:
            return
        compiled_obj = cast(Any, compiled_step).__dict__.get(COMPILED_OBJECT_KEY)
        if compiled_obj is None:
            return
        compiled_obj.state_manager.prefill_device_state_from_host(
            compiled_obj.gm, compiled_obj.flat_state
        )

    def make_checkpointable(self) -> Any:
        """Return a DCP Stateful view of this module (see checkpointable.py).

        Duck-typed hook: mvai_cloud's CheckpointManager uses this if present, so
        the recir/JAX-specific checkpoint logic stays out of the shared framework.
        """
        from recir.embedding.in_graph import DP_TABLE_PREFIX, SHARDED_TABLE_PREFIX

        from .checkpointable import gather_shard_order, RecirCheckpointable

        return RecirCheckpointable(
            self,
            num_processes=self._num_processes,
            sharded_prefixes=(SHARDED_TABLE_PREFIX,),
            table_prefixes=(SHARDED_TABLE_PREFIX, DP_TABLE_PREFIX),
            warmup=self._warmup,
            host_buffers=self._host_buffers,
            dense_optimizer=self._dense_opt,
            embedding_optimizer=self._emb_opt,
            shard_order=(
                gather_shard_order(self._num_processes)
                if self._num_processes > 1
                else None
            ),
        )


class RankFMMvaiModelFamily(BlueReelsVDDModelFamily):
    """RecModelFamily that swaps GPU DMP for the recir embedding backend.

    The trainer/model knobs that the base RecModelFamily does not carry
    (embedding backend choice, batching, replay paths, padding budgets) are
    injected via :meth:`configure` before RecTrainer constructs the family.
    They default to a synthetic single-host CPU-backend run.
    """

    # Class-level injection point. RecTrainer instantiates the family with the
    # fixed (config, pg, device) signature, so the rankfm-specific knobs travel
    # as class attributes set by the entrypoint via ``configure(...)``.
    _trainer_cfg: RankFMTrainerConfig = RankFMTrainerConfig()
    _model_args: RankFMModelArgs = RankFMModelArgs()

    @classmethod
    def configure(
        cls,
        trainer_cfg: RankFMTrainerConfig,
        model_args: RankFMModelArgs,
    ) -> None:
        """Inject the rankfm CLI knobs the base RecModelFamily ctor can't carry."""
        cls._trainer_cfg = trainer_cfg
        cls._model_args = model_args

    # -- helpers -----------------------------------------------------------

    def _module_config(self) -> BlueReelsVDDModuleConfig:
        return cast(BlueReelsVDDModuleConfig, self.config.module_config)

    def _feature_config(self) -> BlueReelsVDDFeatureConfig:
        return cast(BlueReelsVDDFeatureConfig, self.config.feature_config)

    def _build_data_pipe(self) -> Optional[BatchReplayPipe]:
        """Build a BatchReplayPipe when --gcs_data_path is set, else None.

        Rank/world are taken from the embedding backend's PG view (already
        initialized by the time the family is built).
        """
        trainer_cfg = self._trainer_cfg
        if not trainer_cfg.gcs_data_path:
            logger.info("Using synthetic data (no --gcs_data_path)")
            return None

        # gloo rank/world are resolved by the entrypoint and stored on the
        # cfg-adjacent attributes below; fall back to single-host (0, 1).
        gloo_rank = getattr(self, "_gloo_rank", 0)
        gloo_world = getattr(self, "_gloo_world", 1)

        explicit_files: Optional[List[str]] = None
        if trainer_cfg.data_file_template:
            explicit_files = [trainer_cfg.data_file_template.format(i=1)]
            logger.info(
                f"Direct-read mode: file_list={explicit_files} (template "
                f"'{trainer_cfg.data_file_template}')"
            )
        data_pipe = BatchReplayPipe(
            path=trainer_cfg.gcs_data_path,
            rank=gloo_rank,
            world_size=gloo_world,
            replicate_across_ranks=trainer_cfg.data_replicate_across_ranks,
            file_list=explicit_files,
        )
        # Fail here rather than on the first select_rows_slice, which happens
        # after the model is built and traced. Also reports how many files a
        # step will merge, which is the only place that number is visible.
        batches_per_step = data_pipe.batches_per_step(trainer_cfg.local_batch_size)
        logger.info(
            f"Replay pipe: {len(data_pipe)} batches for rank {gloo_rank} "
            f"(replicate={trainer_cfg.data_replicate_across_ranks}, "
            f"{batches_per_step} file(s)/step)"
        )
        return data_pipe

    # -- overrides ---------------------------------------------------------

    def create_vanilla_train_module(self, batch_size: Optional[int]) -> torch.nn.Module:
        """Build the byte-identical BlueReelsVDDTrainV3 via the rankfm builder.

        The base BlueReelsVDDModelFamily.create_vanilla_train_module builds the
        same BlueReelsVDDTrainV3 from module_config, but WITHOUT the TPU-required
        kernel/device settings that the recir lowering path depends on
        (use_cuda_attention=False, custom_kernel=False, _hammer_kernel=PYTORCH,
        model on the build device). build_model_from_config does
        exactly these mutations, so we delegate to it here to stay byte-identical
        on the model that actually traces+lowers on TPU. bf16_training is threaded
        through so the bench's --bf16_training recipe applies. The architecture
        (module hierarchy / state_dict keys) is identical to the base builder --
        only the kernel/device/precision knobs differ.
        """
        module_config = self._module_config()
        # For the in_graph backend with on_device_init, build the embedding tables
        # on meta (no host alloc): backend.setup() replaces them with on-device
        # HBM tables, so materializing the vanilla host tables is pure waste and,
        # at large hash sizes (uncapped 100M rows -> ~100GB/table), stalls model
        # construction on a single-threaded CPU init before compile.
        embeddings_on_meta = (
            self._trainer_cfg.embedding_backend == "in_graph"
            and self._trainer_cfg.on_device_init
        )
        attention_backend = resolve_attention_backend()
        self._attention_backend = attention_backend
        return build_model_from_config(
            module_config,
            torch.device("cpu"),
            bf16_training=self._trainer_cfg.bf16_training,
            embeddings_on_meta=embeddings_on_meta,
            attention_backend=attention_backend,
        )

    def create_train_module_and_optimizer(
        self,
        batch_size: Optional[int] = None,
        publish_destination_world_size: Optional[int] = None,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """Build the vanilla module + trace batch + data-pipe state, then hand
        them to RecirXLACompiledModule, which owns the recir construction
        (get_override + backend.setup + padding + compile). Returns the module's
        dense optimizer (recir owns the step; the trainer never calls opt.step()).
        """
        trainer_cfg = self._trainer_cfg

        # rankfm builds on CPU; recir/JAX places dense compute on TPU.
        device = torch.device("cpu")

        # 1. Embedding backend + PG (gloo for cpu, PJRT-only for tpu / in_graph).
        # RecTrainer.__init__ (use_gpu=False path) already created the default
        # gloo PG. Reuse it instead of letting the cpu backend re-init the
        # default PG -- a second init raises "default process group already
        # initialized" on multi-host cpu runs (e.g. the 128-chip yaml). The tpu /
        # in_graph backends and single-host cpu don't init a PG, so init_pg() is
        # the fallback when nothing has been set up yet.
        module_config = self._module_config()
        backend = build_embedding_backend(
            trainer_cfg.embedding_backend,
            on_device_init=trainer_cfg.on_device_init,
            embedding_tables=module_config.embedding_tables,
        )
        if dist.is_available() and dist.is_initialized():
            gloo_rank, gloo_world = dist.get_rank(), dist.get_world_size()
        else:
            gloo_rank, gloo_world = backend.init_pg()
        self._gloo_rank: int = gloo_rank  # pyre-ignore[16]
        self._gloo_world: int = gloo_world  # pyre-ignore[16]

        # Mesh shapes. num_ranks = jax.device_count() when on real TPU, else
        # gloo_world.
        num_tpu_devices = jax.device_count()
        num_ranks = num_tpu_devices if num_tpu_devices > 1 else gloo_world
        num_processes = gloo_world

        feature_config = self._feature_config()
        # Both dumps here, where both objects are in hand, so one grep of the
        # log has the whole config surface of this leg.
        config_dump("module_config", module_config)
        config_dump("feature_config", feature_config)

        # 2. Vanilla module (inherited builder -> BlueReelsVDDTrainV3).
        torch.manual_seed(0)
        model = self.create_vanilla_train_module(batch_size)
        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

        num_candidates_per_sample: int = _resolve_nro_per_sample(
            module_config, trainer_cfg.enable_padding
        )

        # 3. KJT key -> num_embeddings BEFORE backend.setup (which the wrapper
        #    runs; post-DMP the EC is wrapped and embedding_configs() disappears).
        key_to_size: Dict[str, int] = compute_kjt_key_to_size(model)

        # 4. Build data_pipe (before pulling the trace batch).
        data_pipe = self._build_data_pipe()

        # 5. Trace batch -> the wrapper builds the override sample from it AND
        #    traces on it.
        torch.manual_seed(0)
        if data_pipe is not None:
            sample_iter = replay_batch_iter(
                data_pipe,
                key_to_size,
                max_steps=1,
                local_batch_size=trainer_cfg.local_batch_size,
                num_candidates=num_candidates_per_sample,
            )
            batch = next(sample_iter)
            logger.info(
                "[OVERRIDE+TRACE] using first replay batch (real data) for both "
                "override-build sample and FX trace"
            )
        else:
            batch = create_uniform_train_batch(
                feature_config,
                device,
                batch_size=trainer_cfg.local_batch_size,
            )

        # When no data_pipe, swap the uniform sample for a synthetic ROO (jagged)
        # trace batch.
        if (
            trainer_cfg.enable_padding
            and feature_config is not None
            and data_pipe is None
        ):
            torch.manual_seed(0)
            batch = create_random_train_batch(
                feature_config,
                device,
                batch_size=trainer_cfg.local_batch_size,
            )
            logger.info("[TRACE] using synthetic batch for trace (no data_pipe)")

        # 6. Stash the replay-iter inputs so create_train_datapipe (called later
        #    by RecTrainer.train, on this same family instance) reuses the exact
        #    data_pipe / key_to_size created here -- without rebuilding the
        #    BatchReplayPipe (which would re-enumerate / re-auth GCS).
        self._gcs_data_pipe: Optional[BatchReplayPipe] = data_pipe  # pyre-ignore[16]
        self._gcs_key_to_size: Dict[str, int] = key_to_size  # pyre-ignore[16]

        ckpt_path = trainer_cfg.init_state_dict_path
        seeds_tables_from_ec = trainer_cfg.embedding_backend == "in_graph"
        if ckpt_path and seeds_tables_from_ec:
            if trainer_cfg.on_device_init:
                raise ValueError(
                    "in_graph with --init_state_dict_path needs "
                    "--no_on_device_init: the tables are seeded from the "
                    "embedding collection by copying into them, and "
                    "on_device_init leaves them as meta params with no storage "
                    "to copy into. _seed_in_graph_tables_from_ec would skip "
                    "every table and the run would train from random "
                    "embeddings while reporting a successful checkpoint load."
                )
            _load_parity_checkpoint(model, ckpt_path)

        # 7. The module owns the recir construction (override, backend.setup,
        #    padding, compile). It exposes the dense opt we return to the trainer.
        wrapper = RecirXLACompiledModule(
            model=model,
            backend=backend,
            trace_batch=batch,
            module_config=module_config,
            optimizer_config=self.config.optimizer_config,
            trainer_cfg=trainer_cfg,
            num_ranks=num_ranks,
            num_processes=num_processes,
            device=device,
            attention_backend=self._attention_backend,
        )
        if ckpt_path and not seeds_tables_from_ec:
            _load_parity_checkpoint(wrapper, ckpt_path)

        return wrapper, wrapper.dense_opt

    def create_train_datapipe(  # pyre-ignore[15]
        self, data_source_config: DataSourceConfig
    ) -> IterableDataset:
        """Reuse the synthetic/replay data path instead of the base RandomDataPipe.

        The compiled train_step (built by the provider) is what actually
        consumes batches; RecTrainer just iterates this datapipe and hands
        each batch to ``train_pipeline.progress``. We therefore yield the same
        MTMLROOTrainBatch objects the compiled step consumes: replay batches
        (when --gcs_data_path is set) or synthetic uniform/random batches,
        capped at ``max_steps``.

        The data_pipe / key_to_size were already created in
        create_train_module_and_optimizer and live on this family instance, so
        we re-derive the iterator here from the same inputs to keep this method
        self-contained (RecTrainer calls it independently).
        """
        trainer_cfg = self._trainer_cfg
        module_config = self._module_config()
        feature_config = self._feature_config()
        num_candidates_per_sample = _resolve_nro_per_sample(
            module_config, trainer_cfg.enable_padding
        )
        data_pipe = getattr(self, "_gcs_data_pipe", None)
        key_to_size = getattr(self, "_gcs_key_to_size", {})

        return _RankFMIterableDataset(
            trainer_cfg=trainer_cfg,
            feature_config=feature_config,
            num_candidates=num_candidates_per_sample,
            data_pipe=data_pipe,
            key_to_size=key_to_size,
        )


class _RankFMIterableDataset(IterableDataset):
    """Yields MTMLROOTrainBatch objects mirroring the per-step batch source.

    Three modes:
      1. data_pipe set => replay_batch_iter (real jagged data per step).
      2. enable_padding + feature_config => fresh random jagged batch / step.
      3. otherwise => the closure-captured uniform batch every step.
    Capped at ``max_steps`` so RecTrainer's loop terminates via StopIteration.
    """

    def __init__(
        self,
        trainer_cfg: RankFMTrainerConfig,
        feature_config: BlueReelsVDDFeatureConfig,
        num_candidates: int,
        data_pipe: Optional[BatchReplayPipe],
        key_to_size: Dict[str, int],
    ) -> None:
        self._trainer_cfg = trainer_cfg
        self._feature_config = feature_config
        self._num_candidates = num_candidates
        self._data_pipe = data_pipe
        self._key_to_size = key_to_size
        # Global data pointer (batches consumed so far), checkpointed via
        # state_dict so a resumed run continues from the next batch.
        self._batch_index = 0

    def __iter__(self) -> Any:
        trainer_cfg = self._trainer_cfg
        device = torch.device("cpu")
        if self._data_pipe is not None:
            start = self._batch_index
            source = replay_batch_iter(
                self._data_pipe,
                self._key_to_size or {},
                max_steps=trainer_cfg.max_steps,
                local_batch_size=trainer_cfg.local_batch_size,
                num_candidates=self._num_candidates,
                start_index=start,
                load_workers=trainer_cfg.data_load_workers,
            )
            depth = trainer_cfg.data_prefetch_buffer
            prefetch = PrefetchingIterator(source, depth=depth) if depth > 0 else None
            try:
                for i, batch in enumerate(prefetch if prefetch is not None else source):
                    self._batch_index = start + i + 1
                    yield batch
            finally:
                if prefetch is not None:
                    prefetch.close()
            return

        # Synthetic mode: same uniform batch unless padding (then fresh random).
        start = self._batch_index
        torch.manual_seed(0)
        base_batch = create_uniform_train_batch(
            self._feature_config,
            device,
            batch_size=trainer_cfg.local_batch_size,
        )
        for i in range(trainer_cfg.max_steps):
            pos = start + i
            if trainer_cfg.enable_padding and self._feature_config is not None:
                torch.manual_seed(pos + 1)
                batch = create_random_train_batch(
                    self._feature_config,
                    device,
                    batch_size=trainer_cfg.local_batch_size,
                )
            else:
                batch = base_batch
            self._batch_index = pos + 1
            yield batch

    def state_dict(self) -> Dict[str, Any]:
        """DCP-checkpointed data-stream position (see __iter__). The
        CheckpointManager saves this and restores it via load_state_dict on
        resume so real-data runs continue from the next unread batch. Mirrors
        mvai_cloud PreprocOutputReplayDataPipe.state_dict."""
        logger.info(f"[Replay] checkpoint data cursor: batch_index={self._batch_index}")
        return {"batch_index": self._batch_index}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._batch_index = int(state_dict["batch_index"])
        logger.info(f"[Replay] restored data cursor: batch_index={self._batch_index}")
