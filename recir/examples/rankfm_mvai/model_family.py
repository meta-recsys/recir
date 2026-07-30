#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
from contextlib import contextmanager
from typing import Any, cast, Dict, Generator, List, Optional, Tuple

import jax
import torch
import torch.distributed as dist
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.recir_core import (
    compile_gspmd_step,
    recir_train_step,
    RecirCompiledModuleBase,
    RecirCompileSpec,
    RecirMode,
    run_step_and_sync,
    setup_dense_optimizer_and_warmup,
    WarmupController,
)
from hammer.v3.models.udd_fm_v3.recir_core.compile import compile_step
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.models.udd_fm_v3.udd_fm_family import BlueReelsVDDModelFamily
from hammer.v3.modules.mot.tokenizers.mot_sparse_arch import ECSparseArch
from minimal_viable_ai.core.dataloader.dataloader_configs import DataSourceConfig
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.tests.xla.xsurface_test_utils import (
    _create_random_train_batch,
    _create_uniform_train_batch,
    build_model_from_config,
)
from torch.utils.data import IterableDataset
from torchrec.optim.keyed import KeyedOptimizerWrapper

# Local recir backend factory + data pipe helpers + the static-NRO resolver and
# the shared override/padding helpers. These are the exact same wiring the
# recir-compiled paths and the parity check use.
from .config import dense_parameter_fingerprint, RankFMModelArgs, RankFMTrainerConfig
from .data_pipe import BatchReplayPipe, compute_kjt_key_to_size, replay_batch_iter
from .embedding import build_embedding_backend
from .xsurface_helpers import (
    _resolve_nro_per_sample,
    make_padding_region_transforms,
    prepare_override_build_sample,
)

logger: logging.Logger = logging.getLogger(__name__)

# Per-step device->host state sync (sync_device_state_to_torch) skips state
# entries whose FQN contains any of these substrings. The on-device embedding
# tables (InGraphSparseArch's "_sharded_tables" / "_dp_tables" module dicts) are
# GBs each and are kept resident in the cached JAX device_state across steps, so
# mirroring them to host every step is O(table-size) waste (D2H copy + layout
# transpose) that dominates the step at large hash sizes. Dense params / LR /
# metric buffers don't match and still sync. Full sync (skip_fqns=None) is used
# before checkpointing/inspecting the tables.
_EMB_SYNC_SKIP_FQNS: Tuple[str, ...] = ("_sharded_tables", "_dp_tables")


@contextmanager
def _preserve_sparse_init_rng() -> Generator[None, None, None]:
    original_init = ECSparseArch.__init__

    def patched_init(self: ECSparseArch, *args: Any, **kwargs: Any) -> None:
        with torch.random.fork_rng(devices=[]):
            original_init(self, *args, **kwargs)

    ECSparseArch.__init__ = patched_init  # pyre-ignore[8]
    try:
        yield
    finally:
        ECSparseArch.__init__ = original_init  # pyre-ignore[8]


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
    candidate_sum: torch.Tensor,
    world_candidate_sum: torch.Tensor,
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
        candidate_sum,
        world_candidate_sum,
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
    candidate_sum: torch.Tensor,
    world_candidate_sum: torch.Tensor,
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
        candidate_sum,
        world_candidate_sum,
    )


class RecirXLACompiledModule(RecirCompiledModuleBase):
    """Self-compiling GSPMD/TPU recir train module (built by RankFMMvaiModelFamily).

    Owns the recir construction for the TPU cell, so the family stays focused on
    model + data wiring. In ``__init__`` it: expands the override sample from the
    trace batch, builds the preproc / sparse-arch dispatchers
    (``backend.get_override``), sets up embedding storage (``backend.setup``),
    builds the UNIFIED dense optimizer (``setup_dense_optimizer_and_warmup`` --
    the same ``TensorLRAdamW`` + 0-d LR buffer + ``WarmupController`` mechanism as
    the GPU cell), and -- for the in_graph backend -- a dedicated embedding
    optimizer (SGD @ embedding_lr) over the in-graph table params, wires
    ``set_recir_kjt_keys``, and builds the ``PaddingRegionTransform``. On the
    first forward it lazily compiles (``compile_gspmd_step`` for the single dense
    opt, or ``compile_step`` with (dense_opt, emb_opt) for the in_graph two-opt
    program) and runs each step via ``run_step_and_sync`` / the runner.

    Exposes ``dense_opt`` so the family can return it to the trainer (recir owns
    the step; the trainer never calls opt.step()), avoiding a NoOpOptimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: Any,
        trace_batch: Any,
        module_config: BlueReelsVDDModuleConfig,
        trainer_cfg: RankFMTrainerConfig,
        num_ranks: int,
        num_processes: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        num_candidates_per_sample: int = _resolve_nro_per_sample(
            module_config, trainer_cfg.enable_padding
        )
        global_bs = trainer_cfg.local_batch_size * num_processes
        padded_num_devices: int = math.lcm(
            num_ranks, backend.sc_modulus() * num_processes
        )

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
        dispatchers = backend.get_override(  # pyre-ignore[16]
            model.recblock,
            batch_sample,
            device,
            override_bs,
            num_ranks,
            num_processes,
        )

        # Embedding storage (DMP for cpu / SparseCore for tpu / in-graph tables
        # for in_graph). For cpu/tpu the sparse update is owned by the backend
        # off-GM; for in_graph the tables are plain params trained by emb_opt
        # (below) inside the compiled graph. The dense optimizer is built below.
        model = backend.setup(
            model,
            table_configs,
            trainer_cfg.embedding_lr,
            num_processes,
        )
        # pyre-ignore[29]: backend.setup returns Module; set_recir_kjt_keys is dynamic.
        model.set_recir_kjt_keys(ro_kjt_keys, nro_kjt_keys, unary_kjt_keys)

        # Unified dense optimizer: TensorLRAdamW + 0-d LR buffer + WarmupController,
        # the SAME mechanism as the GPU (HDP) cell. recir_train_step rebinds the
        # dense group['lr'] to LR_DENSE_BUFFER during the trace, the controller
        # advances it each step.
        dense_params = list(
            backend.dense_named_parameters(model).values()
        )  # pyre-ignore[16]
        dense_opt, warmup = setup_dense_optimizer_and_warmup(
            model, dense_params, trainer_cfg.dense_optimizer_config, device
        )

        # in_graph backend only: the in-graph embedding tables are FSDP params
        # looked up + grad'd inside the compiled graph, so give them their OWN
        # optimizer (stateless SGD @ embedding_lr, separate from the dense LR --
        # matching production where sparse embeddings train separately). SGD is
        # used because it traces in-graph as-is; a stateful algorithm would need
        # capturable mode + optimizer-state pre-materialization. For cpu / tpu
        # backends embedding_named_parameters() is absent / empty, so emb_opt
        # stays None and the single-optimizer path is taken.
        emb_named_fn = getattr(backend, "embedding_named_parameters", None)
        emb_params = emb_named_fn(model) if emb_named_fn is not None else {}
        emb_opt: Optional[torch.optim.Optimizer] = None
        if emb_params:
            if dist.is_initialized():
                emb_opt = KeyedOptimizerWrapper(
                    dict(emb_params),
                    lambda params: torch.optim.SGD(
                        params, lr=trainer_cfg.embedding_lr, foreach=True
                    ),
                )
            else:
                emb_opt = torch.optim.SGD(
                    list(emb_params.values()),
                    lr=trainer_cfg.embedding_lr,
                    foreach=True,
                )

        region_transforms: List[Any] = []
        if trainer_cfg.enable_padding:
            region_transforms = make_padding_region_transforms(
                num_candidates_per_sample, padded_num_devices, num_processes
            )

        self._model = model
        self._dense_opt: torch.optim.Optimizer = dense_opt
        self._emb_opt: Optional[torch.optim.Optimizer] = emb_opt
        self._warmup: WarmupController = warmup
        self._dispatchers: List[Any] = dispatchers
        self._trace_batch = trace_batch
        self._num_ranks = num_ranks
        self._num_processes = num_processes
        self._spec = RecirCompileSpec(
            mode=RecirMode.GSPMD_TPU,
            device=device,
            num_ranks=num_ranks,
            num_processes=num_processes,
            loss_reduction="sum",
            region_transforms=region_transforms,
        )
        self._model.train(True)

    @property
    def dense_opt(self) -> torch.optim.Optimizer:
        return self._dense_opt

    def _build_compiled_step(self, first_batch: Any) -> Any:
        # Trace with the family-provided (synthetic / replay, padded) trace batch,
        # not the incoming real batch, so the override + padding shapes the
        # dispatchers were built for stay consistent.
        if self._emb_opt is None:
            # Single dense optimizer (cpu / tpu backends).
            return compile_gspmd_step(
                self._model,
                self._dense_opt,
                self._trace_batch,
                self._dispatchers,
                self._spec,
                train_step_fn=_udd_fm_v3_train_step,
            )
        # in_graph backend: trace the dense optimizer AND the dedicated embedding
        # optimizer into the single GSPMD program. compile_gspmd_step hard-wires a
        # single optimizer, so use the shared multi-optimizer driver compile_step
        # directly, with a GSPMDParallel that knows the in-graph FSDP table prefix.
        from .embedding.in_graph_embedding_backend import SHARDED_TABLE_PREFIX

        emb_opt = self._emb_opt
        assert emb_opt is not None
        parallel_mode = GSPMDParallel(
            "replicate",
            loss_reduction=self._spec.loss_reduction,
            num_ranks=self._num_ranks,
            num_processes=self._num_processes,
            fsdp_table_substr=SHARDED_TABLE_PREFIX,
        )
        return compile_step(
            _udd_fm_v3_train_step_two_opt,
            self._model,
            (self._dense_opt, emb_opt),
            self._trace_batch,
            parallel_mode=parallel_mode,
            backend=XLAGSPMDBackend(),
            dispatchers=self._dispatchers,
            region_transforms=self._spec.region_transforms,
        )

    def _run_step(
        self, compiled_step: Any, batch: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._emb_opt is None:
            return run_step_and_sync(compiled_step, self._model, self._dense_opt, batch)
        # Two-optimizer runner: runner(model, dense_opt, emb_opt, *flat_batch).
        result: Tuple[torch.Tensor, Dict[str, torch.Tensor]] = compiled_step(
            self._model,
            self._dense_opt,
            self._emb_opt,
            *flatten_batch(batch),
        )
        return result

    def flush_host_state(self) -> None:
        """Mirror device-resident state back to torch for host consumers."""
        compiled_step = self._compiled_step
        if compiled_step is not None:
            sync_device_state_to_torch(compiled_step, skip_fqns=_EMB_SYNC_SKIP_FQNS)

    def _post_step(self) -> None:
        # Advance the dense warmup LR (writes the next iter's value into the 0-d
        # LR buffer in place; no-op multiplier until a schedule is configured).
        self._warmup.step()


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
        logger.info(
            f"Replay pipe: {len(data_pipe)} batches for rank {gloo_rank} "
            f"(replicate={trainer_cfg.data_replicate_across_ranks})"
        )
        return data_pipe

    # -- overrides ---------------------------------------------------------

    def create_vanilla_train_module(self, batch_size: Optional[int]) -> torch.nn.Module:
        """Build the byte-identical BlueReelsVDDTrainV3 via the rankfm builder.

        The base BlueReelsVDDModelFamily.create_vanilla_train_module builds the
        same BlueReelsVDDTrainV3 from module_config, but WITHOUT the TPU-required
        kernel/device settings that the recir lowering path depends on
        (use_cuda_attention=False, custom_kernel=False, _hammer_kernel=PYTORCH,
        dropout zeroed, model on the build device). build_model_from_config does
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
        with _preserve_sparse_init_rng():
            model = build_model_from_config(
                module_config,
                torch.device("cpu"),
                bf16_training=self._trainer_cfg.bf16_training,
                embeddings_on_meta=embeddings_on_meta,
            )
        logger.info(
            "RankFM dense initialization fingerprint: %s",
            dense_parameter_fingerprint(model),
        )
        return model

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
        backend = build_embedding_backend(
            trainer_cfg.embedding_backend,
            embedding_eps=trainer_cfg.embedding_eps,
            ringshift_min_rows=trainer_cfg.ringshift_min_rows,
            fsdp_min_rows=trainer_cfg.fsdp_min_rows,
            on_device_init=trainer_cfg.on_device_init,
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

        module_config = self._module_config()
        feature_config = self._feature_config()

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
            batch = _create_uniform_train_batch(
                feature_config,
                device,
                batch_size=trainer_cfg.local_batch_size,
            )

        # 6. ro_per_key_budgets for the replay datapipe (from the batch keys,
        #    same pre/post the trace-batch swap below).
        ro_kjt_keys: List[str] = list(batch.ro_id_list_features.keys())
        ro_per_sample_budgets: Dict[str, int] = (
            module_config.static_ro_per_sample_budgets
            or dict(module_config.contextual_feature_to_max_length)
        )
        ro_per_key_budgets: List[int] = [
            trainer_cfg.local_batch_size
            * ro_per_sample_budgets.get(k, int(module_config.hstu_max_seq_len))
            for k in ro_kjt_keys
        ]

        # When no data_pipe, swap the uniform sample for a synthetic ROO (jagged)
        # trace batch.
        if (
            trainer_cfg.enable_padding
            and feature_config is not None
            and data_pipe is None
        ):
            torch.manual_seed(0)
            batch = _create_random_train_batch(
                feature_config,
                device,
                batch_size=trainer_cfg.local_batch_size,
            )
            logger.info("[TRACE] using synthetic batch for trace (no data_pipe)")

        # 7. Stash the replay-iter inputs so create_train_datapipe (called later
        #    by RecTrainer.train, on this same family instance) reuses the exact
        #    data_pipe / key_to_size / budgets created here -- without rebuilding
        #    the BatchReplayPipe (which would re-enumerate / re-auth GCS).
        self._gcs_data_pipe: Optional[BatchReplayPipe] = data_pipe  # pyre-ignore[16]
        self._gcs_key_to_size: Dict[str, int] = key_to_size  # pyre-ignore[16]
        self._gcs_ro_per_key_budgets: List[int] = ro_per_key_budgets  # pyre-ignore[16]

        # 8. The module owns the recir construction (override, backend.setup,
        #    padding, compile). It exposes the dense opt we return to the trainer.
        wrapper = RecirXLACompiledModule(
            model=model,
            backend=backend,
            trace_batch=batch,
            module_config=module_config,
            trainer_cfg=trainer_cfg,
            num_ranks=num_ranks,
            num_processes=num_processes,
            device=device,
        )
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

        The data_pipe / key_to_size / budgets were already created in
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
        ro_per_key_budgets = getattr(self, "_gcs_ro_per_key_budgets", None)

        return _RankFMIterableDataset(
            trainer_cfg=trainer_cfg,
            feature_config=feature_config,
            num_candidates=num_candidates_per_sample,
            data_pipe=data_pipe,
            key_to_size=key_to_size,
            ro_per_key_budgets=ro_per_key_budgets,
        )


class _RankFMIterableDataset(IterableDataset):
    """Yields MTMLROOTrainBatch objects mirroring the per-step batch source.

    Three modes:
      1. data_pipe set => replay_batch_iter (real data, padded per step).
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
        ro_per_key_budgets: Optional[List[int]],
    ) -> None:
        self._trainer_cfg = trainer_cfg
        self._feature_config = feature_config
        self._num_candidates = num_candidates
        self._data_pipe = data_pipe
        self._key_to_size = key_to_size
        self._ro_per_key_budgets = ro_per_key_budgets

    def __iter__(self) -> Any:
        trainer_cfg = self._trainer_cfg
        device = torch.device("cpu")
        if self._data_pipe is not None:
            yield from replay_batch_iter(
                self._data_pipe,
                self._key_to_size or {},
                max_steps=trainer_cfg.max_steps,
                local_batch_size=trainer_cfg.local_batch_size,
                num_candidates=self._num_candidates,
                ro_per_key_budgets=self._ro_per_key_budgets,
            )
            return

        # Synthetic mode: same uniform batch unless padding (then fresh random).
        torch.manual_seed(0)
        base_batch = _create_uniform_train_batch(
            self._feature_config,
            device,
            batch_size=trainer_cfg.local_batch_size,
        )
        for step in range(1, trainer_cfg.max_steps + 1):
            if trainer_cfg.enable_padding and self._feature_config is not None:
                torch.manual_seed(step)
                yield _create_random_train_batch(
                    self._feature_config,
                    device,
                    batch_size=trainer_cfg.local_batch_size,
                )
            else:
                yield base_batch
