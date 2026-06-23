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
  FBGEMM-fused-optimizer path in RecModelFamily, it wires rankfm's recir
  embedding backend (cpu | tpu) exactly as rankfm/train.py does *before* the
  recir compile: build model on CPU, mark params trainable, compute the
  KJT key->num_embeddings map, pull the trace/override batch, build overrides
  via ``backend.get_override``, run ``backend.setup`` (DMP for cpu /
  SparseCore for tpu), and ``model.set_recir_kjt_keys``. All the artifacts the
  RecirXLATrainPipeline needs to run ``recir.trace`` / ``recir.compile_from_gm``
  are stashed on the returned model under ``_recir_compile_ctx`` (the provider
  only receives ``model``).

* ``create_train_datapipe``: reuses rankfm's data path (synthetic uniform /
  random batches, or barnfs/GCS replay via BatchReplayPipe) instead of the
  base RandomDataPipe.

``model_type()`` is inherited from BlueReelsVDDModelFamily ('udd_fm').
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, cast, Dict, List, Optional, Tuple

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
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.models.udd_fm_v3.udd_fm_family import BlueReelsVDDModelFamily
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from minimal_viable_ai.core.dataloader.dataloader_configs import DataSourceConfig

# Reuse rankfm's recir backend factory + data pipe helpers + the static-NRO
# resolver (do NOT copy). These are the exact same wiring the bespoke
# rankfm/train.py loop uses.
from recir.examples.rankfm.config import RankFMModelArgs, RankFMTrainerConfig
from recir.examples.rankfm.data_pipe import (
    BatchReplayPipe,
    compute_kjt_key_to_size,
    replay_batch_iter,
)
from recir.examples.rankfm.embedding import build_embedding_backend
from recir.examples.rankfm.train import _resolve_nro_per_sample
from recir.region_transforms.padding import (
    compute_padded_shape,
    JaggedPairConfig,
    PaddingRegionTransform,
)
from recir.tests.xla.xsurface_test_utils import (
    _create_random_train_batch,
    _create_uniform_train_batch,
    build_model_from_config,
    resize_batch_inplace,
    resize_kjt_per_key_lengths,
)
from torch.utils.data import IterableDataset
from torchrec.optim.keyed import KeyedOptimizerWrapper

logger: logging.Logger = logging.getLogger(__name__)


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
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Named-signature train step for the GSPMD/TPU compile.

    PaddingRegionTransform.bind_signature resolves batch_size_arg / jagged-pair
    arg names by inspecting the step function's parameter names. The shared
    recir_train_step is variadic (mod, *opts_and_batch), so those names are not
    visible and bind_signature raises e.g. "'ro_float_features' is not in list".
    This thin wrapper exposes the udd_fm_v3 forward arg names (matching
    rankfm/train.py's train_step_fn) and delegates to recir_train_step so the
    dense-opt step / LR-rebind / detached named-tensors behavior is identical.
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
    )


class RecirXLACompiledModule(RecirCompiledModuleBase):
    """Self-compiling GSPMD/TPU recir train module (built by RankFMMvaiModelFamily).

    Owns the recir construction for the TPU cell, so the family stays focused on
    model + data wiring (mirrors the GPU ``RecirHDPCompiledModule``, where the
    construction lives in the module, not the family). In ``__init__`` (run while
    the family builds it, so timing matches the previous in-family construction)
    it: expands the override sample from the trace batch, builds the preproc /
    sparse-arch dispatchers (``backend.get_override``), sets up embedding storage
    (``backend.setup`` -- the sparse update is owned by the embedding backend
    off-GM: SparseCore Adagrad on TPU / FBGEMM-fused DMP on CPU), builds the
    UNIFIED dense optimizer (``setup_dense_optimizer_and_warmup`` -- the same
    ``TensorLRAdamW`` + 0-d LR buffer + ``WarmupController`` mechanism as the GPU
    cell, NOT the backend's plain SGD), wires ``set_recir_kjt_keys``, and builds
    the ``PaddingRegionTransform``. On the first forward it lazily compiles via
    ``recir_core.compile_gspmd_step`` (single dense opt in the GM) and runs each
    step via ``run_step_and_sync``.

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
        if trainer_cfg.enable_padding:
            padded_ro_bs = compute_padded_shape(global_bs, padded_num_devices)
            padded_nro_bs = padded_ro_bs * num_candidates_per_sample
            expanded_batch = copy.deepcopy(trace_batch)
            resize_batch_inplace(
                expanded_batch,
                target_b_ro=padded_ro_bs,
                target_b_nro=padded_nro_bs,
            )
            if module_config.static_ro_per_sample_budgets:
                expanded_batch.ro_id_list_features = resize_kjt_per_key_lengths(
                    expanded_batch.ro_id_list_features,
                    module_config.static_ro_per_sample_budgets,
                    batch_size=padded_ro_bs,
                )
            override_bs = padded_ro_bs
        else:
            expanded_batch = copy.deepcopy(trace_batch)
            resize_batch_inplace(
                expanded_batch,
                target_b_ro=global_bs,
                target_b_nro=global_bs * num_candidates_per_sample,
            )
            override_bs = global_bs

        all_labels = expanded_batch.all_labels
        assert all_labels is not None
        batch_sample = _inputs_to_feature_bundle(
            expanded_batch.ro_id_list_features,
            None,
            expanded_batch.nro_id_list_features,
            expanded_batch.num_candidates,
            all_labels,
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

        # Embedding storage only (DMP for cpu / SparseCore for tpu). The sparse
        # update is owned by the backend off-GM; the dense optimizer is built
        # below, NOT by the backend.
        model = backend.setup(
            model,
            table_configs,
            trainer_cfg.embedding_lr,
            num_processes,
        )
        # pyre-ignore[29]: backend.setup returns Module; set_recir_kjt_keys is dynamic.
        model.set_recir_kjt_keys(ro_kjt_keys, nro_kjt_keys, unary_kjt_keys)

        # Unified dense optimizer: TensorLRAdamW + 0-d LR buffer + WarmupController,
        # the SAME mechanism as the GPU (HDP) cell (recir_train_step rebinds the
        # dense group['lr'] to LR_DENSE_BUFFER during the trace, the controller
        # advances it each step). Replaces the backend's plain-SGD dense opt so
        # dense numerics match across the GPU and TPU paths.
        dense_params = list(
            backend.dense_named_parameters(model).values()
        )  # pyre-ignore[16]
        dense_opt, warmup = setup_dense_optimizer_and_warmup(
            model, dense_params, trainer_cfg.dense_optimizer_config, device
        )

        region_transforms: List[Any] = []
        if trainer_cfg.enable_padding:
            region_transforms = [
                PaddingRegionTransform(
                    jagged_pairs={
                        "unary_values": JaggedPairConfig(
                            "unary_lengths",
                            per_sample_budgets=1,
                            batch_size_arg="nro_float_features",
                            count_arg="num_candidates",
                            max_per_sample_count=num_candidates_per_sample,
                        ),
                    },
                    batch_size_arg="ro_float_features",
                    num_devices=padded_num_devices,
                    num_processes=num_processes,
                    skip_args={
                        "ro_id_list_values",
                        "ro_id_list_lengths",
                        "nro_id_list_values",
                        "nro_id_list_lengths",
                    },
                ),
            ]

        self._model = model
        self._dense_opt: torch.optim.Optimizer = dense_opt
        self._warmup: WarmupController = warmup
        self._dispatchers: List[Any] = dispatchers
        self._trace_batch = trace_batch
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
        return compile_gspmd_step(
            self._model,
            self._dense_opt,
            self._trace_batch,
            self._dispatchers,
            self._spec,
            train_step_fn=_udd_fm_v3_train_step,
        )

    def _run_step(
        self, compiled_step: Any, batch: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return run_step_and_sync(compiled_step, self._model, self._dense_opt, batch)

    def _post_step(self) -> None:
        # Advance the dense warmup LR (writes the next iter's value into the 0-d
        # LR buffer in place; no-op multiplier until a schedule is configured).
        self._warmup.step()


class RankFMMvaiModelFamily(BlueReelsVDDModelFamily):
    """RecModelFamily that swaps GPU DMP for rankfm's recir embedding backend.

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

        Mirrors rankfm/train.py:719-744. Rank/world are taken from the backend
        init_pg() values that the entrypoint recorded on the trainer cfg's
        sibling knobs; here we re-derive them from the embedding backend's PG
        view (which is already initialized by the time the family is built).
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
        _bf16_training=False, dropout zeroed, model on the build device). rankfm's
        train.py uses build_model_from_config for exactly these mutations, so we
        delegate to it here to stay byte-identical to rankfm on the model that
        actually traces+lowers on TPU. The architecture (module hierarchy /
        state_dict keys) is identical to the base builder -- only the
        kernel/device knobs differ.
        """
        module_config = self._module_config()
        return build_model_from_config(module_config, torch.device("cpu"))

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

        # 1. Embedding backend + PG (gloo for cpu, PJRT-only for tpu).
        # RecTrainer.__init__ (use_gpu=False path) already created the default
        # gloo PG. Reuse it instead of letting the cpu backend re-init the
        # default PG -- a second init raises "default process group already
        # initialized" on multi-host cpu runs (e.g. the 128-chip yaml). The tpu
        # backend and single-host cpu don't init a PG, so init_pg() is the
        # fallback when nothing has been set up yet.
        backend = build_embedding_backend(trainer_cfg.embedding_backend)
        if dist.is_available() and dist.is_initialized():
            gloo_rank, gloo_world = dist.get_rank(), dist.get_world_size()
        else:
            gloo_rank, gloo_world = backend.init_pg()
        self._gloo_rank: int = gloo_rank  # pyre-ignore[16]
        self._gloo_world: int = gloo_world  # pyre-ignore[16]

        # Mesh shapes (rankfm/train.py:700-709). num_ranks = jax.device_count()
        # when on real TPU, else gloo_world.
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
        #    traces on it (rankfm/train.py:175-194).
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
        # trace batch (rankfm/train.py:260-267).
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
        """Reuse rankfm's data path instead of the base RandomDataPipe.

        The compiled train_step (built by the provider) is what actually
        consumes batches; RecTrainer just iterates this datapipe and hands
        each batch to ``train_pipeline.progress``. We therefore yield the same
        MTMLROOTrainBatch objects rankfm's training loop feeds its compiled
        step: replay batches (when --gcs_data_path is set) or synthetic
        uniform/random batches, capped at ``max_steps``.

        The data_pipe / key_to_size / budgets were already created in
        create_train_module_and_optimizer and live on the compile context, so
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
    """Yields MTMLROOTrainBatch objects mirroring rankfm's per-step batch source.

    Three modes (rankfm/train.py::_run_training_loop):
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
