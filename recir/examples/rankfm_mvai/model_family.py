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
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple

import jax
import torch
import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
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
from recir.sparsecore.backend import EmbeddingBackend
from recir.tests.xla.xsurface_test_utils import (
    _create_random_train_batch,
    _create_uniform_train_batch,
    build_model_from_config,
    resize_batch_inplace,
    resize_kjt_per_key_lengths,
)
from torch.utils.data import IterableDataset

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class RecirCompileContext:
    """Everything the RecirXLATrainPipeline needs to compile + run the step.

    Produced by RankFMMvaiModelFamily.create_train_module_and_optimizer (which
    has already run backend.get_override + backend.setup), consumed by
    RecirXLATrainPipelineProvider.create_train_pipeline. Stashed on the model
    as ``model._recir_compile_ctx`` because the provider only receives the
    model object.
    """

    backend: EmbeddingBackend
    overrides: List[Any]
    trace_batch: Any
    region_transforms: List[Any]
    num_ranks: int
    num_processes: int
    enable_padding: bool
    # Replay-mode state (None for synthetic mode).
    feature_config: Optional[BlueReelsVDDFeatureConfig]
    data_pipe: Optional[BatchReplayPipe]
    key_to_size: Dict[str, int]
    ro_per_key_budgets: List[int]
    local_batch_size: int
    max_steps: int


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
        """Build the vanilla module + rankfm recir embedding backend.

        Faithful re-implementation of the pre-compile portion of
        rankfm/train.py::_build_compiled_train_step. Returns (model, opt); the
        recir compile itself runs in RecirXLATrainPipelineProvider, fed via the
        RecirCompileContext stashed on the model.
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
        # when on real TPU, else gloo_world. global_bs = local_bs * gloo_world.
        num_tpu_devices = jax.device_count()
        num_ranks = num_tpu_devices if num_tpu_devices > 1 else gloo_world
        num_processes = gloo_world
        global_bs = trainer_cfg.local_batch_size * gloo_world

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
        padded_num_devices: int = math.lcm(
            num_ranks, backend.sc_modulus() * num_processes
        )

        # 3. KJT key -> num_embeddings BEFORE backend.setup (post-DMP the EC is
        #    wrapped and embedding_configs() disappears).
        key_to_size: Dict[str, int] = compute_kjt_key_to_size(model)

        # 4. Build data_pipe (before pulling the trace batch).
        data_pipe = self._build_data_pipe()

        # 5. Initial batch -> drives BOTH the override-build sample and the FX
        #    trace batch (rankfm/train.py:175-194).
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

        # 6. Expanded (global / padded) override-build sample.
        if trainer_cfg.enable_padding:
            padded_ro_bs = compute_padded_shape(global_bs, padded_num_devices)
            padded_nro_bs = padded_ro_bs * num_candidates_per_sample
            expanded_batch = copy.deepcopy(batch)
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
            expanded_batch = copy.deepcopy(batch)
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

        ro_kjt_keys: List[str] = list(batch.ro_id_list_features.keys())
        nro_kjt_keys: List[str] = list(batch.nro_id_list_features.keys())
        _unary = batch.unary_features
        assert _unary is not None
        unary_kjt_keys: List[str] = list(_unary.keys())

        ro_per_sample_budgets: Dict[str, int] = (
            module_config.static_ro_per_sample_budgets
            or dict(module_config.contextual_feature_to_max_length)
        )
        ro_per_key_budgets: List[int] = [
            trainer_cfg.local_batch_size
            * ro_per_sample_budgets.get(k, int(module_config.hstu_max_seq_len))
            for k in ro_kjt_keys
        ]

        # When no data_pipe, swap the uniform override sample for a synthetic
        # ROO (jagged) trace batch (rankfm/train.py:260-267).
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

        # 7. Table configs from EC BEFORE backend.setup mutates the model.
        # pyre-ignore[16]: dynamic submodule traversal through hammer v3 union
        ec = model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection
        table_configs = ec.embedding_configs()

        # 8. Build overrides BEFORE backend.setup (override builder runs the
        #    sparse arch eagerly against the unmodified model).
        overrides = backend.get_override(  # pyre-ignore[16]
            model.recblock,
            batch_sample,
            device,
            override_bs,
            num_ranks,
            num_processes,
        )

        # 9. Embedding storage (DMP for cpu / SparseCore for tpu).
        model, dense_opt = backend.setup(
            model,
            table_configs,
            trainer_cfg.embedding_lr,
            trainer_cfg.dense_lr,
            num_processes,
        )

        # pyre-ignore[29]: backend.setup returns Module; set_recir_kjt_keys is dynamic.
        model.set_recir_kjt_keys(ro_kjt_keys, nro_kjt_keys, unary_kjt_keys)

        # 10. Region transforms (PaddingRegionTransform) when padding is on.
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

        # Stash the replay-iter inputs so create_train_datapipe (called later
        # by RecTrainer.train, on this same family instance) reuses the exact
        # data_pipe / key_to_size / budgets created here -- without rebuilding
        # the BatchReplayPipe (which would re-enumerate / re-auth GCS).
        self._gcs_data_pipe: Optional[BatchReplayPipe] = data_pipe  # pyre-ignore[16]
        self._gcs_key_to_size: Dict[str, int] = key_to_size  # pyre-ignore[16]
        self._gcs_ro_per_key_budgets: List[int] = ro_per_key_budgets  # pyre-ignore[16]

        # 11. Stash the compile context for the train pipeline provider.
        model._recir_compile_ctx = RecirCompileContext(  # pyre-ignore[16]
            backend=backend,
            overrides=overrides,
            trace_batch=batch,
            region_transforms=region_transforms,
            num_ranks=num_ranks,
            num_processes=num_processes,
            enable_padding=trainer_cfg.enable_padding,
            feature_config=(feature_config if trainer_cfg.enable_padding else None),
            data_pipe=data_pipe,
            key_to_size=key_to_size,
            ro_per_key_budgets=ro_per_key_budgets,
            local_batch_size=trainer_cfg.local_batch_size,
            max_steps=trainer_cfg.max_steps,
        )

        return model, dense_opt

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
