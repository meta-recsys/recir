#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Prove the recir example's ``--prod_faithful`` config matches the light job's.

The hammer v3 MAST/light run and the recir example both end up constructing
``BlueReelsVDDTrainV3``, but historically from unrelated configs (prod's
3042-token / 94-task model vs a 32-token / 81-task bench toy). Nothing caught
the drift, because both paths "work" in isolation -- they just train different
models, which makes any NE comparison between them meaningless.

These tests pin the two together. They need the full //minimal_viable_ai
package, so they run under buck on a devserver rather than inside the rankfm
TPU image.

Two checks, cheapest first:

1. ``test_module_config_matches_prod`` -- field-level diff of the two
   ``BlueReelsVDDModuleConfig`` dataclasses. This is what tells you WHICH field
   drifted.
2. ``test_state_dict_matches_prod`` -- parameter/buffer key + shape parity of
   the actually-constructed models. This is the check that matters, since it
   exercises what the model really consumes rather than what the config claims.

The deliberate divergences are listed in ``_ALLOWED_DIVERGENCES`` and asserted
to be exactly those -- a new divergence fails rather than being tolerated.
"""

from __future__ import annotations

import dataclasses
import io
import os
import pickle
import pickletools
import re
import tempfile
import unittest.mock
import zipfile
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import zstandard
from hammer.utils import HammerKernel
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.launch.light_replay_config_shim import (
    ConfigProvider as LightConfigProvider,
)
from hammer.v3.models.udd_fm_v3.launch.prod_config_shim import (
    ConfigProvider as ProdXSurfaceConfigProvider,
)
from hammer.v3.models.udd_fm_v3.launch.train_module_wrapper import (
    _translate_module_config,
    BlueReelsVDDv4HSTUStandaloneXSurfaceTrainV3,
)
from hammer.v3.models.udd_fm_v3.parity.digest import (
    capture_activations,
    capture_grad_outputs,
    capture_module_inputs,
    compare_tensors,
    dump_tensors,
)
from hammer.v3.models.udd_fm_v3.recir_core import (
    flatten_batch,
    setup_dense_optimizer_and_warmup,
)
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from later.unittest import TestCase
from minimal_viable_ai.core.model_family_api.configs import LocationType

# _apply_optimizer_wrappers is private, and imported anyway: it is the function
# that layers WarmupOptimizer + GradientClippingOptimizer onto prod's dense
# optimizer (optimizer.py:972-1062). Re-deriving that layering here would make
# the light leg in this test a hand-written approximation of the light job
# rather than the light job.
from minimal_viable_ai.core.model_family_api.optimizer import (
    _apply_optimizer_wrappers,
    DENSE_OPTIM_PREFIX,
)
from minimal_viable_ai.core.model_family_api.optimizer_configs import OptimConfig
from minimal_viable_ai.core.ranking_common.configs import MTMLROOFeatureConfig
from minimal_viable_ai.models.blue_reels_vdd.model_configs import (
    BlueReelsVDDModuleConfig as ProdBlueReelsVDDModuleConfig,
)
from pyre_extensions import none_throws
from recir.embedding.in_graph import InGraphEmbeddingBackend
from recir.embedding.split_collection import partition_by_sharding
from recir.examples.rankfm_mvai import prod_config_artifact as _artifact

# Private, and imported anyway: this is the function the recir CPU backend uses
# to attach the in-backward embedding optimizer on its single-node branch
# (cpu.py:194-196). Both legs update embeddings inside backward -- light via
# torchrec's fused EmbeddingFusedOptimizer under DMP -- so reusing it keeps the
# recir leg exact and gives the light leg the same mechanism.
from recir.examples.rankfm_mvai.config import (
    _load_prod_module_config,
    _sync_l2_max_seq_len,
    build_module_config,
    build_optimizer_config,
    RankFMModelArgs,
)
from recir.examples.rankfm_mvai.data_pipe import (
    compute_kjt_key_to_size,
    remap_batch_ids,
)
from recir.examples.rankfm_mvai.eager_step import EagerCPUStep
from recir.examples.rankfm_mvai.prod_config_artifact import (
    ARTIFACT_PATH_ENV,
    export_prod_config,
    load_prod_config,
    load_prod_optimizer_config,
)
from recir.examples.rankfm_mvai.rankfm_embedding_backends import build_embedding_backend
from recir.examples.rankfm_mvai.rankfm_model_utils import build_model_from_config
from torchrec.distributed.train_pipeline import TrainPipelineBase
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.warmup import _get_multiplier, _lr_stages, WarmupOptimizer

# export_prod_config reaches the fork with importlib, so autodeps cannot see it
# and strips the deps on every run; without them the export raises "no
# mvai_cloud counterpart module".
# @dep=//mvai_cloud/minimal_viable_ai/core/model_family_api:optimizer_configs
# @dep=//mvai_cloud/minimal_viable_ai/models/ranking_hstu:hstu_config

# The recir --prod_faithful path is now expected to match the light job's
# config exactly. Any field that differs is a regression -- there are no
# sanctioned divergences left.
#
# inter_arch_type used to live here: the recir side forced the DCN off because
# mrs/.../dense_sparse was thought to be absent from the rankfm TPU image. It is
# not (scripts/build_image_podman.sh:162 stages it), so prod-faithful keeps the
# DCN and the two models are now parameter-identical.
_ALLOWED_DIVERGENCES: Set[str] = set()

# Prod's post-id table is 3B rows and the viewer-rid table 1B; neither
# materializes on a test host. Both sides get the same clamp so the comparison
# stays honest.
_HASH_SIZE_CAP: int = 100_000

# Shapes the dump at manifold://mrs_recsys_gcp/tree/czheo_test/udd_fm_3d_20260722
# was preprocessed with, read off rank_0/batch_0/0.pt.zst. If the config stops
# agreeing with these, the model can no longer consume that dump.
#
# Note the module config stores the RO+NRO SUM in float_feature_num and the NRO
# part separately (train_module_wrapper.py:246-257), so the RO width is the
# difference.
#
# Deliberately NOT asserted: the dump's 94 supervision rows vs the config's task
# count. Those are not the same thing -- _supervision_remap
# (udd_fm_train_v3.py:1325) looks each of the model's task names up in
# all_labels BY NAME and hard-fails on any gap, so a dump packing a superset in
# a different order is the designed-for case, not a mismatch.
_DUMP_RO_FLOAT_FEATURES: int = 149
_DUMP_NRO_FLOAT_FEATURES: int = 366

# One real dump batch on local disk, used by ForwardParityTest. The test does
# not fetch it -- pulling 17MB from Manifold does not belong in a unit test:
#   manifold get mrs_recsys_gcp/tree/czheo_test/udd_fm_3d_20260722\
#     /rank_0/batch_0/0.pt.zst /tmp/b0.pt.zst
_DUMP_BATCH_PATH: str = os.environ.get("PARITY_DUMP_BATCH", "/tmp/b0.pt.zst")

# Longest user history ForwardParityTest keeps, in tokens.
#
# The shared model runs HammerKernel.PYTORCH, whose attention
# (hammer/ops/pytorch/ragged_hstu_attention.py:437-438) pads the jagged
# sequences to dense [B, H, N, N] before the einsum and masks afterwards, so
# the max_attn_len window buys nothing. At prod's N=3067 that is
# 512*4*3067^2 fp32 = 77GB of scores per layer per leg; the run does not
# finish. mot_preprocessor.truncate_ro_features keeps the most recent
# hstu_max_seq_len events per row and N follows from those lengths, so capping
# the history is what makes this runnable on CPU.
#
# Both legs share one model and one batch, so the cap applies identically to
# each and cannot bias the comparison -- it only shortens the histories the
# comparison covers. Keep it above max_attn_len (256) so the sliding window is
# still exercised.
_PARITY_MAX_SEQ_LEN: int = int(os.environ.get("PARITY_MAX_SEQ_LEN", "512"))

# Both entrypoints run the same weights through the same kernels, so the only
# expected difference is float accumulation order. A real logic divergence lands
# orders of magnitude above this.
_FWD_RTOL: float = 1e-5


def _diff_embedding_tables(a: Any, b: Any) -> List[str]:
    """Per-table, per-field differences between two ``embedding_tables`` dicts.

    Dataclass ``__eq__`` on a whole table collapses to one unreadable repr, and
    the artifact REBUILDS every table by design (pickle records classes by
    module path, so the internal ``EmbeddingTable`` cannot travel --
    prod_config_artifact.py:16-33). The useful question is therefore never
    "are they equal" but WHICH field moved, and whether that field is one the
    model reads.
    """
    out: List[str] = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            out.append(f"{name}: present on one side only")
            continue
        ta, tb = a[name], b[name]
        for f in dataclasses.fields(ta):
            va = getattr(ta, f.name, None)
            vb = getattr(tb, f.name, None)
            if va != vb:
                out.append(f"{name}.{f.name}: {va!r:.60} != {vb!r:.60}")
    return out


def _cap_seq_len(module_config: Any) -> None:
    """Clamp the history the model consumes. Same rule on either config class.

    Shared so the light and recir legs cannot end up at different history
    lengths: they would then build different-length sequences from the same
    batch and every downstream comparison would be measuring that instead.
    """
    module_config.hstu_max_seq_len = min(
        module_config.hstu_max_seq_len, _PARITY_MAX_SEQ_LEN
    )
    # The L2 window has to stay strictly inside the history, or the L1 stack
    # gets zero-length sequences: minus_k = clamp(ro_lengths - l2_max_seq_len,
    # min=0) collapses to 0 once L2 covers the whole row
    # (mot_preprocessor.py:1069).
    module_config.hstu_l2_max_seq_len = min(
        module_config.hstu_l2_max_seq_len, _PARITY_MAX_SEQ_LEN // 2
    )


def _prod_family_config(disable_seq_len_sampling: bool = False) -> Any:
    """Prod's RecModelFamilyConfig, with embedding tables and history clamped.

    Prod's 3B-row table will not materialize on a test host; both paths get the
    identical clamp and ids are remapped modulo it, so the comparison stays exact.
    The history clamp is what makes the forward tractable at all.

    ``disable_seq_len_sampling`` must be set BEFORE construction: the preprocessor
    builds its sampling module from this field at construction time, so clearing it
    afterwards leaves the live sampler running. Reseeding around one call is enough
    for ONE model, but not across two driven through different entrypoints, whose
    RNG consumption differs.
    """
    prod_config = ProdXSurfaceConfigProvider().create_config()
    for table in prod_config.embedding_tables.values():
        table.num_embeddings = min(table.num_embeddings, _HASH_SIZE_CAP)

    # RecModelFamilyConfig types these loosely, so pyre widens module_config to
    # `object` and rejects every field access below.
    module_config: ProdBlueReelsVDDModuleConfig = cast(
        ProdBlueReelsVDDModuleConfig, prod_config.module_config
    )
    _cap_seq_len(module_config)
    if disable_seq_len_sampling:
        module_config.hstu_seq_len_sampling_configs = None
    return prod_config


def _translate_prod(prod_config: Any) -> BlueReelsVDDModuleConfig:
    """Run the light job's prod -> hammer-v3 translation.

    ``BlueReelsRankingModelFamilyConfig`` types its ``module_config`` /
    ``feature_config`` attributes loosely, so cast to what
    ``_translate_module_config`` actually requires.
    """
    return _translate_module_config(
        prod_module_config=cast(
            ProdBlueReelsVDDModuleConfig, prod_config.module_config
        ),
        feature_config=cast(MTMLROOFeatureConfig, prod_config.feature_config),
        embedding_tables=prod_config.embedding_tables,
    )


def _prod_module_config() -> BlueReelsVDDModuleConfig:
    """The module config the hammer v3 light/MAST job builds."""
    return _translate_prod(ProdXSurfaceConfigProvider().create_config())


def _recir_configs(
    disable_seq_len_sampling: bool = False,
) -> Tuple[BlueReelsVDDModuleConfig, BlueReelsVDDFeatureConfig]:
    """The module + feature config the recir example builds under --prod_faithful."""
    return build_module_config(
        RankFMModelArgs(
            prod_faithful=True,
            disable_seq_len_sampling=disable_seq_len_sampling,
        ),
        enable_padding=False,
    )


def _recir_module_config(
    disable_seq_len_sampling: bool = False,
) -> BlueReelsVDDModuleConfig:
    """The module config the recir example builds under --prod_faithful."""
    module_config, _feature_config = _recir_configs(disable_seq_len_sampling)
    return module_config


def _clamp_tables(module_config: BlueReelsVDDModuleConfig) -> None:
    for table in module_config.embedding_tables.values():
        table.num_embeddings = min(table.num_embeddings, _HASH_SIZE_CAP)


def _values_equal(a: Any, b: Any) -> bool:
    """Value equality that tolerates the artifact's cross-tree rebuild.

    ``_resolve`` prefers the mvai_cloud fork so the artifact loads to the same class
    in the image and on a devserver. The consequence here is that the live leg holds
    ``minimal_viable_ai.<X>`` while the artifact leg holds
    ``mvai_cloud.minimal_viable_ai.<X>`` -- different class objects, so
    ``dataclass.__eq__`` returns NotImplemented, ``==`` falls back to identity, and
    identical values report as different.

    Matching QUALNAME is still required before the field-wise compare. A different
    qualname means the artifact rebuilt the value as another type -- same name,
    wrong tree, silently different object -- which is this path's headline failure
    mode and must still fail.
    """
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        if type(a) is not type(b):
            if type(a).__qualname__ != type(b).__qualname__:
                # Rebuilt as a different type entirely.
                return False
            a_names = {f.name for f in dataclasses.fields(a)}
            b_names = {f.name for f in dataclasses.fields(b)}
            if a_names != b_names:
                # Same name, drifted fields: the two definitions are supposed
                # to be two halves of one wire format.
                return False
            return all(
                _values_equal(getattr(a, name), getattr(b, name))
                for name in sorted(a_names)
            )
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_values_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except (RuntimeError, ValueError):
        # Tensor-valued fields compare elementwise.
        return bool(torch.equal(a, b))


def _diff_dataclasses(
    a: BlueReelsVDDModuleConfig, b: BlueReelsVDDModuleConfig
) -> Dict[str, Tuple[Any, Any]]:
    """Field name -> (a_value, b_value) for every field that differs.

    Compares shallowly per field rather than via ``dataclasses.asdict``, which
    would recurse into nested configs and produce an unreadable diff on the
    first mismatch.
    """
    diffs: Dict[str, Tuple[Any, Any]] = {}
    for f in dataclasses.fields(a):
        a_val = getattr(a, f.name)
        b_val = getattr(b, f.name)
        if not _values_equal(a_val, b_val):
            diffs[f.name] = (a_val, b_val)
    return diffs


class ProdFaithfulConfigTest(TestCase):
    def test_module_config_matches_prod(self) -> None:
        prod = _prod_module_config()
        recir = _recir_module_config()

        diffs = _diff_dataclasses(prod, recir)
        unexpected = sorted(set(diffs) - _ALLOWED_DIVERGENCES)
        self.assertEqual(
            unexpected,
            [],
            "recir --prod_faithful config drifted from the light job's config "
            f"on {unexpected}. Per-field (prod, recir): "
            + "; ".join(f"{k}={diffs[k]!r:.200}" for k in unexpected),
        )

    def test_allowed_divergences_are_all_real(self) -> None:
        """Keep _ALLOWED_DIVERGENCES honest.

        If a divergence gets fixed upstream, the entry should be deleted rather
        than left behind masking a future regression.
        """
        prod = _prod_module_config()
        recir = _recir_module_config()
        diffs = _diff_dataclasses(prod, recir)
        stale = sorted(_ALLOWED_DIVERGENCES - set(diffs))
        self.assertEqual(
            stale,
            [],
            f"{stale} are listed as allowed divergences but now match; "
            "remove them from _ALLOWED_DIVERGENCES.",
        )

    def test_config_can_consume_the_dump(self) -> None:
        """The prod-faithful config must match the dump's actual tensor widths.

        These are the widths that made the standalone bench config unusable
        against the Manifold dump: it declares 515 total float features with 0
        of them NRO, i.e. a 515-wide RO tensor, against the dump's 149.
        """
        recir = _recir_module_config()
        self.assertEqual(
            recir.nro_float_feature_num,
            _DUMP_NRO_FLOAT_FEATURES,
            "NRO float width does not match the dump",
        )
        self.assertEqual(
            recir.float_feature_num - recir.nro_float_feature_num,
            _DUMP_RO_FLOAT_FEATURES,
            "RO float width does not match the dump",
        )

    def test_state_dict_matches_prod(self) -> None:
        """Both stacks must build parameter-identical models.

        Path A goes through the light job's wrapper; path B through the recir
        example's builder. Identical key sets and shapes is the precondition for
        copying weights across and comparing a forward pass.
        """
        prod_config = ProdXSurfaceConfigProvider().create_config()
        prod_module_config = _translate_prod(prod_config)
        _clamp_tables(prod_module_config)

        recir_module_config = _recir_module_config()
        _clamp_tables(recir_module_config)

        # Path A: the light job's adapter owns the translation internally, so
        # feed it the raw prod config and let it redo the work.
        path_a = BlueReelsVDDv4HSTUStandaloneXSurfaceTrainV3(
            embedding_tables=prod_config.embedding_tables,
            module_config=cast(ProdBlueReelsVDDModuleConfig, prod_config.module_config),
            feature_config=cast(MTMLROOFeatureConfig, prod_config.feature_config),
        )._impl
        # Path B: what the recir CPU-eager leg actually runs.
        path_b = build_model_from_config(
            recir_module_config,
            device=torch.device("cpu"),
            bf16_training=False,
        )

        a_shapes = {k: tuple(v.shape) for k, v in path_a.state_dict().items()}
        b_shapes = {k: tuple(v.shape) for k, v in path_b.state_dict().items()}

        only_a = sorted(set(a_shapes) - set(b_shapes))
        only_b = sorted(set(b_shapes) - set(a_shapes))
        self.assertEqual(only_a, [], f"params only in the light-job model: {only_a}")
        self.assertEqual(only_b, [], f"params only in the recir model: {only_b}")

        mismatched = {
            k: (a_shapes[k], b_shapes[k])
            for k in sorted(set(a_shapes) & set(b_shapes))
            if a_shapes[k] != b_shapes[k]
        }
        self.assertEqual(
            mismatched, {}, f"shape mismatches (light, recir): {mismatched}"
        )


def _materialize_meta_params(model: torch.nn.Module, device: torch.device) -> None:
    """Move a model with meta-device embedding tables onto ``device``.

    MVAI's embedding factory builds tables on ``meta`` while dense params and
    explicit buffers land on CPU with real values. ``to_empty`` materializes the
    meta tensors but wipes the real ones, and ``apply(reset_parameters)`` skips
    custom modules that lack that method -- so snapshot the real tensors first,
    then ``to_empty`` + reset, then copy the snapshots back.

    Mirrors RecirHDPCompiledModule.__init__ in
    hammer/v3/models/udd_fm_v3/launch_recir/recir_model_family.py.
    """
    param_snapshots: Dict[str, torch.Tensor] = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.device.type != "meta"
    }
    buffer_snapshots: Dict[str, torch.Tensor] = {
        name: b.detach().clone()
        for name, b in model.named_buffers()
        if b.device.type != "meta"
    }
    model.to_empty(device=device)

    def _reset(m: torch.nn.Module) -> None:
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()

    model.apply(_reset)
    # In-place copy_ so the Parameter/buffer objects themselves survive; any
    # optimizer or trace that captured a reference keeps working.
    for name, value in param_snapshots.items():
        model.get_parameter(name).data.copy_(value.to(device))
    for name, value in buffer_snapshots.items():
        model.get_buffer(name).data.copy_(value.to(device))


class ForwardParityTest(TestCase):
    """One step, one batch, one set of weights, test whether the two forwards agree.
    This is a CPU test.

      * the light/MAST job drives
        ``BlueReelsVDDv4HSTUStandaloneXSurfaceTrainV3.forward``
        (train_module_wrapper.py:558-607), which bypasses
        ``BlueReelsVDDTrainV3.forward`` entirely: it calls
        ``_compute_nro_feature_embeddings``, then ``recblock`` directly, filters
        ``output_segments``, then ``_forward_post_hstu``;
      * recir calls the raw model with ``flatten_batch(batch)`` positional args.

    Both are driven off ONE model instance here, so weights, kernels, precision
    and dropout are identical by construction rather than by careful setup. Any
    delta is therefore a real behavioural difference between the entrypoints,
    not a confound.
    """

    def _load_batch(self) -> Any:
        with open(_DUMP_BATCH_PATH, "rb") as fh:
            raw = zstandard.ZstdDecompressor().decompress(fh.read())
        return torch.load(io.BytesIO(raw), weights_only=False)

    def _build_shared_model(self) -> Tuple[Any, Any]:
        """Return (wrapper, raw_model) where ``wrapper._impl is raw_model``."""
        prod_config = _prod_family_config()
        torch.manual_seed(0)
        wrapper = BlueReelsVDDv4HSTUStandaloneXSurfaceTrainV3(
            embedding_tables=prod_config.embedding_tables,
            module_config=cast(ProdBlueReelsVDDModuleConfig, prod_config.module_config),
            feature_config=cast(MTMLROOFeatureConfig, prod_config.feature_config),
        )
        raw_model = wrapper._impl
        _materialize_meta_params(raw_model, torch.device("cpu"))
        # CPU-runnable kernels. _hammer_kernel is dispatch-only
        # (test_state_dict_matches_prod proves it does not change parameters), so
        # flipping it post-build keeps ONE set of weights.
        raw_model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
        # udd_fm_train_v3 hardcodes bf16_training=True on the STU stacks and the
        # loss module (:340, :388), which casts activations to bf16 while the
        # dense weights stay fp32 -- fine under GPU autocast, a dtype mismatch on
        # CPU. fp32 both also removes precision noise, so any delta below is pure
        # logic. Mirrors rankfm_model_utils.build_model_from_config:101.
        raw_model.recursive_setattr("_bf16_training", False)
        # Dropout must be off or the two calls draw different masks.
        raw_model.recursive_setattr("_dropout_ratio", 0.0)
        raw_model.recursive_setattr("_dropout_rate", 0.0)
        for m in raw_model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        return wrapper, raw_model

    def test_forward_matches_between_entrypoints(self) -> None:
        if not os.path.exists(_DUMP_BATCH_PATH):
            self.skipTest(
                f"no dump batch at {_DUMP_BATCH_PATH}; see the module docstring"
            )

        wrapper, raw_model = self._build_shared_model()
        batch = self._load_batch()
        # Remap ids into the clamped tables, exactly as the recir data path does.
        batch = remap_batch_ids(batch, compute_kjt_key_to_size(raw_model))

        # flatten_batch ships only values/lengths (trace_sample.py:84-89); the
        # raw-tensor forward rebuilds each KJT from the key lists stashed here,
        # which default to empty (udd_fm_train_v3.py:441-443). Skip this and the
        # recir leg builds key-less KJTs and dies in
        # extract_group_from_kjt_keys. Taking the keys off the post-remap batch
        # keeps them in the same order flatten_batch serialized the values in.
        raw_model.set_recir_kjt_keys(
            ro_kjt_keys=batch.ro_id_list_features.keys(),
            nro_kjt_keys=batch.nro_id_list_features.keys(),
            unary_kjt_keys=none_throws(batch.unary_features).keys(),
        )

        # Same mode for both calls: _prepare_loss_labels_and_weights rewrites
        # retention labels only when self.training.
        raw_model.train(True)

        with torch.no_grad():
            # Reseed before each leg. Anything stochastic reachable from the
            # forward -- notably the UIH sequence-length sampler, which runs
            # whenever self.training -- would otherwise draw different values on
            # the second call and show up as a spurious delta.
            torch.manual_seed(0)
            light_losses, light_named = wrapper(batch)
            # flatten_batch's trailing slot is loss_denom_source, but the light
            # entrypoint passes no denominator at all, so pass None to compare
            # like for like (_candidate_axis_mean, udd_fm_train_v3.py:109-112).
            recir_args = list(flatten_batch(batch))
            recir_args[-1] = None
            torch.manual_seed(0)
            recir_losses, recir_named = raw_model(*recir_args)

        light_total = float(light_losses.detach().to(torch.float64).sum())
        recir_total = float(recir_losses.detach().to(torch.float64).sum())
        abs_diff = abs(light_total - recir_total)
        rel_diff = abs_diff / max(abs(light_total), 1e-12)
        print(
            f"[FWD-PARITY] total loss light={light_total:.8f} "
            f"recir={recir_total:.8f} abs={abs_diff:.3e} rel={rel_diff:.3e}"
        )
        self.assertEqual(
            tuple(light_losses.shape),
            tuple(recir_losses.shape),
            "per-task loss vectors have different shapes",
        )
        # Per-task, so one bad head is named rather than averaged away.
        per_task = (
            (light_losses.detach() - recir_losses.detach()).abs().to(torch.float64)
        )
        worst = int(torch.argmax(per_task))
        print(
            f"[FWD-PARITY] worst task idx={worst} "
            f"light={float(light_losses[worst]):.8f} "
            f"recir={float(recir_losses[worst]):.8f} "
            f"abs={float(per_task[worst]):.3e}"
        )
        n_bad = int((per_task > 1e-6).sum())
        print(f"[FWD-PARITY] tasks with abs diff > 1e-6: {n_bad}/{per_task.numel()}")

        for key in sorted(set(light_named) & set(recir_named)):
            a, b = light_named[key], recir_named[key]
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                continue
            if a.shape != b.shape:
                print(f"[FWD-PARITY] {key}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            print(
                f"[FWD-PARITY] {key}: "
                f"max_abs={float((a.detach() - b.detach()).abs().max()):.3e}"
            )

        self.assertLess(
            rel_diff,
            _FWD_RTOL,
            f"light and recir forwards disagree: light={light_total} "
            f"recir={recir_total} rel={rel_diff:.3e}",
        )


# Tolerance on the recir/light gradient l2 ratio. The cluster comparison puts
# the median at 0.49, i.e. this is currently the assertion that FAILS and the
# reason the test exists. bf16 is off on both legs here, so the honest bar is
# fp32 accumulation-order noise, not the 3.9e-3 bf16 epsilon.
_GRAD_L2_RTOL: float = float(os.environ.get("PARITY_GRAD_RTOL", "0.02"))
# The two legs reach the fused tables through different code -- light through
# torchrec's WarmupOptimizer, recir by copying the shared controller's buffer
# into the same param_group -- so the rate agrees exactly and only float
# accumulation order is left.
_EMB_UPDATE_RTOL: float = 0.01

# Which denominator the recir leg passes as loss_denom_source. True mirrors
# --light_loss_norm (None -> _candidate_axis_mean falls back to torch.mean,
# which is exactly what the light entrypoint gets); False passes the group
# candidate count the way train_pipeline.py:631 does by default. At one rank
# and one micro-batch the two are the same quantity, so the default removes
# loss normalization as a suspect.
_LIGHT_LOSS_NORM: bool = os.environ.get("PARITY_LIGHT_LOSS_NORM", "1") == "1"


def _warmup_lr(cfg: OptimConfig, it: int = 0) -> float:
    """The LR a torchrec ``WarmupOptimizer`` applies at iteration ``it``.

    Not the nominal ``cfg.lr``. ``WarmupOptimizer.__init__`` calls
    ``_set_lr(0, 0)`` (warmup.py:139) before any step, and prod's LINEAR stage
    has ``value=0.1``, so the first step runs at a TENTH of the configured
    rate on both the dense and the sparse side. Reading cfg.lr directly is a
    10x error on every post-step weight delta.
    """
    stages = _lr_stages(list(cfg.lr_warmup_schedule or []))
    return cfg.lr * _get_multiplier(stages[0], it)


def _embedding_collection(model: torch.nn.Module) -> Tuple[str, Any]:
    """The model's ``EmbeddingCollection`` and the FQN prefix of its parameters.

    Discovered rather than hardcoded (model_family.py:320 and
    rankfm_embedding_backends.py:52 spell the same path out) so a rename in the
    hammer module tree fails here loudly instead of silently leaving every
    embedding table in the DENSE bucket, where it would be stepped by AdamW on
    both legs and the sparse-optimizer difference under test would vanish.

    Must be read BEFORE the embedding backend runs: post-DMP the EC is wrapped
    and ``embedding_configs()`` disappears.
    """
    matches = [
        (n, m) for n, m in model.named_modules() if isinstance(m, EmbeddingCollection)
    ]
    assert len(matches) == 1, f"expected one EmbeddingCollection, found {matches}"
    return matches[0]


def _capture_param_grads(
    model: torch.nn.Module, store: Dict[str, List[torch.Tensor]]
) -> List[Any]:
    """Record each parameter's gradient AS AUTOGRAD PRODUCES IT.

    Not ``p.grad`` after the step, for two reasons, both of which have already
    produced wrong readings on this project:

      * embeddings have no ``.grad`` at all once an in-backward optimizer owns
        them -- it consumes and clears the gradient inside backward -- so a
        ``.grad`` sweep silently drops exactly the tensors where the first
        divergence was reported;
      * ``clip_grad_value_`` mutates ``.grad`` in place, so two legs that
        capture on opposite sides of the clip disagree wherever |g| > 1.0 for
        reasons that have nothing to do with the model.

    A tensor hook fires before accumulation, before clipping and before any
    optimizer, which is the one point both legs can be compared at.
    """
    handles: List[Any] = []

    def mk(name: str) -> Callable[[torch.Tensor], None]:
        def hook(grad: torch.Tensor) -> None:
            store.setdefault(name, []).append(grad.detach().clone())

        return hook

    for name, p in model.named_parameters():
        if p.requires_grad:
            handles.append(p.register_hook(mk(name)))
    return handles


def _embedding_optimizer_buckets(
    model: torch.nn.Module, dense_names: Set[str], emb_prefix: str
) -> Dict[str, str]:
    """Which optimizer owns each embedding table on this leg.

    ``dense`` means the table kept a real ``.grad`` and is stepped by the dense
    optimizer -- what production's DATA_PARALLEL tables do. ``sparse`` means an
    in-backward optimizer consumed the gradient, which is what its ROW_WISE
    tables do. The two run at learning rates ~50x apart, so a table in the wrong
    bucket trains at the wrong rate while every shape, key and state-dict entry
    still matches.

    Table name is the second-to-last FQN segment, the convention torchrec's
    per-table parameter generators and ``_attach_single_node_sparse_optimizers``
    both use.
    """
    buckets: Dict[str, str] = {}
    for name, _ in model.named_parameters():
        if not name.startswith(emb_prefix):
            continue
        buckets[name.split(".")[-2]] = "dense" if name in dense_names else "sparse"
    return buckets


def _in_graph_table_buckets(
    model: torch.nn.Module, dense_names: Set[str], emb_prefix: str
) -> Dict[str, str]:
    """``_embedding_optimizer_buckets`` for the in_graph backend's FQN layout.

    in_graph keeps its tables in two ``nn.ParameterDict``s (``_sharded_tables``
    / ``_dp_tables``), so a table's parameter is ``<prefix>._dp_tables.<table>``
    and the name is the LAST segment -- not the second-to-last that torchrec's
    ``embeddings.<table>.weight`` puts it in. Reusing the torchrec reader here
    returns two entries keyed by the ParameterDict names and every table
    disappears from the comparison.
    """
    buckets: Dict[str, str] = {}
    for name, _ in model.named_parameters():
        if not name.startswith(emb_prefix):
            continue
        buckets[name.split(".")[-1]] = "dense" if name in dense_names else "sparse"
    return buckets


@dataclasses.dataclass
class _Leg:
    """One leg's native train step, reduced to "run it and hand back the loss"."""

    name: str
    model: torch.nn.Module
    # Runs zero_grad -> forward -> backward -> optimizer step and returns the
    # per-task loss vector. This is the leg's OWN code, not a reimplementation:
    # torchrec's TrainPipelineBase for light, EagerCPUStep for recir.
    run: Callable[[], torch.Tensor]
    # Names of the parameters the DENSE optimizer owns. Compared across legs:
    # a parameter that is dense on one leg and not on the other is trained by a
    # different optimizer, or not at all.
    param_names: Set[str]
    dense_lr: float
    sparse_lr: float


class _LossCapturingPipeline(TrainPipelineBase):
    """``TrainPipelineBase`` that hands back the per-task loss vector.

    ``progress()`` returns the model's second output (the named-tensor dict),
    and the comparison needs the loss vector that was actually differentiated.
    Overriding ``_backward`` to record it and delegate keeps torchrec's own
    reduction -- ``torch.sum(losses, dim=0).backward()`` -- as the thing that
    runs, which is the entire point of using this pipeline instead of writing
    the step out.
    """

    losses: Optional[torch.Tensor] = None

    def _backward(self, losses: torch.Tensor) -> None:
        self.losses = losses.detach()
        super()._backward(losses)


def _build_light_leg(
    model: torch.nn.Module,
    wrapper: Any,
    batch: Any,
    opt_cfg: Any,
    embedding_tables: Dict[str, Any],
) -> _Leg:
    """The light/MAST step: MVAI's optimizer stack driven by torchrec's pipeline.

    Every piece is the object the light job uses, not a description of it: the step
    is ``torchrec.TrainPipelineBase`` (whose ``_backward`` is a plain unweighted
    ``sum(losses, dim=0)``, task weights already being inside the model), the dense
    optimizer goes through prod's ``_apply_optimizer_wrappers`` so clipping runs in
    place before AdamW at the warmed-up LR, and embeddings step inside backward
    under a fused TBE.

    Two forced departures, neither numerical: AdamW runs unfused because prod's
    factory hardcodes the CUDA-only ``fused=True``, and prod's CUDA FBGEMM TBE does
    not exist on CPU, so the sharded tables use the CPU one on both legs.

    Per-table PLACEMENT is deliberately not held constant. This leg derives it
    from prod's own ``EmbeddingTable.sharding`` while the recir leg derives it
    from the exported artifact, so the two can disagree about which tables the
    sparse optimizer owns -- an artifact that lost ``sharding`` once moved seven
    tables to the wrong optimizer at 50x the learning rate, and this test could
    not see it.

    The sparse LR is the warmed-up one: prod's ``WarmupOptimizer`` sets the
    iteration-0 LR before any step, so for a single step baking the multiplier into
    the config is equivalent -- and it is the only option, since ``backend.setup``
    returns no handle to the fused optimizer.
    """
    sparse_cfg: OptimConfig = opt_cfg.sparse.default
    dense_cfg: OptimConfig = opt_cfg.dense.default

    sparse_lr = _warmup_lr(sparse_cfg)
    setup = build_embedding_backend("cpu", embedding_tables=embedding_tables).setup(
        model,
        _embedding_collection(model)[1].embedding_configs(),
        sparse_cfg,
        torch.device("cpu"),
        1,
    )
    assert setup.model is model, (
        "the embedding backend replaced the module; the light wrapper still "
        "holds the old one in _impl and would run an unsharded model"
    )
    dense_named = dict(setup.dense_named_parameters)
    # Snapshot BEFORE wrapping: KeyedOptimizer holds the params dict by
    # reference and WarmupOptimizer inserts a fake "<prefix>.__warmup"
    # parameter into it (warmup.py:140), which is optimizer state, not a model
    # parameter. Reading the names afterwards reports one parameter more than
    # the model has and makes the two legs look structurally different.
    dense_param_names = set(dense_named)
    dense_lr = _warmup_lr(dense_cfg)

    def factory(params: List[torch.Tensor]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            lr=dense_cfg.lr,
            betas=(dense_cfg.beta1, dense_cfg.beta2),
            eps=dense_cfg.eps,
            weight_decay=dense_cfg.weight_decay,
            # Citrine C2: multi-tensor AdamW. Also matches the recir leg's
            # foreach=True, so the two legs share an accumulation order.
            foreach=True,
        )

    opt = _apply_optimizer_wrappers(
        KeyedOptimizerWrapper(dense_named, factory), dense_cfg, DENSE_OPTIM_PREFIX
    )
    # Combine dense + sparse warmup, which is the shape prod's RecOptimizer has
    # (optimizer.py:1618-1786: a CombinedOptimizer over the warmup-wrapped fused
    # embedding optimizers plus the dense one). It matters beyond tidiness:
    # TrainPipelineBase steps whatever optimizer it is handed, and a sparse
    # warmup nobody steps freezes at its iteration-0 LR instead of ramping.
    fused = setup.sparse_runtime.fused_optimizer
    sparse_cfg_here = setup.sparse_runtime.config
    sparse_warmup = (
        WarmupOptimizer(
            fused,
            list(sparse_cfg_here.lr_warmup_schedule),
            lr=sparse_cfg_here.lr,
            param_name="__sparse_warmup",
        )
        if fused is not None and sparse_cfg_here.lr_warmup_schedule
        else None
    )
    stepped: Any = (
        opt
        if sparse_warmup is None
        else CombinedOptimizer([("", opt), ("", sparse_warmup)])
    )
    pipeline = _LossCapturingPipeline(
        model=wrapper, optimizer=stepped, device=torch.device("cpu")
    )

    def run() -> torch.Tensor:
        pipeline.progress(iter([batch]))
        return none_throws(pipeline.losses)

    return _Leg(
        name="light",
        model=model,
        run=run,
        param_names=dense_param_names,
        dense_lr=dense_lr,
        sparse_lr=sparse_lr,
    )


def _build_recir_leg(
    model: torch.nn.Module, batch: Any, embedding_tables: Dict[str, Any]
) -> _Leg:
    """The recir CPU-eager leg, assembled by the recir CPU-eager code.

    Mirrors ``RecirXLACompiledModule.__init__`` call for call: read the embedding
    configs off the EC BEFORE the backend wraps it, ``backend.setup`` (which on the
    single-node branch attaches the in-backward embedding optimizer), then
    ``setup_dense_optimizer_and_warmup``.

    The step is ``EagerCPUStep``, which is what ``EagerCPUTrainPipeline`` runs; the
    pipeline class itself is unimportable here (it subclasses a fork-only base), but
    its numerics do not depend on that.

    Both optimizer configs come from ``build_optimizer_config``, the same call
    the leg makes, so the test cannot drift from what the leg gets.
    """
    optimizer_config = build_optimizer_config()
    sparse_cfg: OptimConfig = optimizer_config.sparse.default
    backend = build_embedding_backend(
        "cpu",
        # This leg's OWN tables, off the artifact -- not prod's live ones. The
        # two sources are supposed to agree; letting each leg read its own is
        # what makes the comparison able to say they do not.
        embedding_tables=embedding_tables,
    )
    setup = backend.setup(
        model,
        _embedding_collection(model)[1].embedding_configs(),
        sparse_cfg,
        torch.device("cpu"),
        1,
    )
    dense_named = dict(setup.dense_named_parameters)

    # Same shape as the leg: ONE controller drives both LRs, and the fused side
    # reads its rate off a plain buffer this copies into param_groups.
    sparse_lr_buffer = torch.zeros((), dtype=torch.float32)
    tensor_lr_opt, warmup = setup_dense_optimizer_and_warmup(
        setup.model,
        list(dense_named.values()),
        optimizer_config.dense.default,
        torch.device("cpu"),
        sparse_cfg=setup.sparse_runtime.config,
        sparse_lr_buffer=sparse_lr_buffer,
    )
    step = EagerCPUStep.from_parts(
        setup.model,
        tensor_lr_opt,
        warmup,
        fused_opt=setup.sparse_runtime.fused_optimizer,
        sparse_lr_buffer=sparse_lr_buffer,
        light_loss_norm=_LIGHT_LOSS_NORM,
        # One dump batch, one rank: accumulation is out of scope here.
        grad_accum_steps=1,
    )

    return _Leg(
        name="recir",
        model=setup.model,
        run=lambda: step.run_step([batch])[0],
        param_names=set(dense_named),
        dense_lr=float(step.lr_buffer.item()),
        sparse_lr=_warmup_lr(sparse_cfg),
    )


def _run_native_step(
    leg: _Leg,
) -> Tuple[Dict[str, Dict[str, List[torch.Tensor]]], torch.Tensor]:
    """Run one full native step, capturing everything the cluster harness does.

    Returns the same ``kind -> name -> [tensor]`` store layout
    ``parity.digest``'s ``dump_tensors`` / ``compare_tensors`` consume, so a local
    repro prints the identical report to a two-pod run.
    """
    acts: Dict[str, List[torch.Tensor]] = {}
    boundary: Dict[str, List[torch.Tensor]] = {}
    gradouts: Dict[str, List[torch.Tensor]] = {}
    pgrads: Dict[str, List[torch.Tensor]] = {}
    handles = capture_activations(leg.model, acts)
    handles += capture_module_inputs(leg.model, boundary, ["", "recblock"])
    handles += capture_grad_outputs(leg.model, gradouts)
    handles += _capture_param_grads(leg.model, pgrads)

    winit = {n: [_plain(p).clone()] for n, p in leg.model.named_parameters()}

    # Reseed even with sampling disabled: anything else stochastic reachable
    # from the forward would otherwise draw a different stream on the second
    # leg and land in the comparison as a model difference.
    torch.manual_seed(0)
    losses = leg.run()

    weights = {n: [_plain(p).clone()] for n, p in leg.model.named_parameters()}
    for h in handles:
        h.remove()
    return {
        "winit": winit,
        "input": boundary,
        "act": acts,
        "gradout": gradouts,
        "pgrad": pgrads,
        "weight": weights,
    }, losses.detach()


def _plain(t: Any) -> torch.Tensor:
    """A plain host tensor for anything the sharded legs hand back.

    Under DMP the embedding tables come back as torchrec ``ShardedTensor``,
    which routes ``numel``/arithmetic through ``__torch_function__`` and raises
    ``RuntimeError: torch function 'numel' ... not supported``. The comparison
    only needs the values, and at world size 1 there is exactly one shard
    covering the whole tensor. Mirrors ``batch_loader.py:302``.
    """
    shards: Any = getattr(t, "local_shards", None)
    if callable(shards):
        local: List[Any] = cast(List[Any], shards())
        assert len(local) == 1, f"expected one local shard, got {len(local)}"
        return local[0].tensor.detach()
    return t.detach()


def _moved_params(
    winit: Dict[str, List[torch.Tensor]], weights: Dict[str, List[torch.Tensor]]
) -> Set[str]:
    """Names of the parameters the step actually changed.

    A vacuity guard. Every ratio in this test is computed over whole tensors,
    and an embedding table is mostly untouched rows, so a real difference on
    the handful of rows a batch hits is diluted to 1.0000 in an l2 ratio. If a
    parameter did not move at all, the comparison of it proved nothing --
    which is the case whenever an optimizer silently failed to be wired up.
    """
    moved: Set[str] = set()
    for name, before in winit.items():
        after = weights.get(name)
        if after is None or before[0].numel() == 0:
            continue
        if float(
            (after[0].to(torch.float64) - before[0].to(torch.float64)).abs().max()
        ):
            moved.add(name)
    return moved


def _deltas(
    winit: Dict[str, List[torch.Tensor]], weights: Dict[str, List[torch.Tensor]]
) -> Dict[str, List[torch.Tensor]]:
    """What the step actually did: ``w_after - w_before``, per parameter.

    Comparing post-step WEIGHTS across legs is nearly vacuous. Both legs start
    from the same checkpoint, so the ratio of their weight norms is ~1.0000
    whatever the optimizers did -- and it is exactly 1.0000 if neither stepped
    at all. The update is the quantity the optimizer produced, and it is the
    one an optimizer difference (a different LR, say) shows up in undiluted.
    """
    out: Dict[str, List[torch.Tensor]] = {}
    for name, before in winit.items():
        after = weights.get(name)
        if after is None or before[0].numel() == 0:
            continue
        out[name] = [after[0].to(torch.float64) - before[0].to(torch.float64)]
    return out


def _summed(seq: List[torch.Tensor]) -> torch.Tensor:
    out = seq[0].to(torch.float64)
    for t in seq[1:]:
        out = out + t.to(torch.float64)
    return out


def _ratio_report(
    kind: str,
    a_store: Dict[str, List[torch.Tensor]],
    b_store: Dict[str, List[torch.Tensor]],
    top: int = 10,
) -> Optional[float]:
    """Print the recir/light l2-ratio distribution for one kind, return its median.

    The l2 RATIO, not a relative delta: sorting by relative delta on a
    reduction puts near-zero tensors on top (sum 1e-5 vs 3e-2 reads as "440x
    off" and means nothing), which produced two wrong readings of this
    comparison before the distribution was added.
    """
    rows: List[Tuple[str, float, float, float]] = []
    for name in sorted(set(a_store) & set(b_store)):
        a, b = _summed(a_store[name]), _summed(b_store[name])
        if a.shape != b.shape:
            print(f"[1STEP] {kind} {name}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if a.numel() == 0:
            continue
        na = float(a.norm())
        if na == 0.0:
            continue
        scale = float(a.abs().max()) or 1.0
        srel = (a - b).abs() / scale
        rows.append(
            (name, float(b.norm()) / na, float(srel.median()), float(srel.max()))
        )
    if not rows:
        print(f"[1STEP] {kind}: nothing compared")
        return None

    ratios = sorted(r[1] for r in rows)

    def pct(q: float) -> float:
        return ratios[min(len(ratios) - 1, int(q * len(ratios)))]

    median = pct(0.5)
    print(
        f"[1STEP] {kind}: n={len(rows)} l2 ratio recir/light "
        f"p10={pct(0.1):.4f} median={median:.4f} p90={pct(0.9):.4f}"
    )
    for name, ratio, p50, mx in sorted(rows, key=lambda r: abs(r[1] - 1.0))[-top:]:
        print(
            f"[1STEP]   {kind} {name}: l2_ratio={ratio:.4f} "
            f"p50_scaled={p50:.3e} max_scaled={mx:.3e}"
        )
    return median


class TwoModelOneStepParityTest(TestCase):
    """Two SEPARATELY BUILT models, one shared checkpoint, one NATIVE step each.

    ``ForwardParityTest`` drives ONE instance through both entrypoints, so weights
    and precision are identical by construction -- it settled that the forward
    bodies agree and is structurally incapable of saying anything about the
    backward, which is where the legs disagree. An earlier version of THIS test
    called ``losses.sum().backward()`` on both legs and never stepped an optimizer,
    so everything downstream of the forward was symmetric code the test wrote
    itself; symmetric code cannot reproduce an asymmetry.

    This runs each leg's OWN step:

    ====================  =============================  =========================
    stage                 light                          recir
    ====================  =============================  =========================
    forward entrypoint    ``wrapper(batch)``             ``model(*flatten_batch)``
    loss reduction        ``sum(losses, dim=0)``         ``losses.sum()``
    table placement       from prod's live config        from the exported artifact
    embedding update      in-backward, warmed-up LR      in-backward, flat LR
    dense update          Warmup+Clip+AdamW wrappers     clip then AdamW, warmup
    ====================  =============================  =========================

    **What is still shared, and what that costs.** Both legs build their
    embeddings through ``CPUEmbeddingBackend``, because prod's embedding TBE is
    the CUDA FBGEMM one and there is no CUDA here. The MECHANISM is therefore
    held constant and this test cannot disagree about it.

    What it CAN disagree about, and what an earlier version of this test could
    not, is the per-table PLACEMENT: each leg reads its own source for
    ``EmbeddingTable.sharding`` -- prod's live config on the light side, the
    exported artifact on the recir side -- and placement decides the compute
    kernel, which decides whether a table is trained by the sparse optimizer
    inside backward or by the dense one at ``step()``, at learning rates ~50x
    apart. While both legs shared one setup that difference was impossible by
    construction, and a real artifact bug that moved seven of ten tables to the
    wrong optimizer went unnoticed here and was only visible on the cluster.
    ``test_one_full_step_matches_from_a_shared_checkpoint`` asserts the buckets
    agree, and separately that they are not all the same bucket.

    Both legs single-rank on purpose: no DDP, no DMP, no accumulation, so a
    remaining disagreement is in the model or optimizer wiring rather than the
    distributed layer.

    Reads ``PARITY_MAX_SEQ_LEN`` (512), ``PARITY_LIGHT_LOSS_NORM`` (1) and
    ``PARITY_GRAD_RTOL`` (0.02). Start at 128 -- memory goes as N^2 and both legs
    are resident at once.
    """

    def _prepare(self, model: Any) -> None:
        """Make a model deterministic and CPU-runnable, as ForwardParityTest does."""
        _materialize_meta_params(model, torch.device("cpu"))
        model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
        model.recursive_setattr("_bf16_training", False)
        model.recursive_setattr("_dropout_ratio", 0.0)
        model.recursive_setattr("_dropout_rate", 0.0)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

    def _build_models(self) -> Tuple[Any, Any, Any, Any, Any]:
        """Both models, each built its own leg's way. Weights are NOT shared here.

        The shared checkpoint is copied by the caller, after both embedding
        backends have run -- ``backend.setup`` rebuilds the embedding
        collection, so anything loaded at this point is silently discarded.

        Returns ``(wrapper, model_a, model_b, prod_config, recir_config)``. The
        models are typed Any because both carry hammer-v3 methods
        (``recursive_setattr``, ``set_recir_kjt_keys``) that
        ``nn.Module.__getattr__`` types as Tensor.
        """
        prod_config = _prod_family_config(disable_seq_len_sampling=True)

        # Path A: built the way the light job builds it.
        torch.manual_seed(0)
        wrapper = BlueReelsVDDv4HSTUStandaloneXSurfaceTrainV3(
            embedding_tables=prod_config.embedding_tables,
            module_config=cast(ProdBlueReelsVDDModuleConfig, prod_config.module_config),
            feature_config=cast(MTMLROOFeatureConfig, prod_config.feature_config),
        )
        model_a = wrapper._impl
        self._prepare(model_a)

        # Path B: built the way the recir CPU-eager leg builds it, from the
        # ARTIFACT -- setUp points PROD_CONFIG_ARTIFACT at one for the whole
        # test, which is the branch the cluster takes. Exercising it also turns
        # the config comparison below into a real check: with both legs on the
        # live path it compared two objects the same code had just produced.
        recir_module_config = _recir_module_config(disable_seq_len_sampling=True)
        # The same two clamps as path A: without the seq-len one, path B builds
        # sequences from prod's full 3042-token history against path A's
        # truncated one and the comparison is meaningless (and does not finish).
        _clamp_tables(recir_module_config)
        _cap_seq_len(recir_module_config)
        # hstu_l2_max_seq_len is ALSO mirrored into the L2 stack's constructor
        # kwargs, because SingleSeqL2HSTU reads it there while MoTPreprocessor
        # reads the config field. Capping only the field leaves the two
        # disagreeing and the jagged truncation indexes out of range mid-
        # forward. config.py's own syncer is what keeps them together; path A
        # gets this for free because _translate_module_config builds the stacks
        # after the cap.
        _sync_l2_max_seq_len(recir_module_config)

        # Both configs, capped, must still be the same config. Without this the
        # first symptom of a missed derived field is an IndexError minutes into
        # the recir forward, or worse, a silently different model.
        #
        # embedding_tables is compared separately: the artifact rebuilds every
        # table against the local class, so whole-dataclass equality asks the
        # wrong question and reports one truncated repr instead of the field.
        light_config = _translate_prod(prod_config)
        table_diffs = _diff_embedding_tables(
            light_config.embedding_tables, recir_module_config.embedding_tables
        )
        self.assertEqual(
            table_diffs,
            [],
            "the artifact's embedding tables differ from prod's:\n  "
            + "\n  ".join(table_diffs),
        )
        diffs = _diff_dataclasses(light_config, recir_module_config)
        diffs.pop("embedding_tables", None)
        self.assertEqual(
            sorted(diffs),
            [],
            "the two legs' module configs diverged under the parity clamps: "
            + "; ".join(f"{k}={diffs[k]!r:.200}" for k in sorted(diffs)),
        )

        torch.manual_seed(0)
        model_b = build_model_from_config(
            recir_module_config, device=torch.device("cpu"), bf16_training=False
        )
        self._prepare(model_b)
        return wrapper, model_a, model_b, prod_config, recir_module_config

    def setUp(self) -> None:
        """Give the embedding backend the process group the real legs have.

        ``RecTrainer.__init__`` ALWAYS initializes one -- gloo whenever the
        device is CPU (mvai_cloud trainer.py:70-92) -- so on the cluster
        ``dist.is_initialized()`` is True and ``CPUEmbeddingBackend.setup``
        takes its DMP / FUSED-TBE branch. Without a PG here it silently takes
        the single-node branch instead, which attaches a plain torchrec
        ``RowWiseAdagrad`` that NEITHER leg runs -- and that branch passes
        ``learning_rate=`` to an optimizer whose parameter is ``lr``, so the
        configured rate is swallowed by ``**unused`` and both legs would run
        at torchrec's 1e-2 default.
        """
        self._owns_pg: bool = not dist.is_initialized()
        if self._owns_pg:
            store = os.path.join(tempfile.mkdtemp(), "pg")
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{store}",
                world_size=1,
                rank=0,
            )
        # Point the recir leg at a prod config ARTIFACT for the whole test,
        # because that is what the cluster does: inside the TPU image the full
        # //minimal_viable_ai chain is absent (~1.15M buck targets, rooted in
        # the very file mvai_cloud forked), so _load_prod_module_config and
        # build_optimizer_config both read a translated .pt instead of building
        # live (prod_config_artifact.py:1-42).
        #
        # Set in setUp rather than around the config call: build_optimizer_config
        # is consulted later, when the leg is assembled, and a narrower scope
        # silently dropped it back to the standalone udd_fm provider -- which
        # carries no sparse warmup schedule, so the very divergence under test
        # disappeared from the recir side.
        artifact = os.path.join(tempfile.mkdtemp(), "prod_cfg.pt")
        export_prod_config(artifact)
        self._artifact_env: Any = unittest.mock.patch.dict(
            os.environ, {ARTIFACT_PATH_ENV: artifact}
        )
        self._artifact_env.start()

    def tearDown(self) -> None:
        self._artifact_env.stop()
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    def test_one_full_step_matches_from_a_shared_checkpoint(self) -> None:
        if not os.path.exists(_DUMP_BATCH_PATH):
            self.skipTest(f"no dump batch at {_DUMP_BATCH_PATH}")

        wrapper, model_a, model_b, prod_config, recir_config = self._build_models()

        # Read off the UNSHARDED models: the embedding backend replaces the
        # EmbeddingCollection with a sharded one, and ``embedding_configs()``
        # disappears with it (model_family.py:318-321 does the same).
        emb_prefix = _embedding_collection(model_a)[0]
        key_to_size: dict[str, int] = compute_kjt_key_to_size(model_a)
        self.assertEqual(
            key_to_size,
            compute_kjt_key_to_size(model_b),
            "the legs clamped their embedding tables differently, so the same "
            "id would land on different rows",
        )

        # ONE batch, remapped ONCE, fed to both legs. Remapping per leg would
        # let an input difference masquerade as a step difference.
        with open(_DUMP_BATCH_PATH, "rb") as fh:
            raw = zstandard.ZstdDecompressor().decompress(fh.read())
        batch = torch.load(io.BytesIO(raw), weights_only=False)
        batch = remap_batch_ids(batch, key_to_size)

        light = _build_light_leg(
            model_a,
            wrapper,
            batch,
            prod_config.optimizer_config,
            prod_config.embedding_tables,
        )
        recir = _build_recir_leg(model_b, batch, recir_config.embedding_tables)

        # THE SHARED CHECKPOINT, copied AFTER both backends have run. Loading
        # before is the failure that cost a full day on the cluster:
        # backend.setup rebuilds the embedding collection, so anything loaded
        # first is silently thrown away while both legs still log a successful
        # load. test_state_dict_matches_prod pins that the key sets and shapes
        # agree, so this is a plain copy -- no FQN remapping.
        missing, unexpected = model_b.load_state_dict(
            model_a.state_dict(), strict=False
        )
        self.assertEqual(list(missing), [], f"keys missing from path B: {missing}")
        self.assertEqual(
            list(unexpected), [], f"unexpected keys in path B: {unexpected}"
        )
        sa, sb = model_a.state_dict(), model_b.state_dict()
        worst_w0 = max(
            float(
                (_plain(sa[k]).to(torch.float64) - _plain(sb[k]).to(torch.float64))
                .abs()
                .max()
            )
            for k in sa
            if _plain(sa[k]).is_floating_point() and _plain(sa[k]).numel()
        )
        self.assertEqual(worst_w0, 0.0, "weights differ before the step")

        # Only the recir leg needs these: flatten_batch ships values/lengths
        # and drops the KJT keys (trace_sample.py:84-89), and the raw-tensor
        # forward rebuilds each KJT from this list. The light entrypoint takes
        # the batch object and never looks at it. After backend.setup, as
        # model_family.py:341-342 does it.
        model_b.set_recir_kjt_keys(
            ro_kjt_keys=batch.ro_id_list_features.keys(),
            nro_kjt_keys=batch.nro_id_list_features.keys(),
            unary_kjt_keys=none_throws(batch.unary_features).keys(),
        )
        # TrainPipelineBase gates zero_grad / backward / step on
        # ``self._model.training``, and its model is the WRAPPER, so training
        # mode has to be set there rather than on the inner module.
        wrapper.train(True)
        model_b.train(True)

        for leg in (light, recir):
            print(
                f"[1STEP] {leg.name}: dense={len(leg.param_names)} params @ "
                f"lr={leg.dense_lr:.3e} sparse lr={leg.sparse_lr:.3e}"
            )
        # By NAME, not by count: a parameter the dense optimizer owns on one leg
        # and not on the other is trained differently, or not at all. The key
        # sets of the two state_dicts already agree, so the difference is a
        # tensor that is a Parameter on one leg and a buffer on the other --
        # which no state_dict comparison can see.
        only_light = sorted(light.param_names - recir.param_names)
        only_recir = sorted(recir.param_names - light.param_names)
        self.assertEqual(
            (only_light, only_recir),
            ([], []),
            "the legs' dense optimizers own different parameters",
        )

        # Per-table optimizer bucketing, the thing this test was blind to for
        # as long as both legs shared one embedding setup. Each leg now derives
        # its placement from its OWN source -- prod's live config for light, the
        # exported artifact for recir -- so a table can land in different
        # buckets on the two legs, which is exactly the failure to catch.
        buckets_a = _embedding_optimizer_buckets(model_a, light.param_names, emb_prefix)
        buckets_b = _embedding_optimizer_buckets(model_b, recir.param_names, emb_prefix)
        for table in sorted(set(buckets_a) | set(buckets_b)):
            print(
                f"[1STEP] table {table}: light={buckets_a.get(table, '-')} "
                f"recir={buckets_b.get(table, '-')}"
            )
        self.assertEqual(
            buckets_a,
            buckets_b,
            "the legs put the same embedding table in different optimizer "
            "buckets; a sparse-bucket table trains at the sparse LR (~50x the "
            "dense one) with no shape or key difference to show for it",
        )
        # Guard against a vacuous pass. If every table ends up in one bucket the
        # comparison above holds trivially, which is what it did while both legs
        # ran the same row-count heuristic and put all ten on the sparse path.
        self.assertEqual(
            sorted(set(buckets_a.values())),
            ["dense", "sparse"],
            "prod splits these tables across BOTH optimizers, so a run with "
            f"only one bucket means the placement was lost: {buckets_a}",
        )

        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "light_tensors.pt")
            light_store, losses_a = _run_native_step(light)
            print(f"[1STEP] light dumped {dump_tensors(ref, light_store)} tensors")
            # Free the big captures before the second leg runs: the PYTORCH
            # attention kernel keeps every [B, H, N, N] score matrix alive for
            # backward, and holding both legs' activations at once is what
            # gets a run oomd-killed.
            grads_a = light_store["pgrad"]
            weights_a = light_store["weight"]
            winit_a = light_store["winit"]
            moved_a = _moved_params(winit_a, weights_a)
            del light_store

            recir_store, losses_b = _run_native_step(recir)
            # The same elementwise walk the cluster prints, in light's recorded
            # execution order, so the FIRST reported entry is where the legs
            # begin to differ.
            compare_tensors(ref, recir_store)
        moved_b = _moved_params(recir_store["winit"], recir_store["weight"])

        total_a = float(losses_a.to(torch.float64).sum())
        total_b = float(losses_b.to(torch.float64).sum())
        print(
            f"[1STEP] loss light={total_a:.8f} recir={total_b:.8f} "
            f"abs={abs(total_a - total_b):.3e}"
        )
        grad_median = _ratio_report("pgrad", grads_a, recir_store["pgrad"])
        delta_a = _deltas(winit_a, weights_a)
        delta_b = _deltas(recir_store["winit"], recir_store["weight"])
        _ratio_report("weight-delta", delta_a, delta_b)
        print(
            f"[1STEP] parameters moved by the step: light={len(moved_a)} "
            f"recir={len(moved_b)} of {len(weights_a)}"
        )
        # The embedding tables specifically. They are the only parameters the
        # How FAR the embedding tables moved, which is the only number here that
        # depends on the sparse learning rate: gradients do not, so a leg
        # stepping at 10x still matches on every gradient ratio above. That is
        # exactly how a 10x sparse LR gap survived this test once, showing up
        # only as 2.563e+00 against 2.563e+01 in a line nobody asserted on.
        emb_moves: Dict[str, float] = {}
        for label, moved, delta in (
            ("light", moved_a, delta_a),
            ("recir", moved_b, delta_b),
        ):
            emb = sorted(n for n in delta if n.startswith(emb_prefix))
            emb_moves[label] = sum(float(delta[n][0].norm()) for n in emb)
            print(
                f"[1STEP] {label}: {sum(n in moved for n in emb)}/{len(emb)} "
                f"embedding tables moved, total update l2={emb_moves[label]:.6e}"
            )
        self.assertGreater(
            emb_moves["light"],
            0.0,
            "the embedding tables did not move, so the sparse rate is untested",
        )
        self.assertAlmostEqual(
            emb_moves["recir"] / emb_moves["light"],
            1.0,
            delta=_EMB_UPDATE_RTOL,
            msg=(
                f"the legs moved their embedding tables by different amounts "
                f"(light={emb_moves['light']:.6e} recir={emb_moves['recir']:.6e}); "
                "with identical starting weights and matching gradients, that is "
                "the sparse learning rate differing between them"
            ),
        )
        # Both legs must have trained the SAME parameters. A parameter that
        # moves on one leg only means an optimizer is wired up on one leg and
        # not the other -- and every "identical" reading for it is vacuous.
        self.assertEqual(
            (sorted(moved_a - moved_b), sorted(moved_b - moved_a)),
            ([], []),
            "the step changed different parameters on the two legs",
        )
        self.assertNotEqual(
            moved_a, set(), "the step changed nothing; nothing was proven"
        )

        self.assertLess(
            abs(total_a - total_b) / max(abs(total_a), 1e-12),
            _FWD_RTOL,
            f"the two legs' losses disagree before any gradient is involved: "
            f"light={total_a} recir={total_b}",
        )
        self.assertIsNotNone(grad_median, "no gradients were compared")
        self.assertAlmostEqual(
            none_throws(grad_median),
            1.0,
            delta=_GRAD_L2_RTOL,
            msg=(
                "the two legs' gradients disagree from identical weights, "
                "identical inputs and an agreeing forward. Read the first "
                "'DIFF gradout/...' line above for where the backward starts "
                "to diverge; parameter gradients only say which modules ended "
                "up wrong."
            ),
        )


class InGraphTablePlacementTest(TestCase):
    """Which optimizer the in_graph backend gives each embedding table.

    Production places tables individually: ROW_WISE -> the sparse optimizer
    (RowWiseAdagrad @ 0.002), DATA_PARALLEL -> the dense one (AdamW @ 4e-5).
    ~50x apart, and a misplaced table keeps its shape, its key and its
    state_dict entry -- it just trains at the wrong rate. udd_fm_v3 is three
    ROW_WISE and seven DATA_PARALLEL.

    The cluster check for this is the weight delta (a sparse-trained table moves
    ~50x further in one step), which needs a pod, two legs and instrumentation.
    This asks the same question off the parameter split alone, so the answer is
    available before anything is submitted.

    Placement is read off the ARTIFACT, not prod's live config, because that is
    what the pod reads. It also puts ``partition_by_sharding``'s bare-string
    guard on the real path: the artifact round-tripped ``sharding`` by NAME
    once, which compares unequal to every ShardingType member while still being
    non-None, so every reader saw "no placement" and said nothing.
    """

    # The parity leg's device count (rankfm_mvai-parity-4chip). Only reaches the
    # row padding for dim-0-sharded tables, but a table's rows are padded up to
    # a multiple of it, so it has to be the real one.
    _WORLD_SIZE: int = 4

    def setUp(self) -> None:
        artifact = os.path.join(tempfile.mkdtemp(), "prod_cfg.pt")
        export_prod_config(artifact)
        self._artifact_env: Any = unittest.mock.patch.dict(
            os.environ, {ARTIFACT_PATH_ENV: artifact}
        )
        self._artifact_env.start()

    def tearDown(self) -> None:
        self._artifact_env.stop()

    def test_in_graph_places_tables_the_way_the_config_asks(self) -> None:
        module_config = _recir_module_config(disable_seq_len_sampling=True)
        _clamp_tables(module_config)
        _cap_seq_len(module_config)
        _sync_l2_max_seq_len(module_config)
        torch.manual_seed(0)
        model = build_model_from_config(
            module_config, device=torch.device("cpu"), bf16_training=False
        )
        # Read the configs off the EC BEFORE setup() swaps it out.
        emb_prefix, ec = _embedding_collection(model)
        table_configs = ec.embedding_configs()

        sharded, replicated = partition_by_sharding(
            cast(Dict[str, object], module_config.embedding_tables)
        )
        expected = {
            **dict.fromkeys(sharded, "sparse"),
            **dict.fromkeys(replicated, "dense"),
        }
        self.assertEqual(
            sorted(expected),
            sorted(tc.name for tc in table_configs),
            "the config's tables and the collection's tables are different sets, "
            "so the comparison below would silently skip whichever is missing",
        )
        # Guard against a vacuous pass: if the source says every table is the
        # same placement, the assertion at the end holds no matter what the
        # backend does. Ten tables in one bucket IS the bug this exists for.
        self.assertEqual(
            sorted(set(expected.values())),
            ["dense", "sparse"],
            "prod splits these tables across BOTH optimizers, so a single "
            f"bucket here means the placement never reached the config: {expected}",
        )

        backend = build_embedding_backend(
            "in_graph",
            embedding_tables=module_config.embedding_tables,
            # Real params, not meta: the meta branch exists so huge tables can
            # be generated straight into HBM by jax, and there is no HBM here.
            on_device_init=False,
        )
        # setup() refuses to run without these. Placement does not depend on
        # them -- InGraphSparseArch only stores them for the lookup path -- so
        # one token per feature keeps the test off the data pipeline.
        # build_embedding_backend is typed as the base EmbeddingBackend, which
        # has no set_feature_seqlens; only the in_graph one does.
        assert isinstance(backend, InGraphEmbeddingBackend)
        backend.set_feature_seqlens(
            {f: 1 for tc in table_configs for f in tc.feature_names}
        )
        setup = backend.setup(
            model,
            table_configs,
            build_optimizer_config().sparse.default,
            torch.device("cpu"),
            self._WORLD_SIZE,
        )

        actual = _in_graph_table_buckets(
            setup.model, set(setup.dense_named_parameters), emb_prefix
        )
        # The whole point of the failure is WHICH tables moved; the default
        # 640-char truncation replaces exactly that with "[335 chars]".
        self.maxDiff = None
        self.assertEqual(
            actual,
            expected,
            "the in_graph backend trains these tables with a different "
            "optimizer than the config asks for; a table that should be dense "
            "and lands in the sparse bucket runs at ~50x its intended learning "
            "rate, with no shape, key or state_dict difference to show for it",
        )


class LightShimConfigTest(TestCase):
    """The light leg's A100 shim must actually override what it claims to.

    Cheap guard against silent no-ops: the two A100 blockers are plain literals
    inside the parent's ``_module_config`` rather than overridable getters, so
    the shim unsets them after the fact. If the parent ever renames or restructures
    those, the override stops applying and the failure would otherwise only show
    up ~9 minutes into a GPU launch as an FP8 kernel error.
    """

    def test_a100_blockers_are_unset(self) -> None:
        config = LightConfigProvider().create_config()
        # Loosely typed on RecModelFamilyConfig; pyre widens it to `object`.
        module_config: ProdBlueReelsVDDModuleConfig = cast(
            ProdBlueReelsVDDModuleConfig, config.module_config
        )
        self.assertIsNone(
            module_config.fp8_config,
            "fp8_config must be None; A100 has no FP8 support",
        )
        self.assertFalse(
            module_config.use_cuda_hstu_attention,
            "use_cuda_hstu_attention must be off; those kernels are Hopper+",
        )

    def test_tables_are_shrunk_to_the_configured_cap(self) -> None:
        """Every table must end up at or under the cap, prod's are 3B/3B/1B.

        Asserted on the tables the shim actually produces, not on the
        ``get_*_hash_size`` getters: the shim deliberately does not override
        those -- ``RecModelFamilyConfig.shrink`` clamps
        ``embedding_tables[*].num_embeddings`` instead and never writes back to
        the provider, so the getters keep reporting prod's numbers by design.
        An earlier version of this test read them and could not pass.
        """
        config = LightConfigProvider().create_config()
        cap = LightConfigProvider.MAX_HASH_SIZE
        config.shrink(cap)  # what RecTrainer does before building the model
        tables = config.embedding_tables  # pyre-ignore[16]
        self.assertTrue(tables, "no embedding tables to check")
        biggest = max(t.num_embeddings for t in tables.values())
        self.assertLessEqual(
            biggest,
            cap,
            f"largest table is {biggest} rows against a cap of {cap}",
        )
        self.assertGreater(
            max(
                t.num_embeddings
                for t in LightConfigProvider().create_config().embedding_tables.values()
            ),
            cap,
            "nothing was above the cap, so this proves nothing about shrinking",
        )

    def test_force_hbm_moves_uvm_tables(self) -> None:
        """FORCE_HBM must actually flip table placement.

        UVM init allocates the full table as virtual memory and takes 30+ min to
        fault in on a devgpu, so this is the difference between a run that
        starts in seconds and one that looks hung.
        """

        class _NoForceHbm(LightConfigProvider):
            FORCE_HBM: bool = False

        before = _NoForceHbm()._embedding_tables()
        uvm_before = [k for k, t in before.items() if t.location == LocationType.UVM]
        self.assertNotEqual(uvm_before, [], "expected some UVM tables to begin with")

        after = LightConfigProvider()._embedding_tables()
        uvm_after = [k for k, t in after.items() if t.location == LocationType.UVM]
        self.assertEqual(uvm_after, [], f"still on UVM under FORCE_HBM: {uvm_after}")


class ProdConfigArtifactTest(TestCase):
    """The exported config artifact must reconstruct the live config exactly.

    The artifact exists so ``--prod_faithful`` works inside the TPU image, where
    the full //minimal_viable_ai chain is absent. Its embedding tables are
    translated field-by-field against whichever ``EmbeddingTable`` class the
    loading side declares, because the internal and mvai_cloud definitions are
    not compatible. A silent mismatch there would give the recir leg a model
    that looks configured correctly and trains differently -- exactly the class
    of bug this whole comparison exists to detect -- so a clean load is not
    sufficient evidence.
    """

    def test_artifact_round_trips(self) -> None:
        live_mc, _live_fc = _load_prod_module_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prod_cfg.pt")
            export_prod_config(path)
            loaded_mc, loaded_fc = load_prod_config(path)

        self.assertEqual(
            sorted(loaded_mc.embedding_tables),
            sorted(live_mc.embedding_tables),
            "artifact and live config disagree on which embedding tables exist",
        )
        for name, live_table in live_mc.embedding_tables.items():
            got = loaded_mc.embedding_tables[name]
            self.assertEqual(
                got.num_embeddings,
                live_table.num_embeddings,
                f"table {name}: num_embeddings drifted",
            )
            self.assertEqual(
                list(got.features),
                list(live_table.features),
                f"table {name}: feature list drifted",
            )
            # Enum fields must come back as ENUM MEMBERS, not their names.
            # ``sharding`` is declared ``Optional[ShardingType]``, and an
            # unwrap-less rehydrate hands back the bare string 'ROW_WISE'. That
            # is worse than a crash: ``== ShardingType.ROW_WISE`` is False while
            # ``is not None`` is True, so EmbeddingTable.__post_init__'s
            # DATA_PARALLEL-requires-HBM assert silently stops firing and
            # sharding_helpers' ``table.sharding.value`` waits to raise
            # AttributeError.
            self.assertEqual(
                type(got.sharding),
                type(live_table.sharding),
                f"table {name}: sharding rehydrated as {type(got.sharding)}, "
                f"expected {type(live_table.sharding)}",
            )
            self.assertEqual(
                got.sharding, live_table.sharding, f"table {name}: sharding drifted"
            )

        # Scalars that decide model shape. If any of these differ the two legs
        # are not the same model, however healthy the run looks.
        for field_name in (
            "hstu_max_seq_len",
            "hstu_l2_max_seq_len",
            "float_feature_num",
            "nro_float_feature_num",
        ):
            self.assertEqual(
                getattr(loaded_mc, field_name),
                getattr(live_mc, field_name),
                f"module_config.{field_name} drifted across the artifact",
            )

        self.assertIsNotNone(loaded_fc, "feature config did not survive the trip")

    def test_artifact_carries_prods_optimizer_not_the_standalone_one(self) -> None:
        """--prod_faithful must train with PROD's optimizer.

        ``build_optimizer_config`` used to always return
        ``ConfigProvider().create_config().optimizer_config`` -- the STANDALONE
        udd_fm provider -- even under --prod_faithful. That gave a prod-faithful
        MODEL trained by a different optimizer, and the two legs diverged
        steadily on every personalized task (recir converging faster, gap
        growing monotonically) while the unpersonalized head, far less
        embedding-driven, agreed to within 3%. An LR difference compounds every
        step, so nothing about it looks like a bug until the curves are laid
        side by side.
        """
        from recir.examples.rankfm_mvai.config import (
            build_optimizer_config,
            ConfigProvider,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prod_cfg.pt")
            export_prod_config(path)
            artifact_opt = load_prod_optimizer_config(path)
            with unittest.mock.patch.dict(os.environ, {ARTIFACT_PATH_ENV: path}):
                via_builder = build_optimizer_config()

        self.assertEqual(
            repr(via_builder),
            repr(artifact_opt),
            "build_optimizer_config ignored the artifact and fell back to the "
            "standalone provider",
        )

        # Prod's must actually differ from the standalone one, or this test
        # would pass vacuously if the fallback ever came back.
        standalone = ConfigProvider().create_config().optimizer_config
        self.assertNotEqual(
            repr(artifact_opt),
            repr(standalone),
            "prod and standalone optimizer configs are identical, so this test "
            "cannot detect the regression it exists for -- re-check the source "
            "of build_optimizer_config",
        )

    def test_artifact_preserves_configs_that_change_model_OUTPUT(self) -> None:
        """The artifact must carry every config the model's numerics depend on.

        The artifact is a translation, so any field it drops silently changes
        what the recir leg computes while every log line still looks healthy.
        This is not hypothetical: with shapes and inputs matched, the FIRST
        elementwise divergence between the legs was
        ``metadata_tokenizer._implicit_tokenizers.0._video_wt_bucket_embeddings``
        -- a BUCKET embedding. Both legs load the same checkpoint, so the table
        is identical and only the bucket INDEX can differ, which is decided by
        these encoding configs.

        Compares the fields that select a bucket, not object identity: the
        artifact deliberately rebuilds objects, so an equality check on the
        instances would fail for the wrong reason.
        """
        from recir.examples.rankfm_mvai.prod_config_artifact import (
            artifact_path,
            load_prod_config,
        )

        path = artifact_path()
        if path is None:
            self.skipTest("PROD_CONFIG_ARTIFACT not set")

        light_mc = ProdXSurfaceConfigProvider().create_config().module_config
        recir_mc, _fc = load_prod_config(path)

        for attr in ("watch_time_encoding_config", "video_length_encoding_config"):
            a = getattr(light_mc, attr, None)
            b = getattr(recir_mc, attr, None)
            self.assertEqual(
                a is None,
                b is None,
                f"{attr}: present on one leg only (light={a is None=}, "
                f"recir={b is None=}) -- bucket indices would differ",
            )
            if a is None:
                continue
            for field in sorted(
                f
                for f in dir(a)
                if not f.startswith("_")
                and isinstance(getattr(a, f, None), (int, float, bool, str))
            ):
                self.assertEqual(
                    getattr(a, field),
                    getattr(b, field, None),
                    f"{attr}.{field} differs between the legs; a bucketization "
                    f"parameter mismatch selects different embedding rows",
                )

    def test_artifact_feature_config_matches_prod(self) -> None:
        """The artifact substitutes a DEFAULT feature config for prod's.

        ``load_prod_config`` returns ``BlueReelsVDDFeatureConfig()`` with a
        docstring asserting prod's is "neither carried nor needed". That claim
        was never tested. The feature config carries per-feature shapes and
        truncation budgets, so if it differs, the two legs build sequences
        differently -- which is exactly the open divergence, localised to
        contextual_embedding_tokenizer's padded-value construction.

        Compares scalar fields only; the artifact rebuilds objects, so identity
        comparison would fail for the wrong reason.
        """
        from recir.examples.rankfm_mvai.prod_config_artifact import (
            artifact_path,
            load_prod_config,
        )

        path = artifact_path()
        if path is None:
            self.skipTest("PROD_CONFIG_ARTIFACT not set")
        _mc, artifact_fc = load_prod_config(path)
        prod_fc = ProdXSurfaceConfigProvider().create_config().feature_config

        def scalars(o: object) -> Dict[str, object]:
            out: Dict[str, object] = {}
            for f in sorted(x for x in dir(o) if not x.startswith("_")):
                try:
                    v = getattr(o, f)
                except Exception:
                    continue
                if isinstance(v, (int, float, bool, str)):
                    out[f] = v
                elif isinstance(v, (list, tuple)) and len(v) < 64:
                    out[f] = tuple(
                        x for x in v if isinstance(x, (int, float, bool, str))
                    )
            return out

        a, b = scalars(prod_fc), scalars(artifact_fc)
        keys = sorted(set(a) | set(b))
        diffs = [k for k in keys if a.get(k) != b.get(k)]
        print(
            f"\n[feature_config] prod={type(prod_fc).__name__} "
            f"artifact={type(artifact_fc).__name__} "
            f"fields={len(keys)} differing={len(diffs)}"
        )
        for k in diffs[:20]:
            print(f"  {k:<40} prod={a.get(k)!r:<28} artifact={b.get(k)!r}")
        self.assertEqual(
            diffs,
            [],
            f"{len(diffs)} feature-config fields differ between prod and the "
            f"artifact's default; the legs do not build features identically",
        )

    def test_undeclared_field_drops_when_default_and_raises_when_set(self) -> None:
        """The asymmetric rule for a field the loading environment lacks.

        This is not hypothetical. fbcode's ``torchrec.optim.warmup.WarmupStage``
        declares ``end_value`` and ``start_interpolating_iters``; the pip
        torchrec staged into the image does not. Prod leaves both at None --
        they belong to the INTERPOLATE policy and prod runs LINEAR -- so
        carrying them must not fail the load. A prod job that actually set one
        must fail it, because the image's class cannot hold the value and the
        LR curve would silently differ from the light leg's.

        Exercised against a locally-defined pair of classes rather than
        torchrec's, so the test states the rule instead of restating today's
        version skew, and keeps passing when the pip version catches up.
        """

        @dataclasses.dataclass
        class _Wide:
            kept: int = 1
            extra: Optional[float] = None

        @dataclasses.dataclass
        class _Narrow:
            kept: int = 1

        node = _artifact._encode(_Wide(kept=7), "probe")
        # The exporter's view: `extra` is untouched, `kept` is not.
        self.assertIn("extra", node[_artifact._DEFAULTS_KEY])
        self.assertNotIn("kept", node[_artifact._DEFAULTS_KEY])

        narrow = dict(node)
        narrow[_artifact._TYPE_TAG] = f"{_Narrow.__module__}.{_Narrow.__qualname__}"
        with unittest.mock.patch.object(_artifact, "_resolve", return_value=_Narrow):
            # Default in a field this environment lacks: dropped, load succeeds.
            cls, values = _artifact._decode_dataclass(narrow, "probe")
            self.assertEqual(values, {"kept": 7})
            self.assertEqual(cls(**values), _Narrow(kept=7))

            # Same field, actually set: must not load.
            narrow_set = dict(narrow)
            narrow_set[_artifact._FIELDS_KEY] = {
                **narrow[_artifact._FIELDS_KEY],
                "extra": 0.05,
            }
            narrow_set[_artifact._DEFAULTS_KEY] = ["kept"]
            with self.assertRaisesRegex(ValueError, "prod sets extra=0.05"):
                _artifact._decode_dataclass(narrow_set, "probe")

    def test_artifact_pickles_no_custom_class_at_all(self) -> None:
        """The payload must be primitives and containers, nothing else.

        A class reference in the stream is a MODULE PATH, and a module path can resolve
        to a different class of the same name on the other side -- which is the whole
        reason the conversion moved to the devserver.

        Asserted by WALKING the pickle stream rather than loading it, since loading
        resolves the names and hides the coupling. The failure it pins actually
        happened: enums were tested after primitives in ``_encode``, IntEnum subclasses
        int, so fbgemm's ``WeightDecayMode`` was handed back unconverted. Every local
        test stayed green, because on a devserver both sides resolve to the same fbgemm.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prod_cfg.pt")
            export_prod_config(path)
            with zipfile.ZipFile(path) as zf:
                name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
                raw = zf.read(name)

            demanded: List[str] = []

            class _Census(pickle.Unpickler):
                def find_class(self, module: str, name: str) -> Any:
                    demanded.append(f"{module}.{name}")
                    return type(name, (), {"__init__": lambda s, *a, **k: None})

                def persistent_load(self, pid: Any) -> Any:
                    return None

            _Census(io.BytesIO(raw)).load()
            self.assertEqual(
                sorted(set(demanded)),
                [],
                "the artifact pickles custom classes, so its contents are "
                "coupled to which tree resolves those module paths. Encode "
                "them as tagged values instead; check the ordering of the "
                "isinstance tests in _encode if these are enums.",
            )

    def test_artifact_is_loadable_against_the_fork(self) -> None:
        """Every class the artifact pickles must exist in mvai_cloud.

        This is the check the fidelity test above cannot make. That one runs on
        a devserver, where ``minimal_viable_ai`` IS the internal package, so
        every class resolves and unpickling trivially succeeds -- it proves the
        values are right, never that the artifact is loadable where only the
        fork exists. Portability is the entire reason the artifact exists.

        Without this, a class missing from the fork only surfaces as a
        ModuleNotFoundError inside a pod, after an image build, a push and a
        JobSet submit. It cost exactly that once: the artifact carried a
        pure-mvai ``MTMLROOFeatureConfig`` whose object graph needed seven
        classes the fork does not have.

        Reads the fork's files off disk rather than importing them, so it works
        from a devserver.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prod_cfg.pt")
            export_prod_config(path)
            with zipfile.ZipFile(path) as zf:
                pickle_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
                raw = zf.read(pickle_name)

        # Collect every (module, name) the pickle will resolve at load time.
        needed: Set[Tuple[str, str]] = set()
        operand_stack: List[Any] = []
        for op, arg, _pos in pickletools.genops(raw):
            if op.name == "GLOBAL":
                module, _, name = str(arg).partition(" ")
                needed.add((module, name))
            elif op.name in ("SHORT_BINUNICODE", "BINUNICODE"):
                operand_stack.append(arg)
            elif op.name == "STACK_GLOBAL" and len(operand_stack) >= 2:
                needed.add((str(operand_stack[-2]), str(operand_stack[-1])))

        fork_root = os.path.join(
            os.environ.get("FBCODE_DIR", "/data/users/linjianma/fbsource/fbcode"),
            "mvai_cloud",
        )
        if not os.path.isdir(fork_root):
            self.skipTest(f"mvai_cloud not found at {fork_root}")

        unresolvable: List[str] = []
        for module, name in sorted(needed):
            if not module.startswith("minimal_viable_ai"):
                continue
            source = os.path.join(fork_root, module.replace(".", "/") + ".py")
            if not os.path.exists(source):
                unresolvable.append(f"{module}.{name} (no module {source})")
                continue
            if not re.search(
                rf"^class {re.escape(name)}\b", open(source).read(), re.MULTILINE
            ):
                unresolvable.append(f"{module}.{name} (module exists, class does not)")

        self.assertEqual(
            unresolvable,
            [],
            "the artifact references classes mvai_cloud does not provide, so it "
            "will fail to unpickle in the image:\n  " + "\n  ".join(unresolvable),
        )


class DummyStackCarriesExportsTest(TestCase):
    """The trace-time stack swap has to preserve FeatureBundle exports.

    `udd_fm_train_v3._post_processing` hard-indexes `exported_tensors[name]` for
    every name in `module_config.output_segments`, and prod sets
    `["unsampled_inputs"]`. On the recir path `install_eager` replaces the real
    `PreprocessorStack` with `DummyPreprocessorStack` for the trace, and exports
    ride a separate str-keyed channel on FeatureBundle that
    the dummy's tensor/KJT/nontensor rebuild does not reach. Dropping them there
    killed the trace with KeyError('unsampled_inputs') before a single step ran.

    Reproduces that locally instead of on 128 chips.
    """

    def test_dummy_stack_keeps_the_export_through_the_trace_swap(self) -> None:
        """The link after the capture: install_eager swaps the real stack for
        DummyPreprocessorStack, and whatever that returns is what the trace --
        and therefore _post_processing -- actually sees."""
        from hammer.v3.modules.core.preprocessors.preprocessor_stack import (
            PreprocessorStack,
        )
        from hammer.v3.modules.mot.preprocessors.mot_preprocessor import MoTPreprocessor
        from recir.examples.rankfm_mvai.rankfm_dispatchers import (
            capture_sparse_arch_io,
            PreprocessorDispatcher,
        )

        recir_module_config = _recir_module_config()
        _clamp_tables(recir_module_config)
        model = build_model_from_config(
            recir_module_config,
            device=torch.device("cpu"),
            bf16_training=False,
        )
        recblock = model.recblock
        preprocessor = cast(
            MoTPreprocessor, recblock.preprocessor_stack.mot_preprocessor
        )
        sample_input = preprocessor.sample_inputs(
            batch_size=2,
            device=torch.device("cpu"),
            max_uih_seq_len=8,
            num_candidates=2,
        )
        _, _, preproc_output = capture_sparse_arch_io(recblock, sample_input)

        dispatcher = PreprocessorDispatcher(
            torch.device("cpu"),
            batch_size=2,
            max_uih_seq_len=8,
            num_candidates=2,
            sample_feature_bundle=sample_input,
            precomputed_output=preproc_output,
        )
        dispatcher.install_eager(model)

        swapped = type(model.recblock.preprocessor_stack).__name__
        self.assertNotIsInstance(
            model.recblock.preprocessor_stack,
            PreprocessorStack,
            f"install_eager did not swap the stack (still {swapped}); the trace "
            "would then run the real preprocessor, not the dummy",
        )

        out = model.recblock.preprocessor_stack(sample_input)
        self.assertIn(
            "unsampled_inputs",
            out.exports,
            f"the swapped-in {swapped} dropped the export; captured "
            f"{sorted(preproc_output.exports.keys())} but emitted "
            f"{sorted(out.exports.keys())}",
        )
