# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared DLRMv3 model definitions and helpers for XLA and EC sharding tests."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.fx
import torch.nn as nn
import torch.utils._pytree as pytree
from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.dlrm_v3.configs import get_hstu_configs
from generative_recommenders.modules.dlrm_hstu import (
    _get_supervision_labels_and_weights,
    DlrmHSTU,
    DlrmHSTUConfig,
    SequenceEmbedding,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from recir.ops.eager_dispatch import install_graph_break_dispatch, mark_output_with_bwd
from recir.region_transforms.training_to_inference import mark_inference_output
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


DLRMV3_PREPROCESS_MARKER_PREFIX = "recir.examples.dlrmv3.preprocess"


def _dlrmv3_marker_prefix(override_id: int) -> str:
    return f"{DLRMV3_PREPROCESS_MARKER_PREFIX}::{override_id}"


# ── Tiny model constants (mirrors debug.gin structure with reduced dims) ──
SMALL_HASH_SIZE = 100
SMALL_EMBEDDING_DIM = 64
SMALL_HIDDEN_DIM = 32
SMALL_TRANSDUCER_DIM = 64
SMALL_ATTN_DIM = 32
NUM_LAYERS = 2
NUM_HEADS = 2

# ── Static data constants ──
LOCAL_BS = 2  # per-device batch size; global_bs = LOCAL_BS * num_ranks
STATIC_UIH_LEN = 50
STATIC_NUM_CANDIDATES = 4
NUM_ITERS = 2


# ---------------------------------------------------------------------------
# Flattened preprocess output format
# ---------------------------------------------------------------------------
# SequenceEmbedding is a NamedTuple which may cause tracing issues.
# We flatten the preprocess output into plain tensors:
#   - emb_keys: ordered list of feature names (static, not traced)
#   - emb_lengths: list of [B] int64 tensors, one per feature
#   - emb_values: list of [total_len, D] float tensors, one per feature
#   - payload_features: Dict[str, Tensor]
#   - max_uih_len: int
#   - uih_seq_lengths: [B] int64
#   - max_num_candidates: int
#   - num_candidates: [B] int64


class DlrmV3Preprocess(nn.Module):
    """Wraps model.preprocess() and flattens SequenceEmbedding into plain tensors."""

    def __init__(
        self,
        model: DlrmHSTU,
        emb_keys: List[str],
        uih_kjt_keys: List[str],
        cand_kjt_keys: List[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.emb_keys = emb_keys
        self.uih_kjt_keys = uih_kjt_keys
        self.cand_kjt_keys = cand_kjt_keys

    def forward(
        self,
        uih_values: torch.Tensor,
        uih_lengths: torch.Tensor,
        cand_values: torch.Tensor,
        cand_lengths: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],  # emb_lengths per feature
        List[torch.Tensor],  # emb_values per feature
        Dict[str, torch.Tensor],  # payload_features
        int,  # max_uih_len
        torch.Tensor,  # uih_seq_lengths
        int,  # max_num_candidates
        torch.Tensor,  # num_candidates
        int,  # total_targets (pre-computed, avoids .item() in main_forward)
        int,  # total_uih_len (pre-computed)
    ]:
        uih_features = KeyedJaggedTensor(
            keys=self.uih_kjt_keys, values=uih_values, lengths=uih_lengths
        )
        cand_features = KeyedJaggedTensor(
            keys=self.cand_kjt_keys, values=cand_values, lengths=cand_lengths
        )
        (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        ) = self.model.preprocess(uih_features, cand_features)
        emb_lengths = [seq_embeddings[k].lengths for k in self.emb_keys]
        emb_values = [seq_embeddings[k].embedding for k in self.emb_keys]
        # Pre-compute data-dependent sizes on CPU (safe — not traced).
        total_targets = int(num_candidates.sum().item())
        # total_uih_len matches what _user_forward computes:
        # source_timestamps.numel() - total_targets, where source_timestamps
        # comes from concat_2D_jagged(uih_offsets, candidate_offsets).
        # Its numel = sum(uih_offsets lengths) + sum(candidate_offsets lengths)
        uih_total = int(payload_features["uih_offsets"][-1].item())
        cand_total = int(payload_features["candidate_offsets"][-1].item())
        total_uih_len = uih_total + cand_total - total_targets
        return (
            emb_lengths,
            emb_values,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
            total_targets,
            total_uih_len,
        )


class DlrmV3Dense(nn.Module):
    """Dense forward path: replicates main_forward + _user_forward logic
    using pre-computed total_targets/total_uih_len (Python ints from
    preprocess) to avoid .item() calls incompatible with make_fx tracing.
    """

    def __init__(self, model: DlrmHSTU, emb_keys: List[str]) -> None:
        super().__init__()
        self.model = model
        self.emb_keys = emb_keys
        # When True, tag the multitask predictions with an inference-output
        # marker so a training-step trace can be fabricated into an inference
        # graph (see recir.region_transforms.training_to_inference).  Off by
        # default so training / JAX / serving callers are unaffected.
        self._mark_inference_outputs: bool = False

    def forward(
        self,
        emb_lengths: List[torch.Tensor],
        emb_values: List[torch.Tensor],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
        total_targets: int,
        total_uih_len: int,
    ) -> Dict[str, torch.Tensor]:
        model = self.model
        configs = model._hstu_configs
        seq_embeddings = {
            k: SequenceEmbedding(lengths=emb_lengths[i], embedding=emb_values[i])
            for i, k in enumerate(self.emb_keys)
        }
        # ── main_forward: merge UIH + candidate embeddings ──
        for uih_feat, cand_feat in configs.merge_uih_candidate_feature_mapping:
            if uih_feat in seq_embeddings:
                seq_embeddings[uih_feat] = SequenceEmbedding(
                    lengths=uih_seq_lengths + num_candidates,
                    embedding=concat_2D_jagged(
                        max_seq_len=max_uih_len + max_num_candidates,
                        max_len_left=max_uih_len,
                        offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            uih_seq_lengths
                        ),
                        values_left=seq_embeddings[uih_feat].embedding,
                        max_len_right=max_num_candidates,
                        offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            num_candidates
                        ),
                        values_right=seq_embeddings[cand_feat].embedding,
                        kernel=model.hammer_kernel(),
                    ),
                )
        candidates_item_embeddings = model._item_forward(seq_embeddings)
        # ── user_forward (inlined, using pre-computed total_targets) ──
        source_lengths = seq_embeddings[configs.uih_post_id_feature_name].lengths
        source_timestamps = concat_2D_jagged(
            max_seq_len=max_uih_len + max_num_candidates,
            max_len_left=max_uih_len,
            offsets_left=payload_features["uih_offsets"],
            values_left=payload_features[
                configs.uih_action_time_feature_name
            ].unsqueeze(-1),
            max_len_right=max_num_candidates,
            offsets_right=payload_features["candidate_offsets"],
            values_right=payload_features[
                configs.candidates_querytime_feature_name
            ].unsqueeze(-1),
            kernel=model.hammer_kernel(),
        ).squeeze(-1)
        embedding = seq_embeddings[configs.uih_post_id_feature_name].embedding
        dtype = embedding.dtype
        if (not model.is_inference) and model._bf16_training:
            embedding = embedding.to(torch.bfloat16)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not model.is_inference) and model._bf16_training,
        ):
            candidates_user_embeddings, _ = model._hstu_transducer(
                max_uih_len=max_uih_len,
                max_targets=max_num_candidates,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_embeddings=embedding,
                seq_lengths=source_lengths,
                seq_timestamps=source_timestamps,
                seq_payloads=model._construct_payload(
                    payload_features=payload_features,
                    seq_embeddings=seq_embeddings,
                ),
                num_targets=num_candidates,
            )
        candidates_user_embeddings = candidates_user_embeddings.to(dtype)
        # ── multitask_module ──
        supervision_labels, supervision_weights = _get_supervision_labels_and_weights(
            supervision_bitmasks=payload_features[
                configs.candidates_weight_feature_name
            ],
            watchtime_sequence=payload_features[
                configs.candidates_watchtime_feature_name
            ],
            task_configs=model._multitask_configs,
        )
        mt_preds, _, _, mt_losses = model._multitask_module(
            encoded_user_embeddings=candidates_user_embeddings,
            item_embeddings=candidates_item_embeddings,
            supervision_labels=supervision_labels,
            supervision_weights=supervision_weights,
        )
        # ``mt_preds`` (slot 0) is the serving prediction (per-task sigmoid
        # probs); training discards it and keeps only ``mt_losses``.  Tag it so
        # a training-step trace can be fabricated into an inference graph.
        if self._mark_inference_outputs:
            mt_preds = mark_inference_output(mt_preds)
        aux_losses: Dict[str, torch.Tensor] = {}
        if not model._is_inference and model.training:
            for i, task in enumerate(model._multitask_configs):
                aux_losses[task.task_name] = mt_losses[i]
        return aux_losses


class DlrmV3Wrapper(nn.Module):
    def __init__(
        self,
        model: DlrmHSTU,
        emb_keys: List[str],
        uih_kjt_keys: List[str],
        cand_kjt_keys: List[str],
    ) -> None:
        super().__init__()
        self.preprocess_mod = DlrmV3Preprocess(
            model, emb_keys, uih_kjt_keys, cand_kjt_keys
        )
        self.dense_mod = DlrmV3Dense(model, emb_keys)

    def forward(
        self,
        uih_values: torch.Tensor,
        uih_lengths: torch.Tensor,
        cand_values: torch.Tensor,
        cand_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        (
            emb_lengths,
            emb_values,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
            total_targets,
            total_uih_len,
        ) = self.preprocess_mod(uih_values, uih_lengths, cand_values, cand_lengths)
        return self.dense_mod(
            emb_lengths,
            emb_values,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
            total_targets,
            total_uih_len,
        )


def get_table_configs() -> List[EmbeddingConfig]:
    """Return the embedding table configs used by the tiny DLRMv3 model."""
    return [
        EmbeddingConfig(
            num_embeddings=SMALL_HASH_SIZE,
            embedding_dim=SMALL_EMBEDDING_DIM,
            name="post_id",
            feature_names=[
                "uih_post_id",
                "item_post_id",
                "uih_owner_id",
                "item_owner_id",
            ],
        ),
        EmbeddingConfig(
            num_embeddings=SMALL_HASH_SIZE,
            embedding_dim=SMALL_EMBEDDING_DIM,
            name="viewer_id",
            feature_names=["viewer_id"],
        ),
        EmbeddingConfig(
            num_embeddings=SMALL_HASH_SIZE,
            embedding_dim=SMALL_EMBEDDING_DIM,
            name="dummy_contexual",
            feature_names=["dummy_contexual"],
        ),
    ]


def get_hstu_config(
    embedding_dim: int = SMALL_EMBEDDING_DIM,
    hidden_dim: int = SMALL_HIDDEN_DIM,
    transducer_dim: int = SMALL_TRANSDUCER_DIM,
    attn_dim: int = SMALL_ATTN_DIM,
    num_layers: int = NUM_LAYERS,
    num_heads: int = NUM_HEADS,
    uih_len: int = STATIC_UIH_LEN,
    num_candidates: int = STATIC_NUM_CANDIDATES,
) -> DlrmHSTUConfig:
    """Return an HSTU config with configurable dimensions.

    Defaults match the original tiny/debug config for backward compatibility.
    Pass larger values for medium/large model benchmarks.
    """
    hstu_config = get_hstu_configs("debug")
    hstu_config.hstu_embedding_table_dim = embedding_dim
    hstu_config.hstu_preprocessor_hidden_dim = hidden_dim
    hstu_config.hstu_transducer_embedding_dim = transducer_dim
    hstu_config.hstu_attn_linear_dim = attn_dim
    hstu_config.hstu_attn_qk_dim = attn_dim
    hstu_config.hstu_attn_num_layers = num_layers
    hstu_config.hstu_num_heads = num_heads
    hstu_config.max_seq_len = uih_len + num_candidates + 16
    hstu_config.max_num_candidates = num_candidates
    hstu_config.contextual_feature_to_min_uih_length = {
        "viewer_id": 10,
        "dummy_contexual": 10,
    }
    # Disable dropout — JAX jit doesn't allow PRNG key side effects
    # from aten.bernoulli (torchax's get_and_rotate_prng_key leaks a tracer).
    hstu_config.hstu_input_dropout_ratio = 0.0
    hstu_config.hstu_linear_dropout_rate = 0.0
    return hstu_config


def create_dlrmv3_wrapper(
    batch_size: int,
    hstu_config: Optional[DlrmHSTUConfig] = None,
    table_configs: Optional[List[EmbeddingConfig]] = None,
    is_inference: bool = False,
) -> Tuple["DlrmV3Wrapper", DlrmHSTUConfig, List[EmbeddingConfig], List[str]]:
    if hstu_config is None:
        hstu_config = get_hstu_config()
    if table_configs is None:
        table_configs = get_table_configs()
    table_config_dict = {t.name: t for t in table_configs}
    model = DlrmHSTU(
        hstu_configs=hstu_config,
        embedding_tables=table_config_dict,
        is_inference=is_inference,
    )
    # Materialize embedding collection on CPU
    model._embedding_collection = EmbeddingCollection(
        tables=table_configs,
        need_indices=False,
        device=torch.device("cpu"),
    )
    # Use pure PyTorch ops (Triton unsupported in XLA)
    model.set_hammer_kernel(HammerKernel.PYTORCH)
    set_dev_mode(True)
    # Feature keys that flow between preprocess and main_forward
    emb_keys = (
        hstu_config.user_embedding_feature_names
        + hstu_config.item_embedding_feature_names
    )
    # Build wrapper model
    contextual_features = list(hstu_config.contextual_feature_to_max_length.keys())
    uih_seq_features = [
        k for k in hstu_config.hstu_uih_feature_names if k not in contextual_features
    ]
    uih_kjt_keys = contextual_features + uih_seq_features
    cand_kjt_keys = hstu_config.hstu_candidate_feature_names
    wrapper = DlrmV3Wrapper(model, emb_keys, uih_kjt_keys, cand_kjt_keys)
    return wrapper, hstu_config, table_configs, emb_keys


def create_static_batch(
    hstu_config: DlrmHSTUConfig,
    batch_size: int,
    seed: int = 42,
    uih_len_override: Optional[int] = None,
    num_candidates_override: Optional[int] = None,
    hash_size_override: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a synthetic batch with STATIC EQUAL lengths.

    Returns decomposed KJT components (plain tensors) instead of
    KeyedJaggedTensor objects to avoid pytree unflatten issues during
    make_fx tracing.

    Args:
        hstu_config: HSTU model config.
        batch_size: Total batch size.
        seed: Random seed (default 42). Use different seeds per rank
            for multiprocess tests.
    """
    torch.manual_seed(seed)
    uih_len = uih_len_override if uih_len_override is not None else STATIC_UIH_LEN
    num_cands = (
        num_candidates_override
        if num_candidates_override is not None
        else STATIC_NUM_CANDIDATES
    )
    hash_size = (
        hash_size_override if hash_size_override is not None else SMALL_HASH_SIZE
    )
    contextual_features = list(hstu_config.contextual_feature_to_max_length.keys())
    uih_seq_features = [
        k for k in hstu_config.hstu_uih_feature_names if k not in contextual_features
    ]
    uih_keys = contextual_features + uih_seq_features
    uih_lengths = []
    uih_values_parts = []
    for key in uih_keys:
        L = 1 if key in contextual_features else uih_len
        uih_lengths.extend([L] * batch_size)
        uih_values_parts.append(torch.randint(1, hash_size, (batch_size * L,)))
    uih_values_t = torch.cat(uih_values_parts)
    uih_lengths_t = torch.tensor(uih_lengths, dtype=torch.long)
    cand_keys = hstu_config.hstu_candidate_feature_names
    cand_lengths = []
    cand_values_parts = []
    for _ in cand_keys:
        cand_lengths.extend([num_cands] * batch_size)
        cand_values_parts.append(torch.randint(1, hash_size, (batch_size * num_cands,)))
    cand_values_t = torch.cat(cand_values_parts)
    cand_lengths_t = torch.tensor(cand_lengths, dtype=torch.long)
    return uih_values_t, uih_lengths_t, cand_values_t, cand_lengths_t


class DummyPreprocess(nn.Module):
    """Returns zero tensors with correct shapes during tracing.

    Matches the output signature of DlrmV3Preprocess but without calling
    EmbeddingCollection. The shapes are derived from the static batch
    dimensions and model config.

    Wraps each output tensor with :func:`recir.ops.eager_dispatch.mark_output_with_bwd`
    which plants a forward/backward marker pair into the make_fx-traced FX
    graph.  The post-trace orchestrator (:class:`DlrmV3SparseDispatcher` or
    :class:`recir.examples.dlrmv3.jax_embedding.JAXEmbeddingDispatcher`) then
    replaces those markers with a graph-break dispatch back to the real
    preprocess (or a JAX SparseCore variant).
    """

    def __init__(
        self,
        emb_keys: List[str],
        batch_size: int,
        embedding_dim: int,
        static_uih_len: int,
        static_num_candidates: int,
        user_emb_feature_names: List[str],
        item_emb_feature_names: List[str],
        hstu_configs: DlrmHSTUConfig,
        override_id: int = 0,
    ) -> None:
        super().__init__()
        self.emb_keys = emb_keys
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.static_uih_len = static_uih_len
        self.static_num_candidates = static_num_candidates
        self.user_emb_feature_names = user_emb_feature_names
        self.item_emb_feature_names = item_emb_feature_names
        self.hstu_configs: DlrmHSTUConfig = hstu_configs
        self.override_id = override_id
        # Dummy parameter that requires_grad so that _batch_dep requires_grad.
        # This makes the marker autograd Function backward fire during make_fx
        # tracing, producing findable ``::bwd::`` marker nodes in the backward
        # graph for ``install_graph_break_dispatch`` to replace.
        self._grad_anchor = nn.Parameter(torch.zeros(1))

    def _feature_total_len(self, key: str, batch_size: int) -> int:
        """Total elements for a feature across the batch (static lengths)."""
        if key in self.item_emb_feature_names:
            return batch_size * self.static_num_candidates
        # Contextual features have length=1 per sample.
        if key in self.hstu_configs.contextual_feature_to_max_length:
            return batch_size * 1
        return batch_size * self.static_uih_len

    def forward(
        self,
        uih_values: torch.Tensor,
        uih_lengths: torch.Tensor,
        cand_values: torch.Tensor,
        cand_lengths: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        oid = self.override_id
        num_uih_keys = len(self.hstu_configs.contextual_feature_to_max_length) + len(
            [
                k
                for k in self.hstu_configs.hstu_uih_feature_names
                if k not in self.hstu_configs.contextual_feature_to_max_length
            ]
        )
        batch_size = uih_lengths.shape[0] // num_uih_keys
        # Maintain data dependency on batch inputs for GSPMD sharding.
        _batch_dep = (
            uih_values.sum()
            + cand_values.sum()
            + uih_lengths.sum()
            + cand_lengths.sum()
        ).float() * 0.0 + self._grad_anchor * 0.0
        _batch_dep_long = (uih_lengths.sum() + cand_lengths.sum()) * 0
        # Build the raw output structure (same shape as DlrmV3Preprocess).
        emb_lengths = []
        emb_values = []
        for key in self.emb_keys:
            total_len = self._feature_total_len(key, batch_size)
            if key in self.item_emb_feature_names:
                L = self.static_num_candidates
            elif key in self.hstu_configs.contextual_feature_to_max_length:
                L = 1
            else:
                L = self.static_uih_len
            emb_lengths.append(
                torch.full((batch_size,), L, dtype=torch.long, device=uih_values.device)
                + _batch_dep_long
            )
            emb_values.append(
                torch.zeros(total_len, self.embedding_dim, device=uih_values.device)
                + _batch_dep
            )
        uih_seq_lengths = (
            torch.full(
                (batch_size,),
                self.static_uih_len,
                dtype=torch.long,
                device=uih_values.device,
            )
            + _batch_dep_long
        )
        num_candidates = (
            torch.full(
                (batch_size,),
                self.static_num_candidates,
                dtype=torch.long,
                device=uih_values.device,
            )
            + _batch_dep_long
        )
        payload_features: Dict[str, torch.Tensor] = {}
        total_uih = batch_size * self.static_uih_len
        total_cand = batch_size * self.static_num_candidates
        for (
            uih_name,
            cand_name,
        ) in self.hstu_configs.merge_uih_candidate_feature_mapping:
            if (
                cand_name not in self.item_emb_feature_names
                and uih_name not in self.user_emb_feature_names
            ):
                payload_features[uih_name] = torch.zeros(
                    total_uih, dtype=torch.long, device=uih_values.device
                )
                payload_features[cand_name] = torch.zeros(
                    total_cand, dtype=torch.long, device=uih_values.device
                )
        payload_features["uih_offsets"] = torch.arange(
            0,
            total_uih + self.static_uih_len,
            self.static_uih_len,
            dtype=torch.long,
            device=uih_values.device,
        )
        payload_features["candidate_offsets"] = torch.arange(
            0,
            total_cand + self.static_num_candidates,
            self.static_num_candidates,
            dtype=torch.long,
            device=uih_values.device,
        )
        raw_output = (
            emb_lengths,
            emb_values,
            payload_features,
            self.static_uih_len,
            uih_seq_lengths,
            self.static_num_candidates,
            num_candidates,
            batch_size * self.static_num_candidates,
            batch_size * self.static_uih_len,
        )
        # Pytree-flatten, plant a marker pair on each tensor leaf, unflatten
        # back.  This ensures DummyPreprocess and the dispatch fwd_fn (via
        # pytree.tree_leaves) produce tensors in the same order.
        prefix = _dlrmv3_marker_prefix(oid)
        flat, treedef = pytree.tree_flatten(raw_output)
        idx = 0
        anchored = []
        for leaf in flat:
            if isinstance(leaf, torch.Tensor):
                anchored.append(mark_output_with_bwd(leaf, prefix, idx))
                idx += 1
            else:
                anchored.append(leaf)
        return treedef.unflatten(anchored)


class DlrmV3SparseDispatcher:
    """Orchestrator-style replacement for the legacy ``DlrmV3SparseOverride``.

    ``install_eager`` swaps :class:`DlrmV3Preprocess` for the
    :class:`DummyPreprocess` so tracing sees stable, fake-tensor-friendly
    shapes (the dummy plants forward/backward marker pairs).
    ``install_graph`` finds those markers and installs a graph-break dispatch
    back to the captured original preprocess.
    """

    def __init__(self, dummy_preprocess: DummyPreprocess) -> None:
        self._dummy = dummy_preprocess
        self._orig_preprocess: nn.Module | None = None
        self._num_ranks: int = 1  # Set by compile() or test for loss scaling

    def install_eager(self, model: nn.Module) -> None:
        for fqn, mod in list(model.named_modules()):
            if isinstance(mod, DlrmV3Preprocess):
                self._orig_preprocess = mod
                parent_fqn, leaf = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                setattr(parent, leaf, self._dummy)

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[Any],
    ) -> Tuple[torch.fx.GraphModule, List[Any]]:
        assert self._orig_preprocess is not None, (
            "install_eager() must run before install_graph()"
        )
        install_graph_break_dispatch(
            gm,
            _dlrmv3_marker_prefix(self._dummy.override_id),
            self._orig_preprocess,  # type: ignore[arg-type]
            num_state_tensors=len(flat_state),
            num_ranks=self._num_ranks,
        )
        return gm, flat_state


def create_tiny_dlrmv3(
    batch_size: int,
) -> Tuple["DlrmV3Wrapper", DummyPreprocess, DlrmHSTUConfig]:
    wrapper, hstu_config, table_configs, emb_keys = create_dlrmv3_wrapper(batch_size)
    dummy = DummyPreprocess(
        emb_keys=emb_keys,
        batch_size=batch_size,
        embedding_dim=SMALL_EMBEDDING_DIM,
        static_uih_len=STATIC_UIH_LEN,
        static_num_candidates=STATIC_NUM_CANDIDATES,
        user_emb_feature_names=hstu_config.user_embedding_feature_names,
        item_emb_feature_names=hstu_config.item_embedding_feature_names,
        hstu_configs=hstu_config,
    )
    return wrapper, dummy, hstu_config
