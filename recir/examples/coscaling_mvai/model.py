# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""The hammer v3 CoScaling model, shaped for the recir (XLA) compile.

Two changes over the GPU module, neither of which touches the math:

* ``CoScalingRecirModule`` takes the flat tensor arguments ``flatten_batch``
  produces (the same 13 the udd_fm_v3 recir forward takes) and rebuilds the
  batch, so the shared recir_core compile/step helpers work unchanged.
* The dense-token tables are retyped to ``StaticSingleIdEmbeddingCollection``:
  the torchrec lookup reads its per-key split sizes off the device, which XLA
  cannot trace, and the planner must not shard these tables in any case.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
from hammer.ops.bf16_train_utils import set_autocast_factory
from hammer.utils import HammerKernel
from hammer.v3.models.udd_fm_v3.coscaling.coscaling_config import CoScalingModuleConfig
from hammer.v3.models.udd_fm_v3.coscaling.coscaling_train_v3 import (
    BlueReelsVDDCoScalingTrainV3,
)
from minimal_viable_ai.core.ranking_common.configs import MTMLROOTrainBatch
from recir.embedding.split_collection import (
    replicate_embedding_collection,
    ReplicatedEmbeddingCollection,
)
from recir.examples.rankfm_mvai.rankfm_model_utils import _ec_sparse_arch_on
from recir.torchax.amp import hammer_autocast_factory
from torch._subclasses.fake_tensor import is_fake
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

if TYPE_CHECKING:
    # Annotation only: the image's slimmed fork does not define it.
    from minimal_viable_ai.core.ranking_common.configs import MTMLROOFeatureConfig

logger: logging.Logger = logging.getLogger(__name__)

DENSE_TOKEN_COLLECTIONS: Tuple[str, ...] = ("_item_embeddings", "_user_embeddings")


class StaticSingleIdEmbeddingCollection(ReplicatedEmbeddingCollection):
    """Lookup for features that carry exactly one id per row.

    With one id per row, key ``k``'s ids are the ``stride`` values starting at
    ``offsets[k * stride]``, so the lookup is a fixed-size gather and every
    output is ``[stride, embedding_dim]``. The bucketized dense features
    CoScaling embeds are built that way (one bucket per candidate).
    """

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        keys = features.keys()
        stride = features.stride()
        values = features.values()
        offsets = features.offsets()
        lengths = features.lengths()
        positions = torch.arange(stride, device=values.device)
        out: Dict[str, JaggedTensor] = {}
        for config in self.embedding_configs():
            table = self.embeddings[config.name]
            for feature in config.feature_names:
                k = keys.index(feature)
                key_lengths = lengths[k * stride : (k + 1) * stride]
                if not is_fake(key_lengths) and not bool((key_lengths == 1).all()):
                    raise ValueError(
                        f"{feature} must carry exactly one id per row; the static "
                        "lookup would read the wrong ids otherwise"
                    )
                ids = values.index_select(0, offsets[k * stride] + positions)
                out[feature] = JaggedTensor(values=table(ids), lengths=key_lengths)
        return out


class CoScalingRecirModule(BlueReelsVDDCoScalingTrainV3):
    """CoScaling train module with the flat-tensor forward recir traces."""

    def __init__(
        self,
        module_config: CoScalingModuleConfig,
        feature_config: MTMLROOFeatureConfig,
        is_train: bool = True,
    ) -> None:
        super().__init__(
            module_config=module_config,
            feature_config=feature_config,
            is_train=is_train,
        )
        self._recir_ro_kjt_keys: List[str] = []
        self._recir_nro_kjt_keys: List[str] = []
        self._recir_unary_kjt_keys: List[str] = []

    def set_recir_kjt_keys(
        self,
        ro_kjt_keys: Sequence[str],
        nro_kjt_keys: Sequence[str],
        unary_kjt_keys: Sequence[str] = (),
    ) -> None:
        self._recir_ro_kjt_keys = list(ro_kjt_keys)
        self._recir_nro_kjt_keys = list(nro_kjt_keys)
        self._recir_unary_kjt_keys = list(unary_kjt_keys)

    # pyrefly: ignore [bad-override]
    def forward(
        self,
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
        loss_denom_source: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = MTMLROOTrainBatch(
            supervision_label=supervision_label_t.t().contiguous(),
            supervision_weight=supervision_weight_t.t().contiguous(),
            ro_float_features=ro_float_features,
            nro_float_features=nro_float_features,
            ro_id_list_features=KeyedJaggedTensor(
                keys=self._recir_ro_kjt_keys,
                values=ro_id_list_values,
                lengths=ro_id_list_lengths,
            ),
            nro_id_list_features=KeyedJaggedTensor(
                keys=self._recir_nro_kjt_keys,
                values=nro_id_list_values,
                lengths=nro_id_list_lengths,
            ),
            unary_features=KeyedJaggedTensor(
                keys=self._recir_unary_kjt_keys,
                values=unary_values,
                lengths=unary_lengths,
            ),
            num_candidates=num_candidates,
            all_labels=all_labels,
            ro_embedding_features=None,
            nro_embedding_features=None,
            request_id=None,
            score_time=None,
            B_NRO=nro_float_features.size(0),
        )
        loss_norm_denom = (
            loss_denom_source.sum() if loss_denom_source is not None else None
        )
        return super().forward(batch, loss_norm_denom=loss_norm_denom)


def _materialize_meta(module: torch.nn.Module, device: torch.device) -> int:
    """Replace meta params/buffers with real storage; returns how many."""
    n_meta = 0
    for submod in module.modules():
        for name, param in list(submod.named_parameters(recurse=False)):
            if param.device.type == "meta":
                new = torch.empty(param.shape, dtype=param.dtype, device=device)
                torch.nn.init.normal_(new, 0.0, 0.02)
                submod.register_parameter(
                    name, torch.nn.Parameter(new, requires_grad=param.requires_grad)
                )
                n_meta += 1
        for name, buf in list(submod.named_buffers(recurse=False)):
            if buf is not None and buf.device.type == "meta":
                submod.register_buffer(
                    name, torch.zeros(buf.shape, dtype=buf.dtype, device=device)
                )
                n_meta += 1
    return n_meta


def build_model(
    module_config: CoScalingModuleConfig,
    feature_config: MTMLROOFeatureConfig | Any,
) -> CoScalingRecirModule:
    """Build the recir CoScaling module on CPU with pure PyTorch kernels."""
    device = torch.device("cpu")
    with _ec_sparse_arch_on(device):
        model = CoScalingRecirModule(
            module_config=module_config,
            feature_config=feature_config,
            is_train=True,
        )
    # DenseTokenTokenizer builds its collections on meta and relies on DMP to
    # materialize them; here nothing shards them, so materialize explicitly.
    pipeline: Any = model.pipeline
    tokenizer: torch.nn.Module = pipeline.tokenizer_stack.dense_token_tokenizer
    n_meta = _materialize_meta(tokenizer, device)
    for attr in DENSE_TOKEN_COLLECTIONS:
        if getattr(tokenizer, attr) is not None:
            replicate_embedding_collection(
                tokenizer, attr, StaticSingleIdEmbeddingCollection
            )
    model.to(device)

    # Triton/CUDA kernels have no TPU lowering; hammer.autocast must route to
    # torchax AMP. Same settings as rankfm_model_utils.build_model_from_config.
    model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
    model.recursive_setattr("_custom_kernel", False)
    set_autocast_factory(hammer_autocast_factory)
    model.recursive_setattr("_bf16_training", False)
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
    logger.info(
        "[coscaling] built model: %d params, %d dense-token meta tensors materialized",
        sum(p.numel() for p in model.parameters()),
        n_meta,
    )
    return model
