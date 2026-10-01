# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""A self-contained CoScaling config + batch artifact for the TPU image."""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace
from typing import Any, Dict, NamedTuple, Optional

import torch
from recir.examples.rankfm_mvai.prod_config_artifact import (
    _decode,
    _decode_dataclass,
    _dict_to_table,
    _encode,
    _FIELDS_KEY,
    _image_path,
    _table_to_dict,
    _TYPE_TAG,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# Bump whenever the layout changes; a mismatch is a hard error.
ARTIFACT_VERSION: int = 1

_TABLE_FIELDS: tuple[str, ...] = ("embedding_tables", "dense_token_embedding_tables")

# Fields carried as an empty value. stu_stack_configs names the HSTU stacks
# (e.g. CCHSTU) the udd base config inherits: nothing on the CoScaling path
# reads it -- only build_recblock_from_xsurface_config does -- and carrying it
# would make the image resolve classes it does not stage.
_EMPTIED_FIELDS: Dict[str, Any] = {"stu_stack_configs": []}


class CoScalingArtifact(NamedTuple):
    module_config: Any
    feature_config: Any
    optimizer_config: Any
    batch: Any
    provenance: Dict[str, Any]


def _encode_kjt(kjt: Optional[KeyedJaggedTensor]) -> Dict[str, Any]:
    if kjt is None:
        return {
            "keys": [],
            "values": torch.empty(0, dtype=torch.int64),
            "lengths": torch.empty(0, dtype=torch.int32),
        }
    return {
        "keys": list(kjt.keys()),
        "values": kjt.values().cpu(),
        "lengths": kjt.lengths().cpu(),
    }


def _decode_kjt(node: Dict[str, Any]) -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=list(node["keys"]), values=node["values"], lengths=node["lengths"]
    )


def encode_batch(batch: Any) -> Dict[str, Any]:
    """The fields the recir step reads, as tensors and key lists."""
    return {
        "supervision_label": batch.supervision_label.cpu(),
        "supervision_weight": batch.supervision_weight.cpu(),
        "ro_float_features": batch.ro_float_features.cpu(),
        "nro_float_features": batch.nro_float_features.cpu(),
        "ro_id_list_features": _encode_kjt(batch.ro_id_list_features),
        "nro_id_list_features": _encode_kjt(batch.nro_id_list_features),
        "unary_features": _encode_kjt(batch.unary_features),
        "num_candidates": batch.num_candidates.cpu(),
        "all_labels": {k: v.cpu() for k, v in (batch.all_labels or {}).items()},
    }


def decode_batch(node: Dict[str, Any]) -> Any:
    """Rebuild an ``MTMLROOTrainBatch`` against the local class."""
    from minimal_viable_ai.core.ranking_common.configs import MTMLROOTrainBatch

    nro_float = node["nro_float_features"]
    return MTMLROOTrainBatch(
        supervision_label=node["supervision_label"],
        supervision_weight=node["supervision_weight"],
        ro_float_features=node["ro_float_features"],
        nro_float_features=nro_float,
        ro_id_list_features=_decode_kjt(node["ro_id_list_features"]),
        nro_id_list_features=_decode_kjt(node["nro_id_list_features"]),
        unary_features=_decode_kjt(node["unary_features"]),
        num_candidates=node["num_candidates"],
        all_labels=dict(node["all_labels"]),
        ro_embedding_features=None,
        nro_embedding_features=None,
        request_id=None,
        score_time=None,
        B_NRO=nro_float.size(0),
    )


def _encode_module_config(module_config: Any) -> Dict[str, Any]:
    """Encode ``module_config`` with its table dicts carried separately."""
    fields: Dict[str, Any] = {}
    for f in dataclasses.fields(module_config):
        if f.name in _TABLE_FIELDS:
            fields[f.name] = {}
            continue
        if f.name in _EMPTIED_FIELDS:
            fields[f.name] = _EMPTIED_FIELDS[f.name]
            continue
        fields[f.name] = _encode(
            getattr(module_config, f.name), f"module_config.{f.name}"
        )
    return {_TYPE_TAG: _image_path(type(module_config)), _FIELDS_KEY: fields}


def _referenced_paths(node: Any, out: set[str]) -> set[str]:
    """Every class path the payload asks the loading side to resolve."""
    if isinstance(node, dict):
        for tag in ("__type__", "__enum__", "__class__", "__namedtuple__"):
            if isinstance(node.get(tag), str):
                out.add(node[tag])
        for value in node.values():
            _referenced_paths(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _referenced_paths(value, out)
    return out


def save_artifact(
    out_path: str,
    module_config: Any,
    feature_config: Any,
    optimizer_config: Any,
    batch: Any,
    provenance: Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "module_config": _encode_module_config(module_config),
        "tables": {
            field: {
                name: _table_to_dict(table)
                for name, table in getattr(module_config, field).items()
            }
            for field in _TABLE_FIELDS
        },
        "optimizer_config": _encode(optimizer_config, "optimizer_config"),
        # The model reads only these two from its feature config.
        "feature_config": {
            "task_names": list(feature_config.task_names),
            "regression_task_names": list(feature_config.regression_task_names),
        },
        "batch": encode_batch(batch),
        "provenance": provenance,
    }
    torch.save(payload, out_path)
    logger.info(
        "[coscaling] artifact references: %s",
        " ".join(sorted(_referenced_paths(payload, set()))),
    )
    logger.info("[coscaling] wrote artifact %s (provenance=%s)", out_path, provenance)


def load_artifact(path: str) -> CoScalingArtifact:
    from minimal_viable_ai.core.model_family_api.configs import EmbeddingTable

    payload = torch.load(path, weights_only=False)
    version = payload.get("version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"CoScaling artifact {path} is version {version}; this code expects "
            f"{ARTIFACT_VERSION}. Regenerate it with export_artifact."
        )
    cls, values = _decode_dataclass(payload["module_config"], "module_config")
    for field in _TABLE_FIELDS:
        values[field] = {
            name: _dict_to_table(spec, EmbeddingTable, name)
            for name, spec in payload["tables"][field].items()
        }
    module_config = cls(**values)
    artifact = CoScalingArtifact(
        module_config=module_config,
        feature_config=SimpleNamespace(**payload["feature_config"]),
        optimizer_config=_decode(payload["optimizer_config"], "optimizer_config"),
        batch=decode_batch(payload["batch"]),
        provenance=payload.get("provenance", {}),
    )
    logger.info(
        "[coscaling] loaded artifact %s (provenance=%s)", path, artifact.provenance
    )
    return artifact


def check(path: str) -> str:
    artifact = load_artifact(path)
    mc = artifact.module_config
    if not mc.static_shape_mode:
        raise ValueError(f"CoScaling artifact {path} has static_shape_mode off")
    if (
        mc.coscaling_static_max_ro_len is None
        or mc.coscaling_static_max_num_targets is None
    ):
        raise ValueError(f"CoScaling artifact {path} has no encoder static bounds")
    return (
        f"CoScaling, {len(mc.embedding_tables)} tables, "
        f"B_NRO={artifact.batch.nro_float_features.size(0)}, "
        f"static_max_ro_len={mc.coscaling_static_max_ro_len}, "
        f"static_max_num_targets={mc.coscaling_static_max_num_targets}"
    )
