#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""Ship prod's config into the TPU image, AS A CLOUD CONFIG.

Usage:
    # devserver, under buck (needs the full //minimal_viable_ai):
    buck2 run \\
        //recir/examples/rankfm_mvai:export_prod_config -- --out /tmp/prod_cfg.pt

    # in-image: set PROD_CONFIG_ARTIFACT=/app/prod_cfg.pt and pass
    # --prod_faithful; config._load_prod_module_config picks it up.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import logging
import os
from typing import Any, Dict, get_args, List, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

# Bump whenever the layout changes. A mismatch is a hard error: a misread config
# builds a model that looks right and trains differently.
ARTIFACT_VERSION: int = 1

_ENUM_TAG: str = "__enum_name__"

ARTIFACT_PATH_ENV: str = "PROD_CONFIG_ARTIFACT"

# Tags for the value transport. A tagged node is a plain dict, so nothing in the
# payload is a custom class and the stream carries no module paths of its own.
_TYPE_TAG: str = "__type__"
_FIELDS_KEY: str = "fields"
_ENUM_TYPE_TAG: str = "__enum__"
_ENUM_NAME_KEY: str = "name"
_CLASS_TAG: str = "__class__"
_DEFAULTS_KEY: str = "__defaults__"
_NAMEDTUPLE_TAG: str = "__namedtuple__"

# Reached with importlib rather than imported: a plain import would break every
# in-image run, where there is no mvai_cloud package. The BUCK deps are held in
# place by @autodeps-skip, since autodeps cannot see an importlib call.
_FORK_PREFIX: str = "mvai_cloud."

# Values that carry no class identity and can be written as they are.
_PRIMITIVE: Tuple[type, ...] = (bool, int, float, str, type(None))


def _image_path(cls: type) -> str:
    """The dotted path ``cls`` will have INSIDE THE IMAGE.

    The fork is rsynced in as ``minimal_viable_ai/`` (``scripts/Dockerfile:9``),
    so a class this process knows as ``mvai_cloud.minimal_viable_ai.X`` is
    ``minimal_viable_ai.X`` there. Strip the prefix; everything else (hammer,
    torchrec) has the same path in both places.
    """
    module = cls.__module__
    if module.startswith(_FORK_PREFIX):
        module = module[len(_FORK_PREFIX) :]
    return f"{module}.{cls.__qualname__}"


def _resolve(path: str) -> Any:
    """Import a dotted path recorded by :func:`_image_path`."""
    module_path, _, name = path.rpartition(".")
    for candidate in (
        (_FORK_PREFIX + module_path,)
        if module_path.startswith("minimal_viable_ai.")
        else ()
    ) + (module_path,):
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue
        resolved = getattr(module, name, None)
        if resolved is not None:
            return resolved
    raise ImportError(
        f"the artifact references {path!r}, which is not importable here. "
        f"Either the artifact was produced against a different tree, or the "
        f"type needs carrying into mvai_cloud."
    )


def _cloud_counterpart(cls: type) -> Optional[type]:
    module = cls.__module__
    if not module.startswith("minimal_viable_ai."):
        # hammer / torchrec / fbgemm are the same package in both environments.
        return None
    try:
        fork_module = importlib.import_module(_FORK_PREFIX + module)
    except ImportError as e:
        raise TypeError(
            f"{module}.{cls.__name__} has no mvai_cloud counterpart module, so "
            f"the image cannot rebuild it. Carry it into "
            f"mvai_cloud/{module.replace('.', '/')}.py first."
        ) from e
    cloud_cls = getattr(fork_module, cls.__name__, None)
    if cloud_cls is None:
        raise TypeError(
            f"mvai_cloud.{module} exists but does not define {cls.__name__}. "
            f"Carry the class into the fork first."
        )
    if not cloud_cls.__module__.startswith(_FORK_PREFIX):
        raise TypeError(
            f"mvai_cloud.{module}.{cls.__name__} resolves to "
            f"{cloud_cls.__module__}, which is the INTERNAL tree. Converting "
            f"against it would be a no-op that reports success. Fix the fork "
            f"module's imports before using it as a conversion target."
        )
    return cloud_cls


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(type(value), "_fields")


def _field_default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _holds_default(value: Any, f: dataclasses.Field) -> bool:
    default = _field_default(f)
    if default is dataclasses.MISSING:
        return False
    try:
        return bool(value == default)
    except Exception:
        return value is default


def _to_cloud(prod: Any, cloud_cls: type) -> Any:
    """Build ``cloud_cls`` from a prod object of the same name.

    Carries only fields ``cloud_cls`` declares. Dropping the rest is what the
    fork's slimming means -- but a NON-DEFAULT value in a dropped field is an
    ERROR, since it is a prod setting that would silently not apply in the
    image. That rule is the point of the whole change.
    """
    prod_fields = {f.name: f for f in dataclasses.fields(prod)}
    cloud_fields = {f.name: f for f in dataclasses.fields(cloud_cls)}

    dropped: List[str] = []
    for name, f in prod_fields.items():
        if name in cloud_fields:
            continue
        value = getattr(prod, name)
        if not _holds_default(value, f):
            dropped.append(f"{name}={value!r:.120}")
    if dropped:
        raise NotImplementedError(
            f"{type(prod).__module__}.{type(prod).__name__} sets "
            f"{len(dropped)} field(s) that {_image_path(cloud_cls)} does not "
            f"declare, so they would silently not apply in the image: "
            + "; ".join(sorted(dropped))
            + ". Carry the field (and its types) into mvai_cloud first."
        )

    kwargs: Dict[str, Any] = {}
    for name in cloud_fields:
        if name in prod_fields:
            kwargs[name] = _convert_value(getattr(prod, name))
    # Constructed, not copied, so the fork's __init__/__post_init__ validation
    # fires here rather than never.
    return cloud_cls(**kwargs)


def _convert_value(value: Any) -> Any:
    """Recursively convert prod values into their cloud equivalents."""
    # Same ordering caveat as _encode, though harmless here: both branches
    # return the value unchanged. Kept explicit so the two functions read the
    # same way.
    if isinstance(value, enum.Enum) or isinstance(value, _PRIMITIVE):
        return value
    if isinstance(value, type):
        return value
    if _is_namedtuple(value):
        return type(value)(
            **{n: _convert_value(getattr(value, n)) for n in value._fields}
        )
    if isinstance(value, tuple):
        return tuple(_convert_value(v) for v in value)
    if isinstance(value, list):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        counterpart = _cloud_counterpart(type(value))
        if counterpart is None:
            return value
        return _to_cloud(value, counterpart)
    return value


def _encode(value: Any, path: str) -> Any:
    """Turn a config object into primitives and containers. Nothing this returns is a custom class, so the ``torch.save`` stream records
    no module path and nothing in it can resolve to a different class of the
    same name on the other side. Lists and tuples stay distinct because pickle
    represents both natively -- that is why the transport is still torch.save
    and not JSON.
    """
    if isinstance(value, enum.Enum):
        return {
            _ENUM_TYPE_TAG: _image_path(type(value)),
            _ENUM_NAME_KEY: value.name,
        }
    if isinstance(value, _PRIMITIVE):
        return value
    if isinstance(value, type):
        return {_CLASS_TAG: _image_path(value)}
    if _is_namedtuple(value):
        if _cloud_counterpart(type(value)) is not None:
            raise NotImplementedError(
                f"{path}: {_image_path(type(value))} is an mvai NamedTuple with "
                f"a fork counterpart. _to_cloud only converts dataclasses, so "
                f"this would be carried WITHOUT the dropped-field check. "
                f"Measured: no such type exists today; implement the conversion "
                f"before adding one."
            )
        return {
            _NAMEDTUPLE_TAG: _image_path(type(value)),
            _FIELDS_KEY: {
                n: _encode(getattr(value, n), f"{path}.{n}") for n in value._fields
            },
        }
    if isinstance(value, tuple):
        return tuple(_encode(v, f"{path}[{i}]") for i, v in enumerate(value))
    if isinstance(value, list):
        return [_encode(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, (str, int, bool)):
                raise TypeError(
                    f"{path}: dict key {k!r} is a {type(k).__name__}, which "
                    f"cannot be carried as plain data."
                )
        return {k: _encode(v, f"{path}[{k!r}]") for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        # Converted during the walk rather than in a separate pass, so a prod
        # type nested inside a hammer type is converted too.
        counterpart = _cloud_counterpart(type(value))
        if counterpart is not None:
            value = _to_cloud(value, counterpart)
        fields = dataclasses.fields(value)
        return {
            _TYPE_TAG: _image_path(type(value)),
            _FIELDS_KEY: {
                f.name: _encode(getattr(value, f.name), f"{path}.{f.name}")
                for f in fields
            },
            # Which fields hold the default THIS tree declares -- only the
            # exporter knows, and the decoder needs it to tell "safe to drop"
            # from "prod set this". Checking falsiness there instead would miss
            # a non-None default.
            _DEFAULTS_KEY: sorted(
                f.name for f in fields if _holds_default(getattr(value, f.name), f)
            ),
        }
    # Deliberately not a silent skip. A value that cannot be carried is a config
    # setting that would not apply in the image, which is the failure mode this
    # module exists to prevent.
    raise TypeError(
        f"{path}: cannot carry a {type(value).__module__}."
        f"{type(value).__name__} as plain data. If it is inert, give it a "
        f"default and let the dropped-field rule handle it; if it matters, "
        f"carry it into mvai_cloud."
    )


def _decode_dataclass(node: Dict[str, Any], path: str) -> Tuple[type, Dict[str, Any]]:
    """Resolve a ``__type__`` node to its local class and decoded field values."""
    cls = _resolve(node[_TYPE_TAG])
    carried = node[_FIELDS_KEY]
    defaulted = set(node.get(_DEFAULTS_KEY, ()))
    local = {f.name for f in dataclasses.fields(cls)}

    missing = sorted(local - set(carried))
    if missing:
        raise ValueError(
            f"{path}: {node[_TYPE_TAG]} declares {missing} here, which the "
            f"artifact does not carry. Regenerate it against this tree."
        )

    set_but_undeclared = sorted(k for k in set(carried) - local if k not in defaulted)
    if set_but_undeclared:
        raise ValueError(
            f"{path}: prod sets "
            + ", ".join(f"{k}={carried[k]!r:.60}" for k in set_but_undeclared)
            + f" on {node[_TYPE_TAG]}, which this environment's version of that "
            f"class does not declare -- the setting would not apply here. "
            f"Carry the field into this environment, or stop setting it."
        )

    dropped = sorted(set(carried) - local)
    if dropped:
        logger.info(
            "%s: dropping %s from %s -- not declared in this environment's "
            "version, and prod left them at their default",
            path,
            dropped,
            node[_TYPE_TAG],
        )
    return cls, {k: _decode(v, f"{path}.{k}") for k, v in carried.items() if k in local}


def _decode(node: Any, path: str = "payload") -> Any:
    """Rebuild what :func:`_encode` wrote."""
    if isinstance(node, dict):
        if _TYPE_TAG in node:
            cls, values = _decode_dataclass(node, path)
            return cls(**values)
        if _ENUM_TYPE_TAG in node:
            return _resolve(node[_ENUM_TYPE_TAG])[node[_ENUM_NAME_KEY]]
        if _NAMEDTUPLE_TAG in node:
            cls = _resolve(node[_NAMEDTUPLE_TAG])
            carried = node[_FIELDS_KEY]
            local = set(cls._fields)
            if set(carried) != local:
                raise ValueError(
                    f"{path}: the artifact and this environment disagree about "
                    f"{node[_NAMEDTUPLE_TAG]}. Carried but not declared here: "
                    f"{sorted(set(carried) - local)}. Declared here but not "
                    f"carried: {sorted(local - set(carried))}."
                )
            return cls(**{k: _decode(v, f"{path}.{k}") for k, v in carried.items()})
        if _CLASS_TAG in node:
            return _resolve(node[_CLASS_TAG])
        return {k: _decode(v, f"{path}[{k!r}]") for k, v in node.items()}
    if isinstance(node, tuple):
        return tuple(_decode(v, f"{path}[{i}]") for i, v in enumerate(node))
    if isinstance(node, list):
        return [_decode(v, f"{path}[{i}]") for i, v in enumerate(node)]
    return node


def _neutralize(value: Any) -> Any:
    """Convert a table field value into something safe to unpickle anywhere.

    Enums are the problem case: an internal ``LocationType`` member pickles as
    that class, which does not exist in the image. Carry the member NAME and
    re-resolve against whichever enum class the loading side declares.
    """
    if isinstance(value, enum.Enum):
        return {_ENUM_TAG: value.name}
    if isinstance(value, (list, tuple)):
        return [_neutralize(v) for v in value]
    if isinstance(value, dict):
        return {k: _neutralize(v) for k, v in value.items()}
    return value


def _enum_target(target_type: Any) -> Optional[type[enum.Enum]]:
    """The Enum class a field is declared as, looking through ``Optional``/``Union``.

    Annotations arrive as written, so ``Optional[ShardingType]`` fails a plain
    ``issubclass``. Without the unwrap an Optional enum field rehydrates as its
    NAME STRING, giving readers a third state nobody handles: ``== DATA_PARALLEL``
    is False while ``is not None`` is True, and asserts keyed on the enum stop
    firing.
    """
    if isinstance(target_type, type) and issubclass(target_type, enum.Enum):
        return target_type
    for arg in get_args(target_type):
        if isinstance(arg, type) and issubclass(arg, enum.Enum):
            return arg
    return None


def _rehydrate(value: Any, target_type: Any) -> Any:
    if isinstance(value, dict) and _ENUM_TAG in value:
        enum_cls = _enum_target(target_type)
        if enum_cls is not None:
            return enum_cls[value[_ENUM_TAG]]
        # The local class no longer types this field as an enum at all; hand
        # back the name rather than guessing.
        return value[_ENUM_TAG]
    if isinstance(value, list):
        return [_rehydrate(v, None) for v in value]
    if isinstance(value, dict):
        return {k: _rehydrate(v, None) for k, v in value.items()}
    return value


def _table_to_dict(table: Any) -> Dict[str, Any]:
    """Flatten one EmbeddingTable to a plain dict."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(table):
        value = getattr(table, f.name)
        if callable(value) and not isinstance(value, type):
            # e.g. init_fn -- not picklable and not readable in-image.
            continue
        out[f.name] = _neutralize(value)
    return out


def _dict_to_table(spec: Dict[str, Any], table_cls: Any, name: str) -> Any:
    """Rebuild an EmbeddingTable against the LOCAL class definition."""
    local_fields = {f.name: f for f in dataclasses.fields(table_cls)}
    kwargs: Dict[str, Any] = {}
    missing_required: list[str] = []
    for fname, f in local_fields.items():
        if fname in spec:
            kwargs[fname] = _rehydrate(spec[fname], f.type)
            continue
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        )
        if not has_default:
            missing_required.append(fname)
    if missing_required:
        raise ValueError(
            f"embedding table {name!r}: the local EmbeddingTable requires "
            f"{missing_required}, which the artifact does not carry. The "
            f"artifact was produced against a different EmbeddingTable "
            f"definition; regenerate it."
        )
    return table_cls(**kwargs)


def _feature_config_primitives(fc: Any) -> Dict[str, Any]:
    """Primitive-only snapshot of a feature config."""
    out: Dict[str, Any] = {}
    for name in sorted(x for x in dir(fc) if not x.startswith("_")):
        try:
            v = getattr(fc, name)
        except Exception:
            continue
        if callable(v):
            continue
        if isinstance(v, (int, float, bool, str, type(None))):
            out[name] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (int, float, bool, str, type(None))) for x in v
        ):
            out[name] = list(v)
        elif isinstance(v, dict) and all(
            isinstance(k, str) and isinstance(x, (int, float, bool, str, type(None)))
            for k, x in v.items()
        ):
            out[name] = dict(v)
    return out


def _apply_feature_config_fields(fc: Any, carried: Dict[str, Any]) -> Any:
    """Copy carried primitives onto the local feature config; log the gap."""
    # Fields ABSENT on the local class are ADDED, not skipped: the two classes
    # are nearly disjoint, and gating on hasattr() applied only 5 of 43 -- the
    # 38 dropped included feature_merge_weights, which the hammer model reads.
    overwrote, added, failed = [], [], []
    for k, v in carried.items():
        existed = hasattr(fc, k)
        try:
            setattr(fc, k, v)
            (overwrote if existed else added).append(k)
        except Exception:
            failed.append(k)
    logger.info(
        f"feature config: carried {len(carried)} prod fields onto "
        f"{type(fc).__name__} -- overwrote {len(overwrote)}, added "
        f"{len(added)}, failed {len(failed)}"
        + (f" (failed: {sorted(failed)[:8]})" if failed else "")
    )
    return fc


def _encode_module_config(module_config: Any) -> Dict[str, Any]:
    """Encode ``module_config``, with ``embedding_tables`` emptied."""
    fields: Dict[str, Any] = {}
    for f in dataclasses.fields(module_config):
        if f.name == "embedding_tables":
            fields[f.name] = {}
            continue
        fields[f.name] = _encode(
            getattr(module_config, f.name), f"module_config.{f.name}"
        )
    return {_TYPE_TAG: _image_path(type(module_config)), _FIELDS_KEY: fields}


def _decode_module_config(node: Dict[str, Any], tables: Dict[str, Any]) -> Any:
    """Rebuild ``module_config`` with its embedding tables already in place."""
    cls, values = _decode_dataclass(node, "module_config")
    values["embedding_tables"] = tables
    return cls(**values)


def build_prod_config_live(for_parity: bool = False) -> Tuple[Any, Any]:
    """Build ``(module_config, feature_config)`` from the PROD provider, now.
    Devserver only: this is the path that needs the full //minimal_viable_ai.
    :func:`export_prod_config` writes what this returns, and ``config`` falls
    back to it when no artifact is configured -- the two callers of the same
    "no artifact available" case.

    ``for_parity`` swaps in the config the light leg of a two-leg comparison
    builds from, which pins a shorter sequence length and a tighter hash cap
    than prod. Off by default: plain ``--prod_faithful`` means PROD's values,
    and a test asserts this path reproduces them.
    """
    try:
        # @manual keeps the //minimal_viable_ai chain out of :train's buck deps
        # for a path it never takes. pyre resolves imports against the build
        # source database and so cannot see these either.
        # pyre-ignore[21]
        from hammer.v3.models.udd_fm_v3.launch.light_replay_config_shim import (  # @manual
            ConfigProvider as LightReplayConfigProvider,
        )

        # pyre-ignore[21]
        from hammer.v3.models.udd_fm_v3.launch.prod_config_shim import (  # @manual
            ConfigProvider as ProdXSurfaceConfigProvider,
        )

        # pyre-ignore[21]
        from hammer.v3.models.udd_fm_v3.launch.train_module_wrapper import (  # @manual
            _translate_module_config,
        )
    except ImportError as e:
        raise ImportError(
            "--prod_faithful needs the full //minimal_viable_ai package, which "
            "is not staged into the rankfm TPU image. Run under buck on a "
            "devserver, or generate an artifact with "
            "//recir/examples/rankfm_mvai:export_prod_config and point "
            "PROD_CONFIG_ARTIFACT at it."
        ) from e

    provider = LightReplayConfigProvider if for_parity else ProdXSurfaceConfigProvider
    prod_config = provider().create_config()
    module_config = _translate_module_config(
        prod_module_config=prod_config.module_config,
        feature_config=prod_config.feature_config,
        embedding_tables=prod_config.embedding_tables,
    )
    return module_config, prod_config.feature_config


def export_prod_config(out_path: str, for_parity: bool = False) -> None:
    """Build prod's config here, convert it to the CLOUD config, and write it.
    Devserver only -- this is the side that needs the full //minimal_viable_ai.

    ``for_parity`` must match what :func:`build_prod_config_live` is called
    with everywhere else: the artifact is only useful if it rebuilds the same
    model the live path would, and a round-trip test pins that.
    """
    module_config, _feature_config = build_prod_config_live(for_parity=for_parity)
    optimizer_config = _load_prod_optimizer_config()

    tables = {
        name: _table_to_dict(table)
        for name, table in module_config.embedding_tables.items()
    }
    payload: Dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "module_config": _encode_module_config(module_config),
        "optimizer_config": _encode(optimizer_config, "optimizer_config"),
        "embedding_tables": tables,
        "feature_config_fields": _feature_config_primitives(_feature_config),
        "provenance": {
            # The seq len is baked into the config, so record which one this
            # artifact represents -- a mismatch against the dump is silent.
            "hstu_max_seq_len": module_config.hstu_max_seq_len,
            "num_tables": len(tables),
        },
    }
    torch.save(payload, out_path)
    logger.info(
        f"wrote cloud config artifact to {out_path} "
        f"({len(tables)} embedding tables, "
        f"hstu_max_seq_len={module_config.hstu_max_seq_len})"
    )


def _check_version(payload: Dict[str, Any], path: str) -> None:
    version = payload.get("version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"prod config artifact at {path} is version {version}, this code "
            f"expects {ARTIFACT_VERSION}. Regenerate it with "
            f"//recir/examples/rankfm_mvai:export_prod_config."
        )


def _load_prod_optimizer_config() -> Any:
    """Prod's optimizer config, CONVERTED to the mvai_cloud classes."""
    # pyre-ignore[21]
    from hammer.v3.models.udd_fm_v3.launch.prod_config_shim import (  # @manual
        ConfigProvider as ProdXSurfaceConfigProvider,
    )

    prod = ProdXSurfaceConfigProvider().create_config().optimizer_config
    cloud_cls = _cloud_counterpart(type(prod))
    assert cloud_cls is not None, (
        f"{type(prod).__module__} is not an mvai module, so there is nothing "
        f"to convert against; the prod config provider changed shape."
    )
    return _to_cloud(prod, cloud_cls)


def load_prod_config(path: str) -> Tuple[Any, Any]:
    """Load an artifact written by :func:`export_prod_config`."""
    from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
    from minimal_viable_ai.core.model_family_api.configs import EmbeddingTable

    # weights_only stays at its default: there is no custom class in the
    # payload to allow-list any more.
    payload = torch.load(path)
    _check_version(payload, path)

    tables = {
        name: _dict_to_table(spec, EmbeddingTable, name)
        for name, spec in payload["embedding_tables"].items()
    }
    module_config = _decode_module_config(payload["module_config"], tables)
    provenance = payload.get("provenance", {})
    logger.info(
        f"loaded cloud config artifact from {path} "
        f"({len(tables)} embedding tables, provenance={provenance})"
    )
    feature_config = _apply_feature_config_fields(
        BlueReelsVDDFeatureConfig(), payload.get("feature_config_fields", {})
    )
    return module_config, feature_config


def load_prod_optimizer_config(path: str) -> Any:
    """The optimizer config carried by the artifact, rebuilt locally."""
    payload = torch.load(path)
    _check_version(payload, path)
    return _decode(payload["optimizer_config"], "optimizer_config")


def artifact_path() -> Optional[str]:
    """Configured artifact path, if one is set and present on disk."""
    path = os.environ.get(ARTIFACT_PATH_ENV)
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{ARTIFACT_PATH_ENV}={path} is set but the file does not exist."
        )
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="artifact output path")
    parser.add_argument(
        "--parity",
        action="store_true",
        help="Build from the parity config the light leg of a two-leg "
        "comparison uses (shorter sequence length, tighter hash cap) rather "
        "than prod's own values.",
    )
    args = parser.parse_args()
    export_prod_config(args.out, for_parity=args.parity)
