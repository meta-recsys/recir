# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

_META_INIT_KIND_ATTR: str = "_recir_meta_init_kind"
_META_INIT_BOUND_ATTR: str = "_recir_meta_init_bound"


def mark_meta_uniform_init(tensor: Any, bound: float) -> None:
    setattr(tensor, _META_INIT_KIND_ATTR, "uniform")
    setattr(tensor, _META_INIT_BOUND_ATTR, bound)


def mark_meta_zero_init(tensor: Any) -> None:
    if tensor.is_meta:
        setattr(tensor, _META_INIT_KIND_ATTR, "zeros")


def copy_meta_init_attrs(dst: Any, src: Any) -> None:
    if not (dst.is_meta and src.is_meta):
        return
    for attr in (_META_INIT_KIND_ATTR, _META_INIT_BOUND_ATTR):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def get_meta_init_spec(tensor: Any, default_bound: float) -> tuple[str, float]:
    kind = getattr(tensor, _META_INIT_KIND_ATTR, "uniform")
    raw_bound = getattr(tensor, _META_INIT_BOUND_ATTR, None)
    if kind == "uniform" and raw_bound is None:
        # The bound is normally populated via ``mark_meta_uniform_init``. If a
        # caller forgets to mark the tensor, the fallback default (derived from
        # the padded/global row count) may not match the intended TorchRec init
        # scale (which uses the pre-padding ``num_embeddings``). Log so the
        # mismatch doesn't go unnoticed.
        logger.warning(
            "meta tensor missing %s; falling back to default uniform bound "
            "%g, which may not match the intended init scale. Mark the tensor "
            "via mark_meta_uniform_init to set the bound explicitly.",
            _META_INIT_BOUND_ATTR,
            default_bound,
        )
    return kind, float(default_bound if raw_bound is None else raw_bound)
