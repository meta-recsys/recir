# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any

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
    return kind, float(default_bound if raw_bound is None else raw_bound)
