# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class GSPMDMeshContext:
    mesh: Any
    batch_axis_resource: Any
    fsdp_axis_size: int
    hybrid_fsdp: bool


_GSPMD_MESH_CONTEXT: contextvars.ContextVar[GSPMDMeshContext | None] = (
    contextvars.ContextVar("recir_gspmd_mesh_context", default=None)
)


@contextlib.contextmanager
def gspmd_mesh_context(
    mesh: Any,
    batch_axis_resource: Any,
    fsdp_axis_size: int,
    hybrid_fsdp: bool,
) -> Iterator[None]:
    """Expose one GSPMD lowering mesh to nested TorchAX operator lowerings."""
    if fsdp_axis_size <= 0:
        raise ValueError("GSPMD FSDP axis size must be positive")
    token = _GSPMD_MESH_CONTEXT.set(
        GSPMDMeshContext(
            mesh=mesh,
            batch_axis_resource=batch_axis_resource,
            fsdp_axis_size=fsdp_axis_size,
            hybrid_fsdp=hybrid_fsdp,
        )
    )
    try:
        yield
    finally:
        _GSPMD_MESH_CONTEXT.reset(token)


def current_gspmd_mesh_context() -> GSPMDMeshContext | None:
    return _GSPMD_MESH_CONTEXT.get()
