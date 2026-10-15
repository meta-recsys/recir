# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Collection

import torch


MetaParameterMaterializer = Callable[[str, torch.nn.Parameter], torch.Tensor | None]


def materialize_recir_parameters(
    model: torch.nn.Module,
    device: torch.device,
    deferred_param_fqns: Collection[str],
    materialize_meta_parameter: MetaParameterMaterializer,
) -> list[str]:
    """Materialize parameters while preserving shared parameter aliases.

    Returns the FQNs of meta parameters not handled by
    ``materialize_meta_parameter``.
    """
    uninitialized_meta_params: list[str] = []
    materialized_parameters: dict[int, torch.nn.Parameter] = {}
    for fqn, param in list(model.named_parameters(remove_duplicate=False)):
        if fqn in deferred_param_fqns and param.device.type == "meta":
            continue
        atoms = fqn.split(".")
        parent: torch.nn.Module = model
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        attr = atoms[-1]
        if id(param) in materialized_parameters:
            parent._parameters[attr] = materialized_parameters[id(param)]
            continue
        if param.device.type != "meta":
            materialized = param.data.to(device)
        else:
            materialized = materialize_meta_parameter(fqn, param)
            if materialized is None:
                materialized = torch.empty(
                    param.shape, device=device, dtype=param.dtype
                )
                uninitialized_meta_params.append(fqn)
        materialized_parameter = torch.nn.Parameter(
            materialized, requires_grad=param.requires_grad
        )
        materialized_parameters[id(param)] = materialized_parameter
        parent._parameters[attr] = materialized_parameter
    return uninitialized_meta_params


def materialize_recir_buffers(
    model: torch.nn.Module, device: torch.device
) -> list[str]:
    """Move buffers to ``device``. Returns the FQNs of meta buffers, which are
    left as uninitialized ``torch.empty`` for the caller to reject."""
    meta_buffers: list[str] = []
    for fqn, buf in list(model.named_buffers()):
        if buf is None:
            continue
        if buf.device.type == "meta":
            meta_buffers.append(fqn)
        atoms = fqn.split(".")
        parent: torch.nn.Module = model
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        attr = atoms[-1]
        parent._buffers[attr] = (
            buf.to(device)
            if buf.device.type != "meta"
            else torch.empty(buf.shape, device=device, dtype=buf.dtype)
        )
    return meta_buffers
