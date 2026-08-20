# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Weight key remapping for RankAGI checkpoints.

Handles two remapping patterns:

1. **Backbone (safetensors):** Checkpoints saved from a ``Gemma3WithEngram``
   wrapper have an extra ``model.`` prefix on all keys (e.g.,
   ``model.model.layers.0.X`` instead of ``model.layers.0.X``).

2. **ClickHead (click_head.pt):** Legacy checkpoints use different attribute
   names than the current ``ClickHead`` class (``projector.net.0`` vs
   ``proj.0``, ``skip.a`` vs ``skip_a``).
"""

from __future__ import annotations

from typing import Any

import torch


def remap_backbone_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "model.",
) -> dict[str, torch.Tensor]:
    """Strip wrapper prefix from backbone weight keys.

    Args:
        state_dict: Raw state dict from ``model.safetensors``.
        prefix: Prefix to strip (default ``"model."`` from Gemma3WithEngram).

    Returns:
        Remapped state dict with prefix stripped.
    """
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        remapped[k[len(prefix) :] if k.startswith(prefix) else k] = v
    return remapped


def remap_click_head_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Remap legacy ClickHead checkpoint keys to current class attribute names.

    Remaps:
        - ``projector.net.0.{weight,bias}`` → ``proj.0.{weight,bias}``
        - ``mlp.net.{0,3}.{weight,bias}`` → ``mlp.{0,3}.{weight,bias}``
        - ``skip.a`` → ``skip_a``
        - ``skip.b`` → ``skip_b``
        - ``dense_fusion.<X>`` → ``dense_fusion.mlp.<X>`` (skipped when key is
          already in the ``dense_fusion.mlp.`` form, so the rule is idempotent).

    Args:
        state_dict: Raw state dict from ``click_head.pt``.

    Returns:
        Remapped state dict matching ``ClickHead`` class attributes.
    """
    remapped: dict[str, Any] = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("projector.net."):
            new_k = "proj." + k[len("projector.net.") :]
        elif k.startswith("mlp.net."):
            new_k = "mlp." + k[len("mlp.net.") :]
        elif k == "skip.a":
            new_k = "skip_a"
        elif k == "skip.b":
            new_k = "skip_b"
        elif k.startswith("dense_fusion.") and not k.startswith("dense_fusion.mlp."):
            new_k = "dense_fusion.mlp." + k[len("dense_fusion.") :]
        remapped[new_k] = v
    return remapped
