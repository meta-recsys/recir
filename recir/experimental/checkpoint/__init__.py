# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from recir.experimental.checkpoint.safetensors_loader import load_safetensors
from recir.experimental.checkpoint.weight_remapping import (
    remap_backbone_state_dict,
    remap_click_head_state_dict,
)

__all__ = [
    "load_safetensors",
    "remap_backbone_state_dict",
    "remap_click_head_state_dict",
]
