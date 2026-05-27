# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Region Transforms: Graph-level subgraph rewriting for RecIR."""

from recir.region_transforms.base import RegionTransform
from recir.region_transforms.core import (
    _apply_region_transforms,
    _compute_region_io,
    _discover_autograd_functions,
    _discover_called_modules,
    _extract_subgraph,
    _find_forward_roots,
    _forward_pre_hooks,
    _preserve_autograd_function_leaves,
    _replace_region,
    _topo_sort_nodes,
    extract_subgraph,
    replace_region,
)
from recir.region_transforms.recompute import RecomputeTransform
from recir.region_transforms.regather import RegatherTransform

__all__ = [
    "RegionTransform",
    "RegatherTransform",
    "RecomputeTransform",
    "_apply_region_transforms",
    "_compute_region_io",
    "_discover_autograd_functions",
    "_discover_called_modules",
    "_extract_subgraph",
    "_find_forward_roots",
    "_forward_pre_hooks",
    "_preserve_autograd_function_leaves",
    "_replace_region",
    "_topo_sort_nodes",
    "extract_subgraph",
    "replace_region",
]
