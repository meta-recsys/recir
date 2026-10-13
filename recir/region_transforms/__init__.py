# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Region Transforms: Graph-level subgraph rewriting for RecIR."""

from recir.region_transforms.base import RegionTransform
from recir.region_transforms.core import (
    _apply_region_transforms,
    _compute_region_io,
    _discover_called_modules,
    _extract_subgraph,
    _find_forward_roots,
    _forward_pre_hooks,
    _preserve_all_autograd_function_leaves,
    _replace_region,
    _topo_sort_nodes,
    extract_subgraph,
    replace_region,
)
from recir.region_transforms.dropout_prng import DropoutPRNGTransform
from recir.region_transforms.inference_swap import (
    apply_inference_swaps,
    deterministic_argmax,
    dtype_cast_matcher,
    identity_passthrough,
    InferenceSwapTransform,
)
from recir.region_transforms.recompute import RecomputeTransform
from recir.region_transforms.regather import RegatherTransform
from recir.region_transforms.sparse_embedding import SparseEmbeddingTransform
from recir.region_transforms.training_to_inference import (
    fabricate_inference_graph,
    inputs_to_ops,
    select_inference_outputs,
    strip_collectives,
    strip_grad_tags,
)

__all__ = [
    "RegionTransform",
    "RegatherTransform",
    "RecomputeTransform",
    "SparseEmbeddingTransform",
    "DropoutPRNGTransform",
    "InferenceSwapTransform",
    "apply_inference_swaps",
    "deterministic_argmax",
    "dtype_cast_matcher",
    "identity_passthrough",
    "fabricate_inference_graph",
    "inputs_to_ops",
    "select_inference_outputs",
    "strip_collectives",
    "strip_grad_tags",
    "_apply_region_transforms",
    "_compute_region_io",
    "_discover_called_modules",
    "_extract_subgraph",
    "_find_forward_roots",
    "_forward_pre_hooks",
    "_preserve_all_autograd_function_leaves",
    "_replace_region",
    "_topo_sort_nodes",
    "extract_subgraph",
    "replace_region",
]
