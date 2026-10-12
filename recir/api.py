# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility façade for RecIR's public and legacy API imports."""

from __future__ import annotations

import torch

# Pull through the ``GraphPatch`` type alias so consumers don't need to import
# from ``recir.backends`` if they only want the type.
from recir.backends.base import GraphPatch  # noqa: E402, F401
from recir.compiler.compilation import (  # noqa: F401 -- compatibility re-exports
    _bind_to_current_module_state,
    _compile_and_transform,
    _compile_data_parallel_v2,
    _compile_graphs,
    _compile_gspmd,
    _compile_via_backend,
    _CompiledResult,
    _dedup_collectives,
    _finalize_graph,
    compile,
    compile_from_gm,
)
from recir.compiler.process_state import CompiledProcessState  # noqa: E402, F401
from recir.compiler.runtime import (  # noqa: F401 -- compatibility re-exports
    COMPILED_OBJECT_KEY,
    get_compiled_process_state,
    invalidate_device_state,
    make_compiled_runner,
    materialize_compiled_output,
    prepare_compiled,
    PREPARE_COMPILED_ATTR,
    set_compiled_output_deferred,
    sync_device_state_to_torch,
    sync_named_state_to_torch,
)
from recir.compiler.tracing import (  # noqa: F401 -- compatibility re-exports
    _bind_transform_signatures,
    _discover_for_region_transforms,
    _enter_region_transform_contexts,
    _finalize_trace_result,
    _finish_data_parallel_trace,
    _get_trace_ctx,
    _GraphSpec,
    _local_compile,
    _LocalCompiledResult,
    _make_hybrid_sparse_transform,
    _make_trace_input_transform,
    _prepare_data_parallel_trace,
    _prepare_transforms_after_shard,
    _PreparedDataParallelTrace,
    _RegionDiscovery,
    _relative_parametrize_kwargs,
    _rematerialize_optimizer,
    _rematerialize_tensor_attrs,
    _resolve_trace_batch_ratio,
    _set_trace_ctx,
    _tag_batch_inputs,
    _to_caller_flattened_graph_module,
    _TRACE_CTX_ATTR,
    _trace_graphs,
    _trace_module_and_optimizers,
    _trace_without_pre_sharding,
    _TraceContext,
    _TraceResult,
    trace,
)
# ---------------------------------------------------------------------------
# v7.1 public surface: graph manipulation primitives
# ---------------------------------------------------------------------------


# Re-exports so callers can write ``from recir.api import ...`` or
# ``recir.find_markers`` interchangeably.
from recir.ops.marker import find_markers  # noqa: E402, F401
from recir.region_transforms import extract_subgraph, replace_region  # noqa: E402, F401
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode


def add_state(
    gm: fx.GraphModule,
    flat_state: list[torch.Tensor],
    name: str,
    tensor: torch.Tensor,
) -> int:
    """Atomically add a new state tensor + matching placeholder to ``gm``.

    The new placeholder is inserted immediately after the existing trailing
    placeholder so it stays grouped with the rest of the flat state.  The
    tensor is appended to ``flat_state`` and the index of the appended entry
    in ``flat_state`` is returned (callers use this when wiring downstream
    uses against the flat-state list).

    Args:
        gm: the host graph module.  Its graph is mutated in place and
            recompiled.
        flat_state: the flat state list mirroring the leading placeholders of
            ``gm``.  Mutated in place: ``tensor`` is appended.
        name: the placeholder name (suffixed with a counter if it collides
            with an existing node name).
        tensor: the tensor whose value will be passed at runtime.

    Returns:
        The index of the newly-appended entry in ``flat_state`` (i.e.,
        ``len(flat_state) - 1`` after append).  This equals the new
        placeholder's positional index among ``gm``'s placeholders only when
        no non-flat-state placeholders precede it in ``gm``.
    """
    # Locate the insertion point: after the last existing placeholder.  This
    # keeps the placeholder block contiguous so the caller's existing
    # flat_state-to-placeholder pairing stays positional.
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if placeholders:
        anchor = placeholders[-1]
        insert_ctx = gm.graph.inserting_after(anchor)
    else:
        # Empty graph (no existing inputs).  Insert at the very front.
        first_node = next(iter(gm.graph.nodes), None)
        insert_ctx = (
            gm.graph.inserting_before(first_node)
            if first_node is not None
            else gm.graph.inserting_after()
        )

    with insert_ctx:
        new_placeholder = gm.graph.placeholder(name)

    # Attach a fake-tensor val to the new placeholder so downstream passes
    # that inspect ``node.meta["val"]`` (e.g. batch-dim propagation, region
    # transforms) see a concrete shape.  Reuse the FakeTensorMode already
    # attached to ``gm`` (via existing placeholders' meta) when possible so
    # the new fake-tensor shares the same mode identity as the rest of the
    # graph — constructing a fresh mode per call would create fake-tensors
    # tied to a different mode object and break downstream invariants that
    # rely on a single mode per graph.
    fake_mode = torch._guards.detect_fake_mode(
        [n.meta["val"] for n in placeholders if "val" in n.meta]
    )
    if fake_mode is None:
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        new_placeholder.meta["val"] = fake_mode.from_tensor(tensor)
    new_placeholder.meta.setdefault("is_batch_input", False)
    new_placeholder.meta.setdefault("is_weight", False)

    flat_state.append(tensor)

    gm.graph.lint()
    gm.recompile()

    # The new placeholder is now the last placeholder; its positional index
    # among placeholders equals len(flat_state) - 1.
    return len(flat_state) - 1
