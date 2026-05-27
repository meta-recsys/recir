# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Bundle runtime: lower a :class:`LoadedBundle` to an :class:`IterGraphModule`.

Public entry point: :func:`wrap_and_lower`.

The runtime is the consumer-side analogue of
``recir.api._compile_data_parallel_v2``: it takes the FQN-keyed dicts the
loader assembled, threads them through the kwargs-only
:meth:`recir.distributed.parallel_mode.ParallelMode.partition`
signature, and runs
:class:`recir.gm_transformation.GraphModuleTransformation` configured
for :data:`~recir.backend.BackendType.INDUCTOR` to prepare the graph for
lazy first-execution Inductor compilation.

After the :class:`IterGraphModule` is built, the runtime restores the
:data:`~torch.export.unflatten._AttrKind.CONSTANT` registration on every
sub-GM via :func:`recir.bundle_runtime._device_migration.reflip_constants_on_sub_gms`.
The reflip is non-optional: the :class:`IterGraphModule` constructor
calls ``fx.GraphModule(src, graph)`` per sub-GM, and the resulting
``_copy_attr`` walk silently re-registers tensor constants as buffers,
which would defeat the producer's CONSTANT-kind contract and cause
downstream optimisation passes to drop the values on subsequent
``recompile()``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING

from recir.api import _to_caller_flattened_graph_module
from recir.backend import BackendType
from recir.gm_transformation import GraphModuleTransformation
from recir.iter_graph_module import IterGraphModule

from ._device_migration import reflip_constants_on_sub_gms


if TYPE_CHECKING:
    from recir.bundle_runtime.loader import LoadedBundle
    from recir.distributed.parallel_mode import ParallelMode


def wrap_and_lower(
    loaded: "LoadedBundle",
    parallel_mode: "ParallelMode",
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] | None = None,
) -> IterGraphModule:
    """Partition + lower a :class:`LoadedBundle` to an :class:`IterGraphModule`.

    Steps:

      1. ``parallel_mode.partition(gm, *, params=, buffers=,
         named_states=)`` -- the kwargs-only signature accepts the
         FQN-keyed dicts directly, so no live :class:`nn.Module` /
         :class:`torch.optim.Optimizer` reconstruction is required.
         **Constants are not threaded through the call vector**: they
         already live on ``loaded.gm`` as
         :data:`~torch.export.unflatten._AttrKind.CONSTANT` attributes
         (the deserializer seeded them at load time), and
         :meth:`recir.serde.BindingTable.call_vector` excludes them by
         design (per :mod:`recir.serde.schema`).
      2. ``_to_caller_flattened_graph_module`` -- replace the structured
         ``_PyTreeCodeGen`` with ``_PyTreeCodeGenOutputsOnly`` so the
         partitioned GM's forward signature accepts ``*flat_leaves``
         positionally instead of expecting a structured 5-tuple.  This
         matches the production callsite contract used by
         :func:`recir.api._finalize_graph`.
      3. ``GraphModuleTransformation(backend=INDUCTOR)`` -- builds an
         :class:`IterGraphModule` (3 sub-GMs: ``setup_gm``, ``main_gm``,
         ``cleanup_gm``) and runs ``partial_lower`` on each, which
         inserts lazy ``_InductorModule`` subgraphs.  No
         ``compile_fx_inner`` runs at this point -- the cost lands
         inside the first invocation per the lazy-compile design.
      4. :func:`reflip_constants_on_sub_gms` -- restore the
         ``_AttrKind.CONSTANT`` registration on each sub-GM for every
         tensor get_attr, undoing the silent buffer re-registration the
         :class:`fx.GraphModule` constructor performs inside
         :class:`IterGraphModule.__init__`.  Without this step the
         constant attrs would survive as buffers, which downstream
         optimisation passes can drop on the next ``recompile()`` and
         which leaks the wrong attribute kind to consumers reading via
         :meth:`nn.Module.named_buffers`.

    Args:
        loaded: The output of :func:`recir.bundle_runtime.import_bundle`.
        parallel_mode: The :class:`ParallelMode` describing the runtime
            topology (mesh, parallel style, etc.).  The bundle itself
            is topology-free, so the same bundle can be loaded under any
            ``parallel_mode``.
        args: Positional user inputs passed through to ``partition`` for
            shape inference.  Default ``()`` covers the common case where
            the bundle's stored placeholders are sufficient (e.g. forward-
            only DDP graphs where ``partition`` reads shape info off
            ``meta['val']``).
        kwargs: Keyword user inputs (same role as ``args``).  Default
            ``None`` is normalised to ``{}``.

    Returns:
        A lowered :class:`IterGraphModule` whose ``_InductorModule``
        subgraphs are ready to JIT-compile via Inductor on first call.
        Tensor constants are restored as
        :data:`~_AttrKind.CONSTANT` on every sub-GM.
    """
    del args, kwargs  # Reserved for future shape-aware partition flavours.

    # Step 1: distributed-graph rewrite via the parallel-mode partitioner.
    # The kwargs-only signature accepts the FQN-keyed dicts the loader
    # assembled, so no nn.Module / Optimizer reconstruction is needed.
    # Constants are deliberately NOT threaded into ``partition`` -- they
    # live on ``loaded.gm`` as ``_AttrKind.CONSTANT`` attributes (per
    # the producer's contract) and ``BindingTable.call_vector()``
    # excludes them by design.
    partitioned_gm = parallel_mode.partition(
        loaded.gm,
        params=loaded.params,
        buffers=loaded.buffers,
        named_states=loaded.named_states,
    )

    # Step 2: caller-flatten the partitioned GM so its forward signature
    # is ``forward(self, *flat_leaves)`` accepting one positional tensor
    # per placeholder, in placeholder order matching
    # ``loaded.bindings.call_vector()``.  The structured ``_PyTreeCodeGen``
    # otherwise expects a 5-tuple input that the production callsite
    # (which passes a flat tensor vector) does not provide.  Mirrors
    # ``recir.api._finalize_graph``.
    partitioned_gm = _to_caller_flattened_graph_module(partitioned_gm)

    # Step 3: GraphModuleTransformation builds the IterGraphModule and
    # runs partial_lower on each sub-GM, inserting lazy _InductorModule
    # subgraphs without invoking ``compile_fx_inner`` (that lands on
    # first execution).
    transformation = GraphModuleTransformation(backend=BackendType.INDUCTOR)
    iter_gm = transformation(partitioned_gm)
    # ``GraphModuleTransformation`` returns ``Union[IterGraphModule,
    # fx.GraphModule]``: the plain-GM branch fires only when
    # ``enable_iter_graph=False``, which we never set.  Narrow for Pyre
    # so the ``setup_gm``/``main_gm``/``cleanup_gm`` accesses below and
    # the return type are well-typed.
    assert isinstance(iter_gm, IterGraphModule)

    # Step 4: ``IterGraphModule.__init__`` rebuilt each sub-GM via
    # ``fx.GraphModule(src, graph)``, whose ``_copy_attr`` silently
    # re-registered every tensor ``get_attr`` attribute as a buffer.
    # Reflip them back to ``_AttrKind.CONSTANT`` on all three sub-GMs
    # so downstream passes see the producer-declared attribute kind.
    reflip_constants_on_sub_gms(
        [iter_gm.setup_gm, iter_gm.main_gm, iter_gm.cleanup_gm],
        loaded.constants,
    )

    return iter_gm
