# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""InferenceSwapTransform — post-trace training->inference graph surgery.

Operates on an already-traced FX ``GraphModule`` (no retracing), detecting
training-only subgraph regions and replacing them with inference equivalents.
Conforms to the landed :class:`RegionTransform` ABC (``transform_graph`` hook)
so it composes with RecIR's ``_apply_region_transforms`` alongside
``RegatherTransform`` / ``RecomputeTransform``, and also exposes a standalone
:func:`apply_inference_swaps` for direct use on any traced graph.

Two robust strategies, composable in a single pass:

1. **Op-level removal (Level 1)** — match individual aten ops (``native_dropout``
   and friends) and splice in an identity passthrough.  Numerically exact: in
   ``.eval()`` dropout is the identity, so removing it from a ``.train()``-traced
   graph yields the inference graph.  Note: ``eliminate_dead_code`` alone cannot
   remove ``native_dropout`` because it is impure (consumes RNG state), so the
   node must be explicitly replaced.

2. **Module-region replacement (Level 2)** — group nodes by the module *class*
   recorded in ``nn_module_stack`` metadata (populated by ``make_fx`` with
   ``record_module_stack=True``) and replace the whole region with a
   caller-supplied factory.  This handles training modules that trace into many
   ops (e.g. a UIH sampler -> 30+ ops) without enumerating those ops.

A :meth:`InferenceSwapTransform.from_module_swap_registry` bridge converts a
pre-trace ``ModuleSwapRegistry`` (Approach 1) into post-trace region swaps.

Out of scope for this prototype (use Level 2 module swap or a custom matcher):
fused stochastic-depth ``bernoulli -> mul`` patterns, where a bare op-target
match cannot tell which ``mul`` input is the surviving branch.

Usage::

    from recir.region_transforms.inference_swap import apply_inference_swaps

    # Strategy 1: built-in dropout removal (on by default)
    gm = apply_inference_swaps(gm)

    # Strategy 2: module-region swaps keyed by class FQN
    gm = apply_inference_swaps(
        gm,
        module_swaps={"my.pkg.UIHSampler": my_inference_factory},
    )
"""

from __future__ import annotations

import logging
import operator
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
from recir.region_transforms.base import RegionTransform
from recir.region_transforms.core import (
    _compute_region_io,
    _extract_subgraph,
    _replace_region,
)
from recir.utils.graph import OP, rebuild_graph

logger: logging.Logger = logging.getLogger(__name__)

# A region factory takes the extracted sub-GraphModule and returns the callable
# that replaces it.  The callable is invoked with the region's external inputs
# (positional) and its return value replaces the region's external outputs.
RegionFactory = Callable[[fx.GraphModule], Callable[..., Any]]
OpMatcher = Callable[[fx.Node], bool]


def _dropout_op_targets() -> tuple[object, ...]:
    """aten dropout op overloads that act as identity in eval mode."""
    return (
        torch.ops.aten.native_dropout.default,
        torch.ops.aten.dropout.default,
        torch.ops.aten.feature_dropout.default,
        torch.ops.aten.feature_alpha_dropout.default,
        torch.ops.aten.alpha_dropout.default,
    )


def _identity_passthrough_apply(*inputs: Any) -> Any:
    """Return the region's first input unchanged.

    MODULE-LEVEL on purpose. This ends up as a ``call_function`` target in the
    fabricated inference graph, and pickling a GraphModule pickles its node
    targets by qualified name. As a closure (``identity_passthrough.<locals>.
    _apply``) it has no importable path, so ``torch.save`` of a graph containing
    it fails with::

        ObjNotFoundError: <function identity_passthrough.<locals>._apply> was not
        found as recir.region_transforms.inference_swap.identity_passthrough.
        <locals>._apply

    The closure captured nothing, so hoisting it is behaviour-preserving.
    """
    return inputs[0]


def identity_passthrough(region_gm: fx.GraphModule) -> Callable[..., Any]:
    """Replacement factory: return the region's first input unchanged.

    Correct for ops that are the identity at inference time (dropout in eval),
    where the region's single external output is the op's data input.  Also the
    semantic for "skip this tail tower and publish its upstream activations".
    """

    return _identity_passthrough_apply


def dtype_cast_matcher(dtype: torch.dtype = torch.bfloat16) -> OpMatcher:
    """Op matcher for dtype-cast nodes whose target dtype is ``dtype``.

    Pair with :func:`identity_passthrough` in ``custom_op_matchers`` to strip
    training-only casts (e.g. the bf16 autocast DLRMv3 applies only when
    ``not is_inference``) from the inference cone, so the fabricated graph runs
    in the original (fp32) dtype that serving expects::

        fabricate_inference_graph(
            gm,
            inference_outputs=marked_inference_outputs(),
            custom_op_matchers=[(dtype_cast_matcher(), identity_passthrough)],
        )

    Only casts *to* ``dtype`` are matched; the inverse cast back to the original
    dtype becomes a no-op (same-dtype cast) once the bf16 region is gone.
    """
    cast_targets: tuple[object, ...] = (
        torch.ops.aten._to_copy.default,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
        torch.ops.prims.convert_element_type.default,
    )

    def _matches(node: fx.Node) -> bool:
        if node.op != OP.CALL_FUNCTION or node.target not in cast_targets:
            return False
        target_dtype = node.kwargs.get("dtype")
        if target_dtype is None:
            for arg in node.args[1:]:
                if isinstance(arg, torch.dtype):
                    target_dtype = arg
                    break
        return target_dtype is dtype

    return _matches


def _expose_input_factory(index: int) -> RegionFactory:
    """Factory that returns the region input at ``index`` (used by pruning)."""

    def _factory(region_gm: fx.GraphModule) -> Callable[..., Any]:
        def _apply(*inputs: Any) -> Any:
            return inputs[index]

        return _apply

    return _factory


def deterministic_argmax(dim: int = -1) -> RegionFactory:
    """Replacement factory for a stochastic-sampling region: greedy argmax.

    Replaces a sampling head (multinomial / gumbel-softmax / categorical) that
    draws from logits at training time with the deterministic ``argmax`` over
    those logits — the standard greedy-decode inference equivalent.  The
    region's first input is taken to be the logits tensor.
    """

    def _factory(region_gm: fx.GraphModule) -> Callable[..., Any]:
        def _apply(*inputs: Any) -> Any:
            return torch.argmax(inputs[0], dim=dim)

        return _apply

    return _factory


def _sampling_op_targets() -> tuple[object, ...]:
    """Stochastic ops whose draw is replaced by its (expected) input at eval.

    These cover reparameterization noise and stochastic-depth style masks where
    the deterministic inference behavior is to pass the input through.  Sampling
    that selects an *index* (e.g. ``multinomial``) is not here — use a
    module-region swap with :func:`deterministic_argmax` for those.
    """
    return (
        torch.ops.aten.bernoulli.default,
        torch.ops.aten.bernoulli.p,
        torch.ops.aten.poisson.default,
    )


def _is_dropout_node(node: fx.Node, targets: tuple[object, ...]) -> bool:
    return node.op == OP.CALL_FUNCTION and node.target in targets


def _dropout_region(node: fx.Node) -> set[fx.Node]:
    """Region for a dropout node: the node plus any ``getitem`` users.

    ``native_dropout`` returns ``(output, mask)``; both ``getitem`` users must
    be in the region so the mask ``getitem`` (dead at inference) is erased with
    the dropout node rather than left dangling.  Functional ``dropout`` returns
    a single tensor and has no ``getitem`` users.
    """
    region: set[fx.Node] = {node}
    for user in node.users:
        if user.op == OP.CALL_FUNCTION and user.target is operator.getitem:
            region.add(user)
    return region


def _module_class_fqns(node: fx.Node) -> set[str]:
    """Class FQNs recorded for ``node`` in ``nn_module_stack`` metadata.

    ``nn_module_stack`` is an ``OrderedDict`` of
    ``{tracer_key: (module_instance_fqn, module_class)}``.  We key swaps on the
    class (second element), normalizing both string and ``type`` forms.
    """
    stack = node.meta.get("nn_module_stack", None)
    if not stack:
        return set()
    classes: set[str] = set()
    for value in stack.values():
        if not isinstance(value, tuple) or len(value) < 2:
            continue
        cls = value[1]
        if isinstance(cls, str):
            classes.add(cls)
        elif isinstance(cls, type):
            classes.add(f"{cls.__module__}.{cls.__qualname__}")
    return classes


def _group_nodes_by_class(
    gm: fx.GraphModule,
    class_fqns: set[str],
    claimed: set[fx.Node],
) -> dict[str, set[fx.Node]]:
    """Group compute nodes by the target module class FQN they belong to."""
    grouped: dict[str, set[fx.Node]] = defaultdict(set)
    for node in gm.graph.nodes:
        if node in claimed:
            continue
        if node.op not in (OP.CALL_FUNCTION, OP.CALL_METHOD):
            continue
        for cls_fqn in _module_class_fqns(node):
            if cls_fqn in class_fqns:
                grouped[cls_fqn].add(node)
    for cls_fqn in class_fqns:
        if cls_fqn not in grouped:
            logger.warning("InferenceSwapTransform: no nodes for %s", cls_fqn)
    return grouped


def _graph_output_node(gm: fx.GraphModule) -> fx.Node:
    for node in gm.graph.nodes:
        if node.op == OP.OUTPUT:
            return node
    raise ValueError("graph has no output node")


class InferenceSwapTransform(RegionTransform):
    """Post-trace training->inference region replacement.

    Args:
        module_swaps: Map of module *class* FQN -> region factory.  Every node
            whose ``nn_module_stack`` names a matching class is grouped into one
            region and replaced by ``factory(region_gm)``.  Covers both
            training-only module replacement and stochastic-sampling heads
            (pair with :func:`deterministic_argmax`).
        prune_module_classes: Module *class* FQNs to drop entirely (partial
            publish, e.g. "skip overarch").  A tail tower whose outputs feed
            only the graph output is removed and dropped from the outputs; a
            mid-graph block instead falls back to exposing its upstream
            activations (identity).
        enable_dropout_removal: Replace dropout ops with identity (default on).
        enable_sampling_removal: Replace stochastic-noise ops (bernoulli /
            poisson) with identity passthrough (default off — opt in).
        custom_op_matchers: Extra ``(matcher, factory)`` pairs for op-level
            swaps not covered by the built-ins.
    """

    def __init__(
        self,
        module_swaps: dict[str, RegionFactory] | None = None,
        prune_module_classes: set[str] | None = None,
        enable_dropout_removal: bool = True,
        enable_sampling_removal: bool = False,
        custom_op_matchers: list[tuple[OpMatcher, RegionFactory]] | None = None,
    ) -> None:
        self._module_swaps: dict[str, RegionFactory] = module_swaps or {}
        self._prune_module_classes: set[str] = prune_module_classes or set()
        self._enable_dropout_removal = enable_dropout_removal
        self._enable_sampling_removal = enable_sampling_removal
        self._custom_op_matchers: list[tuple[OpMatcher, RegionFactory]] = (
            custom_op_matchers or []
        )

    @classmethod
    def from_module_swap_registry(
        cls,
        registry: Any,
        enable_dropout_removal: bool = True,
    ) -> InferenceSwapTransform:
        """Bridge a pre-trace ``ModuleSwapRegistry`` (Approach 1) to post-trace.

        For each ``(training_cls -> inference_factory)`` entry, builds a region
        factory that instantiates the inference module and calls its forward in
        place of the training region.

        Caveat (design discussion topic #2): post-trace, the inference module's
        weights/config are not recoverable from ``nn_module_stack`` alone.  This
        bridge wires up the *structure*; for stateful modules prefer the
        pre-trace swap (Approach 1).  It is exact for stateless training-only
        modules.
        """
        module_swaps: dict[str, RegionFactory] = {}
        for training_cls in registry.registered_types:
            cls_fqn = f"{training_cls.__module__}.{training_cls.__qualname__}"
            module_factory = registry.get_factory(training_cls)
            if module_factory is None:
                continue
            module_swaps[cls_fqn] = cls._module_region_factory(module_factory)
            logger.info("from_module_swap_registry: bridged %s", cls_fqn)
        return cls(
            module_swaps=module_swaps,
            enable_dropout_removal=enable_dropout_removal,
        )

    @staticmethod
    def _module_region_factory(
        module_factory: Callable[..., torch.nn.Module],
    ) -> RegionFactory:
        def _factory(region_gm: fx.GraphModule) -> Callable[..., Any]:
            inference_module: torch.nn.Module = module_factory()

            def _apply(*inputs: Any) -> Any:
                return inference_module(*inputs)

            return _apply

        return _factory

    # -- RegionTransform hook -------------------------------------------------

    def transform_graph(
        self,
        gm: fx.GraphModule,
        # pyre-ignore[2]: matches the base-class hook signature.
        called_modules: Any,
    ) -> None:
        """In-place rewrite of ``gm``: detect training regions and replace."""
        self._run(gm)

    # -- core driver ----------------------------------------------------------

    def _run(self, gm: fx.GraphModule) -> None:
        regions = self._detect_regions(gm)
        for region_nodes, factory in regions:
            if not region_nodes:
                continue
            region_gm = _extract_subgraph(gm, region_nodes)
            _replace_region(gm, region_nodes, factory(region_gm))
        pruned = 0
        if self._prune_module_classes:
            pruned = self._prune_module_regions(gm)
        rebuild_graph(gm)
        logger.info(
            "InferenceSwapTransform swapped %d region(s), pruned %d",
            len(regions),
            pruned,
        )

    def _detect_regions(
        self,
        gm: fx.GraphModule,
    ) -> list[tuple[set[fx.Node], RegionFactory]]:
        """Pair each detected region with the factory that replaces it.

        Op-level matches are claimed first so module-region grouping does not
        double-match the same node.
        """
        regions: list[tuple[set[fx.Node], RegionFactory]] = []
        claimed: set[fx.Node] = set()
        dropout_targets = _dropout_op_targets()
        sampling_targets = _sampling_op_targets()

        for node in gm.graph.nodes:
            if node in claimed:
                continue

            if self._enable_dropout_removal and _is_dropout_node(node, dropout_targets):
                region = _dropout_region(node)
                regions.append((region, identity_passthrough))
                claimed.update(region)
                continue

            if (
                self._enable_sampling_removal
                and node.op == OP.CALL_FUNCTION
                and node.target in sampling_targets
            ):
                regions.append(({node}, identity_passthrough))
                claimed.add(node)
                continue

            for matcher, factory in self._custom_op_matchers:
                if matcher(node):
                    regions.append(({node}, factory))
                    claimed.add(node)
                    break

        if self._module_swaps:
            grouped = _group_nodes_by_class(gm, set(self._module_swaps), claimed)
            for cls_fqn, nodes in grouped.items():
                regions.append((nodes, self._module_swaps[cls_fqn]))
                logger.info(
                    "InferenceSwapTransform: swap region %s (%d nodes)",
                    cls_fqn,
                    len(nodes),
                )

        return regions

    def _prune_module_regions(self, gm: fx.GraphModule) -> int:
        """Remove regions for ``prune_module_classes`` (partial publish).

        A tail tower whose region outputs feed only the graph output is dropped
        from the outputs and DCE'd.  Anything else falls back to exposing its
        upstream activations via identity (the "skip overarch" semantic when the
        tower is the sole output).
        """
        output_node = _graph_output_node(gm)
        grouped = _group_nodes_by_class(gm, self._prune_module_classes, set())
        pruned = 0
        for cls_fqn, nodes in grouped.items():
            sorted_region = [n for n in gm.graph.nodes if n in nodes]
            region_inputs, region_outputs = _compute_region_io(sorted_region)
            region_out_set = set(region_outputs)
            external_users = {
                u for n in region_outputs for u in n.users if u not in nodes
            }
            ret = output_node.args[0]
            is_tail = bool(external_users) and external_users <= {output_node}
            if is_tail and isinstance(ret, (tuple, list)):
                survivors = [o for o in ret if o not in region_out_set]
                if survivors:
                    new_ret = survivors if isinstance(ret, list) else tuple(survivors)
                    output_node.args = (new_ret,)
                    logger.info("InferenceSwapTransform: pruned tail %s", cls_fqn)
                    pruned += 1
                    continue
            # Fallback: expose the region's upstream activation (skip-overarch).
            # Pick the first non-parameter input so identity does not return a
            # weight/bias get_attr that was lifted in as a region input.
            act_idx = next(
                (i for i, n in enumerate(region_inputs) if n.op != OP.GET_ATTR),
                0,
            )
            region_gm = _extract_subgraph(gm, nodes)
            _replace_region(gm, nodes, _expose_input_factory(act_idx)(region_gm))
            logger.info(
                "InferenceSwapTransform: pruned %s by exposing upstream input %d",
                cls_fqn,
                act_idx,
            )
            pruned += 1
        return pruned


def apply_inference_swaps(
    gm: fx.GraphModule,
    module_swaps: dict[str, RegionFactory] | None = None,
    prune_module_classes: set[str] | None = None,
    module_swap_registry: Any | None = None,
    enable_dropout_removal: bool = True,
    enable_sampling_removal: bool = False,
    custom_op_matchers: list[tuple[OpMatcher, RegionFactory]] | None = None,
) -> fx.GraphModule:
    """Apply inference swaps to ``gm`` in place and return it.

    Standalone driver that runs the detect -> extract -> replace loop without
    the surrounding RecIR compile pipeline.  Provide module-level swaps via
    ``module_swaps`` (class FQN -> factory) or ``module_swap_registry`` (a
    ``ModuleSwapRegistry`` instance, bridged automatically), modules to drop via
    ``prune_module_classes`` (partial publish), and op-level removal toggles.
    """
    if module_swap_registry is not None:
        transform = InferenceSwapTransform.from_module_swap_registry(
            module_swap_registry,
            enable_dropout_removal=enable_dropout_removal,
        )
        transform._prune_module_classes = prune_module_classes or set()
        transform._enable_sampling_removal = enable_sampling_removal
        # Merge caller-provided explicit swaps/matchers on top of the
        # registry-derived ones so they are not silently dropped when combined
        # with ``module_swap_registry`` (registry entries are the base).
        if module_swaps:
            transform._module_swaps = {**transform._module_swaps, **module_swaps}
        if custom_op_matchers:
            transform._custom_op_matchers = [
                *transform._custom_op_matchers,
                *custom_op_matchers,
            ]
    else:
        transform = InferenceSwapTransform(
            module_swaps=module_swaps,
            prune_module_classes=prune_module_classes,
            enable_dropout_removal=enable_dropout_removal,
            enable_sampling_removal=enable_sampling_removal,
            custom_op_matchers=custom_op_matchers,
        )
    transform._run(gm)
    return gm
