# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from recir.backends.base import Backend

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from recir.batch_dim_propagation import propagate_batch_input_meta
from recir.decompositions import DEDUP_TARGETS
from recir.distributed.data_parallel import (
    collectives_insertion,
    DataParallelStyle,
    gradients_tagging,
    parametrize_params,
)
from recir.distributed.parallel_mode import (
    DataParallel,
    GSPMDParallel,
    HybridParallel,
    ParallelMode,
)
from recir.lowering.device_state import DeviceStateManager
from recir.passes.pass_registry import default_registry as default_pass_registry
from recir.pipeline import build_pipeline, PipelineConfig
from recir.process_state import (
    _align_fsdp_sharded_flags,
    _annotate_fsdp_state_input_sharding,
    _annotate_fsdp_state_output_sharding,
    _build_compiled_process_state,
    _build_fsdp_sharded_flags,
    _compute_state_fqns,
    CompiledProcessState,
)
from recir.region_transforms import (
    _apply_region_transforms,
    _discover_called_modules,
    _find_forward_roots,
    RegatherTransform,
    RegionTransform,
    SparseEmbeddingTransform,
)
from recir.runtime import (  # noqa: F401 -- compatibility re-exports
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
from recir.tracing import (  # noqa: F401 -- compatibility re-exports
    _local_compile,
    _LocalCompiledResult,
    _rematerialize_optimizer,
    _rematerialize_tensor_attrs,
    _to_caller_flattened_graph_module,
)
from recir.utils.compile import _enable_compile
from torch import fx
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

logger: logging.Logger = logging.getLogger(__name__)


def _dedup_collectives(gm: fx.GraphModule) -> fx.GraphModule:
    args_to_node: Dict[Tuple[Any, ...], fx.Node] = {}

    for node in gm.graph.nodes:
        # replace all args with the results from the first unique comm op
        args = pytree.arg_tree_leaves(*node.args)

        if node.target in DEDUP_TARGETS:
            args_key = (node.target, *args)
            unique_node = args_to_node.get(args_key, None)
            if unique_node is None:
                # first time seeing this combination, remember it
                args_to_node[args_key] = node
            else:
                # the current node is a duplicate, replace it
                node.replace_all_uses_with(unique_node)
                gm.graph.erase_node(node)

    gm.recompile()

    return gm


@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opts: List[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]
    state_manager: DeviceStateManager
    process_state: Optional[CompiledProcessState] = None
    # GSPMD placements decided when this graph sharded the module; graphs
    # compiled together with it (``recir.compile`` of several functions)
    # reuse them.
    sharded_parameter_fqns: frozenset[str] = frozenset()
    optimizer_state_only_sharded_keys: frozenset[tuple[str, str]] = frozenset()
    # Transforms added by the sub-path (e.g. ``_compile_data_parallel_v2``
    # adds ``SparseEmbeddingTransform``).  Their ``finalize_compiled_graph``
    # is invoked by ``_compile_and_transform`` alongside the caller-supplied
    # ``region_transforms``, in a single consolidated loop.
    internal_transforms: List["RegionTransform"] = field(default_factory=list)


def _make_trace_input_transform(
    input_batch_dim: int,
    ratio: float,
    fake_mode: FakeTensorMode,
    dynamic_batch_dim: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a transform that fakeifies inputs and optionally resizes a dimension.

    Every tensor is converted with ``fake_mode``. When ``dynamic_batch_dim``
    is set, ``input_batch_dim`` remains symbolic even when no resize is needed.
    For ratio > 1 (e.g., GSPMD: local → global batch), ``repeat`` expands.
    For ratio < 1 (e.g., v2 XLA path: multi-device local → per-device batch),
    ``narrow`` shrinks. For ratio == 1 only the FakeTensor conversion occurs.
    """

    input_index = 0

    def _to_fake(arg: torch.Tensor) -> torch.Tensor:
        nonlocal input_index
        if not dynamic_batch_dim or arg.ndim <= input_batch_dim:
            return fake_mode.from_tensor(arg)
        symbolic_context = StatelessSymbolicContext(
            dynamic_sizes=[
                DimDynamic.DYNAMIC
                if dimension == input_batch_dim
                else DimDynamic.STATIC
                for dimension in range(arg.ndim)
            ]
        )
        source = LocalSource(f"recir_batch_input_{input_index}")
        input_index += 1
        return fake_mode.from_tensor(
            arg,
            static_shapes=False,
            source=source,
            symbolic_context=symbolic_context,
        )

    def _get_full_batch_arg(arg: torch.Tensor) -> torch.Tensor:
        # Return the fake input untouched in two no-op cases:
        #   * No mini-batch dim to expand/shrink: 0-d scalar tensors (or any
        #     tensor whose rank is <= input_batch_dim). Indexing the missing
        #     dim below would otherwise IndexError.
        #   * ratio == 1 (documented no-op). Skipping here is also what keeps
        #     dynamic shapes alive: falling through to ``int(curr_batch *
        #     ratio)`` would emit an equality guard that specializes the batch
        #     SymInt to its trace-time value, silently freezing the per-batch
        #     (candidate) dimension on the single-device-per-process path.
        if arg.ndim <= input_batch_dim or ratio == 1:
            return _to_fake(arg)
        # Integer expand (ratio > 1 and integer-valued).
        if isinstance(ratio, int) or (
            isinstance(ratio, float) and ratio > 1.0 and ratio.is_integer()
        ):
            fake_arg = _to_fake(arg)
            n = int(ratio)
            repeat_dims = [1] * arg.ndim
            repeat_dims[input_batch_dim] = n
            return fake_arg.repeat(repeat_dims)
        # Integer fraction shrink (ratio = 1/N, N integer >= 2).
        if isinstance(ratio, float) and 0 < ratio < 1.0:
            inv = 1.0 / ratio
            if inv.is_integer():
                n = int(inv)
                fake_arg = _to_fake(arg)
                target_batch = fake_arg.shape[input_batch_dim] // n
                return fake_arg.narrow(input_batch_dim, 0, target_batch)
        # Any other ratio (non-integer and non-1/N) could only be applied via
        # ``int(curr_batch * ratio)``, which specializes the batch SymInt and
        # silently breaks dynamic shapes. No current caller produces such a
        # ratio (HDP/DP -> 1/N; GSPMD -> integer num_processes), so reject it
        # instead of specializing behind the caller's back.
        raise AssertionError(
            f"_make_trace_input_transform: unsupported batch ratio {ratio!r}; "
            "expected 1, a positive integer, or 1/N for integer N >= 2."
        )

    return _get_full_batch_arg


def _resolve_trace_batch_ratio(
    parallel_mode: Optional[ParallelMode],
) -> Optional[float]:
    """Determine the per-host -> trace batch ratio for ``parallel_mode``.

    Returns ``None`` when ``parallel_mode`` doesn't request a pre-trace
    batch transform (e.g., when ``parallel_mode`` is ``None`` or an
    unrecognized mode).

    For GSPMD-style modes (the partition is a no-op; XLA inflates
    placeholder shardings at compile time), the trace should see global
    batch shapes so HuggingFace's modeling code bakes the correct
    Python-int literals into ``expand``/``view`` calls.  The ratio is
    ``num_processes`` (per-host -> per-process global; for single-host
    runs ``num_processes == 1`` and the transform is a no-op).

    For ``DataParallel`` / ``HybridParallel`` driven through the new
    ``trace`` + ``compile_from_gm`` pipeline, the DTensor partition step
    re-shards user-input placeholders back to per-process shapes, so the
    trace must start from the global batch for the runtime per-process
    inputs to match.  This mirrors the per-host -> global transform that
    the legacy :func:`_compile_data_parallel` applies inline on its own
    path; without it, ``partition_data_parallel`` narrows the placeholder
    to ``per_host_batch // num_processes`` and runtime invocation hits an
    ``aten.mm`` shape mismatch (e.g. ``(N, 10) @ (10/num_processes, K)``).
    """
    if parallel_mode is None:
        return None
    # Use isinstance only when the parallel-mode classes are import-safe.
    # Avoid a hard import at module load to keep recir importable in
    # environments where the XLA/distributed deps are missing.
    if isinstance(parallel_mode, GSPMDParallel):
        # GSPMD: partition is a no-op; XLA lowering inflates placeholders
        # to the global batch via _to_sharded_jax + _jit_gspmd shardings.
        # Tracing with the global batch makes baked Python-int literals
        # (HF Gemma3 / Llama / Qwen modeling code) consistent with the
        # GSPMD runtime shape.
        return float(parallel_mode.num_processes)
    # DataParallel / HybridParallel: the v2 parametrization-based trace path
    # (in :func:`trace`) applies its own per-process -> per-device batch
    # transform via ``1/(world_size // num_processes)``.  Returning ``None``
    # here keeps the GSPMD-only contract of this helper explicit and
    # preserves the original behavior for plain
    # ``recir.compile(parallel_mode=DataParallel(), ...)``.
    return None


def _tag_batch_inputs(
    gm: fx.GraphModule,
    num_state_tensors: int,
    input_batch_dim: int,
    frozen_parameter_fqns: set[str],
) -> None:
    """Tag runtime input roles, including parameters safe for device reuse.

    Only non-trainable model parameters are marked frozen. Buffers and
    optimizer state remain uncached because storage-alias writes can bypass a
    tensor's PyTorch version counter.
    """
    tensor_idx = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            is_state = tensor_idx < num_state_tensors
            is_batch = not is_state and val.dim() > 0
            node.meta["is_state_input"] = is_state
            node.meta["is_batch_input"] = is_batch
            node.meta["is_weight"] = not is_batch
            node.meta["is_frozen_parameter"] = (
                is_state and node.meta.get("fqn") in frozen_parameter_fqns
            )
            if is_batch:
                node.meta["batch_dim"] = input_batch_dim
            tensor_idx += 1
        else:
            node.meta["is_state_input"] = False
            node.meta["is_batch_input"] = False
            node.meta["is_weight"] = False
            node.meta["is_frozen_parameter"] = False


def _finalize_graph(
    gm: fx.GraphModule,
    mod: nn.Module,
    opts: List[torch.optim.Optimizer],
    params_and_buffers: Dict[str, Any],
    named_states: Dict[str, Any],
    input_batch_dim: int,
    *,
    dedup: bool = True,
) -> _CompiledResult:
    """Flatten state, dedup collectives, and tag batch inputs."""
    flat_state = pytree.tree_leaves([params_and_buffers, named_states])
    gm = _to_caller_flattened_graph_module(gm)
    if dedup:
        gm = _dedup_collectives(gm)
    _tag_batch_inputs(
        gm,
        sum(isinstance(value, torch.Tensor) for value in flat_state),
        input_batch_dim,
        {
            fqn
            for fqn, parameter in mod.named_parameters(remove_duplicate=False)
            if not parameter.requires_grad
        },
    )
    # Propagate batch-input meta through ops so intermediate FX nodes that
    # become subgraph boundaries (after segment_and_lower) inherit
    # is_batch_input / batch_dim. Without this, downstream subgraph
    # placeholders default to replicated and JAX sees per-process shape
    propagate_batch_input_meta(gm)
    result = _CompiledResult(gm, mod, opts, flat_state, DeviceStateManager())
    result.state_manager.state_fqns = _compute_state_fqns(
        [params_and_buffers, named_states]
    )
    return result


@dataclass
class _RegionDiscovery:
    """Result of `_discover_for_region_transforms`.

    Holds the called-module map (used to find forward roots) built by running
    the train step under FakeTensorMode hooks.
    """

    called_modules: OrderedDict[str, Tuple[nn.Module, Tuple[Any, ...], Dict[str, Any]]]


def _discover_for_region_transforms(
    transforms: List[RegionTransform],
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    mod: Optional[nn.Module] = None,
    fake_mode: Optional[FakeTensorMode] = None,
) -> Optional[_RegionDiscovery]:
    """Run `_discover_called_modules` to find the forward roots.

    Returns ``None`` (no-op) when ``transforms`` is empty.  When ``mod`` is
    not provided the first :class:`nn.Module` in ``args``/``kwargs`` is used.
    """
    if not transforms:
        return None
    if mod is None:
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, nn.Module):
                mod = arg
                break
        assert mod is not None, (
            "region_transforms requires an nn.Module in the arguments"
        )
    # Inputs may already be FakeTensors produced by the trace's batch
    # transform. Reuse that mode when supplied; FakeTensorMode does not permit
    # tensors allocated by a different active mode.
    discovery_mode = fake_mode or FakeTensorMode(
        allow_non_fake_inputs=True, shape_env=ShapeEnv()
    )
    shape_env = discovery_mode.shape_env
    ignore_fresh_symbols = (
        shape_env.ignore_fresh_unbacked_symbols()
        if shape_env is not None
        else nullcontext()
    )
    with _enable_compile(), ignore_fresh_symbols:
        called_modules = _discover_called_modules(
            mod,
            discovery_mode,
            func,
            *args,
            **kwargs,
        )
    return _RegionDiscovery(called_modules=called_modules)


def _relative_parametrize_kwargs(
    root_fqn: str, parametrize_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    if not root_fqn:
        return parametrize_kwargs
    prefix = f"{root_fqn}."
    return {
        key: {fqn[len(prefix) :] for fqn in fqns if fqn.startswith(prefix)}
        for key, fqns in parametrize_kwargs.items()
    }


def _enter_region_transform_contexts(
    region_stack: ExitStack,  # pyre-ignore
    transforms: List[RegionTransform],
    discovery: Optional[_RegionDiscovery],
    fake_mode: FakeTensorMode,
    *,
    parametrize_cm_factory: Optional[
        Callable[[str, nn.Module], Callable[[], Any]]
    ] = None,
) -> None:
    """Enter `_apply_region_transforms` once per forward root on `region_stack`.

    No-op when ``discovery`` is None or ``transforms`` is empty.
    ``parametrize_cm_factory(root_fqn, root_mod)`` returns a context-manager factory
    used during the per-root trace inside ``_apply_region_transforms``;
    pass ``None`` when no parametrization is needed (e.g. GSPMD path).
    """
    if discovery is None or not transforms:
        return
    default_roots = {
        fqn: (root_mod, root_fake_args, root_fake_kwargs)
        for fqn, root_mod, root_fake_args, root_fake_kwargs in _find_forward_roots(
            discovery.called_modules
        )
    }
    transforms_by_root: OrderedDict[str, list[RegionTransform]] = OrderedDict()
    root_data = dict(default_roots)
    for transform in transforms:
        selected = transform.select_forward_root_fqns(discovery.called_modules)
        root_fqns = default_roots.keys() if selected is None else selected
        for fqn in root_fqns:
            if fqn not in discovery.called_modules:
                raise ValueError(
                    f"Region transform selected an uncalled module FQN: {fqn!r}."
                )
            root_data[fqn] = discovery.called_modules[fqn]
            transforms_by_root.setdefault(fqn, []).append(transform)

    for fqn in sorted(
        transforms_by_root, key=lambda name: name.count("."), reverse=True
    ):
        root_mod, root_fake_args, root_fake_kwargs = root_data[fqn]
        cm_factories = (
            [parametrize_cm_factory(fqn, root_mod)]
            if parametrize_cm_factory is not None
            else []
        )
        region_stack.enter_context(
            _apply_region_transforms(
                root_mod,
                root_fake_args,
                discovery.called_modules,
                transforms_by_root[fqn],
                fake_mode,
                _parametrize_cm_factories=cm_factories,
                fake_kwargs=root_fake_kwargs,
            )
        )


def _compile_and_transform(
    func: Callable[..., Any],
    parallel_mode: ParallelMode,
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
    **kwargs: Any,
) -> _CompiledResult:
    if region_transforms:
        for transform in region_transforms:
            bind = getattr(transform, "bind_signature", None)
            if bind is not None:
                bind(func)

    if isinstance(parallel_mode, GSPMDParallel):
        result = _compile_gspmd(
            func,
            parallel_mode,
            *args,
            region_transforms=region_transforms,
            _tensor_attr_holders=_tensor_attr_holders,
            **kwargs,
        )
    elif isinstance(parallel_mode, (DataParallel, HybridParallel)):
        result = _compile_data_parallel_v2(
            func,
            parallel_mode,
            *args,
            region_transforms=region_transforms,
            _tensor_attr_holders=_tensor_attr_holders,
            **kwargs,
        )
    else:
        raise TypeError(
            f"Unsupported parallel_mode type: {type(parallel_mode).__name__}"
        )
    for transform in result.internal_transforms:
        transform.finalize_compiled_graph(result.gm)
    if region_transforms:
        for transform in region_transforms:
            transform.finalize_compiled_graph(result.gm)

    return result


# ---------------------------------------------------------------------------
# v7.1 public lower-level API: trace + compile_from_gm
# ---------------------------------------------------------------------------
#
# These two functions split the legacy ``_compile_and_transform`` pipeline at
# the partition boundary so external graph-surgery orchestrators can splice
# graph-level edits between the local
# trace (stage 2) and partition (stage 5).
#
# ``trace()`` runs stages 0 + 2 (eager pre-trace context + ``make_fx``) and
# returns ``(gm, flat_state)``.  The bundle of (mod, opt, params, buffers,
# named_states) is stashed on the gm as ``gm._recir_trace_ctx`` so
# ``compile_from_gm`` can recover it without forcing the user to pass it
# through.
#
# ``compile_from_gm()`` runs stages 4-10 (region_transforms.finalize,
# partition, optimize_passes, IterGraph wrap, dispatch) using the trace
# context recovered from the gm, the explicit ``parallel_mode``, and the
# explicit ``backend``.


# Marker attribute used to stash trace context on a graph module.
_TRACE_CTX_ATTR = "_recir_trace_ctx"


@dataclass
class _TraceContext:
    """Per-trace bundle stashed on ``gm._recir_trace_ctx``.

    Holds the references that :func:`compile_from_gm` needs to drive the
    post-trace stages: the original module, optimizer, parameter / buffer /
    optimizer-state dicts, and any embedding-specific data computed during
    the v2 trace (used by :func:`rewrite_sparse_embeddings`).
    """

    mod: nn.Module
    opts: List[torch.optim.Optimizer]
    params: Dict[str, nn.Parameter]
    buffers: Dict[str, torch.Tensor]
    named_states: Dict[str, Any]
    state_fqns: List[str]
    input_batch_dim: int
    embedding_param_indices: Optional[set[int]] = None
    internal_transforms: List[RegionTransform] = field(default_factory=list)


def _make_hybrid_sparse_transform(
    parallel_mode: HybridParallel,
    embedding_param_fqns: set[str],
    mod: nn.Module,
    forward_only: bool = False,
) -> RegionTransform:
    if forward_only:
        # Eval / inference has no embedding backward or optimizer state. Reuse
        # the tables already installed on the module by the training transform
        # (for example, TBE table views) and rewrite only their forward lookups.
        return SparseEmbeddingTransform(
            embedding_param_fqns=embedding_param_fqns,
            mesh=parallel_mode.mesh,
            mod=mod,
            embedding_plan=parallel_mode.embedding_plan,
            forward_only=True,
        )

    # The configurable factory owns the training transform: it may create
    # optimizer-backed table state and rewrite both embedding forward and
    # backward. It must not be invoked for the forward-only branch above.
    training_transform_factory = parallel_mode.sparse_transform_factory
    if training_transform_factory is not None:
        transform = training_transform_factory(
            embedding_param_fqns, parallel_mode.mesh, mod
        )
        if not isinstance(transform, RegionTransform):
            raise TypeError(
                "HybridParallel sparse_transform_factory must return a "
                f"RegionTransform, got {type(transform).__name__}."
            )
        return transform
    return SparseEmbeddingTransform(
        embedding_param_fqns=embedding_param_fqns,
        mesh=parallel_mode.mesh,
        mod=mod,
        embedding_plan=parallel_mode.embedding_plan,
        forward_only=False,
    )


def _prepare_transforms_after_shard(
    transforms: Sequence[RegionTransform],
    mod: nn.Module,
    opts: Sequence[torch.optim.Optimizer],
) -> None:
    for transform in transforms:
        transform.prepare_after_shard(mod, opts)


def _bind_transform_signatures(
    transforms: Sequence[RegionTransform],
    callable_: Callable[..., Any],
) -> None:
    for transform in transforms:
        bind = getattr(transform, "bind_signature", None)
        if bind is not None:
            bind(callable_)


def _set_trace_ctx(gm: fx.GraphModule, ctx: _TraceContext) -> None:
    """Attach ``ctx`` to ``gm`` so :func:`compile_from_gm` can recover it.

    Stored as a Python attribute (not a buffer), so it survives
    ``gm.recompile()`` but is not serialized by ``torch.save``.
    """
    setattr(gm, _TRACE_CTX_ATTR, ctx)


def _get_trace_ctx(gm: fx.GraphModule) -> _TraceContext:
    """Recover the trace context stashed by :func:`trace`."""
    ctx = getattr(gm, _TRACE_CTX_ATTR, None)
    if ctx is None:
        raise RuntimeError(
            "recir.compile_from_gm requires a gm produced by recir.trace "
            "(missing trace context attribute). If you mutated the gm via "
            "extract/replace_region, the stash is preserved automatically; "
            "if you built the gm from scratch you must call recir.trace first."
        )
    return cast(_TraceContext, ctx)


_TraceResult = Tuple[fx.GraphModule, List[torch.Tensor]]


@dataclass
class _PreparedDataParallelTrace:
    """A DP/Hybrid trace after setup and before shared parameter sharding."""

    spec: "_GraphSpec"
    parallel_mode: Union[DataParallel, HybridParallel]
    mod: nn.Module
    optimizers: List[torch.optim.Optimizer]
    mesh: Any
    dp_style: DataParallelStyle
    embedding_param_fqns: Optional[set[str]]
    transforms: List[RegionTransform]
    internal_transforms: List[RegionTransform]
    regather_transform: Optional[RegatherTransform]
    embedding_param_indices: Optional[set[int]]
    params_and_buffers: Dict[str, Any]
    parameter_fqns: set[str]
    fake_mode: FakeTensorMode
    discovery: Optional[_RegionDiscovery]
    trace_args: Tuple[Any, ...]
    trace_kwargs: Dict[str, Any]


def _trace_module_and_optimizers(
    spec: "_GraphSpec",
) -> Tuple[nn.Module, List[torch.optim.Optimizer]]:
    mod: Optional[nn.Module] = None
    optimizers: List[torch.optim.Optimizer] = []
    for arg in pytree.arg_tree_leaves(*spec.args, **spec.kwargs):
        if isinstance(arg, nn.Module) and mod is None:
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            optimizers.append(arg)
    assert mod is not None, "recir.trace requires an nn.Module in the arguments"
    return mod, optimizers


def _prepare_data_parallel_trace(
    spec: "_GraphSpec",
) -> _PreparedDataParallelTrace:
    """Perform all DP/Hybrid setup that must precede shared sharding."""
    parallel_mode = spec.parallel_mode
    assert isinstance(parallel_mode, (DataParallel, HybridParallel))
    mod, optimizers = _trace_module_and_optimizers(spec)
    transforms = list(spec.region_transforms)

    if isinstance(parallel_mode, HybridParallel):
        mesh = parallel_mode.mesh
        dp_style = parallel_mode.dense_style
        fqns = parallel_mode.embedding_plan.param_fqns
        embedding_param_fqns = fqns if fqns else None
    else:
        mesh = parallel_mode.device_mesh
        dp_style = parallel_mode.parallel_style
        embedding_param_fqns = None

    internal_transforms: List[RegionTransform] = []
    regather_transform: Optional[RegatherTransform] = None
    if dp_style == DataParallelStyle.FULLY_SHARD:
        regather_skip = RegatherTransform.skip_fqns_for_recompute(
            mod, transforms, base_skip=embedding_param_fqns
        )
        regather_transform = RegatherTransform(
            skip_param_fqns=regather_skip or None,
            forward_root_fqns=(
                parallel_mode.dense_fsdp_module_fqns
                if isinstance(parallel_mode, HybridParallel)
                else None
            ),
        )
        internal_transforms.append(regather_transform)
    if isinstance(parallel_mode, HybridParallel) and embedding_param_fqns:
        internal_transforms.append(
            _make_hybrid_sparse_transform(
                parallel_mode,
                embedding_param_fqns,
                mod,
                forward_only=not optimizers,
            )
        )
    transforms = [*internal_transforms, *transforms]
    _bind_transform_signatures(transforms, spec.fn)

    params_and_buffers: Dict[str, Any] = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }
    parameter_fqns = set(dict(mod.named_parameters(remove_duplicate=False)))
    embedding_param_indices = (
        parallel_mode._compute_embedding_param_indices(params_and_buffers)
        if isinstance(parallel_mode, HybridParallel)
        else None
    )

    fake_mode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv() if isinstance(parallel_mode, HybridParallel) else None,
    )
    input_transform = _make_trace_input_transform(
        parallel_mode.input_batch_dim,
        ratio=1 / (parallel_mode.num_ranks // parallel_mode.num_processes),
        fake_mode=fake_mode,
        dynamic_batch_dim=isinstance(parallel_mode, HybridParallel),
    )
    trace_args = pytree.tree_map_only(torch.Tensor, input_transform, spec.args)
    trace_kwargs = pytree.tree_map_only(torch.Tensor, input_transform, spec.kwargs)
    discovery = _discover_for_region_transforms(
        transforms,
        spec.fn,
        trace_args,
        trace_kwargs,
        mod=mod,
        fake_mode=fake_mode,
    )
    return _PreparedDataParallelTrace(
        spec=spec,
        parallel_mode=parallel_mode,
        mod=mod,
        optimizers=optimizers,
        mesh=mesh,
        dp_style=dp_style,
        embedding_param_fqns=embedding_param_fqns,
        transforms=transforms,
        internal_transforms=internal_transforms,
        regather_transform=regather_transform,
        embedding_param_indices=embedding_param_indices,
        params_and_buffers=params_and_buffers,
        parameter_fqns=parameter_fqns,
        fake_mode=fake_mode,
        discovery=discovery,
        trace_args=trace_args,
        trace_kwargs=trace_kwargs,
    )


def _finish_data_parallel_trace(
    prepared: _PreparedDataParallelTrace,
    actually_sharded_fqns: set[str],
) -> _TraceResult:
    """Trace one prepared graph after the shared module sharding is complete."""
    dense_sharded_param_fqns: Optional[set[str]] = None
    replicated_param_fqns: set[str] = set()
    if prepared.dp_style == DataParallelStyle.FULLY_SHARD:
        dense_sharded_param_fqns = (
            prepared.parameter_fqns & actually_sharded_fqns
        ) - set(prepared.embedding_param_fqns or ())
        replicated_param_fqns = prepared.parameter_fqns - actually_sharded_fqns
        if prepared.regather_transform is not None:
            prepared.regather_transform.skip_param_fqns.update(replicated_param_fqns)
    _prepare_transforms_after_shard(
        prepared.transforms, prepared.mod, prepared.optimizers
    )

    collectives_fn = collectives_insertion(prepared.dp_style, prepared.mesh)
    parametrize_kwargs: Dict[str, Any] = {}
    if dense_sharded_param_fqns is not None:
        parametrize_kwargs["include_param_fqns"] = dense_sharded_param_fqns
    elif prepared.embedding_param_fqns:
        parametrize_kwargs["exclude_param_fqns"] = prepared.embedding_param_fqns

    with ExitStack() as region_stack:
        _enter_region_transform_contexts(
            region_stack,
            prepared.transforms,
            prepared.discovery,
            prepared.fake_mode,
            parametrize_cm_factory=lambda root_fqn, root_mod: partial(
                parametrize_params,
                root_mod,
                collectives_fn,
                **_relative_parametrize_kwargs(root_fqn, parametrize_kwargs),
            ),
        )
        if prepared.dp_style == DataParallelStyle.FULLY_SHARD:
            outer_parametrize_fns: List[Callable[..., torch.Tensor]] = []
            outer_exclude_fqns: Optional[set[str]] = None
        else:
            outer_parametrize_fns = [collectives_fn]
            outer_exclude_fqns = prepared.embedding_param_fqns
        outer_parametrize_fns.extend(prepared.spec.parametrize_fns)
        if replicated_param_fqns:
            region_stack.enter_context(
                parametrize_params(
                    prepared.mod,
                    collectives_insertion(DataParallelStyle.REPLICATE, prepared.mesh),
                    include_param_fqns=replicated_param_fqns,
                )
            )
        local_result = _local_compile(
            prepared.spec.fn,
            *prepared.trace_args,
            fake_mode=prepared.fake_mode,
            _parametrize_fns=outer_parametrize_fns,
            _exclude_param_fqns=outer_exclude_fqns,
            _tensor_attr_holders=prepared.spec.tensor_attr_holders,
            **prepared.trace_kwargs,
        )
    return _finalize_trace_result(
        local_result,
        input_batch_dim=prepared.spec.resolved_input_batch_dim,
        embedding_param_indices=prepared.embedding_param_indices,
        internal_transforms=prepared.internal_transforms,
    )


def _trace_without_pre_sharding(spec: "_GraphSpec") -> _TraceResult:
    """Trace a GSPMD or non-parallel graph, which has no pre-trace shard phase."""
    mod, _ = _trace_module_and_optimizers(spec)
    transforms = list(spec.region_transforms)
    _bind_transform_signatures(transforms, spec.fn)
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    trace_args = spec.args
    trace_kwargs = spec.kwargs
    trace_batch_ratio = _resolve_trace_batch_ratio(spec.parallel_mode)
    if trace_batch_ratio is not None and trace_batch_ratio != 1:
        input_transform = _make_trace_input_transform(
            spec.resolved_input_batch_dim,
            ratio=trace_batch_ratio,
            fake_mode=fake_mode,
        )
        trace_args = pytree.tree_map_only(torch.Tensor, input_transform, trace_args)
        trace_kwargs = pytree.tree_map_only(torch.Tensor, input_transform, trace_kwargs)
    discovery = _discover_for_region_transforms(
        transforms,
        spec.fn,
        trace_args,
        trace_kwargs,
        mod=mod,
        fake_mode=fake_mode,
    )
    with ExitStack() as region_stack:
        _enter_region_transform_contexts(
            region_stack,
            transforms,
            discovery,
            fake_mode,
        )
        local_result = _local_compile(
            spec.fn,
            *trace_args,
            fake_mode=fake_mode,
            _stateless_func_wrapper=gradients_tagging,
            _parametrize_fns=list(spec.parametrize_fns) or None,
            _tensor_attr_holders=spec.tensor_attr_holders,
            **trace_kwargs,
        )
    return _finalize_trace_result(
        local_result,
        input_batch_dim=spec.resolved_input_batch_dim,
    )


def _finalize_trace_result(
    local_result: _LocalCompiledResult,
    *,
    input_batch_dim: int,
    embedding_param_indices: Optional[set[int]] = None,
    internal_transforms: Sequence[RegionTransform] = (),
) -> _TraceResult:
    flat_state = pytree.tree_leaves(
        [
            {**local_result.params, **local_result.buffers},
            local_result.named_states,
        ]
    )
    _tag_batch_inputs(
        local_result.gm,
        sum(isinstance(value, torch.Tensor) for value in flat_state),
        input_batch_dim,
        {
            fqn
            for fqn, parameter in local_result.params.items()
            if not parameter.requires_grad
        },
    )
    propagate_batch_input_meta(local_result.gm)
    ctx = _TraceContext(
        mod=local_result.mod,
        opts=local_result.opts,
        params=local_result.params,
        buffers=local_result.buffers,
        named_states=local_result.named_states,
        state_fqns=_compute_state_fqns(
            [
                {**local_result.params, **local_result.buffers},
                local_result.named_states,
            ]
        ),
        input_batch_dim=input_batch_dim,
        embedding_param_indices=embedding_param_indices,
        internal_transforms=list(internal_transforms),
    )
    _set_trace_ctx(local_result.gm, ctx)
    return local_result.gm, flat_state


def trace(
    callable_: Callable[..., Any],
    *args: Any,
    region_transforms: Sequence[RegionTransform] = (),
    input_batch_dim: int = 0,
    parallel_mode: Optional[ParallelMode] = None,
    _parametrize_fns: Optional[Sequence[Callable[..., torch.Tensor]]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
    **kwargs: Any,
) -> Tuple[fx.GraphModule, List[torch.Tensor]]:
    """Stage 2 standalone: trace ``callable_`` into an FX graph module.

    Wraps :func:`_local_compile` plus the ``region_transforms`` bind +
    discovery + per-root context setup that the legacy paths perform
    pre-trace.  Returns ``(gm, flat_state)`` where ``flat_state`` is 1:1 with
    the first ``len(flat_state)`` placeholders of the gm.

    The returned gm has an internal ``_recir_trace_ctx`` attribute holding the
    references (``mod``, ``opt``, ``params``, ``buffers``, ``named_states``)
    that :func:`compile_from_gm` uses to drive the post-trace stages.

    After tracing this also runs :func:`_tag_batch_inputs` +
    :func:`propagate_batch_input_meta` so every placeholder carries
    ``is_batch_input`` / ``is_weight`` / ``batch_dim`` metadata.  This makes
    external semantic input mapping work without
    falling back to the (broken) ordinal heuristic that misidentifies
    optimizer-state placeholders (e.g. Adam's 0-dim ``step`` scalar) as
    user inputs.

    Args:
        callable_: the function to trace (typically a ``train_step``).
        *args: positional arguments to ``callable_``.  Must include an
            :class:`nn.Module` (and optionally a :class:`torch.optim.Optimizer`).
        region_transforms: optional :class:`RegionTransform` sequence applied
            to scope-rewrite sub-regions of the forward graph during tracing.
            Their ``bind_signature(callable_)`` is called once before the
            outer trace and their ``_apply_region_transforms`` ExitStack
            wrappers are entered around the trace itself.
        input_batch_dim: the batch dimension on user-input tensors.  Stored on
            each user-input placeholder as ``meta["batch_dim"]`` by
            :func:`_tag_batch_inputs`.  Defaults to ``0`` (matches
            :class:`DataParallel` and most :class:`GSPMDParallel` setups).
        parallel_mode: optional :class:`ParallelMode` driving the pre-trace
            setup.  Three flavors are recognized:

              * :class:`DataParallel` / :class:`HybridParallel`: trace via
                autograd-emitted parametrization (``parametrize_params`` +
                ``collectives_insertion``) so the resulting gm has the
                allgather / reduce-scatter ops baked in.  Inputs are shrunk
                from per-process to per-device shape via the
                ``1/(world_size // num_processes)`` batch transform.
                ``shard_params`` runs BEFORE tracing so the gm captures
                sharded parameter shapes.  For :class:`HybridParallel`, the
                embedding parameters are excluded from collectives at trace
                time and rewritten post-trace by
                :func:`compile_from_gm` via :func:`rewrite_sparse_embeddings`.
              * :class:`GSPMDParallel`: expand per-process batch tensors to
                the global batch (HuggingFace-style modeling code bakes
                ``inputs_embeds.shape[0]`` as a Python int; without this
                pre-trace expansion the literal becomes stale once GSPMD
                lowering inflates placeholders to global shape).  Trace
                under :func:`gradients_tagging` so the gm carries
                ``_spmd::tag_grad`` markers consumed by
                :func:`compile_from_gm`'s GSPMD branch.
              * ``None``: trace as-is (legacy behavior for callers that
                don't request a parallelism strategy).
        _tensor_attr_holders: objects attached to the module that mutate module
            state from inside the traced step (e.g. an LR schedule controller
            holding the 0-d LR buffers).  See
            :func:`_rematerialize_tensor_attrs`.
        **kwargs: keyword arguments to ``callable_``.

    Returns:
        ``(gm, flat_state)``.
    """
    spec = _GraphSpec(
        callable_,
        args,
        parallel_mode,
        input_batch_dim=input_batch_dim,
        kwargs=kwargs,
        region_transforms=region_transforms,
        tensor_attr_holders=_tensor_attr_holders,
        parametrize_fns=_parametrize_fns or (),
    )
    if not isinstance(parallel_mode, (DataParallel, HybridParallel)):
        return _trace_without_pre_sharding(spec)

    prepared = _prepare_data_parallel_trace(spec)
    actually_sharded_fqns = parallel_mode.shard_params(
        prepared.mod,
        prepared.params_and_buffers,
        prepared.optimizers,
    )
    return _finish_data_parallel_trace(prepared, actually_sharded_fqns)


def compile_from_gm(  # noqa: C901 — orchestration: stages 4-8 of the pipeline are intentionally laid out inline so the # Stage N comments stay readable; refactor would split the pipeline across helpers.
    gm: fx.GraphModule,
    flat_state: List[torch.Tensor],
    *,
    parallel_mode: ParallelMode,
    backend: "Backend",
    region_transforms: Sequence[RegionTransform] = (),
    pipeline_config: Optional[PipelineConfig] = None,
    _sharded_by: Optional[_CompiledResult] = None,
    _shard_optimizers: Optional[Sequence[torch.optim.Optimizer]] = None,
) -> _CompiledResult:
    """Stages 4-10: continue the pipeline from a pre-traced gm.

    The full recir compile pipeline is split into ten stages, of which
    :func:`trace` runs stages 0-3 and this function runs stages 4-10:

    Pipeline stages (consult the inline ``# Stage N:`` comments below for
    the source-level boundaries):

    * **Stage 0** -- ``eager_preproc`` hooks (handled by :func:`trace`).
      User-supplied callbacks mutate the original ``nn.Module`` *before*
      tracing (e.g. attach hooks, register buffers, swap non-traceable
      submodules).  Not run by this function.
    * **Stage 2** -- ``make_fx`` trace (handled by :func:`trace`). Produce
      the initial ``fx.GraphModule`` plus its placeholder ↔ flat_state
      binding via ``_local_compile``.  Not run here.
    * **Stage 3** -- post-trace meta tagging (handled by :func:`trace`).
      Run ``_tag_batch_inputs`` + ``propagate_batch_input_meta`` so every
      placeholder carries ``is_batch_input`` / ``is_weight`` / ``batch_dim``
      metadata.  Not run here.
    * **Stage 4** -- ``region_transforms[*].finalize_compiled_graph(gm)``.
      Each :class:`RegionTransform` gets a chance to finalize after the
      graph is partitioned (note: stage-4 runs *after* stage-5 in the code
      below to match the legacy ``_compile_and_transform`` ordering).
    * **Stage 5** -- ``parallel_mode.shard_params`` + ``partition``.
      ``shard_params`` updates ``opt.state`` and may re-key parameters
      in-place; we then re-extract ``named_states`` by walking the
      now-sharded module.  For DP/FSDP/Hybrid the graph is already
      parametrized at trace time; GSPMD needs no DTensor partition pass.
      Also re-flattens the graph via ``_to_caller_flattened_graph_module``
      so the caller can invoke ``gm(*flat_state, *user_inputs)`` directly,
      and rebuilds ``flat_state`` from the post-shard params/buffers/states.
    * **Stage 6** -- ``backend.optimize_passes`` applied to
      ``(gm, flat_state)``.  Each pass may rewrite the graph and/or update
      flat_state.  Passes are dispatched by ``inspect.signature`` arity:
      two-argument passes get ``(gm, flat_state)``; single-argument passes
      get ``(gm,)``.
    * **Stage 7** -- ``IterGraphModule(gm)`` wrap if
      ``backend.uses_iter_graph``.  Wraps the graph in an
      :class:`IterGraphModule` so cross-iteration passes (e.g. comm-fusion,
      lazy-lowering hooks) can operate.
    * **Stage 8** -- ``backend.dispatch(gm)``.  Backend-owned final
      rewrite.  For ``XLA*Backend`` this performs ``partial_lower_xla`` to
      splice ``_XLAModule`` call_modules into the graph; for
      ``InductorBackend`` this is a no-op pass-through.
    * **Stage 9** -- legacy ``_finalize_graph`` (handled by the
      :func:`compile` wrapper for the data-parallel/GSPMD paths, not by
      this function).
    * **Stage 10** -- runtime-time wrapping (handled by :func:`compile` /
      :func:`sync_device_state_to_torch`, not by this function).

    Args:
        gm: a graph module produced by :func:`trace` (must carry the
            internal trace context attribute populated by ``trace`` via
            ``_set_trace_ctx``).
        flat_state: the flat state list mirroring the leading placeholders of
            ``gm``.  Reused by the returned ``_CompiledResult``.
        parallel_mode: how to partition the graph across ranks.  Validated
            against ``backend`` via ``backend.validate_with(parallel_mode)``
            before stage 5 runs.
        backend: owns stages 6-8 (optimize passes, IterGraph wrap, dispatch).
        region_transforms: same sequence passed to :func:`trace`.  Each
            transform's ``finalize_compiled_graph(gm)`` is invoked after
            partition (stage 4 in code-order, but conceptually stage 4 of
            the pipeline).
        _sharded_by: set by :func:`_compile_graphs` for every graph after
            the first, which already sharded the module: reuse its placements
            instead of sharding again.
        _shard_optimizers: all optimizers owned by graphs compiled together.
            The first GSPMD graph shards them with the module even when that
            graph itself is forward-only.

    Returns:
        A :class:`_CompiledResult` ready for runtime use via
        :func:`recir.compile`'s wrapper invocation path.
    """
    # Validate backend ↔ parallel_mode compatibility early.
    backend.validate_with(parallel_mode)

    ctx = _get_trace_ctx(gm)
    region_transforms_list = list(region_transforms)
    trace_state_fqns = ctx.state_fqns
    trace_state_leaf_count = len(trace_state_fqns)
    if len(flat_state) < trace_state_leaf_count:
        raise RuntimeError(
            "compile_from_gm flat_state is missing trace-time state leaves: "
            f"state={len(flat_state)}, trace_state={trace_state_leaf_count}"
        )
    added_state_tail = flat_state[trace_state_leaf_count:]

    # Stage 5: per-mode post-trace assembly.
    #
    # The trace context already contains the (post-shard, for DP/HybridParallel)
    # ``params`` / ``buffers`` / ``named_states`` snapshot captured inside
    # :func:`trace`.  Branch shape:
    #
    #   * :class:`DataParallel` / :class:`HybridParallel`: ``shard_params`` ran
    #     inside :func:`trace` so the autograd-emitted parametrization could
    #     bake the per-parameter collectives into ``gm`` at trace time.  Here
    #     we just consume the ctx snapshot — no further partition pass is
    #     required (the DTensor partitioner is skipped entirely on this
    #     path).  For :class:`HybridParallel`, the embedding parameters were
    #     excluded from collectives at trace time; rewrite them now into
    #     decomposed AllToAll ops via :class:`SparseEmbeddingTransform`.
    #   * :class:`GSPMDParallel`: ``shard_params`` is deferred to this stage
    #     because the GSPMD lowering inflates placeholders to the global
    #     batch post-trace; sharding before the trace would leave the gm
    #     with per-shard literals that conflict with the inflated
    #     placeholders.  GSPMD needs no DTensor partition pass, so after
    #     sharding we scale + erase the autograd-emitted ``_spmd::tag_grad``
    #     markers so XLA lowering can proceed.
    sharded_parameter_fqns: set[str] = set()
    optimizer_state_only_sharded_keys: frozenset[tuple[str, str]] = frozenset()
    fsdp_base_state: Optional[Tuple[Any, ...]] = None
    fsdp_state_flags: Optional[list[bool]] = None
    process_state: Optional[CompiledProcessState] = None
    if isinstance(parallel_mode, (DataParallel, HybridParallel)):
        # Trace already sharded; reuse ctx snapshot.
        params_and_buffers: Dict[str, Any] = {**ctx.params, **ctx.buffers}
        named_states: Dict[str, Any] = ctx.named_states

        # Let the parallel mode stamp any state-layout meta it produced during
        # ``shard_params`` onto the graph and substitute its canonical host store
        # into ``params_and_buffers`` / ``named_states`` (both rebuilt into
        # ``flat_state`` below). No-op for modes that produce no layout; keeps
        # this stage layout-agnostic.
        parallel_mode.apply_state_layout(gm, params_and_buffers, named_states)
        process_state = _build_compiled_process_state(
            gm,
            params_and_buffers,
            named_states,
        )

        # Finalize HybridParallel-internal semantic transforms before state is
        # flattened and before IterGraph scheduling. The default sparse
        # transform preserves the existing ATen embedding path; an alternate
        # transform may materialize a TBE-backed sparse graph.
        for transform in ctx.internal_transforms:
            transform.finalize_compiled_graph(gm)
    else:
        # GSPMDParallel.  Mirror the legacy ``_compile_gspmd`` ordering:
        # shard_params first (updates ``opt.state`` and may re-key
        # parameters), then re-extract named_states by walking the
        # now-sharded module.  GSPMD needs no DTensor partition pass.
        params_and_buffers = {**ctx.params, **ctx.buffers}
        if _sharded_by is None:
            sharded_parameter_fqns = parallel_mode.shard_params(
                ctx.mod,
                params_and_buffers,
                ctx.opts if _shard_optimizers is None else _shard_optimizers,
            )
            optimizer_state_only_sharded_keys = getattr(
                parallel_mode,
                "optimizer_state_only_sharded_keys",
                frozenset(),
            )
        else:
            sharded_parameter_fqns = set(_sharded_by.sharded_parameter_fqns)
            optimizer_state_only_sharded_keys = (
                _sharded_by.optimizer_state_only_sharded_keys
            )
        rebuilt_named_states: Dict[str, Any] = {}
        _opts = ctx.opts
        if ctx.named_states:
            # ``shard_params`` may have re-keyed each ``opt.state`` to the new
            # (post-shard) Parameter identities. Resolve the current FQN bindings,
            # then restore the trace-time dict order: pytree flattening is
            # positional, so module order would misalign state leaves with the
            # graph placeholders when optimizer groups interleave in the module.
            for n, p in ctx.mod.named_parameters(remove_duplicate=False):
                for o in _opts:
                    if p in o.state:
                        rebuilt_named_states[n] = o.state[p]  # type: ignore[index]
                        break
            named_states = {
                n: rebuilt_named_states[n]
                for n in ctx.named_states
                if n in rebuilt_named_states
            }
            dropped = set(ctx.named_states) - set(named_states)
            if dropped:
                raise RuntimeError(
                    "GSPMD named_states rebuild lost trace-time optimizer state; "
                    "cannot safely align flat_state with graph placeholders: "
                    f"dropped={sorted(dropped)!r}"
                )
        else:
            # No optimizer state was lifted into the trace, so there are no
            # optimizer-state placeholders to rebuild after sharding.
            named_states = ctx.named_states

        # GSPMD-specific tag_grad fixup.  GSPMD needs no DTensor partition
        # pass, so the autograd-emitted ``_spmd::tag_grad`` markers survive
        # into the lowered graph; they must be scaled (for
        # ``grad_reduction == "avg"``) and erased here.  Mirrors the
        # post-trace block in the legacy ``_compile_gspmd``:
        #   * scale: GSPMD lowers the autograd-emitted grad accumulation
        #     through an all-reduce SUM.  For sum-loss + "avg" grad
        #     reduction (DDP semantics), divide each tag_grad's output by
        #     ``num_ranks``.  For mean-loss (grad_reduction="sum"), the
        #     traced backward already normalises by global numel so the SUM
        #     alone is correct.
        #   * erase: tag_grad has no XLA lowering, so leaving the markers
        #     in the graph triggers ``_is_xla_compatible`` failures at
        #     ``segment_and_lower`` time (the symptom that surfaces as
        #     ``test_no_graph_breaks_gspmd`` failing).
        if isinstance(parallel_mode, GSPMDParallel):
            num_ranks = parallel_mode.num_ranks
            if num_ranks > 1 and parallel_mode.grad_reduction == "avg":
                for node in list(gm.graph.nodes):
                    if (
                        node.op == "call_function"
                        and node.target is torch.ops._spmd.tag_grad.default
                    ):
                        with gm.graph.inserting_after(node):
                            div_node = gm.graph.call_function(
                                torch.ops.aten.div.Scalar, args=(node, num_ranks)
                            )
                            div_node.meta = node.meta.copy() if node.meta else {}
                            # Filter out div_node itself so the rewrite skips
                            # the freshly-created (node, num_ranks) edge --
                            # otherwise div_node would self-reference and we
                            # would need to restore div_node.args post-hoc.
                            # Default-arg ``dn=div_node`` pins the current
                            # iteration's div_node (avoids B023 late binding).
                            node.replace_all_uses_with(
                                div_node, lambda u, dn=div_node: u is not dn
                            )
            for node in list(gm.graph.nodes):
                if (
                    node.op == "call_function"
                    and node.target is torch.ops._spmd.tag_grad.default
                ):
                    node.replace_all_uses_with(node.args[0])
                    gm.graph.erase_node(node)
            gm.recompile()

    # Move the pytree-flattening responsibility from the gm to the caller so
    # the recir wrapper can invoke ``gm(*flat_inps)`` directly.  Mirrors the
    # legacy ``_finalize_graph`` step; without this, the gm's codegen still
    # expects the original nested-pytree (model, opt, *args, **kwargs)
    # structure and runtime invocation explodes.
    gm = _to_caller_flattened_graph_module(gm)

    # Rebuild flat_state from the post-shard params/buffers/named_states.
    # ``shard_params`` may have substituted parameter objects, so the
    # original ``flat_state`` references can be stale.
    state_structure = [params_and_buffers, named_states]
    rebuilt_state = pytree.tree_leaves(state_structure)
    rebuilt_state_fqns = _compute_state_fqns(state_structure)
    if rebuilt_state_fqns != trace_state_fqns:
        raise RuntimeError(
            "Post-shard state structure no longer matches trace-time graph "
            "placeholders; cannot safely rebuild flat_state: "
            f"trace={trace_state_fqns!r}, rebuilt={rebuilt_state_fqns!r}"
        )
    rebuilt_state.extend(added_state_tail)
    flat_state = rebuilt_state

    # GSPMD/FSDP: mirror legacy ``_compile_gspmd`` by setting
    # ``is_fsdp_sharded`` on placeholders that correspond to FSDP-sharded
    # params/states.  ``_jit_gspmd`` inspects this meta to decide which
    # inputs are local shards (vs replicated tensors).  Without the tag,
    # the GSPMD lowering would treat sharded inputs as replicated and the
    # cross-process gather/all-reduce shape contract breaks.  Only populated
    # for the GSPMD branch above — the DP/HybridParallel branch consumes
    # the parametrized gm directly and does not need placeholder-level
    # FSDP tagging.
    if sharded_parameter_fqns or optimizer_state_only_sharded_keys:
        fsdp_state_flags = _build_fsdp_sharded_flags(
            params_and_buffers,
            named_states,
            sharded_parameter_fqns,
            optimizer_state_only_sharded_keys,
        )
        fsdp_base_state = tuple(flat_state)
        placeholder_flags = _align_fsdp_sharded_flags(
            fsdp_base_state,
            fsdp_state_flags,
            flat_state,
        )
        _annotate_fsdp_state_input_sharding(gm, flat_state, placeholder_flags)
        _annotate_fsdp_state_output_sharding(gm, flat_state, placeholder_flags)

    # Stage 4 (post-partition is the legacy ordering used by
    # ``_compile_and_transform``): region transforms get a chance to finalize
    # after the graph has been partitioned.
    for transform in region_transforms_list:
        transform.finalize_compiled_graph(gm)

    pipeline = pipeline_config or PipelineConfig.none()
    use_iter_graph = backend.uses_iter_graph and not pipeline.enabled
    # Stage 5.5: wrap before running passes that use IterGraph-only APIs.
    if use_iter_graph:
        # Local import to avoid pulling IterGraphModule into the
        # module-import path for callers that don't use this path.
        from recir.iter_graph_module import IterGraphModule

        gm = cast(fx.GraphModule, IterGraphModule(gm))

    # Stage 6: backend-configured optimize passes.  Each pass may rewrite the
    # graph and/or update flat_state.  Some passes accept (gm,) and others
    # accept (gm, flat_state) — dispatch via signature inspection so a
    # ``TypeError`` raised from inside ``opt_pass`` is not silently swapped
    # for the single-arg form.
    #
    # Only count *positional* params toward the arity check.  Otherwise a
    # signature like ``def my_pass(gm, *, mode="x")`` (1 positional + 1
    # kw-only) would falsely set ``wants_state=True`` and we'd call
    # ``opt_pass(gm, flat_state)`` → ``TypeError`` at runtime.
    _POSITIONAL_KINDS = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    # Pass ordering/"apply once" bookkeeping is per graph; reset it so one
    # process can compile several graphs (e.g. a train and an eval step).
    default_pass_registry.reset()
    for opt_pass in backend.optimize_passes:
        try:
            sig_params = inspect.signature(opt_pass).parameters.values()
            positional_params = [p for p in sig_params if p.kind in _POSITIONAL_KINDS]
            wants_state = len(positional_params) >= 2
        except (TypeError, ValueError):
            # ``inspect.signature`` raises ``ValueError`` on builtins /
            # C-extensions and ``TypeError`` on non-callables; in either
            # case fall back to the legacy single-arg form.
            wants_state = False
        if wants_state:
            result = opt_pass(gm, flat_state)
        else:
            result = opt_pass(gm)
        if isinstance(result, tuple) and len(result) == 2:
            gm, flat_state = result  # pyre-ignore[9]
        elif isinstance(result, fx.GraphModule):
            gm = result

    if fsdp_base_state is not None and fsdp_state_flags is not None:
        final_state_flags = _align_fsdp_sharded_flags(
            fsdp_base_state,
            fsdp_state_flags,
            flat_state,
        )
        _annotate_fsdp_state_input_sharding(gm, flat_state, final_state_flags)
        _annotate_fsdp_state_output_sharding(gm, flat_state, final_state_flags)

    # Stage 7: freeze cross-iteration rewrites and initialize extra-output state.
    if use_iter_graph:
        from recir.iter_graph_module import IterGraphModule

        cast(IterGraphModule, gm).finalize_setup()

    scheduled_gm: nn.Module = build_pipeline(gm, pipeline)

    # Stage 8: dispatch (backend.dispatch returns an nn.Module — typically
    # the same gm rewritten as a partial-lowered module, but the contract
    # only requires it be a callable nn.Module compatible with the flat-state
    # invocation convention).
    dispatched = backend.dispatch(cast(fx.GraphModule, scheduled_gm))
    # Preserve the gm reference (the dispatched module is typically the gm
    # itself, rewritten in place).  ``_CompiledResult.gm`` is callable, so
    # we use the dispatched module here.
    final_gm = dispatched

    sm = backend.device_state_manager()
    sm.state_fqns = rebuilt_state_fqns
    return _CompiledResult(
        gm=cast(fx.GraphModule, final_gm),
        mod=ctx.mod,
        opts=ctx.opts,
        flat_state=flat_state,
        state_manager=sm,
        process_state=process_state,
        sharded_parameter_fqns=frozenset(sharded_parameter_fqns),
        optimizer_state_only_sharded_keys=frozenset(optimizer_state_only_sharded_keys),
    )


def _bind_to_current_module_state(ctx: _TraceContext) -> None:
    """Point the trace's params/buffers at the module's current tensors, which
    a later trace or compile may have replaced (sharding, TBE lowering)."""
    params = dict(ctx.mod.named_parameters(remove_duplicate=False))
    buffers = dict(ctx.mod.named_buffers(remove_duplicate=False))
    ctx.params = {fqn: params[fqn] for fqn in ctx.params}
    ctx.buffers = {fqn: buffers[fqn] for fqn in ctx.buffers}


@dataclass
class _GraphSpec:
    """One graph for :func:`_trace_graphs`: ``fn(*args, **kwargs)`` traced as
    :func:`trace` would trace it."""

    fn: Callable[..., Any]
    args: Tuple[Any, ...]
    parallel_mode: Optional[ParallelMode]
    input_batch_dim: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    region_transforms: Sequence[RegionTransform] = ()
    tensor_attr_holders: Sequence[Any] = ()
    parametrize_fns: Sequence[Callable[..., torch.Tensor]] = ()

    @property
    def resolved_input_batch_dim(self) -> int:
        if self.input_batch_dim is not None:
            return self.input_batch_dim
        if self.parallel_mode is not None:
            return self.parallel_mode.input_batch_dim
        return 0


def _trace_graphs(specs: Sequence[_GraphSpec]) -> List[_TraceResult]:
    """Trace several graphs over ONE module that will share its parameters
    (e.g. a train step and a forward-only eval step).

    The module is sharded once, for all of them: DataParallel / HybridParallel
    shard before tracing, so every graph finishes its pre-shard setup, the
    module is sharded with every graph's optimizers, then each is traced.
    GSPMD shards at compile time (:func:`_compile_graphs`). Each returned
    ``(gm, flat_state)`` may be rewritten (graph dispatchers) before compiling.
    """
    mods = {
        id(arg): arg
        for spec in specs
        for arg in pytree.arg_tree_leaves(*spec.args, **spec.kwargs)
        if isinstance(arg, nn.Module)
    }
    if len(mods) != 1:
        raise ValueError("graphs compiled together must take the same module")
    trace_results: List[Optional[_TraceResult]] = [None] * len(specs)
    prepared_traces: List[Tuple[int, _PreparedDataParallelTrace]] = []
    for trace_index, spec in enumerate(specs):
        if isinstance(spec.parallel_mode, (DataParallel, HybridParallel)):
            prepared_traces.append((trace_index, _prepare_data_parallel_trace(spec)))
        else:
            trace_results[trace_index] = _trace_without_pre_sharding(spec)

    if prepared_traces:
        canonical_preparation = prepared_traces[0][1]
        all_optimizers: List[torch.optim.Optimizer] = []
        for _, prepared in prepared_traces:
            all_optimizers.extend(
                optimizer
                for optimizer in prepared.optimizers
                if all(optimizer is not existing for existing in all_optimizers)
            )
        actually_sharded_fqns = canonical_preparation.parallel_mode.shard_params(
            next(iter(mods.values())),
            canonical_preparation.params_and_buffers,
            all_optimizers,
        )
        canonical_state = canonical_preparation.params_and_buffers
        for trace_index, prepared in prepared_traces:
            prepared.params_and_buffers.update(
                {fqn: canonical_state[fqn] for fqn in prepared.params_and_buffers}
            )
            trace_results[trace_index] = _finish_data_parallel_trace(
                prepared, actually_sharded_fqns
            )

    traced: List[_TraceResult] = []
    for result in trace_results:
        assert result is not None
        traced.append(result)
    return traced


def _compile_graphs(
    traced: Sequence[_TraceResult],
    *,
    parallel_modes: Sequence[ParallelMode],
    backend: "Backend",
    region_transforms: Optional[Sequence[Sequence[RegionTransform]]] = None,
    pipeline_configs: Optional[Sequence[Optional[PipelineConfig]]] = None,
) -> List[_CompiledResult]:
    """Compile graphs from :func:`_trace_graphs`; they share one set of
    parameters. Each graph binds to the module's tensors as they are when it
    compiles, after every trace-time rewrite (e.g. TBE lowering) and any
    earlier graph's compile-time sharding; the first graph does the GSPMD
    sharding and every later one reuses its placements. On backends
    with device-resident state (XLA) the graphs also share one device state,
    so each sees the others' updates without a host round trip. Wrap each
    result with :func:`make_compiled_runner` and run them independently.
    """
    count = len(traced)
    transforms = region_transforms or [()] * count
    pipelines = pipeline_configs or [None] * count
    if not len(parallel_modes) == len(transforms) == len(pipelines) == count:
        raise ValueError("need one entry per traced graph")
    all_optimizers: List[torch.optim.Optimizer] = []
    for gm, _ in traced:
        for optimizer in _get_trace_ctx(gm).opts:
            if all(optimizer is not existing for existing in all_optimizers):
                all_optimizers.append(optimizer)
    compiled: List[_CompiledResult] = []
    for (gm, flat_state), mode, graph_transforms, pipeline in zip(
        traced, parallel_modes, transforms, pipelines
    ):
        _bind_to_current_module_state(_get_trace_ctx(gm))
        compiled.append(
            compile_from_gm(
                gm,
                flat_state,
                parallel_mode=mode,
                backend=backend,
                region_transforms=graph_transforms,
                pipeline_config=pipeline,
                _sharded_by=compiled[0] if compiled else None,
                _shard_optimizers=all_optimizers,
            )
        )
    shared_device_state: Dict[str, Any] = {}
    for result in compiled:
        result.state_manager.share_device_state(
            shared_device_state, result.gm, result.flat_state
        )
    return compiled


def _compile_via_backend(
    calls: Sequence[Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]],
    parallel_mode: ParallelMode,
    backend: "Backend",
    *,
    eager_preproc: Sequence[Callable[[nn.Module], None]],
    region_transforms: Optional[List[RegionTransform]],
    _parametrize_fns: Sequence[Callable[..., torch.Tensor]] = (),
) -> List[_CompiledResult]:
    """Internal: drive the trace + compile_from_gm pipeline for ``(func, args,
    kwargs)`` calls over one module, compiled together (one call is the usual
    single-function compile). Used only when the user passes ``backend=`` to
    :func:`compile`.
    """
    rt = list(region_transforms or ())
    graph_region_transforms = [rt] if len(calls) == 1 else [deepcopy(rt) for _ in calls]

    # Stage 0: apply eager pre-trace hooks to every nn.Module argument, once.
    # Multi-module signatures (e.g. model + loss module, teacher/student)
    # all need preprocessing -- not just the first one encountered.
    modules = {
        id(arg): arg
        for _, args, kwargs in calls
        for arg in pytree.arg_tree_leaves(*args, **kwargs)
        if isinstance(arg, nn.Module)
    }
    for module in modules.values():
        for hook in eager_preproc:
            hook(module)

    # Stage 2: trace (sharding the module once for all calls). The parallel
    # mode drives the pre-trace setup, e.g. GSPMD expands per-process tensors
    # to the global batch so modeling code that bakes ``shape[0]`` as an int
    # sees the global size.
    traced = _trace_graphs(
        [
            _GraphSpec(
                func,
                args,
                parallel_mode,
                input_batch_dim=parallel_mode.input_batch_dim,
                kwargs=kwargs,
                region_transforms=graph_rt,
                parametrize_fns=_parametrize_fns,
            )
            for (func, args, kwargs), graph_rt in zip(calls, graph_region_transforms)
        ]
    )

    # Stages 4-10.
    return _compile_graphs(
        traced,
        parallel_modes=[parallel_mode] * len(calls),
        backend=backend,
        region_transforms=graph_region_transforms,
    )


def _compile_data_parallel_v2(  # noqa: C901 -- legacy orchestration kept cohesive
    func: Callable[..., Any],
    parallel_mode: Union[DataParallel, HybridParallel],
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
    **kwargs: Any,
) -> _CompiledResult:
    """Compile with autograd-based collective insertion (no DTensor partition).

    Handles both ``DataParallel`` (dense-only) and ``HybridParallel``
    (dense + embedding sharding) v2 paths.

    Caller-supplied ``region_transforms`` (e.g. activation checkpointing via
    :class:`RecomputeTransform`) are appended to the path-internal transforms
    (``RegatherTransform`` for FULLY_SHARD, ``SparseEmbeddingTransform`` for
    HybridParallel).  All transforms share a single discovery +
    ``_apply_region_transforms`` pass.
    """
    # Extract mesh and DP style from the parallel mode.
    if isinstance(parallel_mode, HybridParallel):
        mesh = parallel_mode.mesh
        dp_style = parallel_mode.dense_style
    else:
        mesh = parallel_mode.device_mesh
        dp_style = parallel_mode.parallel_style
    world_size: int = parallel_mode.num_ranks

    # 1. Pre-extract mod and ALL Optimizer instances (need originals
    #    for pre-trace sharding; multi-opt setups partition params into
    #    sparse + dense buckets and pass two opts).
    mod: Optional[nn.Module] = None
    opts: List[torch.optim.Optimizer] = []
    for arg in pytree.arg_tree_leaves(*args, **kwargs):
        if isinstance(arg, nn.Module):
            mod = arg
        if isinstance(arg, torch.optim.Optimizer):
            opts.append(arg)
    assert mod is not None
    _mod: nn.Module = mod

    # 2. Resolve embedding param FQNs from the plan (HybridParallel only).
    embedding_param_fqns: Optional[set[str]] = None
    if isinstance(parallel_mode, HybridParallel):
        fqns = parallel_mode.embedding_plan.param_fqns
        embedding_param_fqns = fqns if fqns else None

    # 3. Build transforms list (RegatherTransform for FULLY_SHARD + caller
    #    region_transforms).  Embedding params are excluded from the regather
    #    so that the post-compile sparse-embedding rewrite owns them.
    transforms: list[RegionTransform] = []
    regather_transform: Optional[RegatherTransform] = None
    if dp_style == DataParallelStyle.FULLY_SHARD:
        regather_transform = RegatherTransform(
            skip_param_fqns=embedding_param_fqns,
            forward_root_fqns=(
                parallel_mode.dense_fsdp_module_fqns
                if isinstance(parallel_mode, HybridParallel)
                else None
            ),
        )
        transforms.append(regather_transform)
    if isinstance(parallel_mode, HybridParallel) and embedding_param_fqns:
        transforms.append(
            _make_hybrid_sparse_transform(parallel_mode, embedding_param_fqns, _mod)
        )
    # Internal transforms whose ``finalize_compiled_graph`` must be invoked
    # by this path (the caller-supplied ``region_transforms`` are finalized
    # by ``_compile_and_transform``).
    internal_transforms: list[RegionTransform] = list(transforms)
    _bind_transform_signatures(internal_transforms, func)
    if region_transforms:
        transforms.extend(region_transforms)

    # 4. Extract params_and_buffers.  Critical for correctness: shard_params
    #    mutates _mod by FQN, and embedding indices into this dict must map
    #    to placeholder positions in the traced graph (which uses the
    #    post-shard params ordering).
    params_and_buffers: Dict[str, Any] = {
        **dict(_mod.named_parameters(remove_duplicate=False)),
        **dict(_mod.named_buffers(remove_duplicate=False)),
    }
    parameter_fqns = set(dict(_mod.named_parameters(remove_duplicate=False)))

    # 5. Set up external fake mode for tracing.
    #    HybridParallel only: a ShapeEnv is attached so custom ops with
    #    data-dependent output shapes (``recir_torchrec::kjt_split_per_feature``,
    #    ``recir_preproc::mot_preprocessor_forward``) can mint fresh SymInts via
    #    ``torch.library.get_ctx().new_dynamic_size()`` from their fake kernels.
    #    Without a ShapeEnv those calls raise ``DynamicOutputShapeException``.
    fake_mode: FakeTensorMode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv() if isinstance(parallel_mode, HybridParallel) else None,
    )

    # Convert every input through the shared FakeTensorMode and resize it when
    # one process owns multiple devices. HybridParallel currently has one
    # process per GPU, so its ratio is 1: no resize occurs, but this call still
    # keeps input_batch_dim symbolic for candidate/jagged inputs.
    input_transform = _make_trace_input_transform(
        parallel_mode.input_batch_dim,
        ratio=1 / (world_size // parallel_mode.num_processes),
        fake_mode=fake_mode,
        dynamic_batch_dim=isinstance(parallel_mode, HybridParallel),
    )
    args = pytree.tree_map_only(torch.Tensor, input_transform, args)
    kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
    discovery = _discover_for_region_transforms(
        transforms, func, args, kwargs, mod=_mod, fake_mode=fake_mode
    )

    # 6. Shard params on the module and optimizer states.
    #    shard_params mutates both the dict and the module in-place.
    #    It also updates each opt's state + param_groups so that
    #    parameter -> state mappings remain valid after sparse Parameter
    #    identity swaps.
    sharded_parameter_fqns = parallel_mode.shard_params(_mod, params_and_buffers, opts)
    dense_sharded_param_fqns: Optional[set[str]] = None
    replicated_param_fqns: set[str] = set()
    if (
        isinstance(parallel_mode, (DataParallel, HybridParallel))
        and dp_style == DataParallelStyle.FULLY_SHARD
    ):
        dense_sharded_param_fqns = (parameter_fqns & sharded_parameter_fqns) - set(
            embedding_param_fqns or ()
        )
        replicated_param_fqns = parameter_fqns - sharded_parameter_fqns
        if regather_transform is not None:
            regather_transform.skip_param_fqns.update(replicated_param_fqns)
    _prepare_transforms_after_shard(transforms, _mod, opts)

    # 7. Apply region transforms AFTER sharding, with parametrization.
    #    The traced forward includes allgather ops from collectives_insertion
    #    so each per-parameter region encompasses allgather + chain + compute.
    #    Embedding params (if any) are excluded from collectives — they use
    #    AllToAll (inserted in step 9), not allgather.
    collectives_fn = collectives_insertion(dp_style, mesh)
    parametrize_kwargs: Dict[str, Any] = {}
    if dense_sharded_param_fqns is not None:
        parametrize_kwargs["include_param_fqns"] = dense_sharded_param_fqns
    elif embedding_param_fqns:
        parametrize_kwargs["exclude_param_fqns"] = embedding_param_fqns
    with ExitStack() as region_stack:
        _enter_region_transform_contexts(
            region_stack,
            transforms,
            discovery,
            fake_mode,
            parametrize_cm_factory=lambda root_fqn, root_mod: partial(
                parametrize_params,
                root_mod,
                collectives_fn,
                **_relative_parametrize_kwargs(root_fqn, parametrize_kwargs),
            ),
        )

        # 8. Set up parametrize fns and trace via _local_compile.
        #    Region transform context managers scope the rewritten forwards
        #    to this block — originals are restored when the ExitStack exits.
        #
        #    Whether the OUTER trace must parametrize the dense params with
        #    ``collectives_fn`` depends on the dense style, NOT merely on
        #    whether region transforms exist:
        #      * FULLY_SHARD always adds ``RegatherTransform``, which bakes the
        #        dense allgather collectives into its ``gm_fwd``; re-applying
        #        ``collectives_fn`` here would double-allgather, so pass [].
        #      * REPLICATE has NO dense region transform (HybridParallel only
        #        adds the embedding-scoped ``SparseEmbeddingTransform``), so the
        #        dense-grad all_reduce must be inserted by parametrizing dense
        #        params in the outer trace. Embedding params are excluded so
        #        they keep their all_to_all path (``None`` when no embeddings).
        if dp_style == DataParallelStyle.FULLY_SHARD:
            _outer_parametrize_fns: List[Callable[[torch.Tensor], torch.Tensor]] = []
            _outer_exclude_fqns: Optional[set[str]] = None
        else:
            _outer_parametrize_fns = [collectives_fn]
            _outer_exclude_fqns = embedding_param_fqns
        if replicated_param_fqns:
            region_stack.enter_context(
                parametrize_params(
                    _mod,
                    collectives_insertion(DataParallelStyle.REPLICATE, mesh),
                    include_param_fqns=replicated_param_fqns,
                )
            )
        local_result = _local_compile(
            func,
            *args,
            fake_mode=fake_mode,
            _parametrize_fns=_outer_parametrize_fns,
            _exclude_param_fqns=_outer_exclude_fqns,
            _tensor_attr_holders=_tensor_attr_holders,
            **kwargs,
        )

    # 9. Finalize.
    result = _finalize_graph(
        local_result.gm,
        local_result.mod,
        local_result.opts,
        {**local_result.params, **local_result.buffers},
        local_result.named_states,
        parallel_mode.input_batch_dim,
    )
    result.internal_transforms = internal_transforms
    return result


def _compile_gspmd(
    func: Callable[..., Any],
    parallel_mode: GSPMDParallel,
    *args: Any,
    region_transforms: Optional[List[RegionTransform]] = None,
    _tensor_attr_holders: Sequence[Any] = (),
    **kwargs: Any,
) -> _CompiledResult:
    """Compile a train step with GSPMD auto-parallelism.

    Unlike the DataParallel path, this skips DTensor partitioning entirely.
    The graph is traced with global batch shapes (expanded from the local
    mini-batch by world_size) and metadata is attached for the GSPMD
    lowering step.

    When ``region_transforms`` is provided, region-level rewrites (e.g.
    activation checkpointing via :class:`RecomputeTransform`) are applied
    to the user's forward methods during tracing.  They are scoped to the
    trace via an ``ExitStack`` and reverted before this function returns.
    """
    fake_mode: FakeTensorMode = FakeTensorMode()
    transforms = region_transforms or []
    # Apply the per-process → global batch transform up front, so discovery,
    # region-transform inner trace, and the outer ``_local_compile`` trace
    # all see the same (global) input shape.  Otherwise the region's
    # ``gm_fwd`` would bake per-process shapes (e.g. ``aten.view`` target
    # lists) that conflict with the outer trace's expanded inputs.
    input_transform = _make_trace_input_transform(
        parallel_mode.input_batch_dim,
        ratio=parallel_mode.num_processes,
        fake_mode=fake_mode,
    )
    args = pytree.tree_map_only(torch.Tensor, input_transform, args)
    kwargs = pytree.tree_map_only(torch.Tensor, input_transform, kwargs)
    discovery = _discover_for_region_transforms(
        transforms, func, args, kwargs, fake_mode=fake_mode
    )
    with ExitStack() as region_stack:
        _enter_region_transform_contexts(region_stack, transforms, discovery, fake_mode)
        local_result = _local_compile(
            func,
            *args,
            fake_mode=fake_mode,
            _stateless_func_wrapper=gradients_tagging,
            _tensor_attr_holders=_tensor_attr_holders,
            **kwargs,
        )
    gm = local_result.gm
    # GSPMD auto-inserts all-reduce sum for gradients
    # all-reduce average. Use the tag_grad markers to find gradient tensors
    # and scale them by 1/num_ranks to match DDP's averaging semantics.
    # For mean loss (grad_reduction="sum"), skip the scaling since the
    # traced backward already normalises by the global numel — the XLA
    # all-reduce sum alone produces the correct global-mean gradient.
    # Then remove the tag_grad nodes since they've served their purpose.
    num_ranks = parallel_mode.num_ranks
    if num_ranks > 1 and parallel_mode.grad_reduction == "avg":
        for node in list(gm.graph.nodes):
            if (
                node.op == "call_function"
                and node.target is torch.ops._spmd.tag_grad.default
            ):
                with gm.graph.inserting_after(node):
                    div_node = gm.graph.call_function(
                        torch.ops.aten.div.Scalar, args=(node, num_ranks)
                    )
                    div_node.meta = node.meta.copy() if node.meta else {}
                    # Filter out div_node itself so the rewrite skips the
                    # freshly-created (node, num_ranks) edge -- otherwise
                    # div_node would self-reference and we would need to
                    # restore div_node.args post-hoc.
                    # Default-arg ``dn=div_node`` pins the current iteration's
                    # div_node (avoids B023 late binding).
                    node.replace_all_uses_with(
                        div_node, lambda u, dn=div_node: u is not dn
                    )
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops._spmd.tag_grad.default
        ):
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
    gm.recompile()

    params_and_buffers = {**local_result.params, **local_result.buffers}
    # FSDP: shard model params AFTER tracing but BEFORE _finalize_graph.
    # The graph retains global shapes; runtime inputs are local shards
    # (_to_sharded_jax bridges the gap). Sharding before _finalize_graph
    # means flat_state is built from already-sharded tensors.
    sharded_parameter_fqns = parallel_mode.shard_params(
        local_result.mod, params_and_buffers, local_result.opts
    )
    optimizer_state_only_sharded_keys = parallel_mode.optimizer_state_only_sharded_keys
    result = _finalize_graph(
        gm,
        local_result.mod,
        local_result.opts,
        params_and_buffers,
        local_result.named_states,
        input_batch_dim=parallel_mode.input_batch_dim,
        dedup=False,
    )
    result.gm.meta["gspmd_fsdp_axis_size"] = parallel_mode.fsdp_axis_size
    # Set is_fsdp_sharded on placeholder nodes so _jit_gspmd knows
    # which inputs are local shards.
    if sharded_parameter_fqns or optimizer_state_only_sharded_keys:
        is_sharded = _build_fsdp_sharded_flags(
            params_and_buffers,
            local_result.named_states,
            sharded_parameter_fqns,
            optimizer_state_only_sharded_keys,
        )
        is_sharded = _align_fsdp_sharded_flags(
            result.flat_state,
            is_sharded,
            result.flat_state,
        )
        _annotate_fsdp_state_input_sharding(
            result.gm,
            result.flat_state,
            is_sharded,
        )
        _annotate_fsdp_state_output_sharding(
            result.gm,
            result.flat_state,
            is_sharded,
        )
        logger.info(
            "FSDP: %d sharded keys, %d sharded placeholders, flat_state=%d entries",
            len(sharded_parameter_fqns),
            sum(
                1
                for n in result.gm.graph.nodes
                if n.op == "placeholder" and n.meta.get("is_fsdp_sharded", False)
            ),
            len(result.flat_state),
        )
    return result


def compile(  # noqa: C901 — orchestration: stages 0-9 + legacy DP/GSPMD wiring intentionally kept inline; refactor would obscure the pipeline.
    parallel_mode: Optional[ParallelMode] = None,
    region_transforms: Optional[List[RegionTransform]] = None,
    *,
    backend: Optional["Backend"] = None,
    eager_preproc: Sequence[Callable[[nn.Module], None]] = (),
    _parametrize_fns: Sequence[Callable[..., torch.Tensor]] = (),
) -> Callable[..., Any]:
    r"""Compile and optimize a callable, which can be a train step within a training loop.

    Several functions over the same module (e.g. a train step and an eval
    step) can be compiled together so they share its parameters -- sharded
    once, one set of state, one device state on XLA::

        train_step, eval_step = recir.compile(parallel_mode=mode, backend=be)(
            train_fn, eval_fn, example_args=[(model, opt, batch), (model, batch)]
        )

    They are compiled immediately from ``example_args`` (not on first call),
    then each runner runs independently. A function that takes no optimizer
    compiles forward-only. ``region_transforms`` apply to every function; each
    graph gets its own copy so signature-bound transform state stays isolated.

    This is the public top-level entry point for the recir compile pipeline.
    Extracts the :class:`nn.Module` + :class:`torch.optim.Optimizer` from the
    decorated callable's arguments, traces them via :func:`trace`, runs
    stages 3-9 via :func:`compile_from_gm`, then wraps the result in a
    runtime callable that manages device state, flattens user args, threads
    ``flat_state`` through every invocation, and routes ``last_iter=True``
    for IterGraph cleanup on the final step.

    Pipeline stages (the canonical recir vocabulary; consult the inline
    ``# Stage N:`` comments in :func:`compile_from_gm` and the per-mode
    helpers for the source-level boundaries).  **Stages 5-8 (optimize,
    stage, dispatch, build) are owned by the configured**
    :class:`~recir.backends.base.Backend`; the rest are owned by the
    orchestrator.

    * **Stage 0 -- eager_preproc.** User-supplied
      ``Sequence[Callable[[nn.Module], None]]`` hooks that mutate the eager
      module *before* tracing — attach hooks, register buffers, swap
      non-traceable submodules.  Pre-FX; no graph artifact yet.  Driven by
      :func:`_compile_via_backend` on the new path; the legacy path (no
      ``backend=``) also runs ``eager_preproc`` so it remains useful with
      raw :class:`ParallelMode` callers.
    * **Stage 1 -- trace.** :func:`functorch.make_fx` capture (Python -> FX)
      via :func:`trace` and its inner :func:`_local_compile`.  Produces the
      initial :class:`fx.GraphModule` plus a ``flat_state`` list whose
      entries map 1:1 to the leading placeholders. Parallel modes normalize
      trace inputs via ``_make_trace_input_transform``; depending on the mode,
      this may resize the batch dimension or only fakeify and symbolize it.
    * **Stage 2 -- post-trace meta tagging.** :func:`_tag_batch_inputs` plus
      :func:`propagate_batch_input_meta` walk the gm and tag every
      placeholder with ``is_batch_input`` / ``is_weight`` / ``batch_dim``
      meta keys.  Drives downstream correctness for surgical orchestrators
      that need to identify user-input placeholders semantically.
    * **Stage 3 -- transform.**
      ``region_transforms[*].finalize_compiled_graph(gm)``.  Each
      :class:`RegionTransform` (recompute / regather / padding) finalizes
      its rewrite after partition.  Code-order is *after* stage 4 to match
      the legacy ``_compile_and_transform`` ordering; the verb "transform"
      is reserved for these semantic rewrites.
    * **Stage 4 -- partition.** Multi-device graph split:
      ``parallel_mode.shard_params(...)`` (partitioning for DP / FSDP /
      Hybrid is baked in at trace time; GSPMD needs no DTensor pass), then
      re-flattens via :func:`_to_caller_flattened_graph_module` and rebuilds
      ``flat_state``.  The verb "partition" is reserved for this distributed
      graph split.
    * **Stage 5 -- optimize** *(Backend)*.  ``backend.optimize_passes``
      rewrites the gm without changing meaning: comm fusion, dead-code
      elimination, scheduling, sparse-embedding decomposition.
      Two-argument passes get ``(gm, flat_state)``; single-argument passes
      get ``(gm,)``.  Owned by :attr:`Backend.optimize_passes`.
    * **Stage 6 -- stage** *(Backend)*.  Wrap the gm in
      :class:`IterGraphModule` when ``backend.uses_iter_graph`` is ``True``.
      Splits one logical step into setup / main / cleanup sub-graphs so
      cross-iteration passes can operate.  Gated by
      :attr:`Backend.uses_iter_graph` and applied by
      :func:`compile_from_gm`.
    * **Stage 7 -- dispatch** *(Backend)*.  Backend-owned final rewrite.
      XLA backends call ``partial_lower_xla`` to splice ``_XLAModule``
      ``call_module`` nodes into the graph; :class:`InductorBackend` is a
      no-op pass-through.  No backend IR yet — just subgraph carving +
      routing.  The verb "lower" is banned here (it collides with JAX's
      ``.lower()``); see :meth:`Backend.dispatch`.
    * **Stage 8 -- build** *(Backend)*.  First point real backend artifacts
      exist.  XLA: ``_XLAModule._compile_xla`` runs
      ``jax.jit(...).lower(...).compile(...)``.  Inductor:
      ``torch._inductor.compile_fx``.  Lazy — runs on the first forward of
      the dispatched ``call_module``, not at decoration time.  The verb
      "compile" is banned here (it's reserved for ``recir.compile``).
    * **Stage 9 -- prebuild.** Stage 8 run ahead of time on the build host,
      serialized to disk, reloaded on a production host.  Used for TPU
      canaries to avoid paying compile cost in the training pod.  See
      ``recir.lowering.aot.aot_compile_and_serialize``.

    **Runtime wrap (post-stage 9).** :func:`compile` wraps the compiled
    gm in a callable that manages device state via
    :class:`~recir.lowering.device_state.DeviceStateManager` (e.g. for XLA
    so state stays on TPU between steps), pytree-flattens user args,
    threads ``flat_state`` through every invocation, and routes
    ``last_iter=True`` for :class:`IterGraphModule` cleanup on the final
    step.  This wrapping is what :func:`compile` adds on top of the lower-
    level :func:`trace` + :func:`compile_from_gm` pair.

    **Pipeline dataflow** (``[B]`` = owned by :class:`Backend`)::

        nn.Module + args
           |
           v   Stage 0    eager_preproc (user callbacks)
           |
           v   Stage 1    trace (make_fx)              -- (gm, flat_state)
           |
           v   Stage 2    post-trace meta tagging
           |
           v   Stage 4    partition (shard + collectives)
           |
           v   Stage 3    transform (region finalize; code-order is AFTER stage 4)
           |
           v   Stage 5    optimize          [B]  Backend.optimize_passes
           |
           v   Stage 6    stage             [B]  IterGraphModule wrap if Backend.uses_iter_graph
           |
           v   Stage 7    dispatch          [B]  Backend.dispatch -- subgraph carving
           |
           v   Stage 8    build             [B]  JAX .lower-then-compile / Inductor -- lazy on first forward
           |
           v   Stage 9    prebuild               (optional AOT path -- serialize + load on target host)
           |
           v   runtime wrap (DeviceStateManager, input flattening, last_iter routing)

    **Naming discipline.**  In recir, name by the artifact you produce.
    If you emit FX, it is ``trace``.  If you make targeted edits to a
    graph, it is ``patch``.  If you intentionally change model semantics
    by rewriting a region, it is ``transform``.  If you split work across
    devices, it is ``partition``.  If you improve scheduling / fusion /
    elimination without semantic change, it is ``optimize``.  If you split
    into setup / main / cleanup, it is ``stage``.  If you carve
    backend-compatible islands and wrap them, it is ``dispatch`` — never
    ``lower``.  If you produce backend IR or an executable, it is
    ``build``.  If you do that ahead of runtime and persist the result,
    it is ``prebuild``.  Only :func:`recir.compile` may use the verb
    ``compile``.

    Args:
        parallel_mode (Optional[ParallelMode]): a :class:`ParallelMode`
            instance specifying how to parallelize the callable.  Each
            ``ParallelMode`` owns stage 4 (its
            :meth:`~ParallelMode.partition` rewrites the
            graph for the chosen DP / FSDP / GSPMD / Hybrid strategy).
            ``DataParallel()`` is used when ``None`` is passed.
            (Default: ``None``)
        region_transforms (Optional[List[RegionTransform]]): a list of
            :class:`RegionTransform` instances applied at stage 3 to
            rewrite sub-regions of the forward graph (e.g. activation
            checkpointing via :class:`RecomputeTransform`, jagged padding
            via :class:`PaddingRegionTransform`).  Stage 3 is code-order
            *after* stage 4 partition so the region rewrites see the
            post-partition graph.  (Default: ``None``)
        backend (Optional[Backend]): a :class:`recir.backends.Backend`
            instance that owns stages 5-8 (``optimize_passes``,
            ``uses_iter_graph`` IterGraph wrap, ``dispatch``, lazy
            ``build``).  When provided, ``recir.compile`` routes through
            the new :func:`trace` + :func:`compile_from_gm` pipeline.
            When ``None``, the legacy partition-only path is used (no
            stage-5+ rewrites; useful for raw DP / FSDP / GSPMD captures
            that downstream code consumes directly).
            (Default: ``None``)
        eager_preproc (Sequence[Callable[[nn.Module], None]]): a sequence
            of in-place pre-trace hooks that run against the model
            argument (stage 0).  Each hook is invoked once on the first
            :class:`nn.Module` discovered in the decorated callable's
            arguments, before :func:`trace` runs.  (Default: ``()``)

    Returns:
        A decorator that, when applied to a callable, returns a wrapped
        callable.  On the first invocation the wrapped callable runs the
        full pipeline (stages 0-9) and caches the compiled object on
        ``wrapper.__dict__[COMPILED_OBJECT_KEY]``; subsequent invocations
        reuse the cached object and pay only the runtime cost (device
        state sync + flat-state threading + gm forward).

    See Also:
        :func:`trace`: stages 0-2 alone, exposed so graph-surgery orchestrators
            can splice graph-level edits
            between the local trace and partition.
        :func:`compile_from_gm`: stages 3-9 alone, takes a pre-traced gm
            from :func:`trace` and drives partition / optimize / dispatch.
        :mod:`recir.backends`: the :class:`Backend` ABC and concrete
            subclasses (Inductor, XLA Dynamo-bridge, XLA GSPMD, XLA
            ShardMap) that own stages 5-8.
    """

    def compile_calls(
        calls: Sequence[Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]],
    ) -> List[_CompiledResult]:
        mode: ParallelMode = DataParallel() if parallel_mode is None else parallel_mode
        if backend is not None:
            return _compile_via_backend(
                calls,
                mode,
                backend,
                eager_preproc=eager_preproc,
                region_transforms=region_transforms,
                _parametrize_fns=_parametrize_fns,
            )
        if len(calls) > 1:
            raise ValueError("compiling several functions together needs backend=")
        if _parametrize_fns:
            raise ValueError(
                "recir.compile(_parametrize_fns=...) requires "
                "backend= to be set (the legacy partition-only "
                "path does not forward user parametrize fns)."
            )
        ((func, args, kwargs),) = calls
        if eager_preproc:
            for arg in pytree.arg_tree_leaves(*args, **kwargs):
                if isinstance(arg, nn.Module):
                    for hook in eager_preproc:
                        hook(arg)
        return [
            _compile_and_transform(
                func,
                mode,
                *args,
                region_transforms=region_transforms,
                **kwargs,
            )
        ]

    def make_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        def ensure_compiled(
            args: Tuple[Any, ...], kwargs: Dict[str, Any]
        ) -> _CompiledResult:
            compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
            if compiled_obj is None:
                (compiled_obj,) = compile_calls([(func, args, kwargs)])
                wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
            return compiled_obj

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_train_step = kwargs.pop("last_train_step", False) if kwargs else False
            compiled_obj = ensure_compiled(args, kwargs)

            user_inps = pytree.arg_tree_leaves(*args, **kwargs)
            flat_inps = compiled_obj.state_manager.prepare_inputs(
                compiled_obj.gm,
                compiled_obj.flat_state,
                user_inps,
                last_train_step,
            )
            with torch.no_grad():
                # N.B.: we don't need autograd as backward has already been
                # captured in the graph.
                if not last_train_step:
                    output_all = compiled_obj.gm(*flat_inps)
                else:
                    # This is the last train step. Call IterGraphModule.forward()
                    # with the `last_iter` argument and catch the exception in
                    # case the compiled_obj is not wrapped with IterGraphModule.
                    try:
                        output_all = compiled_obj.gm(
                            *flat_inps, last_iter=last_train_step
                        )
                    except TypeError as e:
                        if "last_iter" not in str(e):
                            raise e
                        output_all = compiled_obj.gm(*flat_inps)

                output = output_all[0]
                updated_flat = pytree.tree_leaves(output_all[1:])
                state_manager = compiled_obj.state_manager
                state_manager.commit_step_state(
                    compiled_obj.flat_state, updated_flat, last_train_step
                )
                if not state_manager.defers_output(last_train_step):
                    output = state_manager.materialize_output(output)

                return output

        def _prepare_compiled(*args: Any, **kwargs: Any) -> None:
            if kwargs:
                kwargs.pop("last_train_step", None)
            ensure_compiled(args, kwargs)

        wrapper.__dict__[PREPARE_COMPILED_ATTR] = _prepare_compiled
        return wrapper

    def inner(
        func: Callable[..., Any],
        *more_funcs: Callable[..., Any],
        example_args: Optional[Sequence[Tuple[Any, ...]]] = None,
    ) -> Any:
        funcs = (func, *more_funcs)
        wrappers = [make_wrapper(f) for f in funcs]
        if example_args is None:
            if more_funcs:
                raise ValueError(
                    "compiling several functions together needs example_args= "
                    "(one argument tuple per function); they compile now, not on "
                    "first call"
                )
            return wrappers[0]
        if len(example_args) != len(funcs):
            raise ValueError("example_args= needs one argument tuple per function")
        compiled = compile_calls(
            [(f, tuple(args), {}) for f, args in zip(funcs, example_args)]
        )
        for wrapper, compiled_obj in zip(wrappers, compiled):
            wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
        return tuple(wrappers) if more_funcs else wrappers[0]

    return inner


# Pull through the ``GraphPatch`` type alias so consumers don't need to import
# from ``recir.backends`` if they only want the type.
from recir.backends.base import GraphPatch  # noqa: E402, F401
# ---------------------------------------------------------------------------
# v7.1 public surface: graph manipulation primitives
# ---------------------------------------------------------------------------


# Re-exports so callers can write ``from recir.api import ...`` or
# ``recir.find_markers`` interchangeably.
from recir.ops.marker import find_markers  # noqa: E402, F401
from recir.region_transforms import extract_subgraph, replace_region  # noqa: E402, F401


def add_state(
    gm: fx.GraphModule,
    flat_state: List[torch.Tensor],
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
