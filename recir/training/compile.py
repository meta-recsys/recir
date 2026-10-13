# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""recir compile orchestration: shared compile driver for both execution modes.

Both modes use the SAME eager pipeline via `compile_step` -- trace ->
(dispatcher graph rewrite) -> compile_from_gm(backend) -> make_compiled_runner
(eager: trace runs when compile_step is called, not lazily like recir.compile's
decorator). They differ in parallel_mode + backend + dispatchers AND in the
traced step / optimizer count:

  * HDP_GPU (compile_hdp_step) - HybridParallel + InductorBackend; passes TWO
    optimizers (dense AdamW + sparse RowWiseAdagrad) to the shared variadic
    `recir_train_step`, so BOTH step in the GM; LR via the 0-d LR buffers +
    WarmupController. No graph dispatchers (the GPU caller patches
    EmbeddingCollection / MoTPreprocessor eagerly before compile).
  * GSPMD_TPU (compile_gspmd_step) - GSPMDParallel + XLAGSPMDBackend; passes the
    SINGLE dense optimizer to the same `recir_train_step`, so only the DENSE opt
    steps in the GM. On TPU the sparse embedding update is owned by the embedding
    backend (SparseCore Adagrad on TPU / FBGEMM-fused DMP on CPU) and fires OFF
    the GM via the backend's backward hook during loss.backward(). XLAGSPMDBackend
    is imported function-local so importing this module pulls no XLA/torchax code.

So the compile/run MECHANICS are shared; the difference is sparse-opt execution:
in-GM (GPU, torch RowWiseAdagrad) vs backend-owned off-GM (TPU SparseCore). The
0-d LR buffers (LR_DENSE_BUFFER / LR_SPARSE_BUFFER) land in flat_state via
mod.named_buffers() -> tagged is_weight=True (replicated), GSPMD-safe. (jax is
already pulled transitively via parallel_mode -> xla_lower.)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from recir.backends.inductor import InductorBackend
from recir.compiler.compilation import (
    _compile_and_transform,
    _compile_graphs,
    compile_from_gm,
)
from recir.compiler.runtime import make_compiled_runner, sync_device_state_to_torch
from recir.compiler.tracing import _GraphSpec, _trace_graphs, trace
from recir.distributed.parallel_mode import GSPMDParallel, HybridParallel
from recir.distributed.sharding_plan import EmbeddingShardingPlan
from recir.pipeline import PipelineConfig
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group, _world
from torch.distributed.tensor import DeviceMesh
from torchrec.distributed.model_parallel import DistributedModelParallel

logger: logging.Logger = logging.getLogger(__name__)

# Non-persistent 0-d LR buffers the module must register under these names.
# build_torch_optimizer binds each optimizer's group['lr'] to its buffer at
# construction (setup._build_lr_buffered_optimizer), so make_fx traces against
# the proxy buffer instead of baking the construction-time value.
# WarmupController writes these same buffers from inside the compiled step;
# compile_step declares it to recir, which rewires its captured references to
# the proxies for the duration of the trace (api._rematerialize_tensor_attrs),
# so those writes land in the graph.
LR_DENSE_BUFFER: str = "_lr_dense_buffer"
LR_SPARSE_BUFFER: str = "_lr_sparse_buffer"
LR_ITER_BUFFER: str = "_lr_iter_buffer"
WARMUP_ATTR: str = "_recir_warmup"
IN_GRAPH_METRICS_ATTR: str = "_recir_in_graph_metrics"

# ``flatten_batch(batch, global_loss_denom=...)`` -> the flat step args.
FlattenBatch = Callable[..., Tuple[Any, ...]]
# ``hook(train_gm, model, flat_batch)``, called once on the freshly traced train
# graph before any graph dispatcher or backend rewrites it.
PostTraceHook = Callable[[torch.fx.GraphModule, nn.Module, Sequence[Any]], None]


class RecirMode(Enum):
    """Which recir lowering path to use for the compiled train step."""

    HDP_GPU = "hdp_gpu"
    GSPMD_TPU = "gspmd_tpu"


@dataclass(frozen=True)
class TBESparseConfig:
    """Configuration for TBE-backed sparse embedding lowering."""

    use_index_dedup: bool = True
    use_separate_process_group: bool = False
    # Called module re-traced around the embedding lookups ("" = whole model);
    # None keeps the default, the deepest module that owns every TBE table.
    forward_root_fqn: Optional[str] = None


@dataclass
class RecirCompileSpec:
    """Inputs to compile a train step. Mode-specific fields are Optional.

    Shared config object for both modes; the actual compile entry points differ
    (HDP -> compile_train_step here; TPU -> recir_core.tpu.compile_gspmd_step,
    which lives in a separate Buck target to keep jax/XLA out of the GPU binary).
    """

    mode: RecirMode
    device: torch.device
    world_size: int = 1
    rank: int = 0
    pipeline: PipelineConfig = field(default_factory=PipelineConfig.none)
    # HDP_GPU:
    embedding_plan: Optional[EmbeddingShardingPlan] = None
    dense_style: str = "replicate"
    dense_fsdp_module_fqns: Optional[Set[str]] = None
    # HDP_GPU bring-up fallback: when True, compile via the legacy partition-
    # only path (recir.compile with backend=None) -- trace + partition, then run
    # the FX gm EAGERLY with NO InductorBackend optimize / IterGraph / Inductor
    # lowering stages. Lets the GPU path run end-to-end while the InductorBackend
    # lowering is stabilized. Ignored by the TPU path (compile_gspmd_step always
    # uses the XLA backend).
    use_legacy_compile: bool = False
    tbe_sparse_config: Optional[TBESparseConfig] = None
    tbe_process_group: Optional[ProcessGroup] = None
    # GSPMD_TPU (consumed by recir_core.tpu.compile_gspmd_step):
    num_ranks: int = 1
    num_processes: int = 1
    loss_reduction: str = "sum"
    region_transforms: List[Any] = field(default_factory=list)
    # Params whose FQN contains this substring are row-sharded over all chips
    # (e.g. recir.embedding.in_graph.SHARDED_TABLE_PREFIX); None = all replicated.
    fsdp_table_substr: Optional[str] = None


def create_tbe_process_group(
    tbe_sparse_config: Optional[TBESparseConfig], world_size: int
) -> Optional[ProcessGroup]:
    """The dedicated TBE process group when ``use_separate_process_group`` is set
    and there is more than one rank, else None. Keep a strong reference to it:
    c10d holds non-default process groups weakly."""
    if (
        tbe_sparse_config is not None
        and tbe_sparse_config.use_separate_process_group
        and world_size > 1
    ):
        group = dist.new_group(group_desc="recir_tbe")
        if not isinstance(group, ProcessGroup):
            raise RuntimeError("Every rank must join the dedicated TBE process group.")
        return group
    return None


def build_default_pg_mesh(device_type: str, size: int, rank: int) -> DeviceMesh:
    """Build a DeviceMesh that reuses the default c10d-registered PG by name.

    Functional collectives in recir's compiled GM resolve their PG by name via
    c10d._get_group_size_by_name. The default PG from init_process_group is
    already registered under a c10d-managed name; we reuse that name + handle
    so DeviceMesh-bound collectives find it.
    """
    default_pg = _get_default_group()
    pg_name = _world.pg_names[default_pg]
    mesh = DeviceMesh(
        device_type,
        torch.arange(size),
        _init_backend=False,
        _rank=rank,
    )
    mesh._dim_group_names = [pg_name]
    mesh._pg_registry[pg_name] = default_pg
    return mesh


def in_graph_metrics_enabled(mod: nn.Module) -> bool:
    return getattr(mod, IN_GRAPH_METRICS_ATTR, None) is not None


def recir_train_step(
    mod: nn.Module,
    *opts_and_batch: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Variadic full train step for BOTH cells; recir lowers it into one GM.

    ``compile_step`` invokes ``train_step_fn(model, *optimizers, *flat_batch)``,
    so after ``mod`` come the optimizer(s) -- contiguous, real
    ``torch.optim.Optimizer`` instances -- followed by the flat batch tensors
    (from ``flatten_batch``). We split on that boundary.

    This step is a FULL training step: it runs ``mod(*batch)`` with grad enabled,
    reduces to a scalar ``loss``, calls ``loss.backward()``, and steps every
    optimizer it is handed (``opt.step()``), all traced into the single GM. On
    TPU/GSPMD the dense opt steps in-graph while the backend-owned sparse
    embedding update fires off-GM via its backward hook during ``loss.backward()``.

    LR binding: every optimizer reaching here was built by
    ``build_torch_optimizer`` (including the in_graph ``emb_opt`` -- the backend
    only calls an injected builder), so each one's group['lr'] already IS the
    matching 0-d buffer and make_fx traces the lifted proxy rather than a baked
    constant.

    The schedule advances HERE, in the graph, right after ``opt.step()``: the LR
    update is a per-iteration state update like the optimizer step and belongs
    in the same place. That relies on recir rewiring the controller's captured
    buffer references to the lifted proxies for the duration of the trace
    (``api._rematerialize_tensor_attrs``). Without it the writes land on
    pre-trace tensors, the graph echoes the buffers it was handed, and the LR
    freezes at its seed value -- which is what happened on every TPU run until
    that rewiring existed. See docs/LR_SCHEDULE_NOT_IN_GRAPH.md.

    Returns (loss, named_tensors) detached, so the outer pipeline cannot re-trigger
    autograd on the already-backwarded graph; predictions are available via
    ``named_tensors["prediction"]``.
    """
    # Split the leading optimizer args from the flat batch tensors. During the
    # make_fx trace the batch entries are FakeTensors/proxies (isinstance False),
    # so the scan stops at the first non-optimizer arg.
    opts: List[torch.optim.Optimizer] = []
    for arg in opts_and_batch:
        if isinstance(arg, torch.optim.Optimizer):
            opts.append(arg)
        else:
            break
    batch_args: Tuple[Any, ...] = opts_and_batch[len(opts) :]

    all_losses, named_tensors = mod(*batch_args)
    # Tag the prediction in the traced graph so fabricate_inference_graph can
    # select it post-trace (identity at runtime, a mark_boundary clone). Marking
    # here (not in the shared model forward) keeps it recir-specific and env-gated.
    if (
        os.environ.get("RECIR_PUBLISH_TRAINING_GRAPH") == "1"
        and "prediction" in named_tensors
    ):
        from recir.region_transforms.training_to_inference import mark_inference_output

        named_tensors = {
            **named_tensors,
            "prediction": mark_inference_output(named_tensors["prediction"]),
        }
    loss: torch.Tensor = all_losses.sum()
    loss.backward()
    for opt in opts:
        opt.step()
    warmup = getattr(mod, WARMUP_ATTR, None)
    if warmup is not None:
        warmup.step()
    detached_named: Dict[str, torch.Tensor] = {
        k: (v.detach() if isinstance(v, torch.Tensor) else v)
        for k, v in named_tensors.items()
    }
    # The hook is supplied by the caller rather than imported here, which keeps
    # optional model and metrics dependencies out of the core compile target.
    metrics_fn = getattr(mod, IN_GRAPH_METRICS_ATTR, None)
    if metrics_fn is not None:
        detached_named.update(
            metrics_fn(
                detached_named["prediction"],
                detached_named["label"],
                detached_named["weight"],
            )
        )
        for key in ("prediction", "label", "weight"):
            detached_named.pop(key, None)
    return all_losses.detach(), detached_named


def _declared_tensor_attr_holders(model: nn.Module) -> Tuple[Any, ...]:
    warmup = getattr(model, WARMUP_ATTR, None)
    return () if warmup is None else (warmup,)


def compile_step(
    train_step_fn: Callable[..., Any],
    model: nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    trace_batch: Any,
    *,
    parallel_mode: Any,
    pipeline_config: Optional[PipelineConfig] = None,
    backend: Any = None,
    dispatchers: Sequence[Any] = (),
    region_transforms: Sequence[Any] = (),
    global_loss_denom: bool = True,
    flatten_batch: FlattenBatch,
    eval_step_fn: Optional[Callable[..., Any]] = None,
    post_trace_hook: Optional[PostTraceHook] = None,
) -> Any:
    """Shared compile driver for BOTH modes (eager trace).

    trace(train_step_fn, model, *optimizers, *flat) ->
    dispatcher.install_graph (TPU only) -> compile_from_gm(backend) ->
    make_compiled_runner. ``optimizers`` is passed positionally to train_step_fn
    right after the model: GPU passes (dense_opt, sparse_opt), TPU passes
    (dense_opt,) -- both to the shared variadic ``recir_train_step``.
    The caller supplies the mode-specific `parallel_mode` + `backend` (+ graph
    `dispatchers` for TPU); the GPU caller passes dispatchers=() and patches its
    eager EmbeddingCollection / MoTPreprocessor BEFORE calling this. DMP-wrapped
    models are unwrapped to .module so the trace + dispatcher installs hit the
    bare module. Returns a runner with the same signature as train_step_fn:
    ``runner(model, *optimizers, *flat_args) -> (losses, named)``.

    When ``backend`` is None, falls back to the legacy partition-only path
    (recir's ``_compile_and_transform``, equivalent to ``recir.compile`` with
    ``backend=None``): trace + partition + region-transform finalize, then run
    the FX gm eagerly with no backend optimize / IterGraph / lowering stages.
    Graph ``dispatchers`` are not supported on the legacy path (GPU passes none).

    ``global_loss_denom`` is forwarded to ``flatten_batch`` for the TRACE batch
    and MUST match what the caller passes when it later invokes the runner (see
    ``flatten_batch``: global for GSPMD, per-rank for HDP).

    ``flatten_batch(batch, global_loss_denom=...)`` turns the model's batch type
    into the flat tensor args ``train_step_fn`` takes after the optimizers.

    ``eval_step_fn(model, *flat_args)`` (optional) is compiled together with the
    train step (``recir.compile`` of several functions), traced on the same
    batch: the two share the model's parameters (and, on XLA, its device state).
    Returns ``(train_runner, eval_runner)`` then. ``region_transforms`` and
    ``pipeline_config`` apply to the train step only.

    ``post_trace_hook`` only runs on the single-graph backend path (an explicit
    ``backend`` and no ``eval_step_fn``).
    """
    model_to_train = (
        model.module if isinstance(model, DistributedModelParallel) else model
    )
    tensor_attr_holders = _declared_tensor_attr_holders(model_to_train)
    for dispatcher in dispatchers:
        dispatcher.install_eager(model_to_train)
    if backend is None and eval_step_fn is not None:
        raise ValueError("eval_step_fn needs an explicit recir backend")
    if backend is None:
        if pipeline_config is not None and pipeline_config.enabled:
            raise ValueError("Pipeline scheduling requires an explicit recir backend")
        # Legacy partition-only fallback (eager FX gm, no Inductor lowering).
        assert not dispatchers, (
            "legacy compile (backend=None) does not support graph dispatchers; "
            "the TPU path must pass an explicit backend."
        )
        compiled_obj = _compile_and_transform(
            train_step_fn,
            parallel_mode,
            model_to_train,
            *optimizers,
            *flatten_batch(trace_batch, global_loss_denom=global_loss_denom),
            region_transforms=list(region_transforms) or None,
            _tensor_attr_holders=tensor_attr_holders,
        )
        return make_compiled_runner(compiled_obj)
    flat_batch = flatten_batch(trace_batch, global_loss_denom=global_loss_denom)
    if eval_step_fn is not None:
        graphs = [
            _GraphSpec(
                train_step_fn,
                (model_to_train, *optimizers, *flat_batch),
                parallel_mode,
                region_transforms=region_transforms,
                tensor_attr_holders=tensor_attr_holders,
            ),
            _GraphSpec(
                eval_step_fn,
                (model_to_train, *flat_batch),
                parallel_mode,
                tensor_attr_holders=tensor_attr_holders,
            ),
        ]
        traced = []
        for graph_gm, graph_state in _trace_graphs(graphs):
            for dispatcher in dispatchers:
                graph_gm, graph_state = dispatcher.install_graph(graph_gm, graph_state)
            traced.append((graph_gm, graph_state))
        train_obj, eval_obj = _compile_graphs(
            traced,
            parallel_modes=[parallel_mode, parallel_mode],
            backend=backend,
            region_transforms=[region_transforms, ()],
            pipeline_configs=[pipeline_config, None],
        )
        return make_compiled_runner(train_obj), make_compiled_runner(eval_obj)
    gm, flat_state = trace(
        train_step_fn,
        model_to_train,
        *optimizers,
        *flat_batch,
        parallel_mode=parallel_mode,
        region_transforms=region_transforms,
        _tensor_attr_holders=tensor_attr_holders,
    )
    if post_trace_hook is not None:
        post_trace_hook(gm, model_to_train, flat_batch)
    for dispatcher in dispatchers:
        gm, flat_state = dispatcher.install_graph(gm, flat_state)
    compiled_obj = compile_from_gm(
        gm,
        flat_state,
        parallel_mode=parallel_mode,
        backend=backend,
        pipeline_config=pipeline_config,
        region_transforms=region_transforms,
    )
    return make_compiled_runner(compiled_obj)


def compile_hdp_step(
    model: nn.Module,
    dense_opt: torch.optim.Optimizer,
    sparse_opt: torch.optim.Optimizer,
    trace_batch: Any,
    spec: RecirCompileSpec,
    *,
    flatten_batch: FlattenBatch,
    eval_step_fn: Optional[Callable[..., Any]] = None,
    post_trace_hook: Optional[PostTraceHook] = None,
) -> Any:
    """Eager HDP_GPU compile: HybridParallel + InductorBackend via compile_step.

    HybridParallel(dense_style, embedding_plan) over a mesh that reuses the
    default PG inserts embedding routing collectives and dense-grad all_reduce
    in the partition stage; InductorBackend runs the optimize/build stages (comm
    fusion, scheduling, Inductor codegen). The caller MUST have patched the eager
    embedding collections and MoTPreprocessor on `model` before calling (the GPU
    path uses no graph dispatchers).
    """
    assert spec.mode is RecirMode.HDP_GPU, (
        f"compile_hdp_step is the HDP_GPU entry; got mode={spec.mode}. "
        "For GSPMD_TPU call recir_core.tpu.compile_gspmd_step."
    )
    assert spec.embedding_plan is not None, (
        "HDP_GPU requires an embedding_plan; build one with "
        "embedding_plan.build_embedding_plan."
    )
    sparse_transform_factory: Any = None
    tbe_sparse_config = spec.tbe_sparse_config
    if tbe_sparse_config is None:
        if spec.tbe_process_group is not None:
            raise ValueError("tbe_process_group requires tbe_sparse_config")
    else:
        if (
            tbe_sparse_config.use_separate_process_group
            and spec.world_size > 1
            and spec.tbe_process_group is None
        ):
            raise ValueError("use_separate_process_group requires a TBE process group")
        if (
            not tbe_sparse_config.use_separate_process_group
            and spec.tbe_process_group is not None
        ):
            raise ValueError(
                "tbe_process_group was provided while use_separate_process_group "
                "is disabled"
            )
        from recir.region_transforms.tbe_sparse_embedding import (
            TBESparseEmbeddingTransform,
        )

        def _tbe_sparse_transform_factory(
            fqns: set[str], mesh: DeviceMesh, module: nn.Module
        ) -> Any:
            return TBESparseEmbeddingTransform(
                fqns,
                mesh,
                module,
                process_group=spec.tbe_process_group,
                embedding_plan=spec.embedding_plan,
                use_index_dedup=tbe_sparse_config.use_index_dedup,
                forward_root_fqn=tbe_sparse_config.forward_root_fqn,
            )

        sparse_transform_factory = _tbe_sparse_transform_factory
    parallel_mode = HybridParallel(
        dense_style=spec.dense_style,
        embedding_plan=spec.embedding_plan,
        dense_fsdp_module_fqns=spec.dense_fsdp_module_fqns,
        sparse_transform_factory=sparse_transform_factory,
    )
    parallel_mode._mesh = build_default_pg_mesh(
        spec.device.type, spec.world_size, spec.rank
    )
    # Backend selection (GPU):
    #   * spec.use_legacy_compile -> backend=None: legacy partition-only path
    #     (eager FX gm, no Inductor lowering). Bring-up fallback that runs
    #     end-to-end while the InductorBackend lowering path is stabilized.
    #   * else -> InductorBackend(): trace -> partition -> Inductor lowering
    #     (still under development; revisit later).
    backend = None if spec.use_legacy_compile else InductorBackend()
    return compile_step(
        recir_train_step,
        model,
        (dense_opt, sparse_opt),
        trace_batch,
        parallel_mode=parallel_mode,
        pipeline_config=spec.pipeline,
        backend=backend,
        dispatchers=(),
        region_transforms=spec.region_transforms,
        global_loss_denom=False,
        flatten_batch=flatten_batch,
        eval_step_fn=eval_step_fn,
        post_trace_hook=post_trace_hook,
    )


def current_world_rank() -> Tuple[int, int]:
    """(world_size, rank) from the default PG, or (1, 0) if not initialized."""
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def compile_gspmd_step(
    model: nn.Module,
    dense_opt: torch.optim.Optimizer,
    trace_batch: Any,
    dispatchers: List[Any],
    spec: RecirCompileSpec,
    *,
    train_step_fn: Optional[Callable[..., Any]] = None,
    flatten_batch: FlattenBatch,
    sparse_opt: Optional[torch.optim.Optimizer] = None,
    eval_step_fn: Optional[Callable[..., Any]] = None,
) -> Any:
    """Eager GSPMD_TPU compile: GSPMDParallel + XLAGSPMDBackend via compile_step.

    Builds the GSPMD parallel mode + XLA backend and delegates to
    ``compile_step`` with the preproc + sparse-arch graph ``dispatchers`` (which
    turn the eager sparse arch / preprocessor into XLA-lowerable custom ops via
    install_eager + install_graph; built by the family's
    ``backend.get_override``). Passes the
    SINGLE ``dense_opt`` to the shared ``recir_train_step`` -- only the dense opt
    steps in the GM; the sparse embedding update is owned by the embedding backend
    (SparseCore / FBGEMM-fused DMP) and fires off-GM via its backward hook.
    Returns ``runner(model, dense_opt, *flat_args) -> (losses, named)``.

    ``sparse_opt`` (the in_graph embedding backend's optimizer, whose tables are
    parameters inside the program) also steps in the GM; the runner then takes
    ``(model, dense_opt, sparse_opt, *flat_args)``.

    The XLAGSPMDBackend import is function-local so importing this module pulls no
    XLA backend / torchax code (the GPU path never reaches it).
    """
    from recir.backends.xla import XLAGSPMDBackend

    step_fn: Callable[..., Any] = train_step_fn or recir_train_step
    parallel_mode = GSPMDParallel(
        "replicate",
        loss_reduction=spec.loss_reduction,
        num_ranks=spec.num_ranks,
        num_processes=spec.num_processes,
        fsdp_table_substr=spec.fsdp_table_substr,
    )
    return compile_step(
        step_fn,
        model,
        (dense_opt,) if sparse_opt is None else (dense_opt, sparse_opt),
        trace_batch,
        parallel_mode=parallel_mode,
        backend=XLAGSPMDBackend(),
        dispatchers=dispatchers,
        region_transforms=spec.region_transforms,
        # GLOBAL loss denominator: the GSPMD program is a single global view
        # over the padded global batch (see flatten_batch).
        global_loss_denom=True,
        flatten_batch=flatten_batch,
        eval_step_fn=eval_step_fn,
    )


def run_step_and_sync(
    runner: Any,
    model: nn.Module,
    dense_opt: torch.optim.Optimizer,
    batch: Any,
    *,
    flatten_batch: FlattenBatch,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run one compiled GSPMD step and sync device state back to torch.

    Returns the (detached losses, named_tensors) recir_train_step produces, so
    the pipeline can report a real loss/metrics.
    ``sync_device_state_to_torch`` pulls the updated params back so subsequent
    host-side reads (checkpoint, logging) see the trained weights.
    """
    result: Tuple[torch.Tensor, Dict[str, torch.Tensor]] = runner(
        # GSPMD: global loss denominator, matching compile_gspmd_step's trace.
        model,
        dense_opt,
        *flatten_batch(batch, global_loss_denom=True),
    )
    sync_device_state_to_torch(runner)
    return result
