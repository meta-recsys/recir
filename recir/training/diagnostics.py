# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""GM diagnostics for recir-compiled train steps.

Framework-agnostic: given a compiled recir step (or its FX GraphModule),
enumerate the optimizer-update ops and the collective-communication ops
that got baked into the graph. Used to verify that:
  * the optimizer step(s) actually traced into the GM (not elided), and
  * dense-gradient all_reduce / embedding all_to_all collectives are present.

These run once (caller-gated) after the first compiled forward and depend only
on torch and recir.
"""

import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from recir.pipeline import PipelinedGraphModule

logger: logging.Logger = logging.getLogger(__name__)


# AdamW capturable+foreach path emits _foreach_* ops; Adagrad emits
# addcmul_/sqrt/addcdiv_ (or foreach variants); fused paths emit _fused_adam*.
_OPT_PATTERNS: Tuple[str, ...] = (
    "addcmul",
    "addcdiv",
    "_foreach_mul",
    "_foreach_add",
    "_foreach_addcmul",
    "_foreach_addcdiv",
    "_foreach_sqrt",
    "_foreach_div",
    "_foreach_neg",
    "lerp",
    "_fused_adamw",
    "_fused_adam",
)

# Substring patterns so in-place / coalesced variants
# (all_reduce_, all_reduce_coalesced, ...) are all captured.
_COMM_PATTERNS: Tuple[str, ...] = (
    "all_reduce",
    "all_to_all",
    "reduce_scatter",
    "all_gather",
    "wait_tensor",
)


def gm_from_compiled(compiled_step: Callable[..., Any]) -> Optional[Any]:
    """Pull the FX GraphModule out of a recir-compiled step wrapper.

    recir stashes the underlying _CompiledResult (with .gm) on the wrapper's
    __dict__ under COMPILED_OBJECT_KEY after the first invocation. Returns
    None if recir isn't importable or the wrapper hasn't compiled yet.
    """
    try:
        from recir.compiler.runtime import COMPILED_OBJECT_KEY
    except ImportError:
        logger.warning("[recir gm dump] cannot import COMPILED_OBJECT_KEY")
        return None
    compiled_obj = getattr(compiled_step, "__dict__", {}).get(COMPILED_OBJECT_KEY, None)
    if compiled_obj is None or not hasattr(compiled_obj, "gm"):
        logger.warning(
            "[recir gm dump] compiled wrapper has no %s with .gm; dict keys=%s",
            COMPILED_OBJECT_KEY,
            list(getattr(compiled_step, "__dict__", {}).keys()),
        )
        return None
    return compiled_obj.gm


def dump_optimizer_ops(
    gm: torch.fx.GraphModule, *, warn_if_missing: bool = True
) -> bool:
    """Log the op-target histogram and every optimizer-update node in the GM.

    Zero optimizer-related ops means the opt step was elided during tracing
    or never reached the GM.
    """
    target_counts: "Counter[str]" = Counter()
    for n in gm.graph.nodes:
        if n.op == "call_function":
            target_counts[str(n.target)] += 1
        elif n.op == "call_method":
            target_counts[f"<method>{n.target}"] += 1
    logger.info(
        "[recir gm dump] total nodes=%d, call_function/method=%d, unique targets=%d",
        sum(1 for _ in gm.graph.nodes),
        sum(target_counts.values()),
        len(target_counts),
    )
    top20 = target_counts.most_common(20)
    logger.info(
        "[recir gm dump] top-20 op targets:\n%s",
        "\n".join(f"  {n:>6}  {t}" for t, n in top20),
    )

    opt_nodes: List[Tuple[str, str]] = []
    for n in gm.graph.nodes:
        if n.op not in ("call_function", "call_method"):
            continue
        target_str = str(n.target)
        if any(pat in target_str for pat in _OPT_PATTERNS):
            opt_nodes.append((target_str, n.name))
    logger.info("[recir gm dump] optimizer-related ops in GM: %d total", len(opt_nodes))
    opt_counts: "Counter[str]" = Counter(t for t, _ in opt_nodes)
    for target_str, count in opt_counts.most_common():
        logger.info("  %4d  %s", count, target_str)
    if not opt_nodes and warn_if_missing:
        logger.warning(
            "[recir gm dump] NO optimizer-update ops found in the compiled GM. "
            "AdamW step + Adagrad step were either elided during tracing or "
            "never reached the GM. Check: (1) are dense/sparse params non-empty? "
            "(2) is opt.state populated after first .step()? (3) did the "
            "optimizer rematerialization owned_fqns filter match the opt's FQNs?"
        )
    return bool(opt_nodes)


def dump_collective_ops(
    gm: torch.fx.GraphModule, *, warn_if_missing: bool = True
) -> bool:
    """Log every collective-communication node, focusing on dense-grad all_reduce.

    IMPORTANT - world_size semantics: the DDP partition pass elides dense-grad
    collectives when world_size == 1 (nothing to reduce across a single rank).
    So ZERO all_reduce on a local 1-GPU run is EXPECTED, not a bug; it is only
    a red flag on a multi-rank run. The log states which case applies.
    """
    comm_nodes: Dict[str, List[str]] = {pat: [] for pat in _COMM_PATTERNS}
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        target_str = str(n.target)
        for pat in _COMM_PATTERNS:
            if pat in target_str:
                comm_nodes[pat].append(n.name)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    allreduce_names = comm_nodes["all_reduce"]
    logger.info(
        "[recir gm dump] collective ops (world_size=%d): all_reduce=%d, "
        "all_to_all=%d, reduce_scatter=%d, all_gather=%d, wait_tensor=%d",
        world_size,
        len(allreduce_names),
        len(comm_nodes["all_to_all"]),
        len(comm_nodes["reduce_scatter"]),
        len(comm_nodes["all_gather"]),
        len(comm_nodes["wait_tensor"]),
    )

    if allreduce_names:
        logger.info(
            "[recir gm dump] %d dense-grad all_reduce node(s) in the compiled GM: %s",
            len(allreduce_names),
            ", ".join(allreduce_names),
        )
        for n in gm.graph.nodes:
            if n.op == "call_function" and "all_reduce" in str(n.target):
                logger.info(
                    "  all_reduce %s: target=%s args=%s", n.name, n.target, n.args
                )
    elif world_size > 1 and warn_if_missing:
        logger.warning(
            "[recir gm dump] NO all_reduce ops in the compiled GM at "
            "world_size=%d. Dense gradients are NOT synchronized across ranks - "
            "HybridParallel(dense_style='replicate') should have inserted them. "
            "Check that the DDP partition pass (partition_data_parallel) ran and "
            "that dense params were grad-tagged (_spmd.tag_grad).",
            world_size,
        )
    elif warn_if_missing:
        logger.info(
            "[recir gm dump] no all_reduce ops, but world_size=1 so this is "
            "EXPECTED: the DDP partition pass elides dense-grad collectives with "
            "a single rank. Re-run this dump on a multi-rank job to confirm "
            "all_reduce is baked in."
        )
    return bool(allreduce_names)


def dump_compiled_gm(compiled_step: Callable[..., Any], dump_dir: str = "/tmp") -> None:
    """Top-level diagnostic: dump optimizer + collective ops and the readable GM.

    Safe to call once after the first compiled forward; no-op (with a warning)
    if the GM can't be recovered.
    """
    gm = gm_from_compiled(compiled_step)
    if gm is None:
        return
    is_pipelined = isinstance(gm, PipelinedGraphModule)
    graph_modules = gm.stage_graph_modules() if is_pipelined else (gm,)
    has_optimizer_ops = False
    has_all_reduce = False
    for stage_index, graph_module in enumerate(graph_modules):
        if is_pipelined:
            logger.info(
                "[recir gm dump] pipeline stage %d/%d",
                stage_index + 1,
                len(graph_modules),
            )
        has_optimizer_ops |= dump_optimizer_ops(
            graph_module, warn_if_missing=not is_pipelined
        )
        has_all_reduce |= dump_collective_ops(
            graph_module, warn_if_missing=not is_pipelined
        )

    if is_pipelined and not has_optimizer_ops:
        logger.warning(
            "[recir gm dump] NO optimizer-update ops found across any pipeline "
            "stage. The optimizer step was either elided during tracing or never "
            "reached the GM."
        )

    if is_pipelined and not has_all_reduce:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        log = logger.warning if world_size > 1 else logger.info
        log(
            "[recir gm dump] NO all_reduce ops found across any pipeline stage "
            "at world_size=%d%s",
            world_size,
            "; this is expected for a single rank" if world_size == 1 else "",
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    dump_path = f"{dump_dir}/recir_train_module_gm_rank{rank}.txt"
    try:
        readable = gm.print_readable(print_output=False)
    except Exception as e:
        logger.warning(
            "[recir gm dump] gm.print_readable failed: %s; falling back to str(graph).",
            e,
        )
        readable = str(gm.graph)
    try:
        with open(dump_path, "w") as f:
            f.write(readable)
        logger.info(
            "[recir gm dump] full readable graph written to %s (%d chars). "
            "grep for 'addcdiv' / '_foreach' / 'all_reduce' to verify.",
            dump_path,
            len(readable),
        )
    except Exception as e:
        logger.warning("[recir gm dump] failed to write %s: %s", dump_path, e)
