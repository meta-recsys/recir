# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import tempfile
import time
from functools import wraps
from typing import Any, Callable, DefaultDict, Iterable, Set, Union

import torch.fx as fx
from recir.passes.iter_graph_module import IterGraphModule
from recir.utils.graph import dump_graphs_to_files


logger: logging.Logger = logging.getLogger("graph_optimization")


class PassRegistry:
    """Encapsulates state for graph optimization pass ordering and tracking."""

    def __init__(self) -> None:
        self.optimized_func: Set[str] = set()
        self.prerequisite_sets: DefaultDict[str, Set[str]] = collections.defaultdict(
            set
        )
        self.apply_before_sets: DefaultDict[str, Set[str]] = collections.defaultdict(
            set
        )
        self.dump_graph_folder: str = ""

    def enable_dump(self, folder: str = "") -> None:
        if not folder:
            folder = tempfile.mkdtemp()
        self.dump_graph_folder = folder

    def reset(self) -> None:
        """Reset the registry state so passes can be applied again."""
        self.optimized_func.clear()


default_registry: PassRegistry = PassRegistry()


def enable_graph_optimization_dump(folder: str = "") -> None:
    default_registry.enable_dump(folder)


def graph_optimization_pass(
    prerequisites: Iterable[Callable[..., Any]],
    apply_after: Iterable[Callable[..., Any]],
) -> Callable[..., Any]:
    """Define the contract of a graph optimization pass.

    All the passes should be wrapped with this decorator.
    `prerequisites` is used to annotate the prerequisite passes of the this pass.
    `apply_after` means that this wrapped pass must be applied after the passes
    in `apply_after`. The difference between `prerequisites` and `apply_after`
    is that all the passes in `prerequisites` must be applied to the graph and
    must be applifed before the wrapped pass while the passes `apply_after` are
    optional. But if a pass in `apply_after` is applied to the graph, it has to
    be done before the wrapped pass.
    Optimizer pass developers are required to add these fields accordingly and
    users need to follow the restrictions to avoid the assert.

    Current design has one limitation: users can only apply the optimizations
    once.  In some cases, we may need to run multiple the same optimization
    multiple time, e.g., optimization passes -> profiling the result -> apply
    optimization passes with the profiling result again. This limitation will be
    addressed limitation in the future.

    Args:
        prerequisites (Iterable[Callable]): the list of string to the names of
            passes which are the prerequisites of this pass.
        apply_after (Iterable[Callable]): the list of string to the names of
            passes that can not be applied after the wrapped pass.
    """
    registry: PassRegistry = default_registry

    def inner(func: Callable[..., Any]) -> Callable[..., Any]:
        def make_key(func: Callable[..., Any]) -> str:
            return f"{func.__module__}.{func.__name__}"

        func_key: str = make_key(func)
        registry.prerequisite_sets[func_key] = {make_key(f) for f in prerequisites}
        for apply_after_pass in apply_after:
            registry.apply_before_sets[make_key(apply_after_pass)].add(func_key)

        @wraps(func)
        def pass_wrapper(
            gm: Union[fx.GraphModule, IterGraphModule], *args: Any, **kwargs: Any
        ) -> None:
            begin = time.time()
            assert isinstance(gm, (fx.GraphModule, IterGraphModule)), (
                "The first argument of the pass must be either "
                "fx.GraphModule or IterGraphModule."
            )
            assert func_key not in registry.optimized_func, (
                f"Cannot apply {func_key} twice."
            )
            invalid_passes = registry.apply_before_sets[func_key].intersection(
                registry.optimized_func
            )
            assert not invalid_passes, (
                f"{invalid_passes} must be applied after {func_key}."
            )
            assert registry.prerequisite_sets[func_key].issubset(
                registry.optimized_func
            ), (
                f"{registry.prerequisite_sets[func_key] - registry.optimized_func} are the "
                f"prerequisites of {func_key} but are not applified. "
                f"Applied passes are {registry.optimized_func}."
            )

            func(gm, *args, **kwargs)
            gm.graph.lint()
            gm.graph.eliminate_dead_code()
            gm.recompile()
            registry.optimized_func.add(func_key)

            prefix = f"after_{func.__name__}"
            if registry.dump_graph_folder:
                if isinstance(gm, IterGraphModule):
                    dump_graphs_to_files(
                        {
                            f"{prefix}_setup_gm": gm.setup_gm,
                            f"{prefix}_main_gm": gm.main_gm,
                            f"{prefix}_cleanup_gm": gm.cleanup_gm,
                        },
                        registry.dump_graph_folder,
                    )
                else:
                    dump_graphs_to_files({prefix: gm}, registry.dump_graph_folder)

            logger.info("Spent %f seconds applying %s", time.time() - begin, func_key)

        return pass_wrapper

    return inner
