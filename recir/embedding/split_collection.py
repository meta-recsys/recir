# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Split one ``EmbeddingCollection`` into a sharded half and a replicated half.

Production places embedding tables PER TABLE. For this model prod asks for
``ROW_WISE`` on three tables and ``DATA_PARALLEL`` on seven, and that choice is
not cosmetic -- it decides the compute kernel, and the kernel decides which
optimizer trains the table:

===============  ==================  =====================
                 ROW_WISE            DATA_PARALLEL
===============  ==================  =====================
compute kernel   FUSED               DENSE
optimizer        RowWiseAdagrad      the DENSE optimizer
runs             inside backward     at ``step()``
===============  ==================  =====================

A CPU leg cannot express that by asking torchrec for ``DATA_PARALLEL``. The
DENSE kernel is ``BatchedDenseEmbedding``, which builds a
``DenseTableBatchedEmbeddingBagsCodegen`` with ``pooling_mode=PoolingMode.NONE``
-- the SEQUENCE mode, one output row per id. CUDA implements it (the dense host
template generates ``_nobag`` kernels,
``fbgemm_gpu/codegen/training/backward/embedding_backward_split_host_template.cpp:420``).
CPU does not: ``split_embedding_codegen_forward_cpu``
(``fbgemm_gpu/codegen/training/forward/embedding_forward_split_cpu.cpp:183-210``)
unconditionally allocates ``at::empty({B, total_D})`` and reads ``pooling_mode``
only to tell MEAN from SUM, so ``NONE`` silently POOLS. The lookup then returns
``n_features x batch_size`` rows against jagged lengths that sum to something
else, and the sharded collection dies reconstructing its output:

    RuntimeError: split_with_sizes expects split_sizes to sum exactly to 3584
    ... but got split_sizes=[512, 509, 907, 512, 512, 512, 512]

That is a missing CPU kernel, not a plan or config difference, so no amount of
aligning the two legs avoids it.

What CAN be expressed on CPU is what ``DATA_PARALLEL`` MEANS at world size 1: a
full replica of the table, a real ``.grad``, and the dense optimizer. A plain
unsharded ``nn.Embedding`` is exactly that. Since an ``EmbeddingCollection`` is
planned all-or-nothing -- ``ShardedEmbeddingCollection`` indexes
``module_sharding_plan[table.name]`` for every entry of ``_embedding_configs``
(``torchrec/distributed/embedding.py:1240-1258``) -- expressing the split means
splitting the collection itself, which is what this module does.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

_CollectionT = TypeVar("_CollectionT", bound="EmbeddingCollection")


class ReplicatedEmbeddingCollection(EmbeddingCollection):
    """An ``EmbeddingCollection`` the sharding planner deliberately cannot see.

    Behaviourally identical to its base class -- the point is only the TYPE.
    torchrec matches sharders by exact type (``get_module_to_default_sharders``
    is keyed on ``type(module)``, and the enumerator looks the child's type up
    in that map), so a subclass is enumerated as an ordinary dense module and
    is left alone by both the planner and DMP.

    That is load-bearing rather than incidental, so ``split_embedding_collection``
    asserts afterwards that no plan entry was produced for it. If a future
    torchrec matches sharders by ``isinstance``, this class starts being sharded
    with the FUSED kernel and every table in it silently moves to the sparse
    optimizer -- the exact bug this split exists to fix, and one that no shape
    or key comparison would catch.
    """


class SplitEmbeddingCollection(nn.Module):
    """Two collections behind one ``EmbeddingCollection`` interface."""

    def __init__(
        self,
        sharded: EmbeddingCollection,
        replicated: ReplicatedEmbeddingCollection,
    ) -> None:
        super().__init__()
        self.sharded: EmbeddingCollection = sharded
        self.replicated: ReplicatedEmbeddingCollection = replicated

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        out: Dict[str, JaggedTensor] = self.sharded(features)
        out.update(self.replicated(features))
        return out

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return [*self.sharded.embedding_configs(), *self.replicated.embedding_configs()]

    def need_indices(self) -> bool:
        return self.sharded.need_indices()

    def embedding_dim(self) -> int:
        return self.sharded.embedding_dim()


def _sub_collection(
    source: EmbeddingCollection,
    table_names: Sequence[str],
    cls: type[_CollectionT],
) -> _CollectionT:
    """A collection over ``table_names``, reusing ``source``'s live modules."""
    by_name = {c.name: c for c in source.embedding_configs()}
    missing = [n for n in table_names if n not in by_name]
    if missing:
        raise ValueError(f"tables absent from the embedding collection: {missing}")

    configs = []
    for name in table_names:
        # copy.copy, not dataclasses.replace: replace re-runs __post_init__,
        # and these configs carry defaults it would recompute.
        config = copy.copy(by_name[name])
        config.init_fn = None
        configs.append(config)

    # use_gather_select only exists on fbcode's torchrec, not on the pip
    # torchrec staged into the TPU image, and this module runs under both.
    # Passing it unconditionally is an AttributeError at pod start, which is
    # exactly where a local test cannot see it.
    kwargs: Dict[str, Any] = {}
    getter = getattr(source, "use_gather_select", None)
    if callable(getter):
        kwargs["use_gather_select"] = getter()

    sub = cls(
        tables=configs,
        need_indices=source.need_indices(),
        device=torch.device("meta"),
        **kwargs,
    )
    # Put the init functions back. Suppressing them is only about not running
    # arbitrary initialization against meta tensors during construction; the
    # configs live on past it, and ``ShardedEmbeddingCollection`` asserts every
    # one of them has an ``init_fn`` when DMP calls ``reset_parameters``
    # (torchrec/distributed/embedding.py:1250). Leaving them cleared turns an
    # allocation shortcut into an AssertionError inside DMP.
    for config, name in zip(configs, table_names):
        config.init_fn = by_name[name].init_fn

    for name in table_names:
        sub.embeddings[name] = source.embeddings[name]
    return sub


def split_embedding_collection(
    parent: nn.Module,
    attr: str,
    replicated_tables: Iterable[str],
) -> Optional[SplitEmbeddingCollection]:
    """Replace ``parent.<attr>`` with a split collection, in place."""
    source = getattr(parent, attr)
    if not isinstance(source, EmbeddingCollection):
        raise TypeError(
            f"{attr} is {type(source).__name__}, not an EmbeddingCollection"
        )

    all_names = [c.name for c in source.embedding_configs()]
    # Hoisted: evaluating set(...) in the comprehension's if-clause rebuilds it
    # per element and would consume a one-shot iterator on the first one.
    replicated_set = set(replicated_tables)
    replicated = [n for n in all_names if n in replicated_set]
    if not replicated:
        return None
    sharded = [n for n in all_names if n not in replicated_set]
    if not sharded:
        raise ValueError(
            "every table was marked DATA_PARALLEL, leaving nothing for the "
            "sharding planner; the sparse optimizer would then train nothing."
        )

    split = SplitEmbeddingCollection(
        sharded=_sub_collection(source, sharded, EmbeddingCollection),
        replicated=_sub_collection(source, replicated, ReplicatedEmbeddingCollection),
    )
    setattr(parent, attr, split)
    logger.info(
        "Split %s: %d table(s) to shard %s, %d replicated (dense optimizer) %s",
        attr,
        len(sharded),
        sharded,
        len(replicated),
        replicated,
    )
    return split


def replicate_embedding_collection(
    parent: nn.Module,
    attr: str,
    cls: type[_CollectionT] = ReplicatedEmbeddingCollection,
) -> _CollectionT:
    """Retype ``parent.<attr>`` to ``cls`` in place, keeping its live tables.

    For a whole collection that must stay out of the sharding planner (see
    ``ReplicatedEmbeddingCollection``); ``cls`` may be a subclass of it that
    overrides ``forward``.
    """
    source = getattr(parent, attr)
    if not isinstance(source, EmbeddingCollection):
        raise TypeError(
            f"{attr} is {type(source).__name__}, not an EmbeddingCollection"
        )
    replicated = _sub_collection(
        source, [c.name for c in source.embedding_configs()], cls
    )
    setattr(parent, attr, replicated)
    return replicated


def assert_replicated_half_is_unplanned(
    plan: object,
    split: SplitEmbeddingCollection,
    module_prefix: str,
) -> None:
    """Fail loudly if the planner placed the half that must stay unsharded."""
    fqn = f"{module_prefix}.replicated" if module_prefix else "replicated"
    planned = getattr(plan, "plan", {})
    for path in planned:
        if path == fqn or path.startswith(fqn + "."):
            raise RuntimeError(
                f"the sharding planner placed {path}, which must stay "
                "unsharded: its tables are the ones prod trains with the DENSE "
                "optimizer. torchrec's sharder lookup no longer skips "
                "ReplicatedEmbeddingCollection."
            )


def partition_by_sharding(
    embedding_tables: Dict[str, object],
    data_parallel_value: str = "data_parallel",
) -> Tuple[List[str], List[str]]:
    """Split a table dict into (sharded, replicated) table names."""
    sharded: List[str] = []
    replicated: List[str] = []
    for name, table in embedding_tables.items():
        sharding = getattr(table, "sharding", None)
        if isinstance(sharding, str):
            raise TypeError(
                f"table {name!r} has sharding={sharding!r} as a plain string "
                "rather than a ShardingType member. Something deserialized the "
                "enum by name; comparisons against ShardingType will all be "
                "False and the placement would be silently lost."
            )
        value = getattr(sharding, "value", sharding)
        if value == data_parallel_value:
            replicated.append(name)
        else:
            sharded.append(name)
    return sharded, replicated
