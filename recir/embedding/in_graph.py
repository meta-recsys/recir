# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""In-graph row-wise-sharding embedding backend.

This module bundles the whole in-graph embedding path:

  * ``InGraphSparseArch`` -- an EC-shaped ``nn.Module`` that calls
    ``torch.ops.recir.ringshift_embedding`` for the ROWWISE tables. The op is
    opaque + lowered to the ring-shift kernel INSIDE recir's single compiled
    GSPMD program (see ``recir/torchax/ops/jseq_embedding.py``), so the
    cross-host all-to-all is scheduled with the outer data-parallel collectives
    and does NOT deadlock.
    Every table is a plain ``nn.Parameter([vocab, D])``; its ``TablePlan`` says
    how it is stored and read (ROWWISE -> ring-shift, FSDP ->
    ``recir.fsdp_embedding``, REPLICATED -> ``F.embedding``) and which optimizer
    trains it. The ``_sharded_tables`` params are row-wise (FSDP) sharded by
    ``GSPMDParallel(fsdp_table_substr=SHARDED_TABLE_PREFIX)`` (wired in train.py).

    The two halves of a plan are INDEPENDENT. Sharding is a memory decision;
    the optimizer is a training one, and production's two rates are ~50x apart.
    Nothing here derives either from the row count: a table the config places
    wrongly matches on every shape, key and name, so the choice has to be made
    where someone can read it.

  * ``SparseArchInputBreakDispatcher`` -- cuts a graph break at the sparse-arch
    *input ids* (not the arch): the preprocessor + merged-id-list build run eager
    and the ids are injected back into the graph, while the (ring-shift)
    ``InGraphSparseArch`` stays in-graph consuming them. Keeps the heavy
    id-list handling out of the XLA/GSPMD program (the SPMD partitioner segfaults
    on it at 64-chip) while doing the embedding lookup on TPU.

  * ``InGraphEmbeddingBackend`` -- the ``EmbeddingBackend`` entry point.
    ``setup()`` swaps the model's sparse-arch ``_embedding_collection`` for
    ``InGraphSparseArch`` and builds the sparse optimizer over the tables their
    plan assigns to it (the table is now a normal ``nn.Parameter``). The model
    binding locates that sparse arch via the injected ``sparse_arch_getter`` and
    supplies the optimizer factory via ``sparse_optimizer_builder``, keeping
    this module free of any model-specific module path or import.

Static-shape assumption: per-sample sequence lengths may VARY, but each feature's
TOTAL token count per chip is static (``T_local``, the same on every chip). Since
samples are assigned to chips contiguously, a feature's flat value block (length
``W * T_local``) reshapes directly to ``[W, T_local]`` (W = #chips) -- a
per-chip-static shape XLA accepts -- while the real per-sample ``lengths`` flow
through as dynamic metadata for the output ``JaggedTensor``s. ``feature_seqlens``
(here = per-chip token total ``T_local``) is supplied by the backend (derived
from the cached KJT).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple

import jax
import recir.ops.seq_embedding_op  # noqa: F401  (defines torch.ops.recir.ringshift_embedding)

# Registers the torchax op lowering (recir::ringshift_embedding -> ring-shift).
# Module level so it happens with the backend rather than on the first setup(),
# but nothing depends on WHEN: Environment falls back to the live registry when
# its snapshot misses, so an op registered after default_env() is still found.
# It did once -- a lazily imported lowering was invisible to an Environment
# built before it, and the op fell through to its CPU fallback and hit the
# "torchax Tensors ... environment" assertion, which names nothing useful.
import recir.torchax.ops.jseq_embedding  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from recir.embedding.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
    TableOptimizer,
    TablePlan,
    TableSharding,
)
from recir.ops.eager_dispatch import (
    install_graph_break_dispatch,
    mark_input,
    mark_output_with_bwd,
)
from recir.training.optim import OptimConfig
from recir.utils.meta_tensor_init import (
    default_embedding_init_bound,
    mark_meta_uniform_init,
    table_init_stream_id,
)
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    data_type_to_dtype,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

_PREFIX: str = "seq_sparse_input_break"


SHARDED_TABLE_PREFIX: str = "_sharded_tables"
DP_TABLE_PREFIX: str = "_dp_tables"


def _pooling_of(cfg: BaseEmbeddingConfig) -> str:
    """Map a torchrec table config to the ring-shift pooling argument."""
    pooling = getattr(cfg, "pooling", None)
    if pooling is None or pooling is PoolingType.NONE:
        return "none"
    return pooling.value.lower()


def table_name_from_param(name: str) -> str | None:
    """The table a parameter FQN belongs to, or None if it is not a table.

    Both ParameterDicts name their entries ``<prefix>.<table>``, so the table is
    the segment right after whichever prefix appears. Callers use this to answer
    per-table questions (which optimizer trains it) about a flat
    ``named_parameters()`` walk.
    """
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part in (SHARDED_TABLE_PREFIX, DP_TABLE_PREFIX):
            return parts[i + 1]
    return None


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _to_compute_dtype(x: torch.Tensor) -> torch.Tensor:
    """Widen a narrow lookup result back to float32.

    ``EmbeddingConfig.data_type`` is STORAGE precision. TorchRec stores the
    table narrow and computes in float32, so this backend must too: leaving the
    activation in float16 would push the narrow dtype through HSTU and the
    loss, which is a different model, not a different layout.
    """
    return x.float() if x.dtype != torch.float32 else x


class InGraphSparseArch(nn.Module):
    """In-graph EC replacement. forward(KJT) -> Dict[str, JaggedTensor].

    Args:
        table_configs: the EC's EmbeddingConfig list (name/num_embeddings/dim/
            feature_names).
        feature_seqlens: per-chip token total per feature (= the feature's total
            ids / #chips). Per-sample lengths may vary; only this per-chip total
            is static. From the cached KJT.
        world_size: number of devices (dim-0-sharded table rows are padded to a
            multiple so the ring-shift block reshape ``vocab % W == 0`` holds).
        table_plans: per table, how it is stored/looked up and which optimizer
            trains it. Required for every table in ``table_configs`` -- there is
            no default, because both halves of the plan are silent when wrong: a
            misplaced table matches on every shape, key and name and simply
            trains at the other optimizer's rate.
    """

    def __init__(
        self,
        table_configs: Sequence[BaseEmbeddingConfig],
        feature_seqlens: Dict[str, int],
        world_size: int,
        table_plans: Dict[str, TablePlan],
        on_device_init: bool = True,
    ) -> None:
        super().__init__()
        self._on_device_init: bool = on_device_init
        self._feature_seqlens: Dict[str, int] = dict(feature_seqlens)
        # Input-break state (graph-break at the sparse-arch input), toggled by
        # enable_input_break(). When set, forward() marks the merged ids/lengths
        # as a break boundary + injects per-feature blocks instead of the plain
        # flat path.
        self._input_break: bool = False
        self._break_override_id: int = 0
        self._break_input_kjt_keys: List[str] = []
        self._feature_to_table: Dict[str, str] = {}
        self._table_sharding: Dict[str, TableSharding] = {}
        self._table_pooling: Dict[str, str] = {}
        self._sharded_tables: nn.ParameterDict = nn.ParameterDict()
        self._dp_tables: nn.ParameterDict = nn.ParameterDict()

        missing = sorted(tc.name for tc in table_configs if tc.name not in table_plans)
        if missing:
            raise ValueError(
                f"no TablePlan for {missing}; every table needs an explicit "
                "sharding + optimizer, since a wrong one is invisible in every "
                "shape, key and state_dict entry"
            )

        for tc in table_configs:
            sharding = table_plans[tc.name].sharding
            for f in tc.feature_names:
                self._feature_to_table[f] = tc.name
            self._table_sharding[tc.name] = sharding
            pooling = _pooling_of(tc)
            if pooling != "none" and sharding is not TableSharding.ROWWISE:
                raise ValueError(
                    f"table {tc.name!r} is pooled ({pooling}) but its TablePlan "
                    f"selects {sharding.value}; pooling needs ROWWISE sharding"
                )
            self._table_pooling[tc.name] = pooling
            dtype = data_type_to_dtype(tc.data_type)
            if sharding is TableSharding.ROWWISE and dtype != torch.float32:
                logger.warning(
                    "[in_graph] table '%s': data_type=%s ignored, storing "
                    "float32 -- the ring-shift kernel requires it",
                    tc.name,
                    tc.data_type,
                )
                dtype = torch.float32
            # Match the TorchRec EmbeddingCollection init scale
            init_bound = default_embedding_init_bound(tc.num_embeddings)
            if sharding is TableSharding.REPLICATED:
                p = nn.Parameter(
                    torch.empty(tc.num_embeddings, tc.embedding_dim, dtype=dtype)
                )
                nn.init.uniform_(p, -init_bound, init_bound)
                self._dp_tables[tc.name] = p
            else:
                # ROWWISE and FSDP are both dim-0 sharded -> pad rows to a
                # multiple of the device count (the FSDP shard count; also the
                # ring-shift block reshape needs vocab % W == 0).
                vocab = _round_up(tc.num_embeddings, world_size)
                if self._on_device_init:
                    # Meta param: NO host storage. recir's FSDP materialization
                    # (_to_sharded_jax meta branch) generates each device's shard
                    # directly in HBM via jax.random -- avoids host alloc + H2D for
                    # huge tables (e.g. 1T hash). It reads the param's dtype, so
                    # a narrow table stays narrow there too.
                    p = nn.Parameter(
                        torch.empty(vocab, tc.embedding_dim, device="meta", dtype=dtype)
                    )
                    mark_meta_uniform_init(p, init_bound, table_init_stream_id(tc.name))
                else:
                    p = nn.Parameter(torch.empty(vocab, tc.embedding_dim, dtype=dtype))
                    nn.init.uniform_(p, -init_bound, init_bound)
                self._sharded_tables[tc.name] = p
            logger.info(
                f"[in_graph] table '{tc.name}': rows={tc.num_embeddings} "
                f"D={tc.embedding_dim} dtype={dtype} sharding={sharding.value} "
                f"optimizer={table_plans[tc.name].optimizer.value} "
                f"features={tc.feature_names}"
            )

    def forward(self, id_list_features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """Embed a merged id-list KJT -> per-feature activations.

        Two paths:
          * input-break (in-graph backend, ``_input_break`` set): mark the merged
            ids/lengths as a graph-break boundary and inject per-feature
            ``[W, T_local]`` blocks, so the feature-major flat values never enter
            the compiled graph (slicing a feature-major axis after batch-sharding
            segfaulted XLA's SPMD partitioner -- see bench_results.md). The real
            eager per-feature slice runs in the dispatcher's fwd_fn at install
            time; the marking here only sets the trace-time shapes.
          * flat (direct / unit-test): slice the merged values here.
        Both delegate to :meth:`_forward_presliced`.
        """
        if self._input_break:
            keys = self._break_input_kjt_keys
            inj_ids, inj_lens = _mark_input_break(
                id_list_features.values(),
                id_list_features.lengths(),
                self._break_override_id,
                keys,
                self._feature_seqlens,
            )
            return self._forward_presliced(keys, inj_ids, inj_lens)

        keys = id_list_features.keys()
        ids_per_chip, lengths_1d = _slice_per_feature(
            id_list_features.values(),
            id_list_features.lengths(),
            keys,
            self._feature_seqlens,
        )
        return self._forward_presliced(keys, ids_per_chip, lengths_1d)

    def _forward_presliced(
        self,
        keys: List[str],
        ids_2d: List[torch.Tensor],
        lengths_1d: List[torch.Tensor],
    ) -> Dict[str, JaggedTensor]:
        """Embed already-per-feature-sliced ids.

        Args:
            keys: feature names, in KJT order.
            ids_2d: per-feature ``[W, T_local]`` id tensors (one per key), where
                W = #chips and T_local = per-chip token total. The leading axis is
                the (chip) shard axis, so when injected as a graph input it is
                sharded on axis 0 (aligned) -- unlike the feature-major
                concatenated merged values.
            lengths_1d: per-feature real per-sample ``[B]`` length tensors (one
                per key), used only to build the output JaggedTensors.
        """
        result: Dict[str, JaggedTensor] = {}
        # Pass 1: small tables are looked up immediately (dense F.embedding);
        # big-table features are grouped by table so all features sharing a table
        # do ONE ring-shift call (fewer Mosaic kernels -> smaller HLO; the
        # per-feature loop blew the HLO to ~50k instructions and segfaulted XLA's
        # SPMD partitioner).
        big_ids: Dict[str, List[torch.Tensor]] = {}
        big_lengths: Dict[str, List[torch.Tensor]] = {}
        big_seqs: Dict[str, List[int]] = {}
        big_feat_keys: Dict[str, List[str]] = {}
        pooled_ids: Dict[str, List[torch.Tensor]] = {}
        pooled_lengths: Dict[str, List[torch.Tensor]] = {}
        pooled_feat_keys: Dict[str, List[str]] = {}
        for i, key in enumerate(keys):
            ids = ids_2d[i]  # [W, T_local]
            k_lengths = lengths_1d[i]  # [B]
            seq = ids.shape[1]

            table_name = self._feature_to_table.get(key)
            if table_name is None:
                continue
            if self._table_sharding[table_name] is TableSharding.ROWWISE:
                if self._table_pooling[table_name] == "none":
                    big_ids.setdefault(table_name, []).append(ids)
                    big_lengths.setdefault(table_name, []).append(k_lengths)
                    big_seqs.setdefault(table_name, []).append(seq)
                    big_feat_keys.setdefault(table_name, []).append(key)
                else:
                    pooled_ids.setdefault(table_name, []).append(ids)
                    pooled_lengths.setdefault(table_name, []).append(k_lengths)
                    pooled_feat_keys.setdefault(table_name, []).append(key)
            else:
                # FSDP or REPLICATED table
                flat_ids = ids.reshape(-1).long()
                if self._table_sharding[table_name] is TableSharding.FSDP:
                    emb = torch.ops.recir.fsdp_embedding(
                        flat_ids, self._sharded_tables[table_name]
                    )
                else:  # TableSharding.REPLICATED
                    emb = F.embedding(flat_ids, self._dp_tables[table_name])
                emb = _to_compute_dtype(emb)
                # Pin the flat [B*S, dim] activation batch-sharded on axis 0
                # (mirrors cpu's sharded embedding input) so HSTU consumes a
                # batch-sharded tensor; see Pass 2 for the full rationale.
                emb = torch.ops.recir.shard_batch0(emb)
                result[key] = JaggedTensor(values=emb, lengths=k_lengths)

        # Pass 2: ONE ring-shift per big table over the concatenated features.
        for table_name, ids_list in big_ids.items():
            weight = self._sharded_tables[table_name]
            dim = weight.shape[1]
            ids_cat = torch.cat(ids_list, dim=1)  # [W, sum(T_local)]
            # The ring-shift op treats axis 0 as the shard (chip) axis and axis 1
            # as that chip's flat token list; pass a full per-chip count (every
            # token is valid -- the per-chip total is exact) so it masks nothing.
            n_chips_cat, t_comb = ids_cat.shape[0], ids_cat.shape[1]
            comb_lengths = ids_cat.new_full((n_chips_cat,), t_comb)
            act = _to_compute_dtype(
                torch.ops.recir.ringshift_embedding(ids_cat, comb_lengths, weight)
            )
            # Split the [B, sum(S_f), D] activation back per feature.
            off = 0
            for key, seq, k_lengths in zip(
                big_feat_keys[table_name],
                big_seqs[table_name],
                big_lengths[table_name],
            ):
                a = act[:, off : off + seq, :].reshape(-1, dim)
                # The ring-shift activation is pinned batch-sharded as 3D
                # [B(sharded), S, D] (jseq_embedding with_sharding_constraint),
                # but the slice-on-S + reshape to flat [B*S, D] MERGES the
                # sharded B axis with the open S axis. Shardy will not propagate
                # sharding through a reshape that fuses a sharded dim with a
                # non-sharded dim from an interior constraint, so the flat tensor
                # goes REPLICATED -> the downstream 256->512 projection + the
                # L1<->L2 jagged gather replicate (~28.5G each) -> HBM OOM.
                # Re-pin the flat activation batch-sharded on axis 0, mirroring
                # cpu's flat [B*S, D] sharded graph input (which propagates
                # cleanly through the identical gather and never OOMs).
                a = torch.ops.recir.shard_batch0(a)
                off += seq
                result[key] = JaggedTensor(values=a, lengths=k_lengths)

        # Pooled delta: ids stay flat; lengths become aligned per-chip bag cuts.
        for table_name, ids_list in pooled_ids.items():
            lengths_list = pooled_lengths[table_name]
            feature_keys = pooled_feat_keys[table_name]
            weight = self._sharded_tables[table_name]
            dim = weight.shape[1]
            world = ids_list[0].shape[0]
            bags = lengths_list[0].shape[0]
            for key, lengths in zip(feature_keys, lengths_list):
                if lengths.shape[0] != bags:
                    raise ValueError(
                        f"pooled table {table_name!r}: feature {key!r} has "
                        f"{lengths.shape[0]} bags, expected {bags}"
                    )
            if bags % world:
                raise ValueError(
                    f"pooled table {table_name!r}: batch {bags} is not divisible "
                    f"by the {world} chips its ids are laid out over"
                )
            bags_local = bags // world
            ids_cat = torch.cat(ids_list, dim=1)
            bag_lengths = torch.cat(
                [lengths.reshape(world, bags_local) for lengths in lengths_list],
                dim=1,
            )
            act = torch.ops.recir.ringshift_embedding(
                ids_cat,
                bag_lengths,
                weight,
                pooling=self._table_pooling[table_name],
            ).reshape(world, len(ids_list), bags_local, dim)
            for index, (key, k_lengths) in enumerate(zip(feature_keys, lengths_list)):
                values = torch.ops.recir.shard_batch0(act[:, index].reshape(-1, dim))
                result[key] = JaggedTensor(
                    values=values, lengths=torch.ones_like(k_lengths)
                )
        return result

    def enable_input_break(self, override_id: int, input_kjt_keys: List[str]) -> None:
        """Switch forward() to the graph-break-at-input path. Called by the
        dispatcher's install_eager."""
        self._input_break = True
        self._break_override_id = override_id
        self._break_input_kjt_keys = list(input_kjt_keys)


def _prefix(override_id: int) -> str:
    return f"{_PREFIX}_{override_id}"


def _slice_per_feature(
    values: torch.Tensor,
    lengths: torch.Tensor,
    keys: List[str],
    feature_seqlens: Dict[str, int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    total_per_chip = sum(feature_seqlens[k] for k in keys)
    assert values.shape[0] % total_per_chip == 0, (
        f"merged values size {values.shape[0]} is not a multiple of the per-chip "
        f"token total {total_per_chip} (stale/mismatched feature_seqlens?)"
    )
    n_chips = values.shape[0] // total_per_chip
    batch = lengths.shape[0] // len(keys)
    ids_per_chip: List[torch.Tensor] = []
    lengths_1d: List[torch.Tensor] = []
    offset = 0
    for i, key in enumerate(keys):
        t_local = feature_seqlens[key]
        n = n_chips * t_local  # = feature f's total tokens in this slab
        ids_per_chip.append(values[offset : offset + n].reshape(n_chips, t_local))
        lengths_1d.append(lengths[i * batch : (i + 1) * batch])
        offset += n
    return ids_per_chip, lengths_1d


def _mark_input_break(
    values: torch.Tensor,
    lengths: torch.Tensor,
    override_id: int,
    keys: List[str],
    feature_seqlens: Dict[str, int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Mark the merged (values, lengths) as a graph-break boundary and return the
    per-feature ``[W, T_local]`` id blocks + real per-sample ``[B]`` lengths to
    inject in-graph.

    ``::input::`` markers tag the upstream merged tensors (the eager dispatch
    inputs); ``::output::`` markers (2 per feature, interleaved ids/lens) tag the
    injected blocks. The ``_slice_per_feature`` here is only for TRACE shapes --
    it is replaced at install time by the dispatcher's eager ``fwd_fn`` (same
    slicing), then DCE'd.
    """
    prefix = _prefix(override_id)
    # ::input:: markers -> these (upstream, eager) merged values become
    # _dispatch_fwd's inputs.
    v_in = mark_input(values, prefix, 0)
    l_in = mark_input(lengths, prefix, 1)
    # Zero anchor wires the output markers' data-dependency back to the input
    # markers so FX DCE keeps them alive end-to-end.
    anchor = (v_in.to(torch.int64).sum() * 0) + (l_in.to(torch.int64).sum() * 0)
    ids_2d, lengths_1d = _slice_per_feature(v_in, l_in, keys, feature_seqlens)
    inj_ids: List[torch.Tensor] = []
    inj_lens: List[torch.Tensor] = []
    oidx = 0
    for ids, lens in zip(ids_2d, lengths_1d):
        ids_out = mark_output_with_bwd(
            ids + anchor.to(ids.dtype), prefix, oidx, deps=[v_in, l_in]
        )
        oidx += 1
        lens_out = mark_output_with_bwd(
            lens + anchor.to(lens.dtype), prefix, oidx, deps=[v_in, l_in]
        )
        oidx += 1
        inj_ids.append(ids_out)
        inj_lens.append(lens_out)
    return inj_ids, inj_lens


class SparseArchInputBreakDispatcher:
    """Dispatcher protocol (install_eager / install_graph) that breaks at the input
    of the in-graph ``InGraphSparseArch`` while leaving the arch (ring-shift)
    in-graph."""

    def __init__(
        self,
        input_kjt_keys: List[str],
        override_id: int = 1,
        num_ranks: int = 1,
        feature_seqlens: Dict[str, int] | None = None,
    ) -> None:
        self._input_kjt_keys: List[str] = input_kjt_keys
        self._override_id: int = override_id
        self._num_ranks: int = num_ranks
        self._feature_seqlens: Dict[str, int] = dict(feature_seqlens or {})
        self._enabled: List[Tuple[int, InGraphSparseArch]] = []

    def install_eager(self, model: nn.Module) -> None:
        # Enable graph-break-at-input mode on the InGraphSparseArch that setup()
        # swapped into the sparse arch's _embedding_collection. No module swap -- just
        # flip a flag; the arch's own forward then marks the break (see
        # InGraphSparseArch.forward's input-break branch).
        archs = [
            mod
            for _fqn, mod in model.named_modules()
            if isinstance(mod, InGraphSparseArch)
        ]
        for i, arch in enumerate(archs):
            oid = self._override_id + i
            arch.enable_input_break(oid, self._input_kjt_keys)
            self._enabled.append((oid, arch))
        logger.info(
            "SparseArchInputBreakDispatcher: enabled input-break on %d "
            "InGraphSparseArch(s)",
            len(self._enabled),
        )
        assert self._enabled, (
            "SparseArchInputBreakDispatcher found no InGraphSparseArch -- did "
            "setup() run, or did the model structure change?"
        )

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        if not self._enabled:
            return gm, flat_state

        keys = self._input_kjt_keys
        seqlens = self._feature_seqlens

        def _per_feature_fwd(
            values: torch.Tensor, lengths: torch.Tensor
        ) -> Tuple[torch.Tensor, ...]:
            # Slice the merged KJT per feature EAGERLY and inject each
            # [n_chips, T_local] block + real [B] lengths (interleaved, matching
            # the trace-time output markers ids0, lens0, ids1, lens1, ...).
            ids_2d, lengths_1d = _slice_per_feature(values, lengths, keys, seqlens)
            outs: List[torch.Tensor] = []
            for ids, lens in zip(ids_2d, lengths_1d):
                outs.append(ids)
                outs.append(lens)
            return tuple(outs)

        for oid, _arch in self._enabled:
            install_graph_break_dispatch(
                gm,
                _prefix(oid),
                _per_feature_fwd,
                num_state_tensors=len(flat_state),
                num_ranks=self._num_ranks,
                register_backward=False,
            )
        return gm, flat_state


def _seed_in_graph_tables_from_ec(
    old_ec: nn.Module, new_ec: "InGraphSparseArch"
) -> None:
    """Copy per-table weights from the model's existing torchrec
    EmbeddingCollection into the in-graph table params, so the in_graph init is
    BYTE-identical to the cpu EC.

    Only host-materialized tables can be copied. Sharded meta params
    (on_device_init) materialize on-device via jax.random.
    """
    src: Dict[str, torch.Tensor] = {}
    for n, p in old_ec.named_parameters():
        parts = n.split(".")
        # torchrec EmbeddingCollection names weights "embeddings.<table>.weight";
        # the immediate parent module name is the table (ModuleDict) key. Match
        # that exact shape so unrelated ".weight" params (e.g. from other
        # submodules) can't shadow or overwrite a real table entry.
        if parts[-1] == "weight" and len(parts) >= 3 and parts[-3] == "embeddings":
            src[parts[-2]] = p.detach()
    if not src:
        logger.warning(
            "[seed] no EmbeddingCollection weights found to seed in-graph tables "
            "(old_ec=%s); in_graph tables keep their random init",
            type(old_ec).__name__,
        )
        return
    copied = 0
    skipped_meta = 0
    with torch.no_grad():
        for name, p in new_ec._dp_tables.items():
            if name not in src:
                continue
            if src[name].device.type == "meta" or p.device.type == "meta":
                skipped_meta += 1
                continue
            p.copy_(src[name].to(p.dtype))
            copied += 1
        for name, p in new_ec._sharded_tables.items():
            if name not in src:
                continue
            if src[name].device.type == "meta" or p.device.type == "meta":
                skipped_meta += 1
                continue
            rows = src[name].shape[0]  # logical vocab; skip the padded tail rows
            p[:rows].copy_(src[name].to(p.dtype))
            copied += 1
    logger.info(
        "[seed] copied %d EC table(s) into in-graph params "
        "(%d meta table(s) skipped -> on-device init)",
        copied,
        skipped_meta,
    )


class InGraphEmbeddingBackend(EmbeddingBackend):
    """In-graph ring-shift embedding backend.

    ``sparse_arch_getter`` maps the model to the module that owns
    ``_embedding_collection`` as a direct attribute (``setup()`` replaces it
    with an :class:`InGraphSparseArch`). ``sparse_optimizer_builder`` builds
    the LR-buffered optimizer over the resulting table params and returns
    ``(optimizer, lr_buffer)``. Both are injected so this module stays free
    of model-specific module paths and imports.

    ``table_plans`` is a constructor argument rather than a ``setup()``
    parameter because it is backend business: the other backends express the
    same choice through torchrec's sharding plan and cannot act on this form of
    it. Build it with
    ``recir.embedding.split_collection.table_plans_from_embedding_tables``.

    ``set_feature_seqlens()`` must run before ``setup()``.
    """

    def __init__(
        self,
        sparse_arch_getter: Callable[[nn.Module], nn.Module],
        sparse_optimizer_builder: Callable[
            [nn.Module, List[nn.Parameter], OptimConfig, torch.device],
            Tuple[torch.optim.Optimizer, torch.Tensor],
        ],
        table_plans: Dict[str, TablePlan],
        on_device_init: bool = True,
    ) -> None:
        self._sparse_arch_getter = sparse_arch_getter
        self._sparse_optimizer_builder = sparse_optimizer_builder
        self._table_plans: Dict[str, TablePlan] = dict(table_plans)
        self._feature_seqlens: Dict[str, int] = {}
        self._on_device_init: bool = on_device_init

    def set_feature_seqlens(self, feature_seqlens: Dict[str, int]) -> None:
        """Per-feature per-chip token total, consumed by setup()."""
        self._feature_seqlens = feature_seqlens
        logger.info(f"InGraphEmbeddingBackend: feature_chip_tokens={feature_seqlens}")

    def init_pg(self) -> Tuple[int, int]:
        rank = jax.process_index()
        world = jax.process_count()
        logger.info(
            f"InGraphEmbeddingBackend: JAX PJRT PG (rank={rank}, world={world})"
        )
        return rank, world

    def _params_by_optimizer(
        self, model: nn.Module, want: TableOptimizer
    ) -> Dict[str, nn.Parameter]:
        """Table params whose plan assigns them to ``want``.

        Keyed on the PLAN, not on which ParameterDict the table landed in.
        Sharding and optimizer are independent choices here, so the storage
        prefix cannot answer this: a dim-0-sharded table trained by the dense
        optimizer is exactly what FSDP is.
        """
        out: Dict[str, nn.Parameter] = {}
        for n, p in model.named_parameters():
            table = table_name_from_param(n)
            if table is None:
                continue
            plan = self._table_plans.get(table)
            if plan is None:
                raise RuntimeError(
                    f"parameter {n!r} belongs to table {table!r}, which has no "
                    "TablePlan; it would be trained by neither optimizer"
                )
            if plan.optimizer is want:
                out[n] = p
        return out

    def dense_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        # Everything that is not an embedding table, PLUS the tables whose plan
        # puts them on the dense optimizer (production's DATA_PARALLEL ones).
        dense = {
            n: p
            for n, p in model.named_parameters()
            if table_name_from_param(n) is None
        }
        dense.update(self._params_by_optimizer(model, TableOptimizer.DENSE))
        return dense

    def embedding_named_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        # The tables the dedicated in-graph sparse optimizer trains. Tables
        # planned DENSE are deliberately absent -- they are in
        # dense_named_parameters instead, and training them twice at rates ~50x
        # apart is the failure this split exists to prevent.
        return self._params_by_optimizer(model, TableOptimizer.SPARSE)

    def setup(
        self,
        model: nn.Module,
        table_configs: Sequence[BaseEmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> EmbeddingSetupResult:
        if not self._feature_seqlens:
            raise RuntimeError(
                "InGraphEmbeddingBackend.set_feature_seqlens() must run before setup()."
            )
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}.")

        sparse_arch = self._sparse_arch_getter(model)
        new_ec = InGraphSparseArch(
            table_configs=table_configs,
            feature_seqlens=self._feature_seqlens,
            world_size=world_size,
            table_plans=self._table_plans,
            on_device_init=self._on_device_init,
        )
        logger.info(
            f"InGraphEmbeddingBackend: on_device_init={self._on_device_init} "
            f"(meta param -> on-device HBM random init, no host alloc/H2D)"
        )
        # nn.Module.__getattr__ is typed Module | Tensor; this is the EC.
        old_ec: Any = sparse_arch._embedding_collection
        _seed_in_graph_tables_from_ec(old_ec, new_ec)
        sparse_arch._embedding_collection = new_ec

        embedding_params = self.embedding_named_parameters(model)
        if not embedding_params:
            raise RuntimeError(
                "No table is planned for the sparse optimizer, so there is "
                "nothing for it to train. Every table's TablePlan says DENSE; "
                "if that is really intended this backend needs a way to report "
                "no sparse runtime at all."
            )
        sparse_optimizer, sparse_lr_buffer = self._sparse_optimizer_builder(
            model,
            list(embedding_params.values()),
            sparse_config,
            device,
        )
        dense_tables = sum(
            1
            for plan in self._table_plans.values()
            if plan.optimizer is TableOptimizer.DENSE
        )
        logger.info(
            "InGraphEmbeddingBackend: %d table param(s) on %s, %d on the dense "
            "optimizer",
            len(embedding_params),
            type(sparse_optimizer).__name__,
            dense_tables,
        )
        return EmbeddingSetupResult(
            model=model,
            dense_named_parameters=self.dense_named_parameters(model),
            sparse_runtime=SparseOptimizerRuntime(
                placement=SparseUpdatePlacement.COMPILED_OPTIMIZER,
                config=sparse_config,
                optimizer=sparse_optimizer,
                lr_buffer=sparse_lr_buffer,
            ),
        )
