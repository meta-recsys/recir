#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""RankFM batch synthesis and static-shape preparation helpers."""

from typing import cast, Dict, List

import torch
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from minimal_viable_ai.core.dataloader.dataloader import RandomDataPipe
from minimal_viable_ai.core.dataloader.dataloader_configs import (
    RandomDataSourceConfig,
    RandomROODataSourceConfig,
)
from minimal_viable_ai.core.ranking_common.configs import MTMLROOTrainBatch
from torchrec import KeyedJaggedTensor

from .rankfm_dispatchers import TEST_BATCH_SIZE, TEST_NUM_CANDIDATES


def tile_dim0(t: torch.Tensor, target: int) -> torch.Tensor:
    """Cyclically tile *t* along dim 0 to length ``target`` (or trim)."""
    cur = t.shape[0]
    if cur == target:
        return t
    if cur > target:
        return t[:target].contiguous()
    if cur == 0:
        # Nothing to tile from — fall back to zeros.
        return torch.zeros(target, *t.shape[1:], dtype=t.dtype, device=t.device)
    n_full = target // cur
    n_extra = target - n_full * cur
    parts = [t] * n_full
    if n_extra > 0:
        parts.append(t[:n_extra])
    return torch.cat(parts, dim=0)


def tile_last_dim(t: torch.Tensor, target: int) -> torch.Tensor:
    """Cyclically tile *t* along the last dim to length ``target`` (or trim)."""
    cur = t.shape[-1]
    if cur == target:
        return t
    if cur > target:
        return t[..., :target].contiguous()
    if cur == 0:
        return torch.zeros(*t.shape[:-1], target, dtype=t.dtype, device=t.device)
    n_full = target // cur
    n_extra = target - n_full * cur
    parts = [t] * n_full
    if n_extra > 0:
        parts.append(t[..., :n_extra])
    return torch.cat(parts, dim=-1)


def resize_kjt_batch(kjt: KeyedJaggedTensor, target_b: int) -> KeyedJaggedTensor:
    """Resize a per-key-major KJT to have ``target_b`` entries per key.

    Cyclically tiles each key's per-sample lengths and the corresponding
    per-sample value chunks.  Tiling (rather than zero-padding) keeps the
    synthesized batch "valid" enough to flow through the preprocessor
    and tokenizer in :func:`rankfm_embedding_backends.build_eager_overrides` --
    zero-padded entries short-circuit those paths and the sparse arch hook
    never fires.
    """
    keys = kjt.keys()
    K = len(keys)
    cur_lengths = kjt.lengths()
    cur_b = cur_lengths.shape[0] // K
    if cur_b == target_b:
        return kjt
    cur_values = kjt.values()

    new_lengths_parts: List[torch.Tensor] = []
    new_values_parts: List[torch.Tensor] = []
    cur_offset = 0

    for k in range(K):
        key_lengths = cur_lengths[k * cur_b : (k + 1) * cur_b]
        key_total = int(key_lengths.sum().item())
        key_values = cur_values[cur_offset : cur_offset + key_total]
        cur_offset += key_total

        if target_b < cur_b:
            tiled_lengths = key_lengths[:target_b]
            cum = torch.cat(
                [
                    torch.zeros(1, dtype=key_lengths.dtype, device=key_lengths.device),
                    key_lengths.cumsum(0),
                ]
            )
            tiled_values = key_values[: int(cum[target_b].item())]
        elif cur_b == 0:
            # Nothing to tile from — empty key segment.
            tiled_lengths = torch.zeros(
                target_b, dtype=key_lengths.dtype, device=key_lengths.device
            )
            tiled_values = torch.empty(
                0, dtype=key_values.dtype, device=key_values.device
            )
        else:
            n_full = target_b // cur_b
            n_extra = target_b - n_full * cur_b
            length_parts = [key_lengths] * n_full
            value_parts = [key_values] * n_full
            if n_extra > 0:
                length_parts.append(key_lengths[:n_extra])
                cum = torch.cat(
                    [
                        torch.zeros(
                            1, dtype=key_lengths.dtype, device=key_lengths.device
                        ),
                        key_lengths.cumsum(0),
                    ]
                )
                value_parts.append(key_values[: int(cum[n_extra].item())])
            tiled_lengths = torch.cat(length_parts)
            tiled_values = torch.cat(value_parts) if value_parts else key_values

        new_lengths_parts.append(tiled_lengths)
        new_values_parts.append(tiled_values)

    new_lengths = torch.cat(new_lengths_parts)
    new_values = (
        torch.cat(new_values_parts)
        if new_values_parts
        else torch.empty(0, dtype=cur_values.dtype, device=cur_values.device)
    )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=new_values,
        lengths=new_lengths,
    )


def resize_kjt_per_key_lengths(
    kjt: KeyedJaggedTensor,
    per_key_per_sample_budgets: Dict[str, int],
    batch_size: int,
) -> KeyedJaggedTensor:
    """Resize each key's per-sample length to its configured budget."""
    keys = kjt.keys()
    K = len(keys)
    cur_lengths = kjt.lengths()
    cur_values = kjt.values()
    assert cur_lengths.shape[0] == K * batch_size, (
        f"KJT lengths size {cur_lengths.shape[0]} != "
        f"num_keys ({K}) * batch_size ({batch_size})"
    )

    new_lengths_parts: List[torch.Tensor] = []
    new_values_parts: List[torch.Tensor] = []
    cur_offset = 0

    for k_idx, key in enumerate(keys):
        key_lengths = cur_lengths[k_idx * batch_size : (k_idx + 1) * batch_size]
        key_total = int(key_lengths.sum().item())
        key_values = cur_values[cur_offset : cur_offset + key_total]
        cur_offset += key_total

        if key not in per_key_per_sample_budgets:
            new_lengths_parts.append(key_lengths)
            new_values_parts.append(key_values)
            continue

        budget = int(per_key_per_sample_budgets[key])
        new_key_lengths = torch.full(
            (batch_size,),
            budget,
            dtype=key_lengths.dtype,
            device=key_lengths.device,
        )

        if budget == 0:
            new_key_values = torch.empty(
                0, dtype=key_values.dtype, device=key_values.device
            )
        elif key_total == 0:
            new_key_values = torch.zeros(
                (budget * batch_size,),
                dtype=key_values.dtype,
                device=key_values.device,
            )
        else:
            cum = torch.cat(
                [
                    torch.zeros(1, dtype=key_lengths.dtype, device=key_lengths.device),
                    key_lengths.cumsum(0),
                ]
            )
            sample_parts: List[torch.Tensor] = []
            for b in range(batch_size):
                start = int(cum[b].item())
                end = int(cum[b + 1].item())
                sample_data = key_values[start:end]
                sample_len = end - start
                if sample_len >= budget:
                    sample_parts.append(sample_data[:budget])
                elif sample_len == 0:
                    sample_parts.append(
                        torch.zeros(
                            (budget,),
                            dtype=key_values.dtype,
                            device=key_values.device,
                        )
                    )
                else:
                    n_full = budget // sample_len
                    n_extra = budget - n_full * sample_len
                    parts = [sample_data] * n_full
                    if n_extra > 0:
                        parts.append(sample_data[:n_extra])
                    sample_parts.append(torch.cat(parts))
            new_key_values = torch.cat(sample_parts)

        new_lengths_parts.append(new_key_lengths)
        new_values_parts.append(new_key_values)

    new_lengths = torch.cat(new_lengths_parts)
    new_values = (
        torch.cat(new_values_parts)
        if new_values_parts
        else torch.empty(0, dtype=cur_values.dtype, device=cur_values.device)
    )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=new_values,
        lengths=new_lengths,
    )


def resize_batch_inplace(
    batch: MTMLROOTrainBatch, target_b_ro: int, target_b_nro: int
) -> None:
    """Resize *batch* in place to ``(target_b_ro, target_b_nro)``.

    Synthesizes the override-build batch from a small trace-time batch
    so we don't need a second :class:`RandomDataPipe` call at large
    sizes (that path can SIGABRT inside the data pipe's C++ allocator).

    Strategy: cyclically tile each tensor / KJT to the target size so
    the synthesized batch contains real (repeated) samples — never zeros.
    Zero-padded entries (or a 0 in ``num_candidates``) short-circuit the
    tokenizer's call to the sparse arch in
    :func:`rankfm_embedding_backends.build_eager_overrides`, and
    the forward hook then never fires, leaving the cached output empty.

    ``num_candidates`` is distributed as ``target_b_nro // target_b_ro``
    floor with first ``remainder`` samples getting +1, so
    ``sum(num_candidates) == target_b_nro`` exactly -- mirrors PRT's
    ``pad_count_to_sum`` runtime behavior.
    """
    if batch.ro_float_features.shape[0] != target_b_ro:
        batch.ro_float_features = tile_dim0(batch.ro_float_features, target_b_ro)
        batch.ro_id_list_features = resize_kjt_batch(
            batch.ro_id_list_features, target_b_ro
        )

    if batch.nro_float_features.shape[0] != target_b_nro:
        batch.nro_float_features = tile_dim0(batch.nro_float_features, target_b_nro)
        batch.nro_id_list_features = resize_kjt_batch(
            batch.nro_id_list_features, target_b_nro
        )
        if batch.unary_features is not None:
            batch.unary_features = resize_kjt_batch(batch.unary_features, target_b_nro)
        batch.supervision_label = tile_last_dim(batch.supervision_label, target_b_nro)
        batch.supervision_weight = tile_last_dim(batch.supervision_weight, target_b_nro)
        all_labels = batch.all_labels
        if all_labels is not None:
            for key in list(all_labels.keys()):
                all_labels[key] = tile_dim0(all_labels[key], target_b_nro)

    per_sample = target_b_nro // target_b_ro
    remainder = target_b_nro - per_sample * target_b_ro
    nc = torch.full(
        (target_b_ro,),
        per_sample,
        dtype=batch.num_candidates.dtype,
        device=batch.num_candidates.device,
    )
    if remainder > 0:
        nc[:remainder] = nc[:remainder] + 1
    batch.num_candidates = nc


def create_random_train_batch(
    feature_config: BlueReelsVDDFeatureConfig,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
    num_candidates: int = -1,
    pin_memory: bool = False,
) -> MTMLROOTrainBatch:
    """Generate a random MTMLROOTrainBatch via RandomROODataSourceConfig.

    All keys, shapes, and per-key lengths come from the feature config
    (``_RO_HISTORY_FEATURES``, ``_RO_CONTEXTUAL_FEATURES``,
    ``_NRO_FEATURES``).  ``randgen`` produces jagged num_candidates (up to
    ``num_candidates``) and variable RO history lengths when given a
    RecROORandGenContext.

    ``num_candidates`` overrides the per-request candidate count; when left at
    the ``-1`` sentinel it falls back to ``TEST_NUM_CANDIDATES`` (preserving the
    default used by the xsurface unit tests).

    ``pin_memory`` pins the generated host buffers for faster async H2D transfer.
    """
    B = batch_size
    C = num_candidates if num_candidates > 0 else TEST_NUM_CANDIDATES

    data_source_config = RandomROODataSourceConfig(
        batch_size=B,
        num_batches=1,
        num_generate=1,
        num_candidates=C,
        pin_memory=pin_memory,
    )
    data_pipe = RandomDataPipe(
        feature_config=feature_config,
        data_source_config=data_source_config,
    )
    batch = cast(MTMLROOTrainBatch, next(iter(data_pipe)))
    return batch.to(device, non_blocking=False)


def create_uniform_train_batch(
    feature_config: BlueReelsVDDFeatureConfig,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
) -> MTMLROOTrainBatch:
    """Generate a uniform MTMLROOTrainBatch via RandomDataSourceConfig.

    Uses the non-ROO data source so every sample has exactly
    ``feature_config.num_candidates_per_request`` candidates and
    ``feature_config.history_len`` history items, giving a deterministic
    ``B_NRO = batch_size * num_candidates_per_request``. Callers that need
    ``B_NRO`` to divide evenly across ``num_local`` (e.g., when there is no
    ``PaddingRegionTransform`` to align jagged totals) should use this
    instead of :func:`create_random_train_batch`. The caller is responsible
    for mutating ``feature_config.num_candidates_per_request`` /
    ``history_len`` first if they need shapes other than the dataloader
    defaults.
    """
    data_pipe = RandomDataPipe(
        feature_config=feature_config,
        data_source_config=RandomDataSourceConfig(
            batch_size=batch_size,
            num_batches=1,
            num_generate=1,
            pin_memory=False,
        ),
    )
    batch = cast(MTMLROOTrainBatch, next(iter(data_pipe)))
    return batch.to(device, non_blocking=False)
