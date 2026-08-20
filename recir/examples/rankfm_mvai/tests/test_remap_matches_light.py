#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""Test if RecIR's ID remapping matches light

The two legs clamp dump IDs into the model's hash space differently:

    light   MVAI ``_shrink_sparse_features``  ->  fmod(values, max_hash_size)
            ONE modulus, applied only to keys in the embedding-table set
    recir   ``_remap_kjt_ids``                ->  values.remainder(key_to_size[key])
            PER-KEY modulus

For any table smaller than the cap those produce DIFFERENT IDs, so the two legs
read different embedding rows from the same checkpoint. That is what the parity
comparison measured: all 9 contextual feature tensors disagreeing while the
config, the dump files and the row selection were identical.

Both remappings are pure tensor transforms, so this reproduces the disagreement
and pins the fix locally -- no cluster run, no image rebuild.
"""

from typing import AbstractSet, Dict

import torch
from later.unittest import TestCase
from recir.examples.rankfm_mvai.data_pipe import _remap_kjt_ids
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_CAP = 10_000_000


def _kjt(
    keys: list[str], per_key_ids: list[list[int]], batch: int
) -> KeyedJaggedTensor:
    """One row per batch entry per key, ids spread across the key's range.

    ``batch`` rows are filled by cycling each key's id list, so a caller can ask
    for two interesting ids on one key and one on another while every key still
    contributes the same number of rows -- which the KJT layout requires.
    """
    values, lengths = [], []
    for ids in per_key_ids:
        # Cycling silently drops ids past the batch, and the symptom would be a
        # test that passes without ever exercising the value it names.
        assert len(ids) <= batch, (
            f"{len(ids)} ids will not fit in {batch} rows; raise batch or the "
            "extras never appear in the KJT"
        )
        for b in range(batch):
            values.append(ids[b % len(ids)])
            lengths.append(1)
    return KeyedJaggedTensor(
        keys=keys,
        values=torch.tensor(values, dtype=torch.int64),
        lengths=torch.tensor(lengths, dtype=torch.int64),
    )


def _light_shrink(
    kjt: KeyedJaggedTensor, cap: int, table_keys: AbstractSet[str]
) -> torch.Tensor:
    """The light leg's transform, reduced to what affects values.

    ``table_keys`` is not decoration. MVAI shrinks a feature only
    ``if key in all_emb_table_feature_set``, and a model of light that skipped
    that guard would be LOOSER than light -- it would call a leg that wrongly
    remapped an untabled feature a match.
    """
    per_key = kjt.length_per_key()
    out = kjt.values().clone()
    start = 0
    for key, count in zip(kjt.keys(), per_key):
        if key in table_keys:
            out[start : start + count] = torch.fmod(out[start : start + count], cap)
        start += count
    return out


class RemapMatchesLightTest(TestCase):
    def test_per_key_matches_light_under_the_dumps_invariant(self) -> None:
        """Per-key moduli reproduce light's ids, given how the dump is hashed.

        Light divides by ONE number, the run's hash cap; this leg divides by
        each table's own post-cap size. They agree because both sides cap the
        same way -- ``min(num_embeddings, cap)`` -- which leaves two cases:

          * table above the cap: its size IS the cap, so the two divisors are
            the same number (feat_big below).
          * table below the cap: the divisors differ, but the dump's ids for
            that feature are already below the table, so both moduli are the
            identity (feat_small, feat_mid below).

        The fixture also carries a key with no table at all. Both sides must
        leave it untouched -- remapping it would be a divergence in the
        opposite direction from a wrong modulus, and it is the one path a
        fixture of only-tabled keys cannot reach.

        The second case is the dump's invariant, not an accident of these
        numbers: the dump was preprocessed with production-size hashing, so a
        feature's ids live in [0, prod_size). Measured on rank_0/batch_0, every
        feature obeys it -- 91<100, 492<500, 984<1000, 998,771<1e6,
        998,996,144<1e9.
        """
        keys = ["feat_small", "feat_big", "feat_mid", "feat_untabled"]
        kjt = _kjt(
            keys,
            # Small/mid tables: ids below their own size, as the dump guarantees.
            # Big table: capped, so ids run past it and must wrap.
            # Untabled: a large id BOTH sides must leave alone.
            [[91, 60], [23_456_789, 999_999_999], [492, 111], [77_777_777]],
            batch=8,
        )
        key_to_size: Dict[str, int] = {
            "feat_small": 100,  # never reached the cap
            "feat_big": _CAP,  # was larger, capped to it
            "feat_mid": 500,  # never reached the cap
            # feat_untabled is absent on purpose: no embedding table, no remap.
        }
        got = _remap_kjt_ids(kjt, key_to_size).values()
        light = _light_shrink(kjt, _CAP, set(key_to_size))
        self.assertEqual(
            int((got != light).sum()),
            0,
            "per-key remap no longer reproduces light's ids:\n"
            f"  here:  {got.tolist()}\n"
            f"  light: {light.tolist()}",
        )

    def test_an_id_past_its_uncapped_table_is_where_the_two_part(self) -> None:
        """The boundary of the equivalence above, stated as a test.

        If a dump ever carries an id at or past an UNCAPPED table's size, the
        per-key modulus wraps it to a row light never reaches -- light would
        have left that id alone. Nothing in the pipeline would say so: the run
        succeeds and the two legs' activations simply differ.

        Pinned so the equivalence is not mistaken for an identity, and so the
        symptom is on record if a re-preprocessed dump ever trips it.
        """
        keys = ["feat_small"]
        # 150 >= the table's 100 rows, and 150 < the cap: light leaves it alone.
        kjt = _kjt(keys, [[150]], batch=2)
        got = _remap_kjt_ids(kjt, {"feat_small": 100}).values()
        light = _light_shrink(kjt, _CAP, {"feat_small"})
        self.assertEqual(got.tolist(), [50, 50])
        self.assertEqual(light.tolist(), [150, 150])
        self.assertGreater(
            int((got != light).sum()),
            0,
            "an id past its uncapped table no longer diverges; if the remap "
            "learned about the cap, this test should become an equality",
        )
