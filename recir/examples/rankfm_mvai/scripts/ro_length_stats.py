#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Per-key jagged-length distribution over a PreprocOutput dump.

``inspect_batch`` reports each key's total value count and max ID. That says
nothing about how those values are spread across rows, which is what decides
whether a per-sample RO budget is tight or wasteful. This samples files across
several dump ranks and reports the per-key length distribution, so one file
from one rank is not mistaken for the dataset.

Per-file means are reported alongside the pooled distribution: if they spread
widely, the pooled mean is not a property of the data but of which files were
picked.

    buck2 run //recir/examples/rankfm_mvai/scripts:ro_length_stats -- \\
        --num-ranks 8 --num-batches 4 --budget 3554 --nro-budget 20
"""

import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import zstandard

# Imported for its unpickling side effect: the dump is a torch.save of this
# dataclass, so the module must be in the link tree for torch.load to resolve
# it.
from minimal_viable_ai.core.ranking_common.configs import (  # noqa: F401
    MTMLROOTrainBatch,
)

# Reused rather than re-derived: the rank_X/batch_Y/Z.pt.zst vs legacy flat
# layout choice must not drift from the reader the training job uses.
from recir.examples.rankfm_mvai.data_pipe import _batch_file_name
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_DEFAULT_PATH = "manifold://mrs_recsys_gcp/tree/czheo_test/udd_fm_3d_20260722"
# Weighted to the LOW end on purpose. These lengths are produced by an upstream
# truncation, so the interesting structure is the tail below the ceiling, not
# above it -- an all-high quantile set reports p50 == p99 == max and hides it.
_QUANTILES: Tuple[float, ...] = (0.01, 0.1, 0.25, 0.5, 0.9)


def _make_downloader(path: str) -> Callable[[str], bytes]:
    """Return a name -> bytes reader for a manifold:// or local dump root."""
    if path.startswith("manifold://"):
        from mvai_infra.common.manifold import (  # @manual=//mvai_infra/common:manifold
            download_buffer_from_manifold,
            manifold_path_to_bucket_and_file_path,
        )

        bucket, prefix = manifold_path_to_bucket_and_file_path(path.rstrip("/"))

        def download_manifold(name: str) -> bytes:
            blob = f"{prefix}/{name}" if prefix else name
            return download_buffer_from_manifold(bucket, blob).getvalue()

        return download_manifold

    root = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dump root is not a directory: {root}")

    def download_local(name: str) -> bytes:
        with open(os.path.join(root, name), "rb") as fh:
            return fh.read()

    return download_local


def _spread(count: int, total: int) -> List[int]:
    """``count`` indices spread evenly over ``range(total)``, endpoints included."""
    if count >= total:
        return list(range(total))
    if count <= 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def _load_lengths(
    download: Callable[[str], bytes], name: str, field: str
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Return the KJT's keys, its ``(num_keys, stride)`` length matrix, and the
    batch's ``num_candidates``.

    ``lengths()`` is key-major and flat, so the view is what recovers the
    per-key, per-row structure. Reading it as-is is the mistake that makes a
    bimodal key set look like one population.

    ``num_candidates`` rides along because it is the other half of the padding
    question: it is the per-sample count the NRO budget pads out to, and it
    sits on the batch rather than inside any KJT.
    """
    raw = zstandard.ZstdDecompressor().decompress(download(name))
    batch = torch.load(io.BytesIO(raw), weights_only=False)
    kjt = getattr(batch, field, None)
    if not isinstance(kjt, KeyedJaggedTensor):
        raise TypeError(f"{field} on {type(batch).__name__} is not a KeyedJaggedTensor")
    keys = list(kjt.keys())
    lengths = kjt.lengths().view(len(keys), -1).to(torch.int64)
    return keys, lengths, batch.num_candidates.to(torch.int64)


def _quantiles(sorted_values: torch.Tensor, qs: Sequence[float]) -> List[int]:
    n = int(sorted_values.numel())
    return [int(sorted_values[min(n - 1, int(q * n))]) for q in qs]


def _summarize(values: torch.Tensor) -> Dict[str, float]:
    ordered, _ = torch.sort(values)
    p01, p10, p25, p50, p90 = _quantiles(ordered, _QUANTILES)
    top = int(ordered[-1])
    return {
        "n": int(values.numel()),
        "mean": float(values.to(torch.float64).mean()),
        "std": float(values.to(torch.float64).std()) if values.numel() > 1 else 0.0,
        "min": int(ordered[0]),
        "p01": p01,
        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p90": p90,
        "max": top,
        # What fraction sits exactly at the ceiling. With an upstream
        # truncation this is the single most descriptive number: it splits
        # "capped" rows from the tail of genuinely shorter ones.
        "frac_at_max": float((values == top).to(torch.float64).mean()),
        "frac_zero": float((values == 0).to(torch.float64).mean()),
    }


def _print_report(
    per_key: Dict[str, torch.Tensor],
    per_key_file_means: Dict[str, List[float]],
    per_sample_total: torch.Tensor,
    budget: int,
) -> Dict[str, object]:
    order = sorted(per_key, key=lambda k: -float(per_key[k].to(torch.float64).mean()))

    header = (
        f"{'key':<44}{'mean':>9}{'std':>8}{'min':>7}{'p01':>7}{'p10':>7}"
        f"{'p25':>7}{'p50':>7}{'max':>7}{'%@max':>8}{'%zero':>8}"
        f"{'file mean min..max':>22}"
    )
    print(header)
    print("-" * len(header))
    summaries: Dict[str, Dict[str, float]] = {}
    for key in order:
        s = _summarize(per_key[key])
        summaries[key] = s
        means = per_key_file_means[key]
        spread = f"{min(means):.1f}..{max(means):.1f}"
        print(
            f"{key:<44}{s['mean']:>9.1f}{s['std']:>8.1f}{s['min']:>7}{s['p01']:>7}"
            f"{s['p10']:>7}{s['p25']:>7}{s['p50']:>7}{s['max']:>7}"
            f"{s['frac_at_max']:>7.1%}{s['frac_zero']:>8.1%}{spread:>22}"
        )

    total = _summarize(per_sample_total)
    print()
    print(
        f"per-sample total over all keys: mean {total['mean']:.1f}  "
        f"p10 {total['p10']}  p50 {total['p50']}  max {total['max']}"
    )

    if budget > 0:
        print()
        print(f"against budget {budget}:")
        over = [k for k in order if summaries[k]["max"] > budget]
        print(f"{'key':<44}{'fill@mean':>11}{'fill@max':>10}{'headroom':>10}")
        for key in order:
            s = summaries[key]
            print(
                f"{key:<44}{s['mean'] / budget:>10.1%}{s['max'] / budget:>10.1%}"
                f"{budget - int(s['max']):>10}"
            )
        print()
        if over:
            print(
                f"{len(over)}/{len(order)} keys exceed the budget and are "
                f"truncated: {', '.join(over)}"
            )
        else:
            ceiling = max(int(s["max"]) for s in summaries.values())
            print(
                f"no key reaches the budget. The longest observed length is "
                f"{ceiling}, so {budget - ceiling} of every key's {budget} slots "
                f"are padding in every sample measured."
            )

    return {"per_key": summaries, "per_sample_total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=_DEFAULT_PATH, help="dump root")
    parser.add_argument(
        "--field",
        default="ro_id_list_features",
        help="batch field to measure (ro_id_list_features / nro_id_list_features)",
    )
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="per-sample RO budget to report fill rates against (0 disables)",
    )
    parser.add_argument(
        "--nro-budget",
        type=int,
        default=0,
        help="per-sample NRO candidate budget, i.e. hstu_max_output_len (0 disables)",
    )
    parser.add_argument("--json", default="", help="also write the summary here")
    args = parser.parse_args()

    download = _make_downloader(args.path)
    metadata = json.loads(download("metadata.json"))
    version = metadata.get("version")
    num_ranks = int(metadata["num_ranks"])
    num_batches = int(metadata["num_batches"])
    print(
        f"dump {args.path}\n"
        f"  metadata: {num_ranks} ranks x {num_batches} batches, "
        f"batch_size={metadata.get('batch_size')}, version={version}"
    )

    ranks = _spread(args.num_ranks, num_ranks)
    batches = _spread(args.num_batches, num_batches)
    names = [_batch_file_name(r, b, version) for b in batches for r in ranks]
    print(
        f"  sampling {len(names)} files: ranks {ranks} x batches {batches}, "
        f"field={args.field}"
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(lambda n: _load_lengths(download, n, args.field), names)
        )

    keys = results[0][0]
    for name, (file_keys, _, _) in zip(names, results):
        if file_keys != keys:
            raise ValueError(f"{name} has a different key set than {names[0]}")

    chunks: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
    per_key_file_means: Dict[str, List[float]] = {k: [] for k in keys}
    totals: List[torch.Tensor] = []
    candidates: List[torch.Tensor] = []
    for _, lengths, num_candidates in results:
        totals.append(lengths.sum(dim=0))
        candidates.append(num_candidates)
        for i, key in enumerate(keys):
            row = lengths[i]
            chunks[key].append(row)
            per_key_file_means[key].append(float(row.to(torch.float64).mean()))
    pooled = {k: torch.cat(v) for k, v in chunks.items()}

    print()
    summary = _print_report(pooled, per_key_file_means, torch.cat(totals), args.budget)

    nro = _summarize(torch.cat(candidates))
    print()
    print(
        f"num_candidates per sample: mean {nro['mean']:.2f}  min {nro['min']}  "
        f"p01 {nro['p01']}  p10 {nro['p10']}  p50 {nro['p50']}  max {nro['max']}  "
        f"(%@max {nro['frac_at_max']:.1%}, %zero {nro['frac_zero']:.1%})"
    )
    if args.nro_budget > 0:
        fill = nro["mean"] / args.nro_budget
        print(
            f"  against NRO budget {args.nro_budget}: fill {fill:.1%}. Every "
            f"sample is padded out to {args.nro_budget} candidate rows, so "
            f"{1 - fill:.1%} of every NRO tensor is padding."
        )
    summary["num_candidates"] = nro

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "path": args.path,
                    "field": args.field,
                    "ranks": ranks,
                    "batches": batches,
                    **summary,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
