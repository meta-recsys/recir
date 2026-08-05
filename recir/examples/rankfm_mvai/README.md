# RankFM-MVAI (xsurface V3) on GKE TPU v7x

End-to-end RankFM training driven by the `mvai_cloud` `RecTrainer`, with the
dense graph compiled through recir to XLA/GSPMD on TPU.

| | |
|---|---|
| **Docs** | [`docs/README.md`](docs/README.md) — setup, launch, benchmarks |
| | [`docs/COMPARE_EXPERIMENT.md`](docs/COMPARE_EXPERIMENT.md) — A/B run harness |
| **Manifests** | [`manifests/`](manifests) — JobSet YAMLs (4/64/128-chip, bench, parity) |
| **Scripts** | [`scripts/`](scripts) — image build, launch helpers, bench drivers |
| **Entry point** | `python3 -u -m recir.examples.rankfm_mvai.train <flags>` |

The embedding backends this example runs on are general and live in
[`recir/embedding/`](../../embedding);
[`rankfm_embedding_backends.py`](rankfm_embedding_backends.py) here holds only
the RankFM-specific bindings.
