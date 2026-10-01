# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Named launch configurations for ``mvai-cloud run --launch-config``.

A launch config is the parameter set that used to live in a hand-written JobSet under
``manifests/``. Those carried the knowledge of which flags a production run
needs -- and, just as load-bearing, which flags it must NOT pass -- as prose in
YAML comments, duplicated per chip count, with nothing checking that the copies
agreed. ``manifests/rankfm_mvai-64chip.yaml`` and ``-128chip.yaml`` differed in
exactly one value and had drifted in four others.

Deliberately stdlib-only and importing nothing but :mod:`dump`. Three callers
with three different environments read this: ``mvai-cloud`` (a venv holding
only jinja2), ``scripts/set_dump.py`` (stdlib-only by policy), and the tests.

Sizes are derived, never written down twice:

* ``local_batch_size`` is ``global_batch // hosts``, so one launch config covers every
  chip count. Halving the chips doubles the local batch and the global batch --
  the thing that actually has to match production -- stays put.
* ``max_steps`` is one pass over the dump, computed from :mod:`dump`. This is
  what ``set_dump.py`` used to re-derive with a line-anchored regex over a
  hardcoded list of three manifest filenames, one of which had already been
  deleted.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def _load_sibling(name: str):
    """A sibling module, importable from the launcher venv too.

    ``mvai-cloud`` runs in a venv holding only jinja2, and importing
    ``recir.examples.rankfm_mvai.<name>`` normally executes
    ``recir/__init__.py``, which imports torch. So fall back to loading the
    sibling file directly. Both modules reached this way (``dump``,
    ``prod_config_spec``) are stdlib-only, which is what makes that safe.
    """
    try:
        module = __import__(f"recir.examples.rankfm_mvai.{name}", fromlist=[name])
        return module
    except ImportError:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"rankfm_mvai_{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {path}") from None
        module = importlib.util.module_from_spec(spec)
        # Register before exec: anything resolving __module__ back through
        # sys.modules (dataclasses does) gets None otherwise.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


dump = _load_sibling("dump")
prod_config_spec = _load_sibling("prod_config_spec")
trainer_args = _load_sibling("trainer_args")

# Re-exported so the launcher, which loads this module by path, does not have
# to repeat the sibling-loading dance for the spec as well.
PROD_CONFIG_ARTIFACT_CONTAINER_PATH: str = (
    prod_config_spec.PROD_CONFIG_ARTIFACT_CONTAINER_PATH
)

# Rows the dump holds in total: shards x batches per shard x rows per batch.
TOTAL_ROWS: int = dump.NUM_RANKS * dump.NUM_BATCHES * dump.BATCH_SIZE

# A v7 host is 4 physical chips / 8 JAX devices. Sizes are written in hosts
# because that is what -j takes and what `parallelism:` counts.
CHIPS_PER_HOST: int = 4


@dataclass(frozen=True)
class JobSpec:
    """A launch config with the per-run sizes filled in."""

    name: str
    entrypoint: str
    entrypoint_args: Sequence[str]
    env: Mapping[str, str]
    prod_config: ProdConfigSpec | None
    barnfs_path: str
    priority_class: str | None
    termination_grace_seconds: int | None
    barnfs_encryption: bool
    checkpoint_save_dir: str
    hosts: int
    # The sizes and dataset are None for a launch config with no trainer.
    local_batch_size: int | None
    global_batch: int | None
    max_steps: int | None
    dataset_name: str | None

    def summary(self) -> str:
        if self.local_batch_size is None:
            return (
                f"launch config {self.name}: {self.hosts} hosts x {CHIPS_PER_HOST} "
                f"chips = {self.hosts * CHIPS_PER_HOST} chips, entry point "
                f"{self.entrypoint}"
            )
        return (
            f"launch config {self.name}: {self.hosts} hosts x {CHIPS_PER_HOST} chips "
            f"= {self.hosts * CHIPS_PER_HOST} chips, "
            f"local_batch_size={self.local_batch_size}, "
            f"global_batch={self.global_batch}, max_steps={self.max_steps} "
            f"(one pass over {self.dataset_name})"
        )


@dataclass(frozen=True)
class TrainerSpec:
    """What a launch config needs to know about a training entry point: the
    dataclasses its CLI is built from, and the sharded dataset it reads. It
    describes the trainer; it does not train.
    """

    flag_classes: tuple[type, ...]
    dataset_name: str
    num_shards: int
    # Rows per file: a step reads whole files.
    shard_batch_size: int
    total_rows: int

    def flag_names(self) -> frozenset[str]:
        return frozenset(
            f.name for cls in self.flag_classes for f in dataclasses.fields(cls)
        )

    def check_flags(self, names: Iterable[str], source: str) -> None:
        unknown = sorted(set(names) - self.flag_names())
        if unknown:
            raise ValueError(
                f"{source} flags the trainer does not accept: "
                f"{', '.join('--' + name for name in unknown)}"
            )

    def sizes(self, global_batch: int, hosts: int) -> tuple[int, int]:
        """``(local_batch_size, max_steps)`` for *hosts*, or raise saying why not."""
        if hosts <= 0:
            raise ValueError(f"hosts must be positive, got {hosts}")
        if global_batch % hosts:
            raise ValueError(
                f"global_batch {global_batch} is not divisible by {hosts} "
                f"hosts; pick a host count that divides it so every rank "
                f"carries the same number of rows"
            )
        local_batch_size = global_batch // hosts
        if local_batch_size % self.shard_batch_size:
            raise ValueError(
                f"local_batch_size {local_batch_size} (= global_batch "
                f"{global_batch} / {hosts} hosts) is not a whole multiple "
                f"of {self.dataset_name}'s batch_size {self.shard_batch_size}. "
                f"A step reads whole files, so anything else silently trains "
                f"on a subsample of each one."
            )
        if hosts > self.num_shards:
            raise ValueError(
                f"{hosts} hosts exceeds {self.dataset_name}'s {self.num_shards} "
                f"shards, so pods would repeat each other's data instead of "
                f"sharding it"
            )
        if self.total_rows % global_batch:
            raise ValueError(
                f"{self.dataset_name} has {self.total_rows} rows, which is not "
                f"a whole number of {global_batch}-row global batches"
            )
        return local_batch_size, self.total_rows // global_batch


@dataclass(frozen=True)
class ProdConfigSpec:
    """How to bake a model's prod config into the image, for an entry point
    that cannot build it there (the image has no //minimal_viable_ai).

    The image build runs ``export_target`` on the devserver with ``--out``,
    ``--source-revision`` and, for a dirty working copy, ``--source-dirty``.
    The file lands at PROD_CONFIG_ARTIFACT_CONTAINER_PATH, and the pod gets
    PROD_CONFIG_ARTIFACT pointing at it. Inside the built image, the build then
    calls ``check_module``'s ``check(path)``, which raises if the file is not
    the config it should be and otherwise returns a one-line summary.
    """

    export_target: str
    check_module: str


RANKFM_PROD_CONFIG: ProdConfigSpec = ProdConfigSpec(
    export_target="//recir/examples/rankfm_mvai:export_prod_config",
    check_module="recir.examples.rankfm_mvai.prod_config_artifact",
)


# recir.examples.rankfm_mvai.train on the udd_fm replay dump.
RANKFM_TRAINER_SPEC: TrainerSpec = TrainerSpec(
    flag_classes=(trainer_args.RankFMTrainerConfig, trainer_args.RankFMModelArgs),
    dataset_name=dump.DUMP_NAME,
    num_shards=dump.NUM_RANKS,
    shard_batch_size=dump.BATCH_SIZE,
    total_rows=TOTAL_ROWS,
)


@dataclass(frozen=True)
class LaunchConfig:
    """A named launch configuration.

    ``trainer_flags`` values: ``None`` renders a bare flag, anything else
    renders ``--name value``. ``{job_name}`` in a value is substituted at
    resolve time -- it cannot be left as a shell ``${JOBSET_NAME}`` because the
    launcher passes entrypoint args through ``shlex.join``, which quotes them
    and stops the shell expanding anything.
    """

    description: str
    name: str
    entrypoint: str
    # None for an entry point with its own CLI and no dataset (e.g. a smoke
    # run): trainer_flags then pass through as written.
    trainer_spec: TrainerSpec | None
    trainer_flags: Mapping[str, object]
    env: Mapping[str, str]
    # Required with a trainer_spec, meaningless without one.
    global_batch: int | None = None
    prod_config: ProdConfigSpec | None = None
    barnfs: bool = False
    priority_class: str | None = None
    termination_grace_seconds: int | None = None
    # BarnFS-encrypt DCP checkpoint writes. Off by default: an image built
    # with SKIP_BARNFS=1 fails to IMPORT when it is on.
    barnfs_encryption: bool = False
    # Where checkpoints land under the durable GCS mount. The encrypted and
    # plain runs are kept in separate prefixes so a reader never has to guess
    # which kind a directory holds.
    checkpoint_save_dir: str = "/mnt/gcs-out/checkpoints"

    def __post_init__(self) -> None:
        if (self.trainer_spec is None) != (self.global_batch is None):
            raise ValueError(
                f"launch config {self.name!r}: global_batch must be set exactly "
                f"when trainer_spec is"
            )

    def resolve(
        self,
        *,
        hosts: int,
        job_name: str,
        overrides: Sequence[str] = (),
    ) -> JobSpec:
        """Fill in the per-run sizes, dropping anything *overrides* also sets.

        *overrides* is the raw token list the user passed after ``--``. A flag
        present there replaces the launch config's rather than being appended after
        it: argparse would resolve the duplicate the same way, but the rendered
        command is what someone reads when a run goes wrong, and
        ``--max_steps 320725 --max_steps 20`` does not say which one won.
        """
        flags = dict(self.trainer_flags)
        local_batch_size: int | None = None
        max_steps: int | None = None
        spec = self.trainer_spec
        if spec is not None:
            assert self.global_batch is not None  # __post_init__
            local_batch_size, max_steps = spec.sizes(self.global_batch, hosts)
            spec.check_flags(self.trainer_flags, f"launch config {self.name!r} sets")
            spec.check_flags(_flag_names(overrides), "these override")
            flags["local_batch_size"] = local_batch_size
            flags["max_steps"] = max_steps
        for name in _flag_names(overrides):
            flags.pop(name, None)

        args: list[str] = []
        for flag, value in flags.items():
            args.append(f"--{flag}")
            if value is not None:
                args.append(str(value).format(job_name=job_name))

        return JobSpec(
            name=self.name,
            entrypoint=self.entrypoint,
            entrypoint_args=args,
            env={k: v.format(job_name=job_name) for k, v in self.env.items()},
            prod_config=self.prod_config,
            barnfs_path=dump.GCS_ROOT if self.barnfs else "",
            priority_class=self.priority_class,
            termination_grace_seconds=self.termination_grace_seconds,
            barnfs_encryption=self.barnfs_encryption,
            checkpoint_save_dir=self.checkpoint_save_dir,
            hosts=hosts,
            local_batch_size=local_batch_size,
            global_batch=self.global_batch,
            max_steps=max_steps,
            dataset_name=spec.dataset_name if spec is not None else None,
        )


# Production training run. This is manifests/rankfm_mvai-64chip.yaml and
# -128chip.yaml, unified -- they differed only in local_batch_size, which is
# now derived.
#
# Flags that are ABSENT are as load-bearing as the ones present, and none of
# these absences is enforced by the trainer; config._warn_on_incoherent_flags
# only warns. Do not add:
#
#   --hash_size_cap           would clamp prod's ~3e9-row table and rewrite
#                             feature_hash_sizes to match, leaving the whole
#                             pipeline self-consistent and the divergence
#                             undetectable at runtime. Its absence IS the run.
#   --data_file_template      resolves to exactly one file, recycled through a
#                             rotating window: a throughput benchmark, not
#                             training.
#   --data_replicate_across_ranks
#                             makes every pod read the same shard instead of
#                             sharding across them.
#   --num_candidates          inert under --prod_faithful (the NRO budget comes
#                             from prod's hstu_max_output_len), but reads as if
#                             it applied.
#   --l1_num_layers / --l2_num_layers
#                             also inert; 3 and 11 happen to be prod's values,
#                             so passing them reads as if it set them.
_RANKFM_PROD = LaunchConfig(
    description="Production training run on the udd_fm replay dump.",
    name="rankfm-prod",
    entrypoint="recir.examples.rankfm_mvai.train",
    trainer_spec=RANKFM_TRAINER_SPEC,
    # 32768 = prod's global batch. GSPMD shards it to 128 rows per chip at 64
    # chips, comfortably under prod's own 512 rows per GPU rank.
    global_batch=32768,
    trainer_flags={
        "save_every_n_steps": 5000,
        "keep_last_n": 2,
        "async_checkpoint": None,
        "prod_faithful": None,
        "bf16_training": None,
        "embedding_backend": "in_graph",
        "gcs_data_path": "/mnt/barnfs",
        "data_load_workers": 2,
        "data_prefetch_buffer": 32,
        "metrics_report_interval": 100,
        "in_graph_metrics": None,
        "metric_window_size": 64000000,
        "pipeline_preprocessor": None,
        # Steady state on purpose: a window near step 3 measures compile-warm
        # behaviour the run spends almost none of its time in.
        "profile_start_step": 900,
        "profile_end_step": 906,
        "profile_dir": (
            "gs://cq-mrs-core-shared_mrs_tpu_y317h/tpu-profiles/{job_name}"
        ),
    },
    env={
        # Route HSTU attention to the recir Helion kernels, which trace
        # attention as one ragged op instead of materializing the dense
        # [B, H, N, N] score matrix. That is what makes the per-chip batch fit.
        # Requires semi-local attention and dropout off, which
        # prod_config_shim already does at the config layer.
        "RANKFM_MVAI_ATTENTION_BACKEND": "helion",
        "JAX_ENABLE_X64": "1",
        # Print the PYTHON stack on a fatal signal. A SIGSEGV kills the process
        # before the exception machinery runs, so without this the log carries
        # only native frames reading "(unknown)".
        "PYTHONFAULTHANDLER": "1",
        # libtpu stderr logging. "0" surfaces the compiler's "Largest program
        # allocations in hbm" breakdown, which is the only way to see which
        # buffers an HBM OOM is made of. Very verbose, so off by default.
        "TPU_STDERR_LOG_LEVEL": "3",
        # "1" logs embedding/index/scatter/collective ops and the
        # jagged<->dense lowering decisions; "full" adds the whole FX graph
        # plus HLO. ~30k log lines, so off here.
        "RECIR_DUMP_GRAPHS": "0",
    },
    prod_config=RANKFM_PROD_CONFIG,
    barnfs=True,
    priority_class="high-priority",
    # SIGTERM has to flush the log tail through gcsfuse before the kubelet
    # escalates to SIGKILL. The 30s default is not enough headroom when the
    # mount is under checkpoint load, which is exactly when the evidence
    # matters most.
    termination_grace_seconds=90,
    # Matches manifests/rankfm_mvai-64chip.yaml. Turning encryption on would
    # also strand the existing checkpoints: a resume reads the configured
    # prefix, and the two prefixes are different directories.
    barnfs_encryption=False,
    checkpoint_save_dir="/mnt/gcs-out/checkpoints",
)


# Every launch config the CLI can name. Add yours here.
#
# Derive from an existing one rather than copying it, so a change to the base
# reaches every variant:
#
#     _SEQLEN_2048 = dataclasses.replace(
#         _RANKFM_PROD,
#         name="rankfm-seqlen-2048",
#         description="Longer history; does NE catch up?",
#         trainer_flags={**_RANKFM_PROD.trainer_flags, "seq_len_cap": 2048},
#     )
#
# Worth adding one for a setup that will be re-run, handed over, or used as a
# baseline. A one-off is better served by `-- --seq_len_cap 2048`.
LAUNCH_CONFIGS: dict[str, LaunchConfig] = {
    config.name: config for config in (_RANKFM_PROD,)
}


def _flag_names(tokens: Sequence[str]) -> set[str]:
    """Trainer-config field names set by *tokens*.

    ``--no_<field>`` maps to ``<field>``: config._add_dataclass_args registers
    every bool as a mutually exclusive ``--x`` / ``--no_x`` pair against one
    dest, so ``--no_async_checkpoint`` has to displace the launch config's
    ``--async_checkpoint`` rather than sit next to it.
    """
    names = set()
    for token in tokens:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0]
        names.add(name[3:] if name.startswith("no_") else name)
    return names


def get(name: str) -> LaunchConfig:
    """Look up a launch config, or raise listing what there is."""
    try:
        return LAUNCH_CONFIGS[name]
    except KeyError:
        known = ", ".join(sorted(LAUNCH_CONFIGS)) or "(none)"
        raise ValueError(f"unknown launch config {name!r}; known: {known}") from None


def describe() -> str:
    """One line per launch config, for ``--help`` and error messages."""
    return "\n".join(
        f"  {name}  {config.description}"
        for name, config in sorted(LAUNCH_CONFIGS.items())
    )


__all__ = [
    "CHIPS_PER_HOST",
    "PROD_CONFIG_ARTIFACT_CONTAINER_PATH",
    "ProdConfigSpec",
    "RANKFM_PROD_CONFIG",
    "LAUNCH_CONFIGS",
    "LaunchConfig",
    "JobSpec",
    "RANKFM_TRAINER_SPEC",
    "TOTAL_ROWS",
    "TrainerSpec",
    "describe",
    "get",
]
