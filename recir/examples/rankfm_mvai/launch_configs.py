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
from typing import Mapping, Sequence


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
    prod_config: bool
    barnfs_path: str
    priority_class: str | None
    termination_grace_seconds: int | None
    barnfs_encryption: bool
    checkpoint_save_dir: str
    hosts: int
    local_batch_size: int
    global_batch: int
    max_steps: int

    def summary(self) -> str:
        return (
            f"launch config {self.name}: {self.hosts} hosts x {CHIPS_PER_HOST} chips "
            f"= {self.hosts * CHIPS_PER_HOST} chips, "
            f"local_batch_size={self.local_batch_size}, "
            f"global_batch={self.global_batch}, max_steps={self.max_steps} "
            f"(one pass over {dump.DUMP_NAME})"
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
    global_batch: int
    trainer_flags: Mapping[str, object]
    env: Mapping[str, str]
    prod_config: bool = False
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
        local_batch_size, max_steps = self._sizes(hosts)
        self._check_flag_names()
        _check_override_names(overrides)
        flags = dict(self.trainer_flags)
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
        )

    def _check_flag_names(self) -> None:
        """Refuse a flag the trainer does not have.

        A typo here would otherwise surface as "unrecognized arguments" from
        inside the pod, after the image build and the slice wait.
        """
        known = _trainer_flag_names()
        unknown = sorted(set(self.trainer_flags) - known)
        if unknown:
            raise ValueError(
                f"launch config {self.name!r} sets flags the trainer does not accept: "
                f"{', '.join('--' + name for name in unknown)}"
            )

    def _sizes(self, hosts: int) -> tuple[int, int]:
        """``(local_batch_size, max_steps)`` for *hosts*, or raise saying why not.

        Every check here is a failure that is otherwise silent. A local batch
        that is not a whole multiple of the dump's stride trains on a rotating
        subsample of each file; more pods than the dump has shards makes pods
        repeat each other's data; a global batch that does not divide the dump
        means the last step of an "epoch" is short.
        """
        if hosts <= 0:
            raise ValueError(f"hosts must be positive, got {hosts}")
        if self.global_batch % hosts:
            raise ValueError(
                f"global_batch {self.global_batch} is not divisible by {hosts} "
                f"hosts; pick a host count that divides it so every rank "
                f"carries the same number of rows"
            )
        local_batch_size = self.global_batch // hosts
        if local_batch_size % dump.BATCH_SIZE:
            raise ValueError(
                f"local_batch_size {local_batch_size} (= global_batch "
                f"{self.global_batch} / {hosts} hosts) is not a whole multiple "
                f"of the dump's batch_size {dump.BATCH_SIZE}. A step reads "
                f"whole files, so anything else silently trains on a "
                f"subsample of each one."
            )
        if hosts > dump.NUM_RANKS:
            raise ValueError(
                f"{hosts} hosts exceeds the dump's {dump.NUM_RANKS} shards, so "
                f"pods would repeat each other's data instead of sharding it"
            )
        if TOTAL_ROWS % self.global_batch:
            raise ValueError(
                f"dump has {TOTAL_ROWS} rows, which is not a whole number of "
                f"{self.global_batch}-row global batches"
            )
        return local_batch_size, TOTAL_ROWS // self.global_batch


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
    prod_config=True,
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


def _trainer_flag_names() -> frozenset[str]:
    """Every ``--flag`` the trainer's CLI accepts.

    Derived from the dataclasses rather than listed, because the trainer's own
    parser is derived from them too (``trainer_args._add_dataclass_args``).
    """
    return frozenset(
        field.name
        for cls in (trainer_args.RankFMTrainerConfig, trainer_args.RankFMModelArgs)
        for field in dataclasses.fields(cls)
    )


def _check_override_names(overrides: Sequence[str]) -> None:
    """Refuse an override flag the trainer does not have.

    ``_check_flag_names`` covers what the launch config sets; this covers what
    the caller typed after ``--``. Both reach the same argparse.
    """
    known = _trainer_flag_names()
    unknown = sorted(name for name in _flag_names(overrides) if name not in known)
    if unknown:
        raise ValueError(
            "these override flags are not accepted by the trainer: "
            + ", ".join("--" + name for name in unknown)
        )


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
    "LAUNCH_CONFIGS",
    "LaunchConfig",
    "JobSpec",
    "TOTAL_ROWS",
    "describe",
    "get",
]
