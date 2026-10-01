# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""The rankfm-prod launch config, pinned to what the manifests it replaced carried.

``launch_configs.py`` replaced two hand-written JobSets. The golden test below pins
its expansion to the flag set those manifests had, so the migration is
provably behaviour-preserving rather than approximately so -- the manifests are
deleted, and without this the only record of what they said is the commit that
removed them.

Two properties are checked elsewhere because a unit test is the weaker place
for them:

* that ``launch_configs.py`` stays stdlib-only (``mvai-cloud`` reads it from a venv
  holding only jinja2) -- ``scripts/set_dump.py --check`` loads it under bare
  python3 with no buck at all, so a heavy import fails there.
* that the Dockerfile's COPY and the build script's staging directory agree
  with ``prod_config_spec`` -- ``build_image_podman.sh`` loads the artifact
  from ``PROD_CONFIG_ARTIFACT_CONTAINER_PATH`` inside the built image and
  asserts on it, which checks the actual artifact rather than the literals.
"""

from __future__ import annotations

import unittest
from unittest import mock

from recir.examples.rankfm_mvai import dump, launch_configs, prod_config_spec
from recir.examples.rankfm_mvai.trainer_args import (
    parse_cli,
    RankFMModelArgs,
    RankFMTrainerConfig,
)


# The flags manifests/rankfm_mvai-64chip.yaml passed, verbatim, minus the two
# that are now derived (--local_batch_size, --max_steps), minus --seq_len_cap
# (see _MUST_STAY_ABSENT), and with the profile
# directory's ${JOBSET_NAME} resolved. Written out rather than computed: a test
# that derives its expectation from the code under test proves nothing.
_RETIRED_64CHIP_FLAGS = {
    "--save_every_n_steps": "5000",
    "--keep_last_n": "2",
    "--async_checkpoint": None,
    "--prod_faithful": None,
    "--bf16_training": None,
    "--embedding_backend": "in_graph",
    "--gcs_data_path": "/mnt/barnfs",
    "--data_load_workers": "2",
    "--data_prefetch_buffer": "32",
    "--metrics_report_interval": "100",
    "--in_graph_metrics": None,
    "--metric_window_size": "64000000",
    "--pipeline_preprocessor": None,
    "--profile_start_step": "900",
    "--profile_end_step": "906",
    "--profile_dir": ("gs://cq-mrs-core-shared_mrs_tpu_y317h/tpu-profiles/JOB"),
}

# Flags whose ABSENCE is what makes this a production run. Nothing in the
# trainer enforces any of them -- config._warn_on_incoherent_flags only warns,
# and --hash_size_cap in particular clamps prod's tables and then rewrites
# feature_hash_sizes to match, so the whole pipeline stays self-consistent and
# the divergence is undetectable at runtime.
# --seq_len_cap is here rather than above because the retired manifest DID
# pass it, at 1024. It was never about the sequence: capping to 512 moved
# the HBM peak by 1G of 144G. The real consumer was a [T, B, D_out, E]
# fan-out in FusedMMoE's group-balance branch, removed by scaling after the
# reduction instead of before it (PROD_CONFIG_OVERRIDES.md, C9). Passing it
# now only discards two thirds of an average history and makes NE
# incomparable with prod's, which is the opposite of what this config is for.
_MUST_STAY_ABSENT = (
    "seq_len_cap",
    "hash_size_cap",
    "data_file_template",
    "data_replicate_across_ranks",
    "num_candidates",
    "l1_num_layers",
    "l2_num_layers",
)


def _as_map(tokens) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            result[flag] = tokens[index + 1]
            index += 2
        else:
            result[flag] = None
            index += 1
    return result


class RankfmProdLaunchConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.launch_config = launch_configs.get("rankfm-prod")

    def test_expansion_matches_the_retired_64chip_manifest(self) -> None:
        got = _as_map(
            self.launch_config.resolve(hosts=16, job_name="JOB").entrypoint_args
        )
        derived = {"--local_batch_size", "--max_steps"}
        self.assertEqual(
            _RETIRED_64CHIP_FLAGS, {k: v for k, v in got.items() if k not in derived}
        )

    def test_flags_that_must_stay_absent_are_absent(self) -> None:
        for hosts in (16, 32):
            got = _as_map(
                self.launch_config.resolve(hosts=hosts, job_name="JOB").entrypoint_args
            )
            for name in _MUST_STAY_ABSENT:
                self.assertNotIn(f"--{name}", got, f"at {hosts} hosts")

    def test_sizes_are_derived_from_the_host_count(self) -> None:
        # 64 and 128 chips differ only here, and the global batch -- the thing
        # that has to match production -- is the same for both.
        for hosts, expected_local in ((16, 2048), (32, 1024)):
            resolved = self.launch_config.resolve(hosts=hosts, job_name="JOB")
            self.assertEqual(expected_local, resolved.local_batch_size)
            self.assertEqual(32768, resolved.global_batch)

    def test_max_steps_is_one_pass_over_the_dump(self) -> None:
        resolved = self.launch_config.resolve(hosts=16, job_name="JOB")
        assert resolved.global_batch is not None
        self.assertEqual(
            launch_configs.TOTAL_ROWS // resolved.global_batch, resolved.max_steps
        )
        # What manifests/rankfm_mvai-64chip.yaml hardcoded, for the dump
        # dump.py currently names. A rotation changes this, and updating it
        # here is the intended way to notice.
        self.assertEqual(320725, resolved.max_steps)

    def test_local_batch_is_a_whole_multiple_of_the_dump_stride(self) -> None:
        for hosts in (8, 16, 32):
            local_batch_size = self.launch_config.resolve(
                hosts=hosts, job_name="JOB"
            ).local_batch_size
            assert local_batch_size is not None  # the trainer_spec derives it
            self.assertEqual(
                0,
                local_batch_size % dump.BATCH_SIZE,
                f"{hosts} hosts would train on a subsample of each file",
            )

    def test_host_count_that_does_not_divide_the_batch_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.launch_config.resolve(hosts=17, job_name="JOB")

    def test_more_hosts_than_dump_shards_is_refused(self) -> None:
        # Otherwise pods silently repeat each other's data.
        with self.assertRaises(ValueError):
            self.launch_config.resolve(hosts=dump.NUM_RANKS * 2, job_name="JOB")

    def test_job_name_is_substituted_into_the_profile_dir(self) -> None:
        # It cannot be left as a shell ${JOBSET_NAME}: the launcher passes
        # entrypoint args through shlex.join, which quotes them.
        resolved = self.launch_config.resolve(hosts=16, job_name="my-job-123")
        self.assertIn("tpu-profiles/my-job-123", " ".join(resolved.entrypoint_args))
        self.assertNotIn("{job_name}", " ".join(resolved.entrypoint_args))

    def test_user_flags_displace_the_launch_config_s_own(self) -> None:
        got = _as_map(
            self.launch_config.resolve(
                hosts=16, job_name="JOB", overrides=["--max_steps", "20"]
            ).entrypoint_args
        )
        self.assertNotIn("--max_steps", got)

    def test_no_x_displaces_the_bare_flag(self) -> None:
        # config._add_dataclass_args registers every bool as a mutually
        # exclusive --x / --no_x pair against one dest.
        got = _as_map(
            self.launch_config.resolve(
                hosts=16, job_name="JOB", overrides=["--no_async_checkpoint"]
            ).entrypoint_args
        )
        self.assertNotIn("--async_checkpoint", got)

    def test_every_flag_is_a_real_trainer_field(self) -> None:
        # A typo here would otherwise surface as argparse "unrecognized
        # arguments" inside the pod, after the build and the slice wait.
        fields = {
            f.name
            for cls in (RankFMTrainerConfig, RankFMModelArgs)
            for f in __import__("dataclasses").fields(cls)
        }
        for name in self.launch_config.trainer_flags:
            self.assertIn(name, fields)

    def test_a_misspelled_flag_is_refused(self) -> None:
        bad = launch_configs.LaunchConfig(
            description="x",
            name="bad",
            entrypoint="e",
            trainer_spec=launch_configs.RANKFM_TRAINER_SPEC,
            global_batch=32768,
            trainer_flags={"max_stpes": 5},
            env={},
        )
        with self.assertRaises(ValueError) as caught:
            bad.resolve(hosts=16, job_name="JOB")
        self.assertIn("--max_stpes", str(caught.exception))

    def test_unknown_name_lists_the_known_ones(self) -> None:
        with self.assertRaises(ValueError) as caught:
            launch_configs.get("does-not-exist")
        self.assertIn("rankfm-prod", str(caught.exception))


class NoTrainerSpecLaunchConfigTest(unittest.TestCase):
    """A launch config whose entry point has its own CLI and no dataset.

    No trainer_spec, so no dataset-derived --local_batch_size / --max_steps, no
    flag check, and no host-count constraint.
    """

    def _config(self, **flags: object) -> launch_configs.LaunchConfig:
        return launch_configs.LaunchConfig(
            description="x",
            name="other",
            entrypoint="some.other.entry",
            trainer_spec=None,
            trainer_flags=flags,
            env={"A": "{job_name}"},
        )

    def test_flags_pass_through_as_written(self) -> None:
        resolved = self._config(steps=5, verbose=None).resolve(hosts=1, job_name="JOB")
        self.assertEqual(["--steps", "5", "--verbose"], list(resolved.entrypoint_args))
        self.assertIsNone(resolved.local_batch_size)
        self.assertIsNone(resolved.global_batch)
        self.assertIsNone(resolved.max_steps)

    def test_any_host_count_resolves(self) -> None:
        # 17 would fail the rankfm dump's checks.
        for hosts in (1, 17):
            self.assertEqual(
                hosts, self._config().resolve(hosts=hosts, job_name="J").hosts
            )

    def test_env_reaches_the_job_spec(self) -> None:
        resolved = self._config().resolve(hosts=1, job_name="JOB")
        self.assertIsNone(resolved.prod_config)
        self.assertEqual({"A": "JOB"}, dict(resolved.env))
        self.assertIn("some.other.entry", resolved.summary())

    def test_global_batch_goes_with_a_trainer_spec(self) -> None:
        for spec, global_batch in (
            (launch_configs.RANKFM_TRAINER_SPEC, None),
            (None, 32768),
        ):
            with self.subTest(trainer_spec=spec, global_batch=global_batch):
                with self.assertRaises(ValueError):
                    launch_configs.LaunchConfig(
                        description="x",
                        name="x",
                        entrypoint="e",
                        trainer_spec=spec,
                        global_batch=global_batch,
                        trainer_flags={},
                        env={},
                    )


class OverrideFlagNameTest(unittest.TestCase):
    """A misspelled override must fail here, not inside the pod.

    The launch config's own flags were already checked; the tokens the caller
    types after ``--`` reach the same argparse, and finding out from the pod
    costs an image build and a slice wait.
    """

    def test_a_known_override_resolves(self) -> None:
        resolved = launch_configs.get("rankfm-prod").resolve(
            hosts=16, job_name="JOB", overrides=["--max_steps", "20"]
        )
        self.assertNotIn("--max_steps 320725", " ".join(resolved.entrypoint_args))

    def test_a_misspelled_override_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            launch_configs.get("rankfm-prod").resolve(
                hosts=16, job_name="JOB", overrides=["--max_stpes", "20"]
            )
        self.assertIn("--max_stpes", str(caught.exception))

    def test_the_message_names_every_unknown_flag(self) -> None:
        with self.assertRaises(ValueError) as caught:
            launch_configs.get("rankfm-prod").resolve(
                hosts=16,
                job_name="JOB",
                overrides=["--nope_one", "1", "--max_steps", "20", "--nope_two"],
            )
        message = str(caught.exception)
        self.assertIn("--nope_one", message)
        self.assertIn("--nope_two", message)
        self.assertNotIn("--max_steps", message)


class StartupReprTest(unittest.TestCase):
    """What the pod logs at startup, pinned to what the manifests produced.

    ``train.py`` logs ``Trainer config: {trainer_cfg}`` and ``Model args:
    {model_args}`` before the first step. Those reprs carry EVERY field, so
    pinning them covers the flags the launch config deliberately does not pass
    -- ``hash_size_cap=0``, ``data_file_template=''``, ``num_candidates=8`` --
    which a flag-list comparison cannot see, because an absent flag and a flag
    passed at its default look identical there.

    The strings below were captured from the retired manifests
    (``rankfm_mvai-64chip.yaml`` / ``-128chip.yaml``) by running their argv
    through the same ``parse_cli``. They are the record of what those files
    said, now that they are deleted.
    """

    TRAINER = (
        "RankFMTrainerConfig(local_batch_size={lbs}, max_steps=320725, "
        "save_every_n_steps=5000, keep_last_n=2, async_checkpoint=True, "
        "light_loss_norm=False, grad_accum_steps=1, bf16_training=True, "
        "gcs_data_path='/mnt/barnfs', data_file_template='', "
        "data_replicate_across_ranks=False, data_prefetch_buffer=32, "
        "data_load_workers=2, init_state_dict_path='', "
        "embedding_backend='in_graph', on_device_init=True, enable_padding=True, "
        "prefetch_batches=0, pin_memory=False, pull_metrics_every=10, "
        "metrics_report_interval=100, metric_window_size=64000000, "
        "in_graph_metrics=True, profile_start_step=900, profile_end_step=906, "
        "profile_dir='gs://cq-mrs-core-shared_mrs_tpu_y317h/tpu-profiles/JOB', "
        "enable_memory_log=False, pipeline_preprocessor=True, "
        "budgeted_seq_len_sampling=False, tb_dir='', tb_log_every=10)"
    )
    MODEL = (
        "RankFMModelArgs(hash_size_cap=0, l1_num_layers=3, l2_num_layers=11, "
        "seq_len_cap=0, max_uih_seq_len=32, num_candidates=8, "
        "l2_max_seq_len=0, disable_seq_len_sampling=False, "
        "seq_ro_per_sample_budget=24, prod_faithful=True, "
        "disable_inter_arch=False)"
    )

    def _reprs(self, hosts: int) -> tuple[str, str]:
        resolved = launch_configs.get("rankfm-prod").resolve(
            hosts=hosts, job_name="JOB"
        )
        with mock.patch("sys.argv", ["train", *resolved.entrypoint_args]):
            trainer_cfg, model_args = parse_cli()
        return repr(trainer_cfg), repr(model_args)

    def test_trainer_config_repr_matches_the_retired_64chip_manifest(self) -> None:
        trainer, _ = self._reprs(16)
        self.assertEqual(self.TRAINER.format(lbs=2048), trainer)

    def test_trainer_config_repr_matches_the_retired_128chip_manifest(self) -> None:
        trainer, _ = self._reprs(32)
        self.assertEqual(self.TRAINER.format(lbs=1024), trainer)

    def test_model_args_repr_is_host_count_independent(self) -> None:
        for hosts in (16, 32):
            _, model = self._reprs(hosts)
            self.assertEqual(self.MODEL, model, f"at {hosts} hosts")


class ProdConfigSpecTest(unittest.TestCase):
    def test_container_path_is_built_from_the_other_two_constants(self) -> None:
        self.assertEqual(
            f"/app/{prod_config_spec.PROD_CONFIG_STAGE_DIR}/"
            f"{prod_config_spec.PROD_CONFIG_ARTIFACT_FILENAME}",
            prod_config_spec.PROD_CONFIG_ARTIFACT_CONTAINER_PATH,
        )

    def test_launch_configs_reexports_the_container_path(self) -> None:
        # mvai-cloud loads only launch_configs.py, so it reads the path from there.
        self.assertEqual(
            prod_config_spec.PROD_CONFIG_ARTIFACT_CONTAINER_PATH,
            launch_configs.PROD_CONFIG_ARTIFACT_CONTAINER_PATH,
        )

    def test_prod_config_asks_for_the_baked_artifact(self) -> None:
        resolved = launch_configs.get("rankfm-prod").resolve(hosts=16, job_name="JOB")
        self.assertIs(launch_configs.RANKFM_PROD_CONFIG, resolved.prod_config)

    def test_prod_config_mounts_the_dump_dump_py_names(self) -> None:
        resolved = launch_configs.get("rankfm-prod").resolve(hosts=16, job_name="JOB")
        self.assertEqual(dump.GCS_ROOT, resolved.barnfs_path)
