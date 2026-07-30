#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Launch matched RankFM TPU and B200 runs and overlay their TensorBoards.

The TPU YAML is the experiment source of truth. This script reads its device
count, batch size, model flags, and BarnFS source, then derives a GPU launch
that consumes the same global batch from the same replay files. The two jobs
are launched concurrently, monitored to completion, and copied into one
Manifold TensorBoard logdir under ``tpu/`` and ``gpu/`` sub-runs.

Example:
    python3 recir/examples/rankfm_mvai/compare_tpu_gpu.py

Resume monitoring jobs after an interrupted local process:
    python3 recir/examples/rankfm_mvai/compare_tpu_gpu.py \
        --tpu-jobset rankfm-mvai-128chip-... \
        --gpu-job mvai-user-...
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_FBCODE_ROOT = _HERE.parents[2]
if str(_FBCODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FBCODE_ROOT))

from security.frameworks.python.exec.subprocess import (  # noqa: E402
    TrustedSubprocessWithList,
)


_DEFAULT_TPU_YAML = _HERE / "rankfm_mvai-128chip.yaml"
_DEFAULT_MVAI_CLOUD = _FBCODE_ROOT / "mvai_cloud/.venv/bin/mvai-cloud"
_DEFAULT_TPU_CONTEXT = "mrs-tpu-y317h/us-central1/tpu-prod-training-cluster"
_DEFAULT_GPU_CONTEXT = "gke_mrs-research-colen_us-south1-b_recsys-training"
_DEFAULT_TPU_NAMESPACE = "ns-mrs-core-shared"
_DEFAULT_GPU_NAMESPACE = "default"
_DEFAULT_TPU_TB_BUCKET = "cq-mrs-core-shared_mrs_tpu_y317h"
_DEFAULT_GPU_TB_BUCKET = "mvai_cloud"
_DEFAULT_GPU_DATASET = (
    "gs://mvai_cloud/czheo_test/hammerv3-v0/"
    "dataporter_2:prod:dataporter:xdb.mwfs_orchestration_prod:1516:1812875:v0"
)
_DEFAULT_MANIFOLD_PATH = "mrs_recsys_gcp/tree/tensorboard"
_DEFAULT_TTL_SECONDS = 2_592_000
_CANONICAL_EMBEDDING_LR = 0.02
_CANONICAL_EMBEDDING_EPS = 0.001

_GPU_JOB_RE = re.compile(r"Job launched: (mvai-[A-Za-z0-9-]+)")
_NE_RE = re.compile(
    r"NE/global/window/([A-Za-z0-9_]+)/train at step (\d+): ([0-9.eE+-]+)"
)
_LOSS_RE = re.compile(
    r"loss/global/(window|lifetime)/train at step (\d+): ([0-9.eE+-]+)"
)
_INIT_RE = re.compile(r"RankFM dense initialization fingerprint: ([0-9a-f]+)")
_GCS_URI_RE = re.compile(r"^gs://([A-Za-z0-9._-]+)/([A-Za-z0-9._/:=-]+)/?$")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


@dataclass(frozen=True)
class ExperimentConfig:
    tpu_processes: int
    tpu_chips_per_process: int
    tpu_local_batch_size: int
    gpu_hosts: int
    gpu_processes_per_host: int
    gpu_local_batch_size: int
    max_steps: int
    report_interval: int
    embedding_lr: float
    embedding_eps: float
    num_candidates: int
    hash_size_cap: int
    l1_num_layers: int
    l2_num_layers: int
    embedding_backend: str
    dataset_uri: str

    @property
    def tpu_devices(self) -> int:
        return self.tpu_processes * self.tpu_chips_per_process

    @property
    def gpu_devices(self) -> int:
        return self.gpu_hosts * self.gpu_processes_per_host

    @property
    def global_batch_size(self) -> int:
        return self.tpu_processes * self.tpu_local_batch_size


@dataclass(frozen=True)
class JobStatus:
    done: bool
    succeeded: bool
    detail: str


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    process = TrustedSubprocessWithList.run(
        executable="/usr/bin/env",
        cmd_args=list(args),
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
    )
    output = (process.stdout or "") + (process.stderr or "")
    if check and process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {shlex.join(args)}\n{output}"
        )
    return CommandResult(process.returncode, output)


def _run_streaming(
    args: Sequence[str],
    label: str,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    process = TrustedSubprocessWithList.Popen(
        executable="/usr/bin/env",
        cmd_args=list(args),
        stdout=-1,
        stderr=-2,
        text=True,
        env=dict(env) if env is not None else None,
    )
    lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            lines.append(line)
            sys.stdout.write(f"[{label}] {line}")
            sys.stdout.flush()
    returncode = process.wait()
    output = "".join(lines)
    if returncode != 0:
        raise RuntimeError(
            f"command failed ({returncode}): {shlex.join(args)}\n{output}"
        )
    return output


def _kube_args(
    context: str,
    namespace: str,
    *args: str,
) -> list[str]:
    return ["kubectl", f"--context={context}", "-n", namespace, *args]


def _required_int_flag(yaml_text: str, flag: str) -> int:
    match = re.search(rf"{re.escape(flag)}\s+(\d+)", yaml_text)
    if match is None:
        raise ValueError(f"TPU YAML does not set required flag {flag}")
    return int(match.group(1))


def _required_str_flag(yaml_text: str, flag: str) -> str:
    match = re.search(rf"{re.escape(flag)}\s+([A-Za-z0-9_-]+)", yaml_text)
    if match is None:
        raise ValueError(f"TPU YAML does not set required flag {flag}")
    return match.group(1)


def _required_float_flag(yaml_text: str, flag: str) -> float:
    match = re.search(rf"{re.escape(flag)}\s+([0-9.eE+-]+)", yaml_text)
    if match is None:
        raise ValueError(f"TPU YAML does not set required flag {flag}")
    return float(match.group(1))


def _required_int(yaml_text: str, pattern: str, description: str) -> int:
    match = re.search(pattern, yaml_text)
    if match is None:
        raise ValueError(f"TPU YAML does not define {description}")
    return int(match.group(1))


def _dataset_uri(yaml_text: str) -> str:
    match = re.search(r'sourcePath:\s*"(gs://[^\"]+)/?"', yaml_text)
    if match is None:
        raise ValueError("TPU YAML does not define a BarnFS sourcePath")
    dataset_uri = match.group(1).rstrip("/")
    if _GCS_URI_RE.fullmatch(dataset_uri) is None:
        raise ValueError(f"unsupported BarnFS GCS URI: {dataset_uri}")
    return dataset_uri


def _patch_report_interval(
    yaml_text: str,
    source: Path,
    report_interval: int,
) -> str:
    if report_interval <= 0:
        raise ValueError("--report-interval must be positive")
    if re.search(r"--report_interval\s+\d+", yaml_text) is not None:
        return re.sub(
            r"--report_interval\s+\d+",
            f"--report_interval {report_interval}",
            yaml_text,
            count=1,
        )
    yaml_text, replacements = re.subn(
        r"(?m)^(\s*)(--max_steps\s+\d+)",
        rf"\g<1>\g<2>\n\g<1>--report_interval {report_interval}",
        yaml_text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"expected one --max_steps in {source}")
    return yaml_text


def _ensure_tpu_argument(
    yaml_text: str,
    source: Path,
    flag: str,
    value: str | None = None,
) -> str:
    if re.search(rf"(?m)^\s*{re.escape(flag)}(?:\s|$)", yaml_text) is not None:
        return yaml_text
    argument = flag if value is None else f"{flag} {value}"
    yaml_text, replacements = re.subn(
        r"(?m)^(\s*)(--num_candidates\s+\d+)",
        rf"\g<1>\g<2>\n\g<1>{argument}",
        yaml_text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"expected one --num_candidates in {source}")
    return yaml_text


def _patch_tpu_yaml(
    source: Path,
    max_steps: int,
    report_interval: int,
    dataset_uri: str | None,
    image_tag: str | None,
) -> tuple[Path, str]:
    yaml_text = source.read_text()
    if max_steps > 0:
        yaml_text, replacements = re.subn(
            r"--max_steps\s+\d+",
            f"--max_steps {max_steps}",
            yaml_text,
        )
        if replacements != 1:
            raise ValueError(
                f"expected one --max_steps in {source}, found {replacements}"
            )
    yaml_text = _patch_report_interval(yaml_text, source, report_interval)
    yaml_text = _ensure_tpu_argument(
        yaml_text,
        source,
        "--disable_seq_len_sampling",
    )
    yaml_text = _ensure_tpu_argument(
        yaml_text,
        source,
        "--embedding_lr",
        str(_CANONICAL_EMBEDDING_LR),
    )
    yaml_text = _ensure_tpu_argument(
        yaml_text,
        source,
        "--embedding_eps",
        str(_CANONICAL_EMBEDDING_EPS),
    )
    if dataset_uri is not None:
        normalized = dataset_uri.rstrip("/")
        if _GCS_URI_RE.fullmatch(normalized) is None:
            raise ValueError(f"unsupported BarnFS GCS URI: {dataset_uri}")
        yaml_text, replacements = re.subn(
            r'(sourcePath:\s*")[^\"]+("\s*)',
            rf"\g<1>{normalized}/\g<2>",
            yaml_text,
        )
        if replacements != 1:
            raise ValueError(
                f"expected one BarnFS sourcePath in {source}, found {replacements}"
            )
    if image_tag is not None:
        if re.fullmatch(r"[A-Za-z0-9._-]+", image_tag) is None:
            raise ValueError(f"unsupported TPU image tag: {image_tag}")
        yaml_text, replacements = re.subn(
            r"(rankfm_mvai_tpu):[A-Za-z0-9._-]+",
            rf"\g<1>:{image_tag}",
            yaml_text,
        )
        if replacements != 1:
            raise ValueError(
                f"expected one rankfm_mvai_tpu image in {source}, found {replacements}"
            )

    temp = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="rankfm-tpu-gpu-",
        suffix=".yaml",
        delete=False,
    )
    with temp:
        temp.write(yaml_text)
    return Path(temp.name), yaml_text


def _parse_experiment(
    yaml_text: str,
    gpu_hosts_processes: str,
) -> ExperimentConfig:
    parts = gpu_hosts_processes.split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("--gpu-hosts-processes must use HxP form, for example 16x8")
    gpu_hosts, gpu_processes = (int(part) for part in parts)

    tpu_processes = _required_int(
        yaml_text,
        r"parallelism:\s*(\d+)",
        "worker parallelism",
    )
    tpu_chips = _required_int(
        yaml_text,
        r"google\.com/tpu:\s*(\d+)",
        "TPU chips per worker",
    )
    tpu_local_batch = _required_int_flag(yaml_text, "--local_batch_size")
    tpu_devices = tpu_processes * tpu_chips
    gpu_devices = gpu_hosts * gpu_processes
    if tpu_devices != gpu_devices:
        raise ValueError(
            f"device mismatch: TPU YAML requests {tpu_devices}, GPU launch requests "
            f"{gpu_devices} ({gpu_hosts_processes})"
        )
    global_batch = tpu_processes * tpu_local_batch
    if global_batch % gpu_devices != 0:
        raise ValueError(
            f"TPU global batch {global_batch} is not divisible by {gpu_devices} GPUs"
        )
    gpu_local_batch = global_batch // gpu_devices
    if tpu_local_batch % gpu_local_batch != 0:
        raise ValueError(
            "GPU local batch must evenly partition each TPU replay shard: "
            f"{tpu_local_batch} % {gpu_local_batch} != 0"
        )
    max_steps = _required_int_flag(yaml_text, "--max_steps")
    report_interval = _required_int_flag(yaml_text, "--report_interval")
    if report_interval > max_steps or max_steps % report_interval != 0:
        raise ValueError(
            "--report-interval must not exceed --max-steps and must divide it "
            f"evenly: steps={max_steps}, interval={report_interval}"
        )

    return ExperimentConfig(
        tpu_processes=tpu_processes,
        tpu_chips_per_process=tpu_chips,
        tpu_local_batch_size=tpu_local_batch,
        gpu_hosts=gpu_hosts,
        gpu_processes_per_host=gpu_processes,
        gpu_local_batch_size=gpu_local_batch,
        max_steps=max_steps,
        report_interval=report_interval,
        embedding_lr=_required_float_flag(yaml_text, "--embedding_lr"),
        embedding_eps=_required_float_flag(yaml_text, "--embedding_eps"),
        num_candidates=_required_int_flag(yaml_text, "--num_candidates"),
        hash_size_cap=_required_int_flag(yaml_text, "--hash_size_cap"),
        l1_num_layers=_required_int_flag(yaml_text, "--l1_num_layers"),
        l2_num_layers=_required_int_flag(yaml_text, "--l2_num_layers"),
        embedding_backend=_required_str_flag(yaml_text, "--embedding_backend"),
        dataset_uri=_dataset_uri(yaml_text),
    )


def _gpu_run_args(config: ExperimentConfig) -> list[str]:
    return [
        "--max_steps",
        str(config.max_steps),
        "--report_interval",
        str(config.report_interval),
        "--embedding_lr",
        str(config.embedding_lr),
        "--embedding_eps",
        str(config.embedding_eps),
        "--local-batch-size",
        str(config.gpu_local_batch_size),
        "--num_candidates",
        str(config.num_candidates),
        "--disable_seq_len_sampling",
        "--hash_size_cap",
        str(config.hash_size_cap),
        "--l1_num_layers",
        str(config.l1_num_layers),
        "--l2_num_layers",
        str(config.l2_num_layers),
        "--embedding_backend",
        config.embedding_backend,
        "--data-path",
        "/mnt/barnfs",
        "--data-file-template",
        "rank_{rank}/batch_0/{i}.pt.zst",
        "--batch-index",
        "1",
        "--expected-world-size",
        str(config.gpu_devices),
        "--tpu-data-rank-count",
        str(config.tpu_processes),
        "--tpu-local-batch-size",
        str(config.tpu_local_batch_size),
    ]


def _launch_tpu(
    yaml_path: Path,
    context: str,
    namespace: str,
) -> str:
    result = _run(
        _kube_args(
            context,
            namespace,
            "create",
            "-f",
            str(yaml_path),
            "-o",
            "jsonpath={.metadata.name}",
        )
    )
    jobset = result.output.strip()
    if not jobset:
        raise RuntimeError(f"could not parse TPU JobSet name from: {result.output}")
    print(f"[tpu-launch] JobSet launched: {jobset}", flush=True)
    return jobset


def _launch_gpu(
    mvai_cloud: Path,
    config: ExperimentConfig,
    gpu_dataset_uri: str,
    block_size: int,
) -> str:
    command = _gpu_launch_command(
        mvai_cloud,
        config,
        gpu_dataset_uri,
        block_size,
    )
    output = _run_streaming(command, "gpu-launch")
    match = _GPU_JOB_RE.search(output)
    if match is None:
        raise RuntimeError(
            f"could not parse GPU job name from mvai-cloud output:\n{output}"
        )
    return match.group(1)


def _gpu_launch_command(
    mvai_cloud: Path,
    config: ExperimentConfig,
    gpu_dataset_uri: str,
    block_size: int,
) -> list[str]:
    dataset_match = _GCS_URI_RE.fullmatch(gpu_dataset_uri)
    if dataset_match is None:
        raise ValueError(f"unsupported BarnFS GCS URI: {gpu_dataset_uri}")
    bucket, path = dataset_match.groups()
    command = [
        str(mvai_cloud),
        "run",
        "-j",
        f"{config.gpu_hosts}x{config.gpu_processes_per_host}",
        "--run-module",
        "rankfm_mvai_shared.train_on_gpu",
    ]
    if bucket != "mvai_cloud":
        command.extend(["--gcs-bucket", bucket])
    command.extend(
        [
            "--barnfs-path",
            f"/mnt/gcs/{bucket}/{path.rstrip('/')}",
            "--barnfs-block-size",
            str(block_size),
            "--run-args",
            shlex.join(_gpu_run_args(config)),
        ]
    )
    return command


def _manifest_fingerprint(dataset_uri: str) -> tuple[int, str]:
    if _GCS_URI_RE.fullmatch(dataset_uri) is None:
        raise ValueError(f"unsupported BarnFS GCS URI: {dataset_uri}")
    result = _run(
        [
            "gcloud",
            "storage",
            "objects",
            "describe",
            f"{dataset_uri.rstrip('/')}/manifest.json",
            "--format=json",
        ]
    )
    metadata = json.loads(result.output)
    return int(metadata["size"]), str(metadata.get("md5_hash", metadata.get("md5Hash")))


def _validate_dataset_mirror(tpu_dataset_uri: str, gpu_dataset_uri: str) -> None:
    tpu_fingerprint = _manifest_fingerprint(tpu_dataset_uri)
    gpu_fingerprint = _manifest_fingerprint(gpu_dataset_uri)
    if tpu_fingerprint != gpu_fingerprint:
        raise ValueError(
            "TPU and GPU BarnFS manifests do not match: "
            f"TPU={tpu_fingerprint}, GPU={gpu_fingerprint}"
        )
    print(
        f"Dataset manifests match: size={tpu_fingerprint[0]} md5={tpu_fingerprint[1]}",
        flush=True,
    )


def _get_json(args: Sequence[str]) -> dict[str, Any] | None:
    result = _run(args, check=False)
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.output)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _tpu_status(
    jobset: str,
    context: str,
    namespace: str,
) -> JobStatus:
    resource = _get_json(
        _kube_args(context, namespace, "get", "jobset", jobset, "-o", "json")
    )
    if resource is None:
        return JobStatus(False, False, "not found")
    status = resource.get("status", {})
    terminal = str(status.get("terminalState", ""))
    if terminal:
        succeeded = terminal.lower() in {"complete", "completed", "succeeded"}
        return JobStatus(True, succeeded, terminal)
    conditions = status.get("conditions", [])
    for condition in conditions if isinstance(conditions, list) else []:
        if condition.get("status") != "True":
            continue
        condition_type = str(condition.get("type", ""))
        if condition_type in {"Completed", "Complete", "Failed"}:
            return JobStatus(
                True,
                condition_type in {"Completed", "Complete"},
                condition_type,
            )
    return JobStatus(False, False, "running")


def _gpu_status(
    job: str,
    context: str,
    namespace: str,
) -> JobStatus:
    resource = _get_json(
        _kube_args(context, namespace, "get", "job", job, "-o", "json")
    )
    if resource is None:
        return JobStatus(False, False, "not found")
    spec = resource.get("spec", {})
    status = resource.get("status", {})
    completions = int(spec.get("completions", 1))
    succeeded = int(status.get("succeeded", 0))
    if succeeded >= completions:
        return JobStatus(True, True, f"Complete {succeeded}/{completions}")
    conditions = status.get("conditions", [])
    for condition in conditions if isinstance(conditions, list) else []:
        if condition.get("status") == "True" and condition.get("type") == "Failed":
            return JobStatus(True, False, str(condition.get("reason", "Failed")))
    active = int(status.get("active", 0))
    failed = int(status.get("failed", 0))
    return JobStatus(
        False, False, f"active={active} succeeded={succeeded} failed={failed}"
    )


def _wait_for_jobs(
    tpu_jobset: str,
    gpu_job: str,
    args: argparse.Namespace,
) -> tuple[JobStatus, JobStatus]:
    deadline = time.monotonic() + args.wait_minutes * 60
    while True:
        tpu = _tpu_status(tpu_jobset, args.tpu_context, args.tpu_namespace)
        gpu = _gpu_status(gpu_job, args.gpu_context, args.gpu_namespace)
        print(f"[status] TPU={tpu.detail}; GPU={gpu.detail}", flush=True)
        if tpu.done and gpu.done:
            return tpu, gpu
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"jobs did not both finish within {args.wait_minutes:g} minutes"
            )
        time.sleep(args.poll_seconds)


def _rank_zero_logs(
    *,
    context: str,
    namespace: str,
    selector: str,
    container: str,
    tpu: bool,
) -> str:
    pods = _get_json(
        _kube_args(
            context,
            namespace,
            "get",
            "pods",
            "-l",
            selector,
            "-o",
            "json",
        )
    )
    if pods is None:
        return ""
    items = pods.get("items", [])
    if not isinstance(items, list):
        return ""
    pod_name = ""
    for pod in items:
        metadata = pod.get("metadata", {})
        name = str(metadata.get("name", ""))
        labels = metadata.get("labels", {})
        index = str(labels.get("batch.kubernetes.io/job-completion-index", ""))
        if (tpu and "worker-0-0" in name) or (not tpu and index == "0"):
            pod_name = name
            break
    if not pod_name:
        return ""
    return _run(
        _kube_args(context, namespace, "logs", pod_name, "-c", container),
        check=False,
    ).output


def _latest_metrics(
    log_text: str,
) -> tuple[
    dict[str, tuple[int, float]],
    dict[str, tuple[int, float]],
]:
    ne: dict[str, tuple[int, float]] = {}
    for task, step_text, value_text in _NE_RE.findall(log_text):
        step, value = int(step_text), float(value_text)
        if task not in ne or step > ne[task][0]:
            ne[task] = (step, value)
    losses: dict[str, tuple[int, float]] = {}
    for kind, step_text, value_text in _LOSS_RE.findall(log_text):
        step, value = int(step_text), float(value_text)
        if kind not in losses or step > losses[kind][0]:
            losses[kind] = (step, value)
    return ne, losses


def _comparison(tpu_log: str, gpu_log: str) -> str:
    tpu_ne, tpu_loss = _latest_metrics(tpu_log)
    gpu_ne, gpu_loss = _latest_metrics(gpu_log)
    common_tasks = sorted(set(tpu_ne) & set(gpu_ne))
    lines = ["metric                         TPU          GPU    GPU-TPU"]
    lines.append("-" * len(lines[0]))
    tpu_init = _INIT_RE.findall(tpu_log)
    gpu_init = _INIT_RE.findall(gpu_log)
    if tpu_init and gpu_init:
        init_state = "MATCH" if tpu_init[-1] == gpu_init[-1] else "MISMATCH"
        lines.append(
            f"dense initialization       {tpu_init[-1]:>16} {gpu_init[-1]:>16} "
            f"{init_state:>10}"
        )
    for kind in ("window", "lifetime"):
        if kind not in tpu_loss or kind not in gpu_loss:
            continue
        tpu_value, gpu_value = tpu_loss[kind][1], gpu_loss[kind][1]
        lines.append(
            f"loss/{kind:<17} {tpu_value:>12.6f} {gpu_value:>12.6f} "
            f"{gpu_value - tpu_value:>+10.6f}"
        )
    for task in common_tasks:
        tpu_value, gpu_value = tpu_ne[task][1], gpu_ne[task][1]
        lines.append(
            f"NE/{task:<20} {tpu_value:>12.6f} {gpu_value:>12.6f} "
            f"{gpu_value - tpu_value:>+10.6f}"
        )
    lines.append(
        f"common NE tasks: {len(common_tasks)}; "
        f"TPU tasks: {len(tpu_ne)}; GPU tasks: {len(gpu_ne)}"
    )
    return "\n".join(lines)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _copy_tensorboard(
    source: str,
    destination: Path,
    retries: int,
    delay_seconds: int,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        _run(
            [
                "gcloud",
                "storage",
                "cp",
                "-r",
                f"{source.rstrip('/')}/*",
                f"{destination}/",
            ],
            check=False,
        )
        size = _directory_bytes(destination)
        if size > 200:
            return size
        if attempt < retries - 1:
            print(
                f"[tensorboard] {source}: only {size} bytes; retrying in "
                f"{delay_seconds}s",
                flush=True,
            )
            time.sleep(delay_seconds)
    return _directory_bytes(destination)


def _upload_tensorboard(
    experiment_dir: Path,
    destination: str,
    ttl_seconds: int,
) -> str:
    _run(
        [
            "manifold",
            "putr",
            "--userData=true",
            f"--ttl={ttl_seconds}",
            str(experiment_dir),
            destination,
        ]
    )
    return (
        f"https://www.internalfb.com/intern/tensorboard/?dir=manifold://{destination}"
    )


def _cleanup(
    tpu_jobset: str,
    gpu_job: str,
    args: argparse.Namespace,
) -> None:
    _run(
        _kube_args(
            args.tpu_context,
            args.tpu_namespace,
            "delete",
            "jobset",
            tpu_jobset,
        ),
        check=False,
    )
    _run(
        _kube_args(
            args.gpu_context,
            args.gpu_namespace,
            "delete",
            "job",
            gpu_job,
        ),
        check=False,
    )
    _run(
        _kube_args(
            args.gpu_context,
            args.gpu_namespace,
            "delete",
            "service",
            gpu_job,
        ),
        check=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpu-yaml", type=Path, default=_DEFAULT_TPU_YAML)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="override the TPU YAML and use the same step count on GPU",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=10,
        help="report matched TPU/GPU metrics every N steps",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="optional TPU gs:// BarnFS URI override applied to the TPU YAML",
    )
    parser.add_argument(
        "--tpu-image-tag",
        default=None,
        help="optional rankfm_mvai_tpu image tag applied to the TPU YAML",
    )
    parser.add_argument(
        "--gpu-dataset",
        default=_DEFAULT_GPU_DATASET,
        help="GPU-accessible mirror of the TPU BarnFS dataset",
    )
    parser.add_argument("--gpu-hosts-processes", default="16x8")
    parser.add_argument("--barnfs-block-size", type=int, default=2_097_152)
    parser.add_argument("--mvai-cloud", type=Path, default=_DEFAULT_MVAI_CLOUD)
    parser.add_argument("--tpu-context", default=_DEFAULT_TPU_CONTEXT)
    parser.add_argument("--gpu-context", default=_DEFAULT_GPU_CONTEXT)
    parser.add_argument("--tpu-namespace", default=_DEFAULT_TPU_NAMESPACE)
    parser.add_argument("--gpu-namespace", default=_DEFAULT_GPU_NAMESPACE)
    parser.add_argument("--tpu-jobset", default=None)
    parser.add_argument("--gpu-job", default=None)
    parser.add_argument("--wait-minutes", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/rankfm-tpu-gpu"))
    parser.add_argument("--tpu-tb-bucket", default=_DEFAULT_TPU_TB_BUCKET)
    parser.add_argument("--gpu-tb-bucket", default=_DEFAULT_GPU_TB_BUCKET)
    parser.add_argument("--tb-manifold-path", default=_DEFAULT_MANIFOLD_PATH)
    parser.add_argument("--tb-name", default=None)
    parser.add_argument("--tb-ttl", type=int, default=_DEFAULT_TTL_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.tpu_yaml.is_file():
        raise FileNotFoundError(args.tpu_yaml)
    if not args.mvai_cloud.is_file() and args.gpu_job is None:
        raise FileNotFoundError(args.mvai_cloud)
    if (args.tpu_jobset is None) != (args.gpu_job is None):
        raise ValueError("provide both --tpu-jobset and --gpu-job, or neither")

    patched_yaml, yaml_text = _patch_tpu_yaml(
        args.tpu_yaml,
        args.max_steps,
        args.report_interval,
        args.dataset,
        args.tpu_image_tag,
    )
    config = _parse_experiment(yaml_text, args.gpu_hosts_processes)
    _validate_dataset_mirror(config.dataset_uri, args.gpu_dataset)
    print(
        "Matched experiment: "
        f"TPU={config.tpu_devices} ({config.tpu_processes}x"
        f"{config.tpu_chips_per_process}), GPU={config.gpu_devices} "
        f"({config.gpu_hosts}x{config.gpu_processes_per_host}), "
        f"global_batch={config.global_batch_size}, steps={config.max_steps}, "
        f"report_interval={config.report_interval}, "
        "embedding_optimizer=ROWWISE_ADAGRAD, "
        f"embedding_lr={config.embedding_lr}, embedding_eps={config.embedding_eps}, "
        "sequence_sampling=disabled, "
        f"tpu_dataset={config.dataset_uri}, gpu_dataset={args.gpu_dataset}",
        flush=True,
    )
    if args.dry_run:
        print(f"TPU manifest: {patched_yaml}")
        print(
            "GPU command: "
            + shlex.join(
                _gpu_launch_command(
                    args.mvai_cloud,
                    config,
                    args.gpu_dataset,
                    args.barnfs_block_size,
                )
            )
        )
        patched_yaml.unlink(missing_ok=True)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.tpu_jobset is None:
            with ThreadPoolExecutor(max_workers=2) as executor:
                tpu_future = executor.submit(
                    _launch_tpu,
                    patched_yaml,
                    args.tpu_context,
                    args.tpu_namespace,
                )
                gpu_future = executor.submit(
                    _launch_gpu,
                    args.mvai_cloud,
                    config,
                    args.gpu_dataset,
                    args.barnfs_block_size,
                )
                tpu_jobset = tpu_future.result()
                gpu_job = gpu_future.result()
        else:
            tpu_jobset = args.tpu_jobset
            gpu_job = args.gpu_job
            print(f"Resuming TPU={tpu_jobset}; GPU={gpu_job}", flush=True)

        state_path = args.out_dir / "jobs.json"
        state_path.write_text(
            json.dumps(
                {
                    "tpu_jobset": tpu_jobset,
                    "gpu_job": gpu_job,
                    "tpu_dataset": config.dataset_uri,
                    "gpu_dataset": args.gpu_dataset,
                    "devices": config.tpu_devices,
                    "global_batch_size": config.global_batch_size,
                    "max_steps": config.max_steps,
                    "report_interval": config.report_interval,
                    "embedding_optimizer": "ROWWISE_ADAGRAD",
                    "embedding_lr": config.embedding_lr,
                    "embedding_eps": config.embedding_eps,
                    "sequence_sampling": "disabled",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved job state to {state_path}", flush=True)

        try:
            tpu_status, gpu_status = _wait_for_jobs(tpu_jobset, gpu_job, args)
        except KeyboardInterrupt:
            print(
                "\nMonitoring interrupted; jobs were left intact. Resume with:\n"
                f"  {Path(__file__).name} --tpu-jobset {tpu_jobset} "
                f"--gpu-job {gpu_job}",
                file=sys.stderr,
            )
            return 130
        tpu_log = _rank_zero_logs(
            context=args.tpu_context,
            namespace=args.tpu_namespace,
            selector=f"jobset.sigs.k8s.io/jobset-name={tpu_jobset}",
            container="rankfm-mvai",
            tpu=True,
        )
        gpu_log = _rank_zero_logs(
            context=args.gpu_context,
            namespace=args.gpu_namespace,
            selector=f"job-name={gpu_job}",
            container="main",
            tpu=False,
        )
        (args.out_dir / "tpu.log").write_text(tpu_log)
        (args.out_dir / "gpu.log").write_text(gpu_log)
        report = _comparison(tpu_log, gpu_log)
        (args.out_dir / "comparison.txt").write_text(report + "\n")
        print(f"\n{report}\n", flush=True)

        if not tpu_status.succeeded or not gpu_status.succeeded:
            raise RuntimeError(
                f"comparison jobs failed: TPU={tpu_status.detail}; GPU={gpu_status.detail}"
            )

        experiment_name = args.tb_name or f"rankfm-tpu-gpu-{tpu_jobset}"
        experiment_dir = args.out_dir / experiment_name
        tpu_bytes = _copy_tensorboard(
            f"gs://{args.tpu_tb_bucket}/tensorboard/{tpu_jobset}",
            experiment_dir / "tpu",
            retries=10,
            delay_seconds=30,
        )
        gpu_bytes = _copy_tensorboard(
            f"gs://{args.gpu_tb_bucket}/tensorboard/{gpu_job}",
            experiment_dir / "gpu",
            retries=10,
            delay_seconds=30,
        )
        if tpu_bytes <= 200 or gpu_bytes <= 200:
            raise RuntimeError(
                "TensorBoard export is incomplete: "
                f"TPU={tpu_bytes} bytes, GPU={gpu_bytes} bytes"
            )
        destination = f"{args.tb_manifold_path}/{experiment_name}"
        link = _upload_tensorboard(experiment_dir, destination, args.tb_ttl)
        (args.out_dir / "tensorboard_url.txt").write_text(link + "\n")
        print(f"TENSORBOARD (TPU and GPU overlaid):\n{link}", flush=True)

        if args.cleanup:
            _cleanup(tpu_jobset, gpu_job, args)
        return 0
    finally:
        patched_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
