# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""CPU smoke run: the CoScaling recir smoke (``run.py``) on JAX CPU, with the
config and batch built live on the devserver (``config.py``).

    buck2 run @fbcode//mode/opt fbcode//recir/examples/coscaling_mvai/smoke:cpu_smoke

Knobs: COSCALING_BATCH_PATH (/tmp/coscaling_roo_batch.pt), COSCALING_STEPS (5),
COSCALING_MAX_ROWS (4096), COSCALING_MAX_SEQ (64; 0 disables truncation and
sizes the encoder's static bound for prod).
"""

import os

# Before any jax import this module controls.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# The env var alone is not enough: the par bootstrap imports recir (and so
# jax) before this module runs, and jax reads it at import time.
import jax  # noqa: E402  # @manual

jax.config.update("jax_enable_x64", True)

from recir.examples.coscaling_mvai.smoke.config import (  # usort: skip
    build_batch,
    build_coscaling_config,
    DEFAULT_BATCH_PATH,
)
from recir.examples.coscaling_mvai.smoke.run import run_smoke
from recir.training import tpu_runtime


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def main() -> None:
    tpu_runtime.setup_logging()
    max_rows = _env_int("COSCALING_MAX_ROWS", 4096)
    max_seq = _env_int("COSCALING_MAX_SEQ", 64)
    max_seq_or_none = max_seq if max_seq > 0 else None
    module_config, feature_config, prod = build_coscaling_config(
        max_rows, max_uih_len=max_seq_or_none
    )
    batch = build_batch(
        prod,
        os.environ.get("COSCALING_BATCH_PATH", DEFAULT_BATCH_PATH),
        max_rows,
        max_seq_or_none,
    )
    run_smoke(
        module_config,
        feature_config,
        prod.optimizer_config,
        batch,
        num_steps=_env_int("COSCALING_STEPS", 5),
    )


if __name__ == "__main__":
    main()
