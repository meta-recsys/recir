# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""TPU smoke run: the CoScaling recir smoke (``run.py``) on the pod's TPU
devices, with the config and batch from the prod config artifact the image
build baked in (written on the devserver by ``export_artifact``).

    python3 -u -m recir.examples.coscaling_mvai.smoke.tpu_smoke

Single process (one pod): jax.distributed.initialize() is NOT called -- on a
lone pod it mis-initializes the runtime (see rankfm_mvai/train.py main()), and
jax finds the host's local chips without it.

Knobs: PROD_CONFIG_ARTIFACT (required; the launcher sets it), COSCALING_STEPS (5).
"""

import os

# Before any jax import this module controls.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# The env var alone is not enough: the par bootstrap imports recir (and so
# jax) before this module runs, and jax reads it at import time.
import jax  # noqa: E402  # @manual

jax.config.update("jax_enable_x64", True)

from recir.examples.coscaling_mvai.smoke.artifact import load_artifact  # usort: skip
from recir.examples.coscaling_mvai.smoke.run import run_smoke
from recir.training import tpu_runtime


def main() -> None:
    tpu_runtime.setup_logging()
    artifact = load_artifact(os.environ["PROD_CONFIG_ARTIFACT"])
    run_smoke(
        artifact.module_config,
        artifact.feature_config,
        artifact.optimizer_config,
        artifact.batch,
        num_steps=int(os.environ.get("COSCALING_STEPS", "5")),
    )


if __name__ == "__main__":
    main()
