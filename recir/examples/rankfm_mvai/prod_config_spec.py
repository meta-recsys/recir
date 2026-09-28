# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Where the exported prod config artifact lives, in the build and in the image.

Three things need to agree on this and they cannot import each other:
``scripts/build_image_podman.sh`` (bash, stages the artifact), ``mvai-cloud``
(the launcher venv, which has only jinja2) and the trainer inside the image.
So this module is stdlib-only and dependency-free, and is the single place
any of them reads the path from.

Modelled on
``mvai_cloud/minimal_viable_ai/models/udd_fm_v3/prod_config_spec.py``, which
does the same job for the B200 image. There is no analogue of its
``PROD_CONFIG_REPO_PATH`` / ``PROD_CONFIG_SOURCE_OVERRIDES`` here: that
exporter rewrites a provider *source file* before importing it, while
``prod_config_artifact.export_prod_config`` builds from the live buck-resolved
``ProdXSurfaceConfigProvider``.
"""

# Staged into the Docker build context under this directory name, and COPYed
# to the container path below. The directory is created even when the export
# is skipped, because COPY cannot be made conditional -- the same reason
# ``barnfs_wheel/`` is always staged.
PROD_CONFIG_STAGE_DIR: str = "prod_config"
PROD_CONFIG_ARTIFACT_FILENAME: str = "prod_cfg.pt"
PROD_CONFIG_ARTIFACT_CONTAINER_PATH: str = (
    f"/app/{PROD_CONFIG_STAGE_DIR}/{PROD_CONFIG_ARTIFACT_FILENAME}"
)

# Prod's xsurface config hardcodes HSTU_MAX_SEQ_LEN = 3072 + 512 - 10 - 20
# (model_v5_hstu_standalone_cint_datafm_fm_xsurface_config.py:111). Any other
# value means the export picked up a parity or replay shim, i.e. the artifact
# describes a different model than production. Nothing at runtime checks this,
# so the image build asserts it instead.
PROD_CONFIG_EXPECTED_SEQ_LEN: int = 3554
