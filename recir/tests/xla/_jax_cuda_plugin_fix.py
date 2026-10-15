# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Workaround for JAX cuda12 plugin path lookup on Buck RE workers.

After the 2026-05-14 buckification of `jax-cuda12-pjrt==0.8.0`, the `xla_cuda_plugin.so` PJRT plugin shared
library is shipped as a `pypi_source_library` resource colocated with the
SYMLINK of `jax_plugins/xla_cuda12/__init__.py` in the buck link-tree.

However, the JAX plugin's stock `_get_library_path()` uses
`pathlib.Path(__file__).resolve().parent / 'xla_cuda_plugin.so'`. The
`.resolve()` follows the link-tree symlink to the real source location
under `third-party/pypi/jax-cuda12-pjrt/0.8.0/wheels/_whl_common_files/...`,
where the `.so` does NOT live (the `.so` is in the per-arch wheel directory
`jax_cuda12_pjrt-0.8.0-py3-none-manylinux_2_27_*_whl/...`).  Result:
`_get_library_path()` returns None, `initialize()` silently early-returns,
and the JAX `cuda` backend is never registered.

This module monkey-patches `_get_library_path` to use `os.path.dirname`
(which does NOT dereference the symlink), so the .so is found via the
link-tree symlink that Buck created for us.

Import this module BEFORE any `import jax` so the patch is in effect when
JAX calls `discover_pjrt_plugins()` (which imports `jax_plugins.xla_cuda12`
and calls its `initialize()`).
"""

import logging
import os
import pathlib

logger: logging.Logger = logging.getLogger(__name__)


def _install_jax_cuda_plugin_path_fix() -> None:
    try:
        import jax_plugins.xla_cuda12 as _p  # @manual  # pyre-ignore[21]
    except ImportError:
        # JAX cuda plugin not available in this binary — nothing to patch.
        return

    link_tree_dir = os.path.dirname(_p.__file__)

    def _patched_get_library_path() -> pathlib.Path | None:
        # Stay in the link-tree directory (do NOT .resolve() __file__): the
        # .so is colocated with the symlinked __init__.py, not with its
        # resolved source target.
        candidate = pathlib.Path(link_tree_dir) / "xla_cuda_plugin.so"
        if candidate.exists():
            return candidate
        # Fall back to the upstream implementation (covers wheel-install paths).
        return _orig_get_library_path()

    _orig_get_library_path = _p._get_library_path
    _p._get_library_path = _patched_get_library_path
    logger.debug(
        "Patched jax_plugins.xla_cuda12._get_library_path to use link-tree dir %s",
        link_tree_dir,
    )


_install_jax_cuda_plugin_path_fix()
