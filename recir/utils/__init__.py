# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Small cross-cutting helpers used by other recir packages.

The helpers here are intentionally tiny and focused; anything substantial
gets its own dedicated subpackage.
"""

from .node_kwargs import update_kwargs


__all__ = ["update_kwargs"]
