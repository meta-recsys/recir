# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import setup

_readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
long_description = open(_readme).read() if os.path.exists(_readme) else ""

setup(
    name="recir",
    version="0.1.0",
    description="A unified intermediate representation for recommendation systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    url="https://github.com/meta-recsys/recir",
    packages=[
        "recir",
        "recir.distributed",
        "recir.lowering",
        "recir.passes",
        "recir.torchax",
        "recir.torchax.ops",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4",
    ],
    extras_require={
        "xla": [
            "jax>=0.8.0",
            "jaxlib>=0.8.0",
            "flax>=0.10.5",
            "optax>=0.2.5",
        ],
        "tpu": [
            "jax>=0.8.0",
            "jaxlib>=0.8.0",
            "flax>=0.10.5",
            "optax>=0.2.5",
            "libtpu",
        ],
        "rec": [
            "torchrec",
        ],
        "test": [
            "expecttest>=0.3.0",
            "parameterized>=0.9.0",
        ],
        "all": [
            "jax>=0.8.0",
            "jaxlib>=0.8.0",
            "flax>=0.10.5",
            "optax>=0.2.5",
            "libtpu",
            "torchrec",
            "expecttest>=0.3.0",
            "parameterized>=0.9.0",
        ],
    },
)
