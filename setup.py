# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import os

from setuptools import find_packages, setup

_XLA_REQUIREMENTS = [
    "jax>=0.8.0,<0.11",
    "jaxlib>=0.8.0,<0.11",
    "flax>=0.10.5",
    "optax>=0.2.5",
]

_DLRM_COMMON_REQUIREMENTS = [
    "generative-recommenders",
    "gin-config",
    "pandas",
    "pybind11",
    "tensorboard",
    "torch==2.10.*",
    "torchrec==1.4.*",
    "triton",
]

_DLRM_CPU_REQUIREMENTS = _DLRM_COMMON_REQUIREMENTS + [
    "fbgemm-gpu-cpu==1.5.*",
    *_XLA_REQUIREMENTS,
]

_DLRM_GPU_REQUIREMENTS = _DLRM_COMMON_REQUIREMENTS + [
    "fbgemm-gpu==1.5.*",
]

_DLRM_TPU_REQUIREMENTS = _DLRM_CPU_REQUIREMENTS + ["libtpu"]

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
    packages=find_packages(
        include=["recir", "recir.*"],
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests",
            "tests.*",
            "*.test",
            "*.test.*",
            "test",
            "test.*",
            "*.benchmarks",
            "*.benchmarks.*",
            "recir.fb",
            "recir.fb.*",
        ],
    ),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.4",
    ],
    extras_require={
        "xla": [
            *_XLA_REQUIREMENTS,
        ],
        "tpu": [
            *_XLA_REQUIREMENTS,
            "libtpu",
        ],
        "rec": [
            "torchrec",
        ],
        "test": [
            "expecttest>=0.3.0",
            "parameterized>=0.9.0",
        ],
        "dlrm-cpu": _DLRM_CPU_REQUIREMENTS,
        "dlrm-gpu": _DLRM_GPU_REQUIREMENTS,
        "dlrm-tpu": _DLRM_TPU_REQUIREMENTS,
        "all": [
            *_DLRM_TPU_REQUIREMENTS,
            "expecttest>=0.3.0",
            "parameterized>=0.9.0",
        ],
    },
)
