# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any

import torch


def _is_in_bad_fork() -> bool:
    return False


def manual_seed_all(seed: int) -> None:
    pass


def device_count() -> int:
    return 1


def get_rng_state() -> list[Any]:
    return []


def set_rng_state(new_state: Any, device: Any) -> None:
    pass


def is_available() -> bool:
    return True


def current_device() -> int:
    return 0


def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.bfloat16]
