# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3

from typing import Dict, List

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import get_embedding_names_by_table


class TracableEmbeddingCollection(nn.Module):
    """EmbeddingCollection variant whose forward takes Dict[str, Tensor] (mapping
    feature name -> index values) instead of KeyedJaggedTensor, avoiding the
    non-fx-traceable KeyedJaggedTensor.to_dict() call.

    The __init__ and forward logic mirrors EmbeddingCollection as closely as
    possible; only the input/output types differ.
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device | None = None,
        need_indices: bool = False,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        self._need_indices: bool = need_indices
        for config in tables:
            self.embeddings[config.name] = nn.Embedding(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.embedding_dim,
                device=device,
            )
            if not config.feature_names:
                config.feature_names = [config.name]

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(
            tables
        )
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]

    def forward(
        self,
        jt_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        feature_embeddings: Dict[str, torch.Tensor] = {}
        for i, emb_module in enumerate(self.embeddings.values()):
            feature_names = self._feature_names[i]
            embedding_names = self._embedding_names_by_table[i]
            for j, embedding_name in enumerate(embedding_names):
                feature_name = feature_names[j]
                f = jt_dict[feature_name]
                lookup = emb_module(
                    input=f,
                ).float()
                feature_embeddings[embedding_name] = lookup
        return feature_embeddings
