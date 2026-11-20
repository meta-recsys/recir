# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

from recir.examples.dlrmv3.common.config import DLRMv3TrainConfig
from recir.utils.meta_tensor_init import table_init_bounds


class EmbeddingTablesTest(unittest.TestCase):
    def test_resized_tables_init_for_their_own_row_count(self) -> None:
        tables = DLRMv3TrainConfig(hash_size=1000).embedding_tables()
        bound = math.sqrt(1.0 / 1000)
        for name, config in tables.items():
            self.assertEqual(config.num_embeddings, 1000, name)
            low, high = table_init_bounds(config)
            self.assertAlmostEqual(low, -bound, msg=name)
            self.assertAlmostEqual(high, bound, msg=name)
