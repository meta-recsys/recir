# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from recir.ops.pallas.embeddings.ringshift import (
    lengths_to_segment_ids,
    pool_per_rank,
    ringshift_embedding_lookup,
    ringshift_embedding_lookup_block,
    ringshift_embedding_lookup_block_backward_local,
    ringshift_embedding_lookup_block_local,
    ringshift_embedding_lookup_reference,
    ringshift_embedding_lookup_reference_jax,
)

__all__ = [
    "lengths_to_segment_ids",
    "pool_per_rank",
    "ringshift_embedding_lookup",
    "ringshift_embedding_lookup_block",
    "ringshift_embedding_lookup_block_backward_local",
    "ringshift_embedding_lookup_block_local",
    "ringshift_embedding_lookup_reference",
    "ringshift_embedding_lookup_reference_jax",
]
