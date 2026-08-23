# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


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
