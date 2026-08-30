# TBE Sparse Embedding Region Transform

## Scope

`TBESparseEmbeddingTransform` lowers compatible row-wise-sharded
`aten.embedding` lookups to grouped FBGEMM TBE execution. The transform keeps
the local lookup and fused optimizer update inside opaque RecIR TBE operators,
while making the distributed index, embedding, and gradient routing explicit
in the final FX graph.

This change defines the transform and the RegionTransform lifecycle hooks it
requires. A follow-up change wires the transform into the HybridParallel
compile path.

## Why the rewrite has two stages

The transform operates on two different graph levels:

```text
rank-local sharded module
          |
          | prepare_after_shard()
          v
capture local table and optimizer state
          |
          | inner trace of the selected forward root
          v
transform_graph()
  - find compatible aten.embedding sites
  - group local lookups into LocalTBEGroup
  - migrate weights and RowWiseAdagrad state to TBE
          |
          | trace the full training step
          v
recir::tbe_forward + recir::tbe_backward_update
          |
          | finalize_compiled_graph()
          v
insert row-wise AllToAll routing around the opaque TBE operators
          |
          v
final compiled forward/backward graph
```

`transform_graph()` must run before the full training-step trace. That allows
TBE's custom autograd boundary to emit `recir::tbe_forward` and
`recir::tbe_backward_update`, and makes TBE the owner of the embedding optimizer
update.

`finalize_compiled_graph()` runs after the full forward/backward graph exists.
At that point it can connect the forward routing and backward gradient routing
to the matching opaque TBE operators. It also avoids requiring the inner
FakeTensor trace to execute the dynamic split-list materialization used by
AllToAll.

## Lifecycle

### Post-shard preparation

`prepare_after_shard()` inspects the rank-local module and optimizers after the
embedding parameters have been sharded. For every selected table it captures:

- the local two-dimensional weight parameter;
- the table's storage location;
- the optimizer and owning parameter group;
- the RowWiseAdagrad accumulator;
- learning rate, epsilon, and stochastic-rounding settings.

The transform currently requires one optimizer owner per table, exact
RowWiseAdagrad semantics, no weight decay, and evenly sized row-wise shards.
This information cannot be finalized before sharding because the local
parameter object and local optimizer state do not exist yet.

### Forward-root selection

`select_forward_root_fqns()` chooses the deepest called module that contains all
selected embedding parameters. The inner trace therefore covers the smallest
module that can expose every target lookup, rather than tracing unrelated model
code.

## `transform_graph()`: create the local TBE region

The inner graph initially contains one `aten.embedding` call per feature:

```text
local_weight_0 ---- aten.embedding(ids_0) ---- output_0

local_weight_1 ---- aten.embedding(ids_1) ---- output_1
```

The transform resolves each embedding weight back to its parameter FQN, checks
that the ID tensor is one-dimensional, and groups compatible lookup sites.
Compatibility includes device, weight dtype, embedding dimension, storage
location, and optimizer configuration.

Each compatible group becomes one `LocalTBEGroup` call:

```text
ids_0 --\
         +--> LocalTBEGroup --> tuple --> getitem(0) --> output_0
ids_1 --/                         |
                                  +----> getitem(1) --> output_1
```

The local group owns one TBE table per feature. During construction the
transform:

1. copies each rank-local embedding weight into TBE storage;
2. copies the RowWiseAdagrad accumulator into TBE optimizer state;
3. removes the original embedding parameter from its optimizer group and
   optimizer state;
4. replaces the module parameter with a non-gradient view of the TBE weight.

After this migration, the native fused TBE backward updates both the weight and
its optimizer state. The original dense embedding backward and Python optimizer
step are no longer responsible for those tables.

When the full training step is traced, `LocalTBEGroup` is represented by opaque
RecIR operators:

```text
forward:  recir::tbe_forward
backward: recir::tbe_backward_update
```

## `finalize_compiled_graph()`: add distributed routing

For a single rank, no communication rewrite is needed. For multiple ranks,
finalization finds each recorded group's `recir::tbe_forward` and matching
`recir::tbe_backward_update`, then inserts the row-wise communication around
them.

### Index routing before TBE

The caller starts with global IDs in feature-major order. Each ID is mapped to
its owning rank and rank-local row, then stable-sorted by `(owner rank,
feature)`.

```text
global feature IDs
        |
        | owner rank + local row
        v
stable pack by (destination rank, feature)
        |
        | counts AllToAll
        v
input/output split metadata
        |
        | indices AllToAll
        v
received local IDs
        |
        | reorder to feature-major layout
        v
recir::tbe_forward
```

The counts exchange determines how many IDs this rank sends to and receives
from every peer. `_materialize_splits()` performs one batched device-to-host
copy to produce the Python split lists required by
`all_to_all_single_tensor_splits`. The received IDs are then reordered from
source-major communication order to the feature-major layout expected by TBE.

### Embedding return after TBE

TBE produces embeddings for the IDs owned by the local rank. These results are
sent back to the ranks that originated the requests:

```text
feature-major local TBE output
        |
        | undo received feature reorder
        v
source-major embeddings
        |
        | embeddings AllToAll
        v
destination-major returned embeddings
        |
        | inverse request permutation
        v
original caller feature/ID order
```

The embedding dimension converts ID counts into flattened tensor element
counts for the return AllToAll.

### Gradient routing before fused backward

The backward path reverses the output communication. Gradients arrive in the
caller's original feature/ID order, are packed into the same request order,
and are sent to the rank that owns each row:

```text
caller-order embedding gradients
        |
        | request permutation
        v
destination-major gradients
        |
        | reverse AllToAll
        v
source-major gradients
        |
        | received feature permutation
        v
feature-major gradients
        |
        v
recir::tbe_backward_update
```

The forward routing records and reuses its permutations and split metadata, so
the backward rewrite does not need to rediscover the ownership mapping.

## Resulting graph shape

For one grouped TBE region, the final multi-rank graph has the following logical
shape:

```text
                          FORWARD

global IDs
   |
   +--> counts AllToAll --> indices AllToAll --> recir::tbe_forward
                                                   |
                                                   +--> embeddings AllToAll
                                                            |
                                                            v
                                                       model consumers

                          BACKWARD

consumer gradients
   |
   +--> gradients AllToAll --> recir::tbe_backward_update
                                      |
                                      v
                         fused RowWiseAdagrad update
```

The final graph intentionally contains collectives and opaque TBE operators,
but no `embedding_dense_backward` for the transformed tables.

## Current invariants

- Lookups are unpooled `aten.embedding` operations with one-dimensional IDs.
- Tables are evenly row-wise sharded across the transform's one-dimensional
  device mesh.
- A group contains tables with compatible device, dtype, embedding dimension,
  storage location, and optimizer settings.
- The optimizer is RowWiseAdagrad with no weight decay.
- Each table is owned by exactly one optimizer parameter group.
- The global number of rows is supplied separately from the local shard size.

## Code map

- `TBESparseEmbeddingTransform.prepare_after_shard()` captures local table and
  optimizer state.
- `TBESparseEmbeddingTransform.select_forward_root_fqns()` limits the inner
  trace to the relevant module.
- `TBESparseEmbeddingTransform.transform_graph()` creates grouped local TBE
  calls and migrates optimizer ownership.
- `route_row_wise_indices_grouped()` routes global IDs to their row owners.
- `return_row_wise_embeddings_grouped()` returns lookup results to callers.
- `route_row_wise_gradients_grouped()` sends gradients back to row owners.
- `TBESparseEmbeddingTransform.finalize_compiled_graph()` splices those routing
  helpers around the opaque TBE forward/backward operators.
