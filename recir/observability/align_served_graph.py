# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Trace served RecIR graph nodes back to training nodes.

  align_served_graph PUBLISH_GRAPH_JSON SERVED_GRAPH_JSON [SERVED_NODE ...]

PUBLISH_GRAPH_JSON is ``publish_graph.json`` from ``RECIR_GRAPH_ARTIFACT_DIR``;
SERVED_GRAPH_JSON is the served graph of the same model as a JSON list of node
records, each with ``name``, ``op`` and ``target``.
"""

import collections
import json
import sys

from recir.observability.graph_provenance import (
    align_served_nodes,
    INSERTED_BY_KEY,
    TRAIN_NODE_KEY,
)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    with open(argv[0]) as f:
        publish_records = json.load(f)["<root>"]
    with open(argv[1]) as f:
        served_records = json.load(f)
    mapping = align_served_nodes(publish_records, served_records)
    print(
        f"served nodes={len(served_records)} publish nodes={len(publish_records)} "
        f"aligned={len(mapping)}"
    )
    origins = collections.Counter(
        record.get(INSERTED_BY_KEY) or "training" for record in mapping.values()
    )
    print("aligned served nodes by origin:", dict(origins.most_common()))
    for name in argv[2:]:
        record = mapping.get(name)
        if record is None:
            print(f"\n{name}: not aligned (added on the serving side)")
            continue
        print(f"\n{name} -> {record['name']} ({record['target']})")
        print(f"  training node: {record.get(TRAIN_NODE_KEY)}")
        print(f"  inserted by:   {record.get(INSERTED_BY_KEY)}")
        print(f"  module:        {record.get('module_stack')}")
        print(f"  value:         {record.get('val')}")
        for line in record.get("stack") or []:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
