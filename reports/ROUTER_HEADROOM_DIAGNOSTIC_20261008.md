# Frozen composition router headroom: search-only feasibility check

## Result and scope

The already exposed router-author search packet leaves **zero observed headroom**
for improving routing over identity: identity and the per-task expert oracle both
score 12/12; each specialist scores 6/12. The experts are complementary, but
identity already captures all complementary successes. Nine exact feature
vectors occur across twelve tasks; the three repeated groups have consistent
successful experts. No identical-vector contradictory target is present.

This is arithmetic over recorded search feedback, **not a new independent tensor
replay**. Such replay failed closed at the frozen-runtime hash check, before any
expert execution. No learned-router superiority, unseen-task generalization,
weights evolution, or population qualification is established.

No author/model call, new task generation, experiment campaign, gradient, dataset
download, evaluation/private access, final panel opening, or closed-run edit was
performed. Repository artifact retrieval used the connected GitHub repository.
No harder synthetic research tasks were made. Small unit fixtures only test the
diagnostic's set/argmax/hash mechanics.

## Exact inputs and exposure

All repository reads were pinned to
`Sanokei/CellForge@02beaf18251fca5ed1f1eb55e92f00a3e3820e88`, except the explicitly
identified historical source lookup below. Root prefix:
`runs/weights-native-composition01-20261006/`.

| Input | Exposure/use | SHA256 of fetched bytes |
|---|---|---|
| `AUTHORING/router_founder.txt` | Previously exposed declared search features and successful-expert feedback; sole source for numeric result | `c373d201d3723a9315b199ad5917999b9bd5678109e293a73542387033f1f074` |
| `AUTHORING/router_founder.json` | Hash-bound author packet receipt | `f91f982c7cd243262980f6b4adb7cbfc80b2d69f3946cefedcdf4442578772f5` |
| `MIXED_SEARCH.json` | Previously declared synthetic search demonstrations and query labels; allowed for replay only, labels never inference inputs | `84258c1931d281977a42a8b3fa291f346d89bde3b47f482fa381d9272ac4dfb8` |
| `HOST_EVALUATION/PROTOCOL.json` | Frozen protocol and inherited column tensor | `0df18cd5740902b6ab1b42dc60389fbc7e341d338b51617e935a65d5e7c8ab9a` |
| `HOST_EVALUATION/ROSTER.json` | Search-selected horizontal tensor, expert order and source binding | `e460cb8941e37ec914a2f1040d5100cec5c97db3142611bd3824925576fbb941` |
| `HOST_EVALUATION/runtime/skillcore.py` | Historical host source, retrieved but not executed due to mismatch | `ec3b064c1022a7eb4e97afa4faeadf313c555c9e722266ecb0868fce5fc0b40a` |

The directory name HOST_EVALUATION does not make its protocol and roster held-out
task evidence. Only these listed files were opened; no sealed/final result was
used. AGENTS, LAB, latest capability checkpoint and relevant host source were read
for governance and feature semantics; their historical summary scores were not
used as new feasibility observations.

Frozen expert IDs, in order:
- Column: `778d0678b19cd5d5723db1e988d22dca35182719e210c2b5a0afb4e45064aa34`
- Horizontal: `b5a67f446b3b3a4843cb66718063603125a624ec293342394cb79baa75257ef4`

## Recorded search findings

Features are negative mean leave-one-out demonstration pixel error, ordered by
that immutable roster. Identity takes first argmax, including its tie rule.
The oracle counts a task if any frozen expert is recorded as solving all its
queries; it is an analysis ceiling, never an inference input.

- Twelve tasks, IDs `mixed-search-0` through `mixed-search-11`.
- Column alone: 6; horizontal alone: 6; oracle: 12; identity: 12.
- Expert-solvable tasks missed by identity: none.
- Minimum successful-expert feature advantage: approximately 0.55.
- Nine exact vectors; repeated groups: (0,6), (1,7), (3,9).
- Contradictory exact-vector groups: zero.
- Unrestricted deterministic feature-only routing ceiling: 12/12.

The ceiling sums the greatest expert-success support within each exact feature
class. It does not claim linear-router learnability in general. No vectors were
rounded or coarsened; signed zero is treated as equal. Empty success sets are
expert coverage failures, not contradictory routing targets. Overlapping success
sets do not conflict when a common expert can satisfy them.

## Independent replay blocker and bounded recovery

The signed protocol requires runtime `skillcore.py` SHA256:
`3f6b926ecb7650f041c2e0d8ee42dc0271588c76701ce9e30414b3007bcc5b8c`.

Fetched frozen source instead has SHA256:
`ec3b064c1022a7eb4e97afa4faeadf313c555c9e722266ecb0868fce5fc0b40a`.
Its fetched bytes reproduce Git blob SHA1
`5634f4c2e1b135b32e4ba40513ef4844804fcb7f`, confirming materialization did not
change the connector's returned blob. Current `scripts/skillcore.py` has the same
mismatched SHA256. In-memory LF/CRLF normalization does not recover the required
hash (CRLF hash `d3ca4a91c07a9fb6d66fb9430f401d494393fa0ca24defa37b9462828ded4889`).

Bounded path-specific Git history lookup returned only the Oct 8 import commit
for the frozen file. Current-source history returned that import and
`19b2393f648e98d13326ffce2eb0f80c21e75cac` (Oct 2). The earlier source hashes to
`c8e2d4728d50fb011b3b8f6c4d6560c3f216b9e591e1c7de362b951fd24b5fb6`, also not a
match. No original matching source was recovered. No substitute was executed,
receipt rehashed, or historical file rewritten. The cause of the source mismatch
is not established.

## Reproduction and checks

Packet-only:

```sh
python scripts/router_headroom_diagnostic.py \
  runs/weights-native-composition01-20261006/AUTHORING/router_founder.txt
```

Strict independent verification, after locating only an authorized original
runtime whose bytes match the protocol:

```sh
python scripts/router_headroom_diagnostic.py \
  runs/weights-native-composition01-20261006/AUTHORING/router_founder.txt \
  --verify-search-root runs/weights-native-composition01-20261006 \
  --frozen-runtime runs/weights-native-composition01-20261006/HOST_EVALUATION/runtime/skillcore.py
```

On the pinned snapshot the second command exits with
`ValueError: frozen runtime hash mismatch`. Signed-record hashes, packet receipt,
search bytes, and roster/protocol cross-bindings pass before that check; the
expert roster and tensor execution stages are not reached.

`python -m unittest tests.test_router_headroom_diagnostic -v`: 8/8 passed.
`python -m py_compile scripts/router_headroom_diagnostic.py tests/test_router_headroom_diagnostic.py`: passed.
These are focused diagnostic checks, not ONE_STANDARD/semantic/ARC qualification.
Protected tests and floors were not modified; aggregate qualification is handled
separately by the implementation owner.

## Decision and falsifying conditions

Do not allocate another router-author tranche on this same panel and feature
contract to seek improvement over identity: its observed score is already at the
expert oracle ceiling. Do not deliberately manufacture harder cases to defeat
that baseline. The recorded result does not rule out headroom elsewhere.

A valid independent replay disagreeing with the saved packet would falsify this
packet-based conclusion. An already permitted, separately justified training
panel showing expert success on identity misses would establish a different
feasibility case, subject to new scope approval. Exact equal feature vectors with
disjoint successful-expert sets would demonstrate that no deterministic router
using only those features can solve both, regardless of tensor choice. None is
observed in this packet. Recover the matching original source before claiming
independent tensor verification; preserve the source mismatch as an audit fact.
