# Three-task direct ARC prediction batch, 2026-10-06

All three tools-free direct predictions exactly match their task queries.
Independent fresh-process replays match every saved result. With the separately
recorded bd283c4a baseline, direct prediction is 4/4 on the four existing SEARCH
tasks, using four independent per-task author calls. This is not a shared
learned-weight model, a held-out score, evidence of fresh model knowledge, or an
ARC benchmark completion claim.

| Task | Direct exact | Direct pixel differences | Identity result |
| --- | --- | --- | --- |
| 1a07d186 | 1/1 | 0 | 0/1; 18 differing pixels |
| 3194b014 | 1/1 | 0 | 0/1; different shape |
| f5b8619d | 1/1 | 0 | 0/1; different shape |

Identity is a weak control. These results do not establish an advantage over a
capable task-specific program. In particular, f5b8619d was already solved by the
prior literal ray/tile cell; all four tasks have prior search/evaluation exposure.
Public ARC training tasks may also be familiar to a model from pretraining.

## Scope and provenance

Run: `runs/weights-native-arc-direct-batch02-20261006/`. Each author received only
its training demonstrations, query input and the frozen common instructions.
The parent reports tools-free isolated authoring. The model setting was default,
with xhigh reasoning; exact backend identity and credit usage remain unknown.

| Task | Thread | Turn |
| --- | --- | --- |
| 1a07d186 | 01a11242-3c1b-75c0-83f6-453ca63a101e | 01a11242-4771-714d-92ee-b4af28e6c917 |
| 3194b014 | 01a11242-8819-71aa-b673-26439548a01a | 01a11242-9156-76fe-a106-2e2fbb1a7521 |
| f5b8619d | 01a11242-c59d-73bc-af50-d39bc0e47028 | 01a11242-d124-767a-a5be-a6e0af2c1f4c |

The exact parent-relayed JSON texts were saved as UTF-8 without reserialization
or appended newlines. The multiplication character in the 3194b014 rationale is
preserved as U+00D7. Original platform-byte captures are unavailable; provenance
receipts explicitly distinguish the parent relay from independent capture.

All raw responses and their task-bound receipts were frozen before host label
scoring. Packet/data hashes and author bindings were checked. Every response
consumed its single authorized slot. No query feedback went to authors, no
retry occurred, and no additional call is allowed by this terminal batch.
No reserved report or excluded evaluation data was read. No genome positions
are counted: three direct-baseline calls, separately from the weights ledger.

## Evidence and checks

| Task | Raw response SHA256 | Result hash |
| --- | --- | --- |
| 1a07d186 | 1e13399a20c619a3f79c825a2b9cf5b11a693325480792ac1f231478225bc93f | d03d73a3888ca6767da933c5847f1cb4398b800101d47d413281de45e44c8927 |
| 3194b014 | 221512e90e98aad6ab614d830386782b44d5be0d56e37a5531e21a2f7e7a5ec9 | c8c4686ec3eb41f2181f6f6bb6ed98feff95d743826537dda76ee5965b2d4169 |
| f5b8619d | 022486e54e3156e1da829118a5401bf607127e737406d25526160b3826adb377 | 86d97205083a5b8495085840c64a40231252596ae444ac15d1f81472364bb10c |

The prior strict data-only parser/comparator was bound to these three task IDs;
no new solver framework or template was introduced. Its checks passed for a valid
fixture, ten malformed or misbound payload cases, pixel mismatch and shape
mismatch. The scorer executes no authored source. Query mismatch counts remain
null when shapes differ rather than reporting a misleading pixel count.

All six scoring/replay child processes used one numerical thread and a 30-second
limit. Scoring took 0.359/0.344/0.375 seconds; replay took 0.281/0.281/0.297 seconds.
Log: `native-direct-batch02-score-replay-20261006.log`. Saved results are in the
run's `HOST_EVALUATION/`; `TERMINAL.json` links them and the previous bd283c4a result.

The existing full-standard status remains red/incomplete, carried from
`native-direct-grid-standard-gates-20261006.log` (30.031-second timeout after
standard failures and semantic errors). Per parent direction, those unrelated
suites were not rerun for these scores. No protected tests or floors changed.

The seven-value frequency-packing template was not implemented. ARC-3 installation
is owned by the separate runtime worker; this batch did not change its source,
packages, game files or settings. No remote work, credentials, paid API calls,
deployment, publication or commit occurred here. Research/paper integration
remains parent-owned.
