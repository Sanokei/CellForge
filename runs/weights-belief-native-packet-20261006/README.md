# One belief-cell native author packet

Preparation only: one file-supervised native weight-authoring attempt is allowed
by this packet. No author is invoked by preparation, checks, import or replay.
The existing belief-cell module and documentation are preserved unchanged.

Only `author/AUTHORING_PACKET.json` and `author/RESPONSE_SCHEMA.json` may be
provided to the author. The parent must use a tool-disabled native context with
only those file contents. Do not expose this README, host runtime, evaluator,
protocol, development files or outcomes. Directory separation is not an OS
sandbox; if the author has repository tools, this access boundary is not met.
There is no authorized author retry or development-feedback loop.

The packet requests one JSON object with exactly `schema`, `packet_id`,
`parameters` (293 literal finite binary64-compatible numbers) and `rationale`
(1..4000 characters). Standard JSON only, <=65536 UTF-8 bytes, no duplicate keys,
bool parameters, NaN, infinity, Markdown or executable code. The JSON schema
describes the object; size, duplicate-key and finite-binary64 rules are enforced
by the importer. No parameter hash is demanded from the author: the host hashes
the exact interpreted literals. No programmatic solving or training is used.

## Frozen task and scoring contract

Search uses seeds 2026100611..2026100614: four two-actor/two-object histories,
eight events each. Development uses seeds 2026100621..2026100624: four
four-actor/four-object histories, sixteen events each, with alternating order
of two preservation contexts. Every four-event block has one scored query in
each of seen move, unseen move, unrelated object and observed unknown.

Search has 32 primary queries (eight/category); development has 64
(sixteen/category). All slots still pass through the MLP on every event. Query
selection avoids swamping the main measure with unrelated slots. The author
sees only the 32 search query inputs and targets using reference old beliefs.
The initial beliefs are one-hot host-specified beliefs; they may differ from
actual world state. Events and data are original local synthetic material.
The scenario families/seeds are frozen before author output; no result justifies
a language, social-awareness or independent generalization claim.

Primary scoring is recursive rollout top-1 correctness, with exact ties broken
by lowest location index. A scenario is solved only when every scored query is
correct and none is unresolved. Teacher-forced query accuracy, maximum absolute
probability error >0.05, mean L1, per-category and per-scenario outcomes are
reported separately. Every output distribution and expected target is retained
host-side. Runtime failures count explicitly; remaining rollout queries stay
unresolved in the denominator, with no reference repair or fallback. Teacher
forcing continues independently. Probability tolerance is diagnostic: one-hot
reference targets need not be exactly attainable with finite-logit softmax.

One random control uses seed 2026100631 and the existing module's
uniform[-0.25,0.25] initializer at all 293 positions. Its draw is fixed before
authoring and never selected among alternatives. Both native and random arms
consume 293 literal-position charges per submitted attempt, including an
invalid native response. Zero weights and the host reference rule are
diagnostic controls. The reference is authored code, not learned weights.
Equal positions do not prove equal compute or credit use; native credits are
unknown/null, never zero/free or a fabricated estimate.

## Host operation

The parent must retain `FREEZE.sha256` externally before dispatch. `FREEZE.json`
hashes all static packet, schema, data, protocol, source snapshot, evaluator,
generator, README and STANDARD files. Its own SHA-256 is the external anchor.
Frozen files are written exclusively and are never rewritten; every operation
requires `--expected-freeze` and verifies the hashes before importing the
snapshot or evaluating anything. A writeable checkout is not signed storage;
the externally retained hash and trusted host execution are required.

Prepare once from the repo root (already performed for a delivered packet):

```text
python -B runs/weights-belief-native-packet-20261006/host/prepare.py
```

Run the bounded control/parser check once; it writes `host/CONTROLS_CHECK.json`
without consuming an author attempt. Pass the exact saved hash, not a fresh hash
recomputed after an author ran:

```text
python -B runs/weights-belief-native-packet-20261006/host/evaluate.py --expected-freeze <saved-hash> check
```

After the parent captures the exact native output bytes in a file:

```text
python -B runs/weights-belief-native-packet-20261006/host/evaluate.py --expected-freeze <saved-hash> import --response <exact-native-response.json>
python -B runs/weights-belief-native-packet-20261006/host/evaluate.py --expected-freeze <saved-hash> replay
python -B runs/weights-belief-native-packet-20261006/host/evaluate.py --expected-freeze <saved-hash> record-source --source-thread-id <id> --source-turn-id <id> --model <actual-model> --reasoning <actual-setting>
```

Import creates `host/attempt/` exclusively before reading/parsing, preventing a
second submission. `ATTEMPT.json` records full position charges, unknown
credits, initially unknown source IDs and validation status. Valid output is
preserved as `RESPONSE_CAPTURE.bin`, decoded into `IMPORTED_CELL.json`, and
scored into `RESULT.json`; an invalid attempt still evaluates controls and has
no native score. Oversized responses preserve only the first 65537 bytes,
explicitly marked incomplete with no full-response hash. Missing/unreadable
responses also consume the attempt. An interrupted attempt directory blocks
re-import; it requires host investigation rather than an automatic retry.

`replay` checks valid raw response identity, exact imported literals, attempt
binding and independent recomputation of all arms. `record-source` adds an
exclusive source receipt later without rewriting the frozen protocol/attempt;
it binds parent-reported source IDs/model/reasoning to the attempt and response.
Source claims are not authenticated by a hash. Original native records must be
retained by the parent. Neither command calls a model or admits a cell.

`STANDARD.json` truthfully records the preparation state, no founders/proposals
and a single control seed below the project's three-seed qualification. It is
not updated to pretend a qualifying experiment occurred. Actual later attempt
receipts remain additive. The protected broad standard/semantic/ARC/lane suites
are not part of this bounded check and are not claimed green.

No new dependencies, downloads, host tool actions, deployment, protected suite
changes or commits are part of this packet. No live grid/main/agent-contract or
failure-journal source is edited.
