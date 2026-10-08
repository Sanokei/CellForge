# Fixed-T8 NCA data-only author intake preparation, 2026-10-08

**The founder packet and finite intake/evaluation boundary are frozen. No author
has been dispatched, reserved or consumed. No phase2 candidate or second random
has been evaluated. Actual parent author approval remains false.**

This is preparation for one founder and at most one conditional mutation, not a
weights result. The separate route and quota decision remains pending. Publication
is paused. Existing broader qualification remains red/incomplete.

## Exact packet and contract

Run: `runs/weights-nca-author-tranche-20261008/`

| Artifact | SHA256 |
| --- | --- |
| `FOUNDER_PACKET.json` (16,752 bytes) | `f26d3567f9d02903bbb525072803ed4713714108665a30458df99f2cd1532e37` |
| `CONTRACT.json` | `18f8b7c1be20991b3ea021aa43d7bfd4a821aab67bc547989c2a9f222535d7ee` |
| `AUDIT_ENCODING_SPEC.json` | `45e069f7f5487c7320fa665dd3776eec9276b70ed1512ec58507880b714c36a1` |
| `RESPONSE_SCHEMA.json` (15,494 bytes) | `d7ebc66e50abf32d43db94d4984201725d8dc8c122f2795a2af57aef1d272f72` |
| `RESPONSE_SCHEMA_VALIDATION.json` | `1e2baffdf5dee91496f75a012b7e95b6ac56f73b68d9e5e92f43a1a206317b0d` |
| `scripts/nca_author_intake.py`, also saved under run `source/` | `39ce4aa1798549854299002e02b3a74f12c4c824851686e0bf6c14f001deb0f6` |
| `tests/test_nca_author_intake.py` | `37fc5c144d652431e77c08896553c58dd15937c5453ed1c4b0fa3a07c59eaa3f` |

The packet contains architecture equations and exact shift/channel semantics,
96 lexicographically named tensor-array shapes, all 1,488 required literal
positions, the common vocabulary `[-1,-0.5,-0.25,0,0.25,0.5,1]`, and 17 sanitized
training input/output pairs under opaque stable labels `case_01` through
`case_17`. Original task IDs and all-11/32 provenance are evaluator-only.

No reference or random tensor values, control scores, baseline selection outcome,
original query/test data, original task IDs or executable source are in the
founder packet. The response must be exactly a JSON object with `schema` equal
to `nca-author-tensors/1` and a `tensors` object containing every required array.

The supplemental machine-readable `RESPONSE_SCHEMA.json` is suitable for a
supported CLI `--output-schema` input. It requires the exact root keys and const
schema name, every one of the 96 named arrays, fixed nested lengths, numeric
seven-value enums and no additional properties. Its hash must be bound in parent
route approval separately from the already frozen contract/packet. Installed
`jsonschema` Draft202012 schema checks and 13 parsed-value fixtures agree with
the strict parser; three additional raw-format rejection checks pass. JSON
Schema cannot itself detect raw duplicate keys or malformed UTF-8, so strict
raw intake remains mandatory. No author call or route smoke check was made.

The architecture describes dy-major/dx-minor neighborhood ordering in
`(-1,0,1)`, sampling state at `(y-dy,x-dx)`, 12 channels, three 36-wide affine
partials per hidden row including all three biases, left-associated summation
then ReLU, 12 output linears and the original input mask. There are exactly
eight shared iterations, ten initial one-hot color channels plus two zero
channels, first-ten-channel argmax with lowest-index ties, and fixed height and
width. One global tensor serves all cases, with no demonstration conditioning,
task routing or structure changes.

## Immutable phase1 dependency

The phase2 boundary validates the original phase1 manifest
`11d371121214c1bd0caeaa2c78af0ebc92affab9f5f70c09728c6a6240373326`, runner
`c5c3ea59e389f100693365293edadfe5f5fd1d527f8247917fc2173b15e704c7`, all saved
source/artifact hashes, baseline result
`c2bc1ca4841dca1d9f9999e6bfb7bab825eca362f3b805e7713c63bcdcdd2072`, and successful
replay result `2abe7a62ff7ac3a735a07d4f456fbc375733ca506c641d097c04ff5dd43cb4ff`.

The phase1 runtime, graph descriptor, tensors and control outputs were not
modified. Its loader still permits only its original four arms. The phase2
candidate worker loads the saved phase1 runtime in a fresh subprocess and
constructs the exact saved graph descriptor with every validated author literal
injected unchanged. It does not use the mutable T2 evolution caller or replace
the proposal with reference weights. Float64 bit checks preserve signed zeros;
separate numerical-equivalence fingerprints normalize integer/float spellings
and signed zero only for duplicate attribution.

Candidate scoring retains phase1 leave-one-out context, reference interpreter,
float64, one numerical thread, concurrency one, 30-second arm wall limit,
25-second CPU limit, five-second demo limit, 1 GiB address space, 8 MiB output
file limit and 64 open files. It saves every prediction/error and reports 17-demo
and six-task scores plus full eleven-task/32-demo coverage. Invalid candidates
and controls retain null scores; invalid controls forbid comparisons and further
authoring.

## Finite state and accounting

The module contains no model/network/CLI-author dispatch. Future host use must
first provide an external approval binding the parent identity, exact contract
and packet hashes, and route/quota attestation references. Exclusive fsync-backed
reservation creation precedes any possible external dispatch. Submission intent
then consumes one full attempt before the host can dispatch. No transient
response or transport outcome is treated as a free attempt.

Every consumed invalid, refused, neutral or duplicate submission costs 1,488
author positions and 1,488 paired-random positions, with a maximum of 2,976 per
arm. Invalid/refused responses stop without correction or replacement. Reserved
or unresolved submissions block another author. An explicitly uncertain outcome
halts the whole tranche without retry or refund. Full original response bytes
are written once before parsing; malformed, non-UTF8 and oversized responses
are retained, never repaired or overwritten.

The parser rejects duplicate JSON keys, NaN/infinity, booleans, strings in numeric
positions, out-of-vocabulary values, wrong shapes, missing/extra tensors,
executable fields, wrapper text and Markdown. Receipts bind packet/raw hashes,
parent identity, approval/submission, route/quota references and host audit.

The first paired random is reused from phase1 with its exact frozen tensor and
complete result, never redrawn or re-evaluated. Seed 2026100802 can only be
constructed after a second attempt has actually been consumed. It has not been
constructed in the real phase2 run.

A mutation packet can be prepared only after a valid audited founder has a
complete valid evaluation and at least one failed demo. It contains the same
frozen founder context, exact founder tensors and deterministic training-only
failure predictions/errors with opaque case labels. No control scores are
exported. It is separately packet-hash approved before reservation. No mutation
packet exists yet in the real run.

The comparison requires strictly more exact demos than the best fixed reference
and every charged random, preserving selected-baseline solved demos. A mutation
claim additionally requires strict founder improvement and no regression of
founder solves. Numeric and panel-functional duplicates remain charged and are
labeled explicitly, including the founder comparator. No crossover, admission,
benchmark, held-out/generalization or ONE_STANDARD claim follows.

## Pending route audit, with no preventive-isolation claim

The current route gate is **observed_zero_tool_use**, not a claim that tools are
impossible. A complete captured CLI event stream must be checked against a
conservative allowlist: thread/turn lifecycle, reasoning-summary event hashes,
and one completed final-message event. A tool attempt, unknown event, missing
line, truncation, timeout or uncertain completion consumes the attempt and
terminally halts with no retry or scientific claim. Input-only invocation must
exclude repository/global context carrying forbidden labels. An empty declared
tool list does not meet this gate.

The full raw event trace belongs only to the restricted parent route audit.
Phase2 evidence may contain final response bytes, classified event metadata and
hashes, token usage and audit references, but no private reasoning text. An
invalid audit is retained as a rejection and supplied-audit hash, without saving
unexpected trace/reasoning content. Token usage does not establish credit use;
credits remain unknown.

The parent's exact event-encoding clarification arrived immediately after the
main freeze. It is preserved additively in `AUDIT_ENCODING_SPEC.json`, without
rewriting the packet, source or contract. Parent route approval must explicitly
bind that specification's hash before use:

- `sequence` is the zero-based **host-wrapper physical JSONL line index**, not a
  claimed native Codex field. No original line may be omitted.
- Each event SHA hashes the exact original line bytes **including its original
  line terminator** when present; LF/CRLF remain unchanged. A final line without
  a newline is hashed as received. Never reserialize to compute the hash.
- The restricted trace SHA covers exact complete stdout event-stream bytes.
- Final raw response bytes are the unmodified UTF-8 encoding of the single
  completed final-message text: no trimming, newline insertion, Markdown removal
  or other transformation.

The source validates bound audit receipts; it does not itself capture or verify
a restricted trace. Host statements and reference strings are attestations, not
cryptographically signed platform guarantees. Actual route/event-schema
compatibility and authorization require parent review. This report is not that
approval.

## Verification and remaining gate

`python -m unittest tests.test_nca_author_intake tests.test_nca_feasibility_preflight -v`
passed **50 tests** (32 new strict-intake tests plus 18 unchanged phase1 tests).
The log is `PREPARATION_TESTS.log` in the phase2 run. All author-state tests use
separate temporary mock receipts; actual numeric predictions are only on tiny
synthetic fixtures. Tests cover complete mutation state transition, strict
parser failures, invalid full charging/no retry, unknown submission halt,
forbidden/unknown-event rejection, private-reasoning exclusion, no overwrites,
source/packet/response tamper, source-bound T8 injection and anti-substitution,
no-query/ID leaks, numeric-equivalence attribution, strict comparisons and null
invalid-control semantics.

The frozen verify command passed, reporting zero reservations, zero consumed
attempts and zero charged positions per arm. Source/AST compilation and
`git diff --check` passed. The independent reviewer reran all 32 intake tests and
the frozen verify, matched all 17 packet pairs independently to the saved
training panel, checked exact architecture semantics and found no ID/control/query
leakage. The reviewer confirmed the remaining route boundary: supplied host audit
metadata does not independently establish actual CLI event-schema compatibility;
the parent must inspect the actual engine schema and wrapper before use. No broader protected
qualification rerun was claimed. No protected test/floor, old STOP, production
template, historical result, remote environment or publication was changed.
