# Verified search feedback and second native authoring request

Prepared, but did not dispatch, the next request:

- `runs/weights-native-agent-two-20261006/AUTHORING_PACKET_V2.json`
- `runs/weights-native-agent-two-20261006/RESPONSE_SCHEMA.json`

The response schema is byte-identical to the first request: one object for
`1a07d186`, containing complete literal `sim[10,10]`, `temp[1]`, and `keep[1]`
tensors in [-9,9] plus a bounded rationale. The packet supplies the original
three search demonstrations, previous candidate tensors, and one verified
failure-journal observation. No raw author rationale, query outputs, selection
scores/examples, or report information is in the packet.

Packet SHA256:
`53d65db6ec6f4479bfc6f4c1af2b9e66198c113c36767e0f62c4bf1f403f8f9b`.

Journal head retained in the request manifest and feedback artifact:
`1126af1754b428960eaea2da7c5d4cec859c1a2afafbad46388e77a56b229903`.

## Observed search failure

The completed candidate's observed category is `mismatch_output_values`, with
no query runtime error. The included expected/actual grids are **training
example 0**, predicted using only the other two training examples. Its output
shape is 15 by 14, and 34 pixels differ. For example, with zero-based indices:

| Row | Column | Expected | Actual |
|---|---|---|---|
| 1 | 9 | 0 | 4 |
| 2 | 3 | 2 | 0 |
| 3 | 0 | 2 | 0 |

The complete training input, expected output and actual output are retained in
the 2,514-byte verified feedback text. This demonstration result is separate
from the full search-task outcome. `representation_limit` and `wrong_mapping`
are explicitly unverified competing hypotheses, not established diagnoses.

## Host verification and provenance

`scripts/native_failure_feedback.py` connects the existing failure-journal API
without modifying its module or the original run. Preparation first verifies
the terminal signed native result, charged attempt, exact captured proposal,
frozen request, evaluator sources and pinned public-training task bytes.
It independently recomputes the search scores and compares them with the
terminal result before writing a separate search-only receipt.

A new memory manifest binds all four complete search rows in their original
order: `1a07d186`, `bd283c4a`, `3194b014`, `f5b8619d`. The template identity binds
implementation bytes, interface and tensor layout. The old attempt is recorded
as a founder with no parents; the diagnostic reference cell is not falsely
claimed as its parent. V2 identifies the previous native tensor as the proposed
parent for a future replacement.

The trusted callback verifies candidate, ordered parents, proposal and charged
attempt identities; rechecks the open manifest and exact source/task/receipt
bindings; and recomputes search-only outcomes plus leave-one-out feedback.
It binds the terminal result's bytes without parsing development panels. The
journal is closed after append, then reopened and reverified through `retrieve`
before prompt creation. Full runtime snapshots and receipts are under the new
run's `HOST_MEMORY/` directory. No original evidence was rewritten.

Preparation and all checks completed in 1.032 seconds. Independent replay using
the saved runtime passed in 0.562 seconds. A Python audit hook rejected any
attempt to open a non-search training file; only the four search files were
opened. All four negative provenance checks rejected altered candidate,
parent, attempt and proposal hashes. `git diff --check` passed. Logs:

- `reports/native-feedback-prepare-20261006.log`
- `reports/native-feedback-replay-20261006.log`

Both subprocesses had 30-second timeouts and single-thread BLAS/OpenMP/MKL.
There were zero new model calls. Credit usage remains unknown. The request
reserves no call, performs no automatic dispatch/admission, and requires a
matched 102-position random arm if subsequently authorized and evaluated.

The original importer accepts only the original packet format. A future V2
import must explicitly bind this packet/schema, the actual new author identity
and response, and these parent/feedback identities in its own receipt; do not
rewrite the original request or disguise V2 as the first protocol. The packet
is ready for the parent's next bounded decision. No general-capability or
ONE_STANDARD qualification is claimed.
