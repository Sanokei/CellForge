# Two native text responses imported: invalid first attempt, valid null revision

Desktop host access was verified after reconnection. No denied author
executable was retried and no sandbox, credential or security setting changed.
The two already-completed native responses were preserved and imported through
a separate text-route envelope. **No new model call was made.**

## Measured result

| Arm | Search | Development | Runtime errors |
|---|---|---|---|
| Native text attempt 1 | Invalid top-level object; no candidate scored | Invalid | Not applicable |
| Native text attempt 2 | **0/4** | **0/4** | 0 |
| Hand-coded reference | 1/4 | 0/4 | 0 |
| Zero tensors | 0/4 | 0/4 | 0 |
| Matched random for attempt 1 | 0/3 scored; 1 unknown | 0/3 scored; 1 unknown | 3 |
| Matched random for attempt 2 | 0/3 scored; 1 unknown | 0/3 scored; 1 unknown | 3 |

Each random arm encountered transpose-on-rectangular-input errors: one query
on search task `1a07d186`, and two queries on development task `31d5ba1a`.
Those two task outcomes per arm remain null, not incorrect predictions. The
controls therefore do not establish a fully valid comparison across all eight
tasks. No positive authored-weight result or population admission is claimed.
The hand-coded reference's one search success is the selected task
`f5b8619d`; it remains explicitly host-authored feasibility, not a native genome.

The first response was not silently wrapped or repaired. Its object wrapper
violated the required single-element array and consumed a full attempt. The
second response passed the exact tensor schema; its rationale did not enter
the predictor or determine its grade.

## Accounting and provenance

New run: `runs/weights-native-spatial-text-two-20261006/`.
Both exact parent-supplied canonical JSON strings are preserved, without a
trailing newline, under `TEXT_ROUTE_RECEIPTS/`:

- `ATTEMPT1_RESPONSE.json`, 1,150 bytes, SHA256
  `868c33bcb2df019e329ef2de0897616ee1723fae79cca11c325ec16fb007cc39`.
- `ATTEMPT2_RESPONSE.json`, 1,172 bytes, SHA256
  `3aeee9ba2b3dae94bed3eea3113467383a3a5e8aa7be283a65ed0ccd71beac64`.

The source is the parent's report of native task
`/root/author_spatial_weights_text`, model `gpt-6-luna`, reasoning `low`.
That task name is not a cloud UUID. Author cloud/turn UUIDs remain null.
The source record states that the author used no tools or machine operations,
according to the parent; this was not independently observed by the importer.

The parent transcribed and compacted the original author inputs and supplied
one round of static schema/semantic feedback without hidden evaluation.
The exact actual prompt, feedback and platform response bytes are unavailable.
The actual prompt/feedback hashes remain **null**. Hashes of the archived
reference packet and original `TEXT_AUTHOR_INPUT.txt` are explicitly reference
hashes, not hashes of the actual author context. The response hashes bind the
parent-supplied canonical strings, not an invented direct platform byte capture.

Each response and matched random control is charged all **167** newly authored
positions, including the invalid first response and inactive tile slots.
The text batch therefore charges **334 native + 334 random positions**.
Including the prior two 102-position native ARC proposals, the sequence total
is **538 per arm**. These totals exclude separately accounted experiments such
as the belief trial. Credits remain **unknown/null**; no CLI meter receipt is
claimed. Random seeds are separately fixed at `2026100605` and `2026100606`.

Each attempt receipt is written before parsing/scoring and linked to the prior
attempt hash. Existing or partial attempts refuse replay as a new import.
The original single-attempt spatial protocol, source snapshot, candidate
packet and failed accounting directory are unchanged.

Protocol hash:
`0dc2db5b429adff6cba5f1119ceda3be5fc0f370a4058394a403b0d7faade33c`.
Batch result hash:
`78e28103555cdb992459938afc37fc9b00805c74e605a4b475057af7855c9a7f`.
Per-attempt results and receipts are in the new run's `HOST_EVALUATION/`.

## Search-only explanation of the failed revision

The actual values select stop-policy **0**, despite the rationale claiming 1.
The active 2-by-2 tile orientation indices are **[[0,0],[1,1]]**: two identity
tiles above two horizontal-flip tiles, rather than the four distinct transforms
claimed in the rationale. All ray paint entries are negative, so rays propagate
their source colors.

The three shown search inputs contain no color 8, while their expected outputs
require color 8. Propagating existing colors and transforming/repeating tiles
cannot create that absent color. This is a failure of these values on the
shown examples; the separate successful hand-coded baseline demonstrates that
the fixed template family can express the rule. These observations use only
the response tensors, forward semantics and search training demonstrations.
They are saved in `HOST_EVALUATION/SEARCH_STATIC_REVIEW.json`, separately from
development evaluation. No further author feedback was sent.

## Verification and remaining limits

Two new isolated tests passed in 0.117 seconds. They check complete two-attempt
charges, invalid-then-valid handling, repeat refusal, receipt chaining, replay,
changed-response rejection and refusal to label a reference context hash as
the actual author context. Existing spatial validation/evaluation code was
reused unchanged and hash-verified against its frozen source.

Preparation took 0.375 seconds and actual import 1.063 seconds. Independent
replay in a fresh process using saved source took 0.875 seconds and reproduced
both complete candidate/control results and the combined ledger exactly.
Subprocesses had 30-second timeouts and one BLAS/OpenMP/MKL thread.
`git diff --check` passed. Logs:

- `reports/spatial-text-tests-20261006.log`
- `reports/spatial-text-prepare-20261006.log`
- `reports/spatial-text-result-20261006.log`
- `reports/spatial-text-replay-20261006.log`

No report split, new author call, deployment, server run, gradient, download,
commit or push occurred. This remains host-defined spatial primitive selection
and composition, with task/structure selected using search evidence. Unknown
actual author context and partially invalid random evaluations prevent stronger
provenance/comparison claims. No broad milestone qualification is asserted.
