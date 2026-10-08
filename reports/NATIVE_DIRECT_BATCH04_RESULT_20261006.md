# Direct ARC batch04: three exact submissions, one cancelled slot

All three submitted predictions match exactly, and all three fresh-process
replays match. **Score: 3/3 submitted; coverage: 3/4 planned.** The fourth slot,
e7639916, was cancelled during author creation. The parent reports `list_threads`
confirmed no author was created. It has no response or score and is neither a
synthetic zero nor a success. No retry occurred.

| Version | Task | Status | Direct pixel differences | Identity differences |
| --- | --- | --- | ---: | ---: |
| ARC-AGI | ed36ccf7 | Exact | 0 | 4 |
| ARC-AGI | ef135b50 | Exact | 0 | 7 |
| ARC-AGI-2 | 712bf12e | Exact | 0 | 28 |
| ARC-AGI-2 | e7639916 | Cancelled, not created, unscored | N/A | N/A |

ARC-AGI is 2/2 submitted. ARC-AGI-2 is 1/1 submitted, with one additional
cancelled planned task. Identity scored 0/3. These are newly selected official
public-training tasks, not proven unseen or held-out evaluation. Remote/nontext
history and model pretraining exposure remain unverified. Each prediction is
task-specific direct reasoning, not a stored neural-weight result.

## Protocol, provenance and verification

Run: `runs/weights-native-arc-direct-batch04-20261006/`.

Seed 202610060404 was frozen before task contents. Existing official manifests,
pins and licenses were reused; the four batch03 IDs were added to the prior
exclusion inventory. The ARC-AGI-2 pool additionally excluded every shared ARC1/2
ID. All four selected downloads matched pinned Git blobs. See
`NATIVE_DIRECT_BATCH04_PREPARED_20261006.md` for exact selection and acquisition.
No evaluation/private data was accessed.

The parent reports isolated tools-free authors with exact common instructions
and individual label-free packets. Model setting: default, exact backend
unverified; reasoning: xhigh. Credit usage and actual platform prompt bytes are
unknown. Three completed author calls are recorded; the cancelled slot has zero
created authors by parent verification. The cancelled tool-call event remains
in the provenance rather than being concealed.

| Task | Thread | Turn |
| --- | --- | --- |
| ed36ccf7 | 01a11263-ef6f-76e0-848e-271829be3014 | 01a11263-f787-7503-9f4d-f719934f5441 |
| ef135b50 | 01a11264-3637-7722-b1a1-1009d0984038 | 01a11264-3d63-700b-930d-7269fa2045e6 |
| 712bf12e | 01a11264-aa4f-740b-83bf-86250c4bf02a | 01a11264-b70a-74c1-94d7-756931c32727 |

Exact parent-relayed JSON text was captured as UTF-8, without reserialization or
trailing newlines, then frozen with provenance before host scoring. Original
platform-byte captures are unavailable. No query-feedback revision, retry or
post-query candidate selection occurred. Zero genome positions are attributed
to this baseline.

| Task | Raw response SHA256 | Result hash |
| --- | --- | --- |
| ed36ccf7 | 533fe757666eaffd1bb0f3b5f92fd512cf7e2d176a5dfec0e74c534d8a71bf70 | 68bac5022543ec8ff76d94ed35e850e75ef16516934adc21f219755d26eb1086 |
| ef135b50 | 1af74fdd4051b3880cf96adb73fd294010275b4908786cd2882feb31242d983d | f8fca25cee35347960bde3b7878ee6bf837b8edc5d5018bf213bc88612dd4d25 |
| 712bf12e | 2f3f2b0551f98d65ecc01aa0f892f3c16339afec56ea510ca349cf1692a9475f | de4ccc01b4591007e7ee742ca07f244314e5a6cfab057e463edcfe4c60b89928 |

The previous strict data-only scorer was rebound to current task IDs and pinned
data paths. An explicit cancellation guard refuses the cancelled slot before
query-label access; that behavior and task binding were checked. No authored
source is executed. The original scorer is preserved.

Scoring took 0.344/0.328/0.406 seconds and replay 0.328/0.422/0.406 seconds.
Each child used one numerical thread and a 30-second limit. Logs:
`native-direct-batch04-first-two-score-replay-20261006.log` and
`native-direct-batch04-last-score-replay-20261006.log`. `TERMINAL.json` links all
results and the separate cancellation-resolution record.

## Stop and research checkpoint

Across the direct-prediction trials, 11/11 submitted unique public-training tasks
were exact out of 12 planned; one planned slot remains cancelled/unscored. This
combines four previously exposed search tasks with seven newly selected submitted
tasks and must not be presented as a benchmark or generalization score.

No further author call, cancellation retry, batch05 or larger campaign is
authorized or prepared. `CHECKPOINT_STOP.json` records this stop. The requested
capability/weights checkpoint is saved separately in
`CAPABILITY_WEIGHTS_CHECKPOINT_20261006.md`.

Existing full-standard status remains red/incomplete and was carried forward;
unrelated full suites were not rerun. No protected tests/floors, learned template,
credentials, remote job, deployment or publication was changed. ARC-3 work and
paper integration remain with their separate owners.
