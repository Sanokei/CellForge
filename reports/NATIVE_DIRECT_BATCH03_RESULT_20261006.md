# Newly selected direct ARC batch03 result, 2026-10-06

All four direct predictions matched exactly, and all four fresh-process replays
matched the saved results. The version split is **ARC-AGI 2/2, ARC-AGI-2 2/2**.
No query-feedback revision, retry, candidate selection from query outcomes or
additional author call occurred.

| Version | Task | Direct exact | Direct pixel differences | Identity |
| --- | --- | --- | ---: | --- |
| ARC-AGI | 46442a0e | 1/1 | 0 | Different shape, not exact |
| ARC-AGI | c8cbb738 | 1/1 | 0 | Different shape, not exact |
| ARC-AGI-2 | 984d8a3e | 1/1 | 0 | Same shape, 54 differing pixels |
| ARC-AGI-2 | 0c786b71 | 1/1 | 0 | Different shape, not exact |

These are newly selected official public-training tasks, absent from scanned
local text records. Remote/nontext history and model pretraining exposure remain
unverified. They are not proven unseen, held-out evaluation, or a benchmark score.
Each solution is a separate task-specific direct model response, not a stored
neural-weight cell or a shared learned model. Identity is only a weak baseline.

The preceding panel scored 4/4 on already exposed search tasks. Descriptively,
the direct-prediction trials now total 8/8 unique public-training tasks across
eight independent author calls; preserve the two panels' different exposure
histories instead of presenting that total as generalization evidence.

## Selection and data provenance

Run: `runs/weights-native-arc-direct-batch03-20261006/`.

The selection seed 202610060303 was frozen before new task contents. Official
training metadata supplied 400 ARC-AGI and 1000 ARC-AGI-2 IDs, with 391 shared.
The local inventory conservatively excluded 238 and 644 known-record IDs,
respectively; all shared IDs were also excluded from the ARC-AGI-2 pool. The
remaining pools contained 162 and 203 IDs. Two per version were selected by the
predeclared hash order; all four selections were independently reproduced.

- ARC-AGI commit: `399030444e0ab0cc8b4e199870fb20b863846f34`.
- ARC-AGI-2 commit: `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Both pinned repository licenses: Apache-2.0; original files and exact hashes
  are retained in `data/arc-training-batch03-20261006/`.
- Each task download matched its pinned Git blob; scoring rechecked SHA256.
- No public evaluation, private evaluation or reserved report task was accessed.

See `NATIVE_DIRECT_BATCH03_PREPARED_20261006.md`, `SELECTION.json`, the complete
local inventory, official training manifests and data `SOURCE.json` for coverage
and acquisition details. Earlier blocked receipts remain preserved; one reviewed
retry and the subsequent bounded download were approved before acquisition.

## Authorship and immutable capture

The parent reports each author ran in a new isolated tools-free thread with the
exact common instructions and its individual label-free packet. All used xhigh
reasoning and the default model setting. Exact backend identity, platform prompt
bytes and credit usage are unknown, not fabricated or inferred.

| Task | Thread | Turn |
| --- | --- | --- |
| 46442a0e | 01a11255-2c9d-704e-91c4-42e4995bd03e | 01a11255-34b7-72c3-b5a6-8a418b51a8f9 |
| c8cbb738 | 01a11255-6878-747f-80ec-6cbb4922000e | 01a11255-73d9-712a-953d-2285ce5f89c6 |
| 984d8a3e | 01a11255-a6e3-753a-b994-f8d196957022 | 01a11255-afb9-741b-b20c-d91b7441fd9a |
| 0c786b71 | 01a11255-e701-7564-b319-b9c40103002a | 01a11255-ef96-7056-89a7-fa9016094bb9 |

The exact parent-relayed JSON texts were captured as UTF-8 without reserialization
or trailing newlines, including the multiplication/degree characters. Original
platform-byte captures are unavailable. Each raw response and task-bound
provenance receipt were frozen before its query labels were used for scoring.
Four author calls consumed, zero genome positions; no further call is allowed by
this terminal batch.

| Task | Raw response SHA256 | Result hash |
| --- | --- | --- |
| 46442a0e | 52b6dcfb8e2f287abebfc026d1f1a8b578d821d2763b2e842b2148f063b2326c | 8bd95e7470cbf4029ebde8e1384626047a13fda90b67d075ea5173671e752f51 |
| c8cbb738 | ee0fe69308280689fb0898fbebe9368cac8989b39f97c7e68a46d7488aa55a98 | 2bc7bdf33b7a2d4c18b3c6482129acf717a72a19842d137ed2ac33adcc2e114d |
| 984d8a3e | 9dba6f8620bc8e370f14a483f512a3dc19e15ddf26193310701a6ef2af2ea272 | d48ca4aef61057c234fe0dc3038175cb8690010cd47537acb11837b015466a0a |
| 0c786b71 | 8d45de7a353c159af5bfd36bc1ae896fd6e8707a6df1e661cdcf90e9721d6576 | d8d17ca3ab1f4d32ca7281893106ca0a38d6521a6166cbbc2a9ca646c1b62f08 |

## Verification and limits

The existing strict data-only scorer was rebound to four task IDs, per-task
dispatch receipts and the pinned data paths; its decoder and exact comparator
are unchanged. The new data-path guard confines reads to the frozen batch data
directory. A valid binding fixture passed and cross-task/path-like identifiers
were rejected. No authored source was executed and no new solver framework was
introduced. Original scorer source and binding hashes are preserved.

Scoring took 0.328/0.328/0.375/0.359 seconds in arrival order; replay took
0.313/0.297/0.297/0.297 seconds. Every child used one numerical thread and a
30-second limit. Full evidence:

- `native-direct-batch03-first-three-score-replay-20261006.log`
- `native-direct-batch03-last-score-replay-20261006.log`
- Run `HOST_EVALUATION/*.RESULT.json`, raw/frozen receipts and `TERMINAL.json`.

The existing full-standard status remains red/incomplete and was carried forward;
unrelated broad suites were not rerun. No protected tests/floors or historical
results were changed. No frequency-packing template was implemented. ARC-3 source,
runtime installation and game work remain with the separate owner; this scoring
work did not overlap them. No credentials, remote jobs, deployment, publication
or commit was changed here. Research/paper integration remains parent-owned.
