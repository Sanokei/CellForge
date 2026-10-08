# Four native spatial proposals: one search success, no development gain

The bounded four-author experiment is terminal. Two responses were invalid;
two were valid and distinct. The best native candidate scored **1/4 search and
0/4 development-selection tasks**, matching the hand-coded spatial baseline.
All four matched random controls scored 0/4 on both panels, with no runtime
errors. There is no measured development gain, population admission, chatbot
capability, or broad milestone qualification.

Repository: `D:\Github\CellForge`. Run:
`runs/weights-native-spatial-storm01-20261006/`. No server work, deployment,
commit, push, gradient, download, API call, or additional author call was made
by this host during intake and evaluation. Existing unrelated work is intact.

## Frozen experiment and actual delivery

The parent authorized four native author completions within the user's stated
62,000-credit ceiling. Four desktop attempts failed during sandbox startup,
before reading their packets or producing proposals. Those infrastructure
failures were recorded without candidate charges. No denied executable was
retried and no security setting was changed.

The parent then resumed those same four threads as tools-free text authors,
model `gpt-6-luna`. Reasoning effort and credit use were not supplied and remain
unknown. All four received the author01 payload according to the parent;
the external slot/thread mapping is recorded in `HOST_EVALUATION/TEXT_DELIVERY.json`.
The original slot-specific packets remain unchanged. The host cannot attest
to actual platform prompt bytes: `actual_platform_context_sha256` is null.
The common reference payload's parent-attested SHA256 is
`db8361a9c2fda3bc1000aa84c4efbbaef2fe2c140f6831c924ec96ef1bd4e75d`.

Each author supplied a complete 167-position replacement for the same historical
parent in the fixed `ray_cast -> tile_orient` composition on search task
`f5b8619d`. The packet supplied the search demonstrations, previous tensors,
and one replayed training failure. Query outputs, development results, and
reserved report data were absent. No post-evaluation feedback was sent.

Before authors completed, the proposal vocabulary and matched random sampler
were restricted identically: **103 free values and 64 fixed values**, all 167
charged. Every orientation vector's final four values must equal -9; its first
four values select shape-preserving transforms. This avoids the rectangular
transpose errors from the earlier trial. The saved forward implementation was
unchanged; old results and their failed controls remain intact.

All values, including inactive slots, were required. Duplicate keys, shape or
fixed-value defects receive no repair. Each consumed slot is immutable and
cannot be imported twice. Four random seeds were frozen at 2026100611 through
2026100614. Stop-file handling remains effective.

## Results and selection

| Arm / slot | Validation | Search | Development | Runtime errors |
|---|---|---:|---:|---:|
| Native author01 | Duplicate `task_id` rejected | Not scored | Not scored | Not applicable |
| Native author02 | Valid | 0/4 | 0/4 | 0 |
| Native author03 | `orient[1][1][7]` is -8, required -9 | Not scored | Not scored | Not applicable |
| Native author04 | Valid | 1/4 | 0/4 | 0 |
| Random control, each of four slots | Valid | 0/4 | 0/4 | 0 |
| Historical parent | Diagnostic | 0/4 | 0/4 | 0 |
| Hand-coded reference | Diagnostic | 1/4 | 0/4 | 0 |
| Zero tensors | Diagnostic | 0/4 | 0/4 | 0 |

The prespecified descriptive ranking uses development successes, then search
successes, then ascending cell ID. It selects author04 as the native winner
and author02's matched draw as the random winner. There are two distinct
valid native tensor genomes. Author04's cell ID is
`bcef4e93af6a2418aa4bea08bcb6435cb47816441cbbb9881b3b37ede8de4b45`.

The one search success is an improvement over the historical parent on the
search-selected task family. Development selection remains zero, and the
hand-coded baseline already achieves the same search result. This is an
authored parameter choice within host-written algorithms, not evidence of
algorithm invention, transfer, crossover benefits, or a capable chatbot.
Development is held out from these author payloads, not a final untouched
report split or a guarantee against model pretraining exposure.

## Accounting and evidence integrity

Every completed response, including both invalid responses, consumed **167
native and 167 random positions**. Batch totals are **668 per arm**; cumulative
native ARC totals across the preceding trials and this batch are **1,206 per
arm**. Separate belief experiments are excluded. Position accounting is not a
credit meter; credit use is null, not zero or free.

The exact parent-supplied response strings were captured as UTF-8 without a
trailing newline. Author01's duplicate key and author03's prohibited value
remain intact. Author04's multiplication character is preserved as U+00D7.
These are parent-delivered strings, not a direct capture of platform bytes.

A host transcription error initially omitted one repeated orientation vector
from author02. It was caught before any slot was consumed. The erroneous
capture and initial intake manifest remain preserved. The corrected capture
comes from the same already-completed parent-supplied response, with no model
retry or proposal repair. `HOST_EVALUATION/INTAKE_CORRECTION.json` explicitly
supersedes the initial author02 capture; the correct response is under
`SUPERVISED_TEXT_SUBMISSIONS_CORRECTED/author02/` and copied verbatim to its
consumed slot. The other three initial captures are unchanged.

| Slot | Consumed response bytes | Consumed response SHA256 |
|---|---:|---|
| author01 | 965 | `6d29f9b7a2c73431489c2a9bf0928ffe68c429d3ab38667190019e936a6c625b` |
| author02 | 1001 | `8117c458b878e3b17658f426a92ac057125d0140b868cae8b5077ecc8f9f3f1e` |
| author03 | 855 | `1beb575de88e486bf3ff32a79f9324378cd5663d7434320b7d934abf3c31e0e1` |
| author04 | 950 | `a5228f662d3ebad6950c4613dec953322e53dfc1002d65d4a4d0be07f6b745f9` |

Protocol hash:
`aec8ec5ce406ec718a6b4af32e5e2d413f7880a1d3d212e7ec48720881cee91e`.
Final batch hash:
`f7ee36eb75b19f49e31d1030f9a359bb7f9b382b76cce733483ef9b88ab9ee62`.
The run's `HOST_EVALUATION/author01..04/` contains exact responses, source
receipts, charged attempts, all scored tensors, outcomes, and result hashes.
`HOST_EVALUATION/BATCH_RESULT.json` binds all four results. The preparation-time
`STANDARD.json` is retained as a nonqualifying declaration; it is not a completed
ONE_STANDARD study ledger. Terminal state is recorded separately.

## Implementation and verification

Added `scripts/native_parallel_batch.py` for the fixed four-slot experiment:
freeze payloads and source, validate restrictions, sample matched controls,
charge before parsing, score through the existing importer, refuse repeated
imports, replay saved results, and produce the descriptive group selection.
No model dispatcher or autonomous retry loop was added.

Added `tests/test_native_parallel_batch.py`. Its seven focused tests passed:
167/103/64 accounting, exact control vocabulary, rectangular execution and tie
behavior, immutable response validation, full invalid charges and repeat
refusal, honest context provenance, replay, and stop-file handling.

```text
Ran 7 tests in 0.444s
OK
```

Saved-runtime preflight passed in 0.297 seconds. Individual imports took
0.782-0.968 seconds. A fresh process independently
replayed each saved response and every control exactly; finalization replayed
the four slots again and produced the bound batch result in 2.234 seconds.
Numerical libraries were restricted to one thread; each child had a 30-second
timeout. Source, task, packet, and prior-lineage hashes were checked unchanged.

The required command was run without modifying or weakening tests:

```text
python -m unittest tests.test_one_standard tests.test_standard_semantic -v
TIMEOUT_SECONDS: 30
STATUS: TIMEOUT
ELAPSED_SECONDS: 30.031
```

It recorded seven contract/milestone failures and semantic fixture errors before
the timeout. A bounded diagnostic rerun of the first suite completed:

```text
Ran 15 tests in 0.065s
FAILED (failures=7)
```

Failures: the blocked maker run has zero charged proposals against its declared
budget; the analytical belief reference has fewer than three seeds; and the
repository lacks the required increasing accuracy, crossover children, founder
coverage, mutation-over-random evidence, and 100 distinct recorded genomes.
These checks fail on the actual local checkout; this experiment does not cure
or qualify around them.

One isolated semantic fixture diagnosis completed with:

```text
PermissionError: [WinError 5] Access is denied:
  C:\Users\wkeif\AppData\Local\Temp\tmpl3x91_e7\weights-fake
Ran 1 test in 0.004s
FAILED (errors=1)
```

This is a local temporary-directory access failure, not evidence that semantic
verification passed. No ACL or sandbox change was attempted. ARC goal success
is not asserted. Full output is preserved in:

- `reports/native-storm01-host-tests-20261006.log`
- `reports/native-storm01-prepare-20261006.log`
- `reports/native-storm01-evaluate-20261006.log`
- `reports/native-storm01-replay-20261006.log`
- `reports/native-storm01-standard-gates-20261006.log`
- `reports/native-storm01-standard-failures-20261006.log`
- `reports/native-storm01-semantic-error-20261006.log`

## Next bounded decision

All four authorized candidate slots are consumed; no additional author call is
pending. Repeating the same search-selected task would primarily measure schema
compliance or another parameter rewrite. The useful next research decision is
whether to fund a separate, prespecified transfer/composition experiment with
multiple task families, equal random and hand-coded comparisons, and a fresh
held-out panel. Freeze its criterion and total call/position budget before any
authors see feedback. No such experiment or new data access was started here.

The immediate blockers to broader claims remain zero development gain, the
host-defined algorithm boundary, unknown credit consumption, incomplete broad
milestones, and semantic fixture access. Reserved report data, closed campaigns,
human shutdown control, and existing server runs remain untouched.
