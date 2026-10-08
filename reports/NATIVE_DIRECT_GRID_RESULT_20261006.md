# Direct Codex prediction on bd283c4a, 2026-10-06

The tools-free direct prediction exactly solves the one query of this existing
ARC-AGI-2 public-training SEARCH task. Fresh-process replay matches. This is a
direct reasoning baseline, not an LLM-authored neural-weight result or evidence
of held-out transfer. The host-inferred program also solves the query, so no
advantage over that stronger baseline is established.

| Arm | Exact tasks | Query mismatched pixels | Runtime errors |
| --- | --- | --- | --- |
| Direct Codex prediction | 1/1 | 0 | 0 |
| Frozen host-inferred frequency-packing program | 1/1 | 0 | 0 |
| Identity | 0/1 | 72 | 0 |

No expected output or query feedback was sent to the author. No retry occurred.
All query grids had matching shapes. The frozen response was scored once, then
recomputed in a separate process solely to verify the saved result.

## Provenance and accounting

- Run: `runs/weights-native-arc-iteration01-20261006/`.
- Author thread: `01a11234-8933-7155-b775-4de60ff0e033`.
- Author turn: `01a11234-96ca-72d8-a12f-e2452ae4d1f2`.
- Reasoning: xhigh. Model: default; exact backend was not supplied or inferred.
- One author call consumed. Actual credits remain unknown. No genome positions
  are credited or charged to the separate literal-weight experiment.
- Raw JSON: 355 bytes, parent-relayed exact text preserved without reserialization
  or a trailing newline. Original platform-byte capture is unavailable; the
  distinction is recorded in the receipt.
- Raw SHA256:
  `0a8eedd34eac9f077160905dc083cfcd64b3c6e065f9e846e35766d8531902e8`.
- Frozen response receipt:
  `0bc8cc915ecf88f4450adc8e5a1ae73b7114842137e647b0b5a7e7b04fbb6ef1`.
- Result hash:
  `4b4a307a1e2eeef3412b7553f05fc7c20a2dde3a670096b3754ea5bc599ba2c3`.

The direct baseline protocol was frozen before the author response; the raw
response and provenance were then frozen before scoring. Dataset SHA256 was
checked against the pinned official training source. Query labels were used only
by the trusted data comparator after prediction freeze. Scoring outputs contain
exactness, shape and mismatch counts, not expected grids. Reserved report and
excluded evaluation files remain untouched. Public-training familiarity and
previous search exposure limit the interpretation.

## Verification

The dedicated data-only scorer uses no authored Python. Its strict decoder
passed a valid-grid fixture and rejected eight malformed/duplicate-key cases;
pixel and shape mismatch handling was checked separately. It enforces one query,
ASCII digits, rectangular grids, 30x30 maximum, and bounded raw/rationale size.

Scoring completed in 0.375 seconds and independent replay in 0.344 seconds,
each in a fresh one-thread child with a 30-second limit. Full evidence:
`native-direct-grid-score-replay-20261006.log` and the run's
`HOST_EVALUATION/DIRECT_GRID_RESULT.json`. No source or expected grid was routed
to a model executor, and no sandbox or admission guard was weakened.

Required qualification check remains red and incomplete; standard failures and
semantic errors preceded the bounded timeout. No protected tests were changed.

```text
python -m unittest tests.test_one_standard tests.test_standard_semantic -v
STATUS: TIMEOUT
ELAPSED: 30.031s
```

Full partial output: `native-direct-grid-standard-gates-20261006.log`. The direct
prediction result does not satisfy the weight-population milestones.

## Minimal tensor extension: design only

`HOST_EVALUATION/FREQUENCY_PACK_DESIGN.json` describes one additive trusted
`frequency_pack_v1` function with seven literal values: a frequency rank
coefficient and binary scan-axis/row-direction/column-direction score vectors.
The fixed code would count, sort and place pixels, preserving shape and the
color histogram for all valid tensors, including zero/random controls. It could
reuse existing typed-cell contracts without arbitrary model code execution.

No template code was added and no tensor author was called. This is a coarse
parameter-control space, not neural algorithm learning. Reversing frequency order
and both traversal directions produces the identical result when counts are
distinct, as in both training demonstrations: 16 sign/argmax configurations give
eight behaviors. The resulting random odds are 12.5% for one symmetric draw,
33.0% best-of-three and 95.9% best-of-24. This corrects an earlier design estimate
that counted the reversal-equivalent configurations separately; the original
record is preserved with `FREQUENCY_PACK_DESIGN_CORRECTION.json`. These are
analytical estimates, not observed evaluation results. Host-designed functionality should
remain an explicit baseline rather than being presented as learned capability.

The original proposed program/structure authoring arms remain unlaunched because
their bounded runtime is unavailable. ARC-3 and remote-access blockers remain as
recorded in `ARC_ITERATION_READINESS_20261006.md`. No installation, remote work,
deployment or publication occurred. The baseline is terminal after its single
authorized author call; no additional task or author is silently started.
