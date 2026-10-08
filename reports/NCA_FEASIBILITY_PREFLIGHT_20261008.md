# Fixed-T8 NCA baseline preflight, 2026-10-08

**Result: the four frozen controls execute validly, but all score 0/17 exact
training demonstrations and 0/6 exact tasks. Independent saved-source replay
matches every arm and aggregate result.** This establishes operational baseline
headroom, not positive capacity, an authored-weight gain or generalization.
No author/model call was dispatched and no author phase is authorized by this report.

## Scope and controls

The existing 11-task/32-demonstration panel is retained in full accounting. Every
item is already exposed public-training development evidence. Only derived
training input/output pairs were opened; no original query/test data, evaluation,
private data, new downloads or cancelled e7639916 content was accessed.

- Compatible six tasks, 17 demonstrations: 1a07d186 (3), 712bf12e (3),
  984d8a3e (2), bd283c4a (2), ed36ccf7 (4), ef135b50 (3).
- Dimension-incompatible five tasks, 15 demonstrations: 0c786b71, 3194b014,
  46442a0e, c8cbb738, f5b8619d (three each). These remain explicit structural
  exclusions, not scored failures. Whole-11-panel feasibility remains ruled out
  for this unchanged same-size graph.
- All arms share exactly the existing T8 topology: 48 distinct active cells,
  96 tensor arrays and 1,488 stored scalar positions. One global tensor per arm
  is used across all six tasks; no task-specific tensor selection or routing.
- Wiring, activations, masks, reference interpreter, float64 and dimension/color
  argmax decoding are unchanged. Other demonstrations from the same task are
  supplied in original order; the scored pair is omitted from its own context.
- The 36 hidden chunk biases contain 24 real-arithmetic redundant directions.
  All remain stored and counted. This is not a bitwise float64 equivalence claim.

Before predictions, the parent and independent reviewer inspected the frozen
manifest, harness and passing additive tests. The parent then authorized only
the finite four-arm baseline. The baseline worker ran each arm once; a separate
reviewer performed the single fresh-process saved-source replay.

## Observations

| Arm | Exact demos | Exact tasks | Mean per-demo normalized pixel error | Errors |
| --- | ---: | ---: | ---: | ---: |
| Existing identity | 0/17 | 0/6 | 0.2877535467628347 | 0 |
| Existing neighbor-majority/smoothing | 0/17 | 0/6 | 0.6022913902403066 | 0 |
| Zero | 0/17 | 0/6 | 0.5465838521643475 | 0 |
| First matched full-random tensor | 0/17 | 0/6 | 0.9526492418829879 | 0 |

The frozen tie-break selects identity. Its solved-demonstration preservation set
is empty, so that future constraint currently has no anchors. All four numeric
tensors and all four complete 17-demo prediction vectors are distinct; there are
no numeric or panel-functional duplicate groups. Every pair of arms differs on
every full demonstration prediction. This does not prove global functional
inequivalence.

Per-demo predictions, exact flags, mismatch counts, normalized errors, operation
counts and timings are saved in `baseline/*.json`. All runtime errors and invalid
outcomes would have explicit error records and null score fields, rather than
ordinary misses. The independent reviewer also recomputed all 68 saved per-demo
pixel-error records separately and confirmed the metrics.

## Resource bounds, accounting and future decision

Each arm used an isolated subprocess, one numerical thread and concurrency one,
with 30 seconds wall, 25 seconds CPU, 5 seconds per demo, 1 GiB address space,
8 MiB file size and 64 open-file limits. Measured baseline arm wall times were
0.68–0.76 seconds. This is trusted host code inside the existing cloud sandbox,
not an arbitrary model-code sandbox or proof of OS-level network isolation.
Python 3.12.14 and NumPy 2.3.5 versions are bound; their binary builds are not
cryptographically sealed. Eleven project source files cover the runtime imports.

Zero author attempts and zero author/paired-attempt charged positions were
consumed. The first random preflight computation has 1,488 positions and seed
2026100801; it must be reused without a redraw if a founder is later authorized.
The saved second seed 2026100802 was neither constructed nor evaluated. Credit
usage is unknown; no model call occurred in this phase.

Author and paired-random vocabulary is frozen to
{-1, -0.5, -0.25, 0, 0.25, 0.5, 1}, with sorted parameter names and row-major
positions. Existing smoothing's 1/9 and 1/36 coefficients are explicitly a fixed
reference exception. No gradient, authored Python or new numeric author proposal
was used.

The future protocol is design only: at most two author calls, one founder and
one mutation conditional on a valid founder with failures. Every consumed attempt,
including invalid, refused or neutral responses, charges 1,488 positions to both
the author and its full-random counterpart (maximum 2,976 per arm). An invalid or
refused submission consumes its attempt and stops without retry or repair.

A positive feasibility claim requires strictly more exact demonstrations than
the best fixed reference and every consumed paired random, preserving selected
baseline solves. A mutation claim additionally requires strict improvement over
the founder and no regression of founder-solved demos, with saved lineage.
Founder packets exclude random tensors/scores and baseline selection outcomes;
mutation feedback is restricted to deterministic training failures. No such
packet was dispatched here. No crossover, admission, ARC benchmark, generality
or ONE_STANDARD claim is made.

No perfect-baseline, invalid-control, source-binding or restricted-panel
structural-contradiction stop condition was reached. Whether to authorize a
separate bounded author phase is now a parent decision. Baseline success is not
author dispatch permission. Existing campaigns and STOPs remain closed.

## Evidence and verification

Run: `runs/weights-nca-feasibility-20261008/`

| Artifact | SHA256 |
| --- | --- |
| `MANIFEST.json` | `11d371121214c1bd0caeaa2c78af0ebc92affab9f5f70c09728c6a6240373326` |
| `BASELINE_RESULT.json` | `c2bc1ca4841dca1d9f9999e6bfb7bab825eca362f3b805e7713c63bcdcdd2072` |
| `REPLAY_RESULT.json` | `2abe7a62ff7ac3a735a07d4f456fbc375733ca506c641d097c04ff5dd43cb4ff` |
| `scripts/nca_feasibility_preflight.py` | `c5c3ea59e389f100693365293edadfe5f5fd1d527f8247917fc2173b15e704c7` |
| `tests/test_nca_feasibility_preflight.py` | `47ed5e11ee023d8b6054ef83ca6912b08399f64aaabea75ee4abac4b16ba23b9` |

New checker suite: `python -m unittest tests.test_nca_feasibility_preflight -v`,
18 passed; independent reviewer also reran all 18 successfully. Frozen manifest
verification passed. Replay reports all four arm matches and aggregate match,
`passed=true`. Only timing fields are omitted from semantic equality. Artifacts
also include per-tensor hashes, graph hashes, source snapshots, immutable input
provenance, complete coverage, process logs and `STANDARD.json` honestly recording
zero authors and no scientific acceptance.

The previously required broader qualification remains **red/incomplete**. It was
not rerun for this bounded baseline and no passing full-standard claim is made.
No existing template, protected test/floor, historical run, server, local desktop,
GitHub branch or publication was changed by this preflight.
