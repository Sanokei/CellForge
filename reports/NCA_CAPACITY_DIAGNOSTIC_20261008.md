# Existing NCA: frozen training-only capacity diagnostic

## Decision

The unchanged NCA cannot reproduce the entire frozen panel: **five of eleven
tasks require output dimensions that it cannot emit**, across all fifteen of
those tasks' demonstrations. Do not spend author calls trying to overcome that
structural impossibility with numerical edits.

The other six tasks are **not ruled out**, not established feasible. Every one
of their 2,061 radius-eight input-and-mask neighborhoods is unique. Therefore
the absence of contradictory repeated neighborhoods is vacuous on this panel;
it supplies no positive capacity evidence. No NCA candidate was executed.

This is a necessary-condition diagnostic of public-training demonstrations,
not a new scored weight experiment, capability result, or authorization to
dispatch authors. Author calls and weight proposals remain zero.

## Scope frozen before compatibility inspection

The development owner froze `nca-capacity-20261008/INPUT_INVENTORY.json` before
retrieving packet contents, at repository base
`02beaf18251fca5ed1f1eb55e92f00a3e3820e88`. The panel comprises the eleven completed
direct-author public-training task packets from iteration01 and batches02–04.
Its inventory SHA-256 is
`14ea47a0fbeda01fcff1b39087994175a3b6b6de399df50610e5f01f66110c33`.

The inventory confirmation preceded materialization in the coordination record:
04:51:10 UTC freeze confirmation, 04:52:35 UTC demonstration-only ready notice,
on October 8, 2026. These times are coordination evidence, not a separately
trusted timestamping service. No compatibility-based selection preceded freeze.

Each original packet's Git blob was checked by the materializing owner against
the pinned source tree. `MATERIALIZATION.json` retains packet paths, Git blob
SHA-1, packet SHA-256, derived-file SHA-256 and the transformation. Only training
demonstrations were exported as integer grids. Query inputs and instructions
were stripped; no query/test outputs were present in those author packets.
The diagnostic and independent reviewer read only the derived demo-only files.

The loader independently checks inventory hash, source commit, exact task
membership, provenance references, derived hashes, exact allowed keys, grid
shape/color validity and task identity before applying any compatibility test.
All eleven inventory rows matched the separately available, nontruncated pinned
Git tree metadata. The reconstructed local checkout does not itself have that
base commit as a tree object; original-packet integrity therefore relies on the
connector materialization receipts, not a local `git show` reconstruction.

### Exclusions and exposure accounting

- `e7639916`: cancelled author slot, excluded before content access; not retried,
  replaced, diagnosed, or assigned a synthetic failure.
- iteration01 `AUTHOR_PROGRAM.json` and `AUTHOR_WEIGHTS.json`: alternative
  author routes outside the direct-grid packet inventory. The original task's
  `AUTHOR_DIRECT_GRID.json` is the one selected provenance source; these two
  alternative packets were not opened for this diagnostic.
- Earlier native-agent/spatial packets: duplicate task provenance outside this
  direct-packet inventory; not an additional selected capability panel.
- Other raw task files, unknown-exposure task material, report/private/evaluation
  data and response/result files: excluded. No new official downloads occurred.
- Shape-incompatible demonstrations were **not removed from the panel**. They
  remain explicit failures below. Local-neighborhood color checks are inapplicable
  to their differently shaped targets and are not reported as passes.

All eleven tasks already have known development exposure. Four belong to the
earlier exposed search panel; seven were newly selected for completed direct
trials and are now exposed. This does not establish their absence from model
pretraining or remote/nontext history. None can be relabeled untouched evaluation.

## Source-derived necessary conditions

`src/cellforge/construction/nca_gene.py:39–59` constructs a shared local update:
nine 12-channel neighbors, twelve hidden affine rows, ReLU, twelve output affine
rows, then the fixed input mask. State begins with ten color one-hots and two
zero auxiliary channels. The proposed configuration is fixed T=8.

1. **Same dimensions:** height and width are passed unchanged from the input
   (`nca_gene.py:59–64`; `runtime/grid.py:125–128`). Any size-changing training
   pair is impossible for every tensor assignment.
2. **Locality:** after eight local steps, each output cell depends on at most
   its radius-eight Chebyshev input/mask neighborhood. Two equal neighborhoods
   must yield equal output colors under the shared weights. The checker encodes
   a 17×17 square, with -1 for outside-grid cells and 0–9 for valid colors;
   outside padding is not conflated with black. This square is a conservative
   dependency superset, so lack of a collision cannot prove capacity.
3. **No demonstration adaptation:** the root ignores demonstration labels,
   phase and index, and passes through empty state (`nca_gene.py:19,63–64`).
   Identical complete inputs cannot require different outputs. The checker
   checks this separately across every demonstration, including size changes.

The checker constructs no weights and does not import or execute NCA inference.
Its analytical pixel-error lower bound comes from conflicting required labels
within equivalent neighborhoods, not measured predictions or a fitness score.

## Whole frozen panel

| Task | Demos | Dimension condition | Within-task locality conflicts | Necessary-condition status |
|---|---:|---|---:|---|
| 0c786b71 | 3 | All 3 mismatch: 3×4 → 6×8 | Not applicable | Ruled out |
| 1a07d186 | 3 | Same size | 0 | Not ruled out |
| 3194b014 | 3 | All 3 mismatch: 20×20 → 3×3 | Not applicable | Ruled out |
| 46442a0e | 3 | All 3 mismatch: 2×2 → 4×4; twice 3×3 → 6×6 | Not applicable | Ruled out |
| 712bf12e | 3 | Same size | 0 | Not ruled out |
| 984d8a3e | 2 | Same size | 0 | Not ruled out |
| bd283c4a | 2 | Same size | 0 | Not ruled out |
| c8cbb738 | 3 | All 3 mismatch: 12×11 → 5×5; 10×8 → 3×3; 12×14 → 5×5 | Not applicable | Ruled out |
| ed36ccf7 | 4 | Same size | 0 | Not ruled out |
| ef135b50 | 3 | Same size | 0 | Not ruled out |
| f5b8619d | 3 | All 3 mismatch: 6×6 → 12×12; 3×3 → 6×6; 2×2 → 4×4 | Not applicable | Ruled out |

Totals: eleven tasks, 32 demonstrations, 2,575 target pixels. Seventeen same-size
demonstrations across six tasks contain 2,061 target pixels. Their 2,061 unique
neighborhood classes have maximum multiplicity one: zero repeated classes and
zero classes shared across tasks. There are zero within/across-task neighborhood
conflicts and zero identical-whole-input/different-output conflicts. The
neighborhood-derived unavoidable-error lower bound is consequently zero and
uninformative. These counts were independently recomputed.

At most six of these eleven tasks can have **all training demonstrations**
reconstructed by this unchanged NCA. This is not an upper bound on unseen query
accuracy, and the diagnostic never consulted query labels.

## What this does and does not permit next

The exact same-size candidate set is `{1a07d186, 712bf12e, 984d8a3e, bd283c4a,
ed36ccf7, ef135b50}`. It is reported as the complete surviving necessary-condition
set, not selected as a successful experiment or an untouched benchmark. The
full eleven-task negative remains the primary result.

No further author tranche is justified merely by these zero-conflict counts.
A later bounded weight trial would still require a specified shared behavior,
an explicitly frozen graph/caller, independent source/control admission checks,
valid matched random controls, exact training metrics, a scientific acceptance
criterion, a bounded authorized dispatch budget, and a separately approved
unopened final evaluation plan. None were established by this diagnostic.

The historical caller cannot silently supply that protocol:
`scripts/evolve_nca_gene.py:48` builds T=2, and `nca_gene.py:76–89` permits
activation and wiring mutation. Its hand-coded founders/random-edit search are
not literal LLM founder/mutation evidence. Proposed T=8 must be explicit.

The fresh fixed graph has 1,488 stored scalar positions, but three hidden biases
per row enter through their sum before one ReLU, yielding 24 redundant
directions in real arithmetic. Stored positions remain chargeable; they are
not 1,488 independent or individually prediction-sensitive degrees of freedom.
`nca_genome_contract` only checks slot widths and does not freeze graph structure.

## Verification and artifacts

- `scripts/nca_capacity_diagnostic.py`: additive pure diagnostic and fail-closed
  frozen-panel loader.
- `tests/test_nca_capacity_diagnostic.py`: ten isolated checker regressions.
  These fixtures are not synthetic capability evidence.
- `nca-capacity-20261008/INPUT_INVENTORY.json`: immutable pre-content panel.
- `nca-capacity-20261008/MATERIALIZATION.json`: immutable derivation provenance.
- `nca-capacity-20261008/demonstrations/`: eleven immutable demo-only inputs.
- `nca-capacity-20261008/RESULT.json`: every task, mismatch, conflict list and
  source/data hash; no failed task omitted.
- `nca-capacity-20261008/checker-tests-final.log`: 10 tests, 0.023 seconds, OK.

Reproduction:

```sh
python -m unittest tests.test_nca_capacity_diagnostic -v
python scripts/nca_capacity_diagnostic.py \
  --inventory reports/nca-capacity-20261008/INPUT_INVENTORY.json \
  --materialization reports/nca-capacity-20261008/MATERIALIZATION.json \
  --out /tmp/nca-capacity-independent.json
```

The output destination must be fresh. RESULT SHA-256:
`6972a1d3d3bf129de1c23c424d41aa4180cb072d3c31d9ae9cf335f68a34890c`.
Diagnostic source SHA-256:
`3d74659262e9ff9c853ed9d797f0183a11f5f34c96f7efb776c7b021ec3c4139`.

Independent reviewer verified all eleven derived hashes/membership, the pinned
inventory metadata and every analytical result field. No checker mathematical
defect was identified. No protected tests, existing templates, floors, historical
receipts or cached manuscript payload were changed. Existing full-standard
qualification remains red/incomplete; these ten tests do not replace it.
