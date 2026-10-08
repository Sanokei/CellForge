# NCA locality lower bounds from exposed demonstrations

## Result

**At least four local updates are necessary to reproduce the compatible panel
exactly.** In particular, task `1a07d186` cannot be reproduced exactly with three
or fewer shared masked 3x3 updates, regardless of numerical weights or hidden
width. The existing T8 architecture is not ruled out by this test; its capacity
remains unknown. This does not explain the observed founder identity response.

The original eleven-task / 32-demonstration inventory remains intact. Five tasks
and fifteen demonstrations remain shape-incompatible. This diagnostic analyzes
all 2,061 output pixels in the other six tasks / seventeen demonstrations.
Only frozen, hash-verified derived training pairs were read. No queries, report
outputs, new data, model calls, proposed weights or model predictions were used.

| Updates/radius | Distinct signatures | Repeated classes | Contradictory classes | Unavoidable wrong pixels, lower bound |
|---:|---:|---:|---:|---:|
| 0 | 10 | 10 | 10 | 372 |
| 1 | 755 | 224 | 54 | 112 |
| 2 | 1662 | 91 | 7 | 7 |
| 3 | 1956 | 34 | 2 | 2 |
| 4 | 2030 | 12 | 0 | 0 |
| 5 | 2055 | 2 | 0 | 0 |
| 6 | 2060 | 1 | 0 | 0 |
| 7 | 2061 | 0 | 0 | 0 |
| 8 | 2061 | 0 | 0 | 0 |

A zero lower bound means this check proves no positive lower bound. It does not
mean zero errors are achievable. At radii seven and eight every signature is
unique, so label consistency is vacuous, not positive capacity evidence.

The first radius with no within-task contradiction is 4 for `1a07d186`; 2 for
`712bf12e`, `984d8a3e`, `bd283c4a`, and `ef135b50`; and 1 for `ed36ccf7`.
These are lower bounds on necessary update radius, not sufficient architectures.

## Exact obstruction witnesses

Coordinates and demonstration indices are zero-based.

- `1a07d186`, demonstration 2: positions `(y=6,x=3)` and `(y=6,x=4)` have
  identical radius-three input-plus-mask neighborhoods, but require colors 8
  and 0 respectively. Neighborhood SHA256:
  `85f6e22ccc58cf4888903d1b0574ad03251b03005ae2d5d5da6b50707ffc1fe7`.
- The same task, demonstration 1: `(y=3,x=11)` and `(y=11,x=11)` have identical
  radius-three neighborhoods but require 4 and 0. SHA256:
  `4538e8d33389d88da448680fcd7545bef7e766b6a9fbbd0da27aff4d1a6ff497`.

Thus even treating these tasks independently cannot remove the obstruction.
A separate radius-two witness spans tasks `1a07d186` and `ef135b50`; the JSON
artifact preserves it. Shared weights impose cross-task consistency as well.

## Why the bound is rigorous

The frozen `nca_gene.py` builds each update from a 3x3 neighborhood, pointwise
shared affine/ReLU/affine computation, and the fixed input mask. There are no
coordinates, task IDs, demonstration-dependent tensors or global reductions in
that transition. The output decoder is pointwise color argmax.

At time zero, a pixel's state is its color one-hot with two zero channels.
Outside the original input rectangle all twelve channels are zero. The checker
encodes outside as -1, distinct from color 0: black inside the rectangle has a
nonzero one-hot channel and cannot be confused with padding.

Induction: suppose two centers have identical radius-r input/mask squares.
Each corresponding neighboring center has identical radius-(r-1) squares.
Their states after r-1 updates are therefore identical. The same ordered
pointwise arithmetic and target mask produce identical r-step states, hence
identical argmax colors. This remains true across demonstrations and tasks
because the same weights and transition are used everywhere. It holds for
finite valid floating-point executions as well as exact arithmetic: corresponding
operations and inputs are identical, not merely numerically close.

The full mask is included conservatively. Frozen `op_shift` uses constant-zero
padding, not wraparound. Extending the canvas with identically zero masked cells
produces the same observable states at valid input pixels; the outer 30x30
canvas boundary supplies no additional signal. Height/width output ports do not
enter color-state evolution. Translated colors *without matching mask geometry*
are not assumed equivalent.

For each identical-signature class, any deterministic local model must emit one
color. If the required color counts are n_c, at least sum(n_c)-max(n_c) pixels
must be wrong. Summing over disjoint classes gives the reported lower bound.
This relaxes the actual NCA to an arbitrary local lookup function. This
relaxation can only weaken the obstruction; the reported lower bound cannot
exceed the true minimum error. It does not bound achievable accuracy tightly
or prove sufficiency when zero.

## Consequence for the next study

- Do not switch to the existing T2 caller as a supposedly equivalent cheap
  version: it is provably incapable of exact completion of `1a07d186` here.
- Four steps are necessary for this panel but not known sufficient. The result
  gives no justification to expand T8, add channels, or reopen authorship.
- The closed tranche's founder null and mutation timeout do not prove a T8
  expressivity failure. They remain authoring/operational observations.
- A prospective explicit-objective experiment may still test authoring quality;
  its result must not be described as architecture-capacity certification.

## Artifacts and verification

- `scripts/nca_equivariance_diagnostic.py`: additive, deterministic radius sweep.
- `tests/test_nca_equivariance_diagnostic.py`: six synthetic fixture checks covering
  padding distinction, translated conflicts, cross-task conflicts, consistent
  repeated neighborhoods, full exclusion accounting and query-field rejection.
- `reports/NCA_LOCALITY_LOWER_BOUNDS_20261008.json`: all radii, witnesses, per-task
  bounds, source hash and the frozen inventory/materialization/input hashes.

Commands:

```
python -m unittest tests.test_nca_equivariance_diagnostic -v
python -m scripts.nca_equivariance_diagnostic --out reports/NCA_LOCALITY_LOWER_BOUNDS_20261008.json
```

Six tests passed in 0.001 seconds. The output path uses exclusive creation.
Source comparison confirmed current `nca_gene.py` and `runtime/ops.py` match the
frozen preflight copies. No frozen run, tensor, template, protected gate or
historical result was changed. These are mathematical diagnostics, not a new
scientific run, inference benchmark, or full-standard pass.
