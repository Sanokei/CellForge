# Existing spatial candidate: ray casting followed by whole-grid tiling

The practical next candidate is search task **`f5b8619d`**, using the existing
typed chain **`ray_cast` -> `tile_orient`**, with **167 literal positions**.
The selection and candidate packet are frozen; no author was dispatched and
no query, development or report predictions were made during this review.
This can remain parked while the parent prioritizes the separate belief trial.

## Why this pair

The four previously frozen search tasks were inspected using training
demonstrations only. Task and template selection did not use development or
report outcomes. The original first-task choice had not established that
`assoc_induct` could express the demonstrated rule; the new plan requires a
source-level compatibility argument and a zero-control check first.

| Search task | Existing candidate | Finding |
|---|---|---|
| `1a07d186` | `snap_to_line`, 12 positions | Exactly matches all 3 demonstrations with **zero tensors**. Useful host baseline, unsuitable for claiming authored weights cause the success. |
| `bd283c4a` | No complete match established | No candidate was selected or searched in this bounded review. |
| `3194b014` | No complete match established | No candidate was selected or searched in this bounded review. |
| `f5b8619d` | `ray_cast` -> `tile_orient`, 167 positions | One source-derived hand-coded mechanism probe matches **3/3 demonstrations**; the same chain with zero tensors matches **0/3**. |

The selected demonstrations fill background pixels in columns containing a
nonzero source with color 8, preserve source pixels, and repeat the resulting
whole grid in two rows and two columns. Ray casting can perform the first
operation; `tile_orient` can perform the repetition. Ordinary nearest-neighbor
scaling would enlarge pixels rather than repeat the whole grid, so it is not
an interchangeable smaller second component.

The probes are explicitly **host-constructed feasibility baselines**, not
native model proposals, population members, transfer results or scientific
qualification. There was no parameter enumeration or random-control search.
All nine demonstration predictions completed without runtime errors. A second
process reproduced the saved predictions and independently counted every
literal leaf scalar. Its audit guard allowed only the four search task files.

## Small existing interface and honest boundary

The smallest directly matching interface found for the original task is
`snap_to_line`'s grid-to-grid interface, controlled by `bg[10]` and `orphan[2]`.
Its host algorithm already detects full colored lines, attracts same-colored
pixels and handles orphan removal. The zero tensor solution is why the next
weight-writing trial should not use it.

For the selected task, use `skillcore.typed_chain_cell` with the existing
versioned grid-token interfaces; no new coordinator, specialist architecture,
executor or neural layer is needed. The proposed author writes:

- `ray_cast`: `src[10]`, `dirs[8]`, `stop[3]`, `paint[10]` = **31** positions.
- `tile_orient`: `per[4]`, `pec[4]`, `orient[4,4,8]` = **136** positions.

Host code implements propagation, blocking, geometric transformations and tile
assembly. Tensor signs/argmaxes control sources, directions, stop policy,
paint, repetition counts and orientations. These are mostly choices among
fixed host behaviors; the trial would demonstrate authored literal controls
and typed composition, **not learned spatial algorithms or a capable chatbot**.
All 136 tiling positions remain charged even when some are inactive.

## Frozen candidate and next bounded trial

Ready for parent review, not automatic execution:

- `runs/weights-native-spatial-plan-20261006-fix1/AUTHORING_PACKET_CANDIDATE.json`
- `runs/weights-native-spatial-plan-20261006-fix1/RESPONSE_SCHEMA_CANDIDATE.json`
- `runs/weights-native-spatial-plan-20261006-fix1/SELECTION.json`
- `runs/weights-native-spatial-plan-20261006-fix1/PACKET_MANIFEST.json`

The packet contains only the selected task's three demonstrations and the two
template semantics/layouts. It contains no hand-coded baseline tensors,
query labels, previous selection scores or report information. The proposed
response is one JSON array element with `task_id`, complete nested `params`
for `ray_cast` and `tile_orient`, and a bounded rationale. The registered
literal values remain finite and bounded to [-9,9].

Selection hash:
`177b89f5b495120e48c94d019a3ae0c0a0d1cd823db43027aa1723b8041eada8`.
Packet SHA256:
`ede26dc1d8f95a3d3c65c04ef5f47986efb69ee406df371196f6a35e332b65a5`.
Schema SHA256:
`2792127426ad8e6444cc61da240f75462c4bf6fdb9184da59f8e9a81efa99fde`.

If prioritized, the suggested next tranche is **one native author call**,
followed by strict validation of both leaf tensors and deterministic typed
composition. Charge **167 candidate and 167 random positions**, even for an
invalid/refused response, and draw a fresh full-shape matched random control.
Retain zero and the already exposed hand-coded mechanism as diagnostic
baselines. Use the unchanged search/development panels, with outputs stripped
before prediction; do not present the development panel as newly untouched
evidence after previous evaluations. Keep the report partition unavailable.
There is no automatic retry if the candidate fails or ties controls.

Current model-position totals remain 204 per arm; that hypothetical next
attempt would bring them to 371 per arm. Credit usage is unknown and must
remain unknown without a real meter. Local demonstration checks took 0.516
seconds and replay 0.359 seconds, under 30-second timeouts and one compute
thread. Any model call remains a parent-controlled allowance.

## Blocker and preserved failed check

The current native importer accepts only `assoc_induct`; before dispatch, it
needs a narrowly bounded two-leaf schema/identity/accounting path. Do not submit
this packet through the old importer or relabel it as the old protocol.

The first planning script correctly stopped on an accounting assertion:
`skillcore.positions(chain)` returns zero under its inherited-assembly
convention. That is not a valid charge for authoring both constituent cells.
The partial directory `runs/weights-native-spatial-plan-20261006/` retains its
failure, source and invalid zero-position diagnostic field. The corrected
directory sums all leaf positions, independently verified as 31 + 136 = 167.
The selection hash is identical before and after that accounting correction;
no task, structure or outcome was tuned in response. Existing source and run
evidence were not rewritten.

Reproduction source: `reports/spatial-candidate-review-20261006.py` (creation
refuses an existing destination). Check logs:
`reports/spatial-candidate-review-fix1-20261006.log` and
`reports/spatial-candidate-replay-20261006.log`. No server changes, deployments,
commits, model calls, gradients or downloads occurred.
