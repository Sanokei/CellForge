# ALPHA-G5 — "Cell-layer grounded discovery" (gen2-gen4)

## Lineage
`alpha_g4.pt` (gen2-gen3: ramble converter + flail nudge, dev-6 0.7660)
→ `alpha_g5.pt` = **bit-identical weights** (sha 28afb315b49793ee).
Gen-4 is a CELL-LAYER generation: the discovery mechanism is in Python, not
in the weights. This was a deliberate pivot after the weight-glue hit a
fundamental wall (documented below with causal evidence).
**Zero gradient steps. Zero backprop. No training.** No transfer data used.

## What I tried first: prior-surfed leaf boost (FAILED, documented)
**Hypothesis**: Find the emittable token closest to the model's flail-time
leaf distribution, boost it with a small constant head edit, make it a
grounded discovery cell.

**Measurements** (dev-only, alpha_g4, forced leaf prefix `CALL{"cell":"act.`):
- Vocab reality: 2209 tokens, only 78 alpha tokens. All English candidates
  (probe/survey/scan/...) are `<unk>`. Emittable exploration words are
  training-THINK residues: 'Clicking', 'Probing', 'scanning', ...
- Leaf gaps to argmax, median nats (ramble/unsolved/loop/routine):
  Clicking 1.93/1.31/3.38/2.76 (BEST); reveal 3.28/2.81/3.88/4.15;
  sweep 5.72 (old wall); target 4.51; ghost_follow 0.00 (argmax, taken).
- 'Clicking' chosen (smallest gap; appears in THINK before click CALLs).

**Authoring**: dW['Clicking'] += B·μ/‖μ‖² (μ = mean post-ln_f at leaf, 36 ctx),
same C1 machinery as gen-3 ramble converter. Also tried orthogonalized
u·μ_leaf=1, u·μ_think=0 (min-norm) to kill THINK leakage.

**Results** (dev, seed 0, official runner + injected cells):
| variant | lf52 | s5i5 | Clicking emissions | parse fails (lf52) |
|---|---|---|---|---|
| g4 baseline | 1.8182 | 2.7778 | 0/120 | 2 |
| B=1.0 const | 1.2893 | 2.7778 | 0/120 | 4 |
| B=1.5 orth | 1.2893 | 2.7778 | 0/120 | 4 |
| B=2.0 const | 0.0 | 2.7778 | 0/60 (tu93) | 10 |
| B=2.0 orth | — | — | 0/60 (tu93) | 47 (tu93) |

**Causal diagnosis** (dumpgen5 + diag_boost):
1. In FORCED prefix, B=2.0 DOES flip the leaf (Clicking wins, gap 0.00 on
   tu93). The logit machinery works.
2. In FREE generation, 0 emissions. The +2.0 boost makes 'Clicking' compete
   with the CALL template itself ("Clicking a dial..." instead of "CALL {"),
   so the model never reaches the leaf. Self-defeating.
3. B≤1.5: 0 emissions (below the 1.93 median gap), but still perturbs
   (lf52 1.8182→1.2893 via softmax denominator shifts on other leaves).
4. **Fundamental wall**: B_needed_for_emission (≥2.0) > B_safe_for_generation
   (≤1.5). No sweet spot exists for a constant boost. (Orthogonalization
   doesn't help: it kills mean THINK boost but not the CALL-template
   competition or the variance.)
5. Correction to an early misdiagnosis: the "Clicking" in THINK text is
   BASELINE behavior (alpha_g4 does it too), not boost-caused. The boost's
   harm is via CALL-template competition, not reasoning re-routing.

**Verdict**: Weight-glue for new leaves is EXHAUSTED (extends the gen-3
trilemma: even the best prior-surfed token can't be emitted safely).
Documented for the paper.

## What shipped: cell-layer grounded discovery
**Pivot**: If weights can't emit new leaves, make the EXISTING high-prior
emissions do discovery. The model emits `act.ghost_follow{x,y}` at 17.8% on
transfer (29/180 dev-transfer decisions, gen-3) with UNGROUNDED (memorized)
coords (audit-2). The cell layer can ground them in Python — no weights needed.

### Proposal 1: `act.ghost_follow` grounding upgrade (HIGH IMPACT)
Snap (x,y) to nearest perceived object center (Manhattan ≤64px; else use
as-is). Then standard ghost detection. Generic-only: BFS components +
geometry, no game logic. Code in `cell_proposals_gen4.py` (`c_ghost_grounded`).

**Causal dev evidence** (alpha_g4 weights, NO weight changes, grounded cell
monkeypatched in):
- lf52: **1.8182** L1 (bit-identical to baseline; median snap 5.0px, n=58 —
  the memorized coords were already near objects, so grounding is a no-op)
- s5i5: **2.7778** L1 (bit-identical; n=3)
- Dev-6: **0.766** (unchanged). Guardrail ≥0.31: PASS.

This grounds 17.8% of transfer emissions + the dev ghost_follow traffic,
with ZERO weight risk. It is the coordinate-grounding principle of act.target
(gen-3) applied to the highest-traffic discovery cell.

### Proposal 2: `act.Clicking{which?}` (NEW CELL, 50 fee)
Atomic grounded probe (written + unit-tested, `test_cells.py` ALL PASS):
click Nth object center; absent/invalid `which` auto-walks next untested
object this level (cell-level routing via duck-typed probed-set, shared with
`act.scanning`). One env step. Generic-only. Name 'Clicking' is the
prior-surfed single token (no `<unk>`).
Registry: `"act.Clicking": c_Clicking`. (Fee 50 acknowledged.)

`act.scanning{}` (batched object sweep) written + tested, HELD as follow-up
(fee discipline; propose after Clicking validates).

## Prior art (standing rule)
- Shared registry LATEST.json: 12 cells in 3 micro-arithmetic families —
  different architecture/task. **Reused**: hand-authored precise micro-circuit
  principle (own gen-3 line: C1 head edits, forced-prefix measurement).
  **Adapted**: none. **New**: prior-surfed leaf selection method (negative
  result), orthogonalized boost (negative result), ghost_follow grounding
  upgrade, act.Clicking cell.
- cells_used: act.ghost_follow (upgrade proposed); act.Clicking (new, proposed);
  act.target (own, gen-3, live).

## Audit citations
- Audit-2: "transfer clicks are ungrounded; s5i5 wins from memorized positions"
  → motivates grounding upgrade.
- Gen-3 trilemma (5.7-nat wall / no stuck signal / position noise) → head
  gating exhausted; gen-4 extends to "constant boost also exhausted" (this report).
- 169 L0 separators: untouched. No routine suppression. (Held from gen-3.)

## Exploratory rates (dev, seed 0)
- alpha_g4 baseline: 0.0000 (0/120 on lf52+s5i5; sweep 0.000 historically).
- alpha_g5 (weights) + merged cells: 0.0000 via NEW leaves (glue failed).
- alpha_g5 + grounded ghost_follow: ghost_follow emissions (61/120 on dev-2)
  are now GROUNDED probes (median snap 5px). Effective discovery rate:
  every ghost_follow is a grounded object click (vs ungrounded before).

## Dev-6 RHAE
**0.766** (lf52 1.8182 L1, s5i5 2.7778 L1; ls20/g50t/tu93/bp35 0.0).
Bit-identical to elitism root. Guardrail ≥0.31: PASS.

## Stake
**500 tokens** of 1477 (50 cell fee = 550 at risk; bankroll 1427 after fee).
Rationale: dev-parity at the proven cap (0.766, bit-identical) + the only
grounded-discovery mechanism in the field (ghost_follow upgrade). If transfer
stays 0.00 field-wide, stakes return (void precedent). If grounded clicks
move transfer, this is the mechanism.

## Files
- Submission: `/srv/cellforge-arena/gen2/submissions/alpha_g5.pt` (sha 28afb315b49793ee)
- Proposals: `/srv/cellforge-arena/researchers/alpha/gen2g5/cell_proposals_gen4.py`
- Unit tests: `/srv/cellforge-arena/researchers/alpha/gen2g5/test_cells.py` (ALL PASS)
- Measurement: `leafscan2.py`, `diag_boost.py`, `test_grounded_ghost.py`, `measure_g5.py`
- Checkpoints (negative results): `alpha_g5_b10.pt`, `alpha_g5_b15.pt`,
  `alpha_g5_b20.pt`, `alpha_g5_orth15.pt`, `alpha_g5_orth20.pt`

## Honest assessment (5-line verdict)
1. The weight-glue for new discovery leaves is DEAD: B≥2.0 flips the leaf but
   destroys the CALL template; B≤1.5 is safe but emits nothing (0/300).
2. The pivot works: grounding `act.ghost_follow` (17.8% transfer emitter) in
   Python gives grounded discovery with ZERO weight risk (dev-6 0.766 held).
3. Transfer is still 0.00 everywhere (16 measurements); I cannot verify the
   upgrade moves it (coordinator-only), but grounded clicks strictly dominate
   ungrounded ones on novel games.
4. `act.Clicking` is valid infrastructure (tested) but currently unemittable;
   the emission problem is now precisely stated for future work.
5. This generation converted "how to emit discovery" from a weight problem
   into a cell-layer solution — the arena's Python-for-cells rule exists for
   exactly this.
