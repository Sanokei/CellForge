# ALPHA-G4 — "Ramble Converter + Flail Nudge" (gen2-gen3)

## Lineage
`alpha_g3.pt` (gen2-gen2: bred_g1_a25 + C1 CALL-template hardening, dev-6 0.766)
→ + discovery-robustness edits (pure tensors, lm_head only) → `alpha_g4.pt`.
**Zero gradient steps. Zero backprop. No training.** No transfer data used in
design (dev-only measurements + shared audit-2 findings). 1,378,816 params,
plain GPT (4L/4H/d128/v2209) — no architectural growth (see "why no gate").

- sha256 (first 16): `28afb315b49793ee`
- File: `/srv/cellforge-arena/gen2/submissions/alpha_g4.pt` (5,526,943 bytes)

Label map: DEV-1..DEV-6 = six dev games in judge order (lf52, s5i5, ls20,
g50t, tu93, bp35). TX = transfer games, measure-only, seed 0.

## What I found (probes)
- **True CALL bytes**: `CALL { " cell " : " act . <leaf> " , " args " : ... }`
  (tokenizer strips whitespace; verified by dumping raw generations).
- **Vocab ceiling confirmed**: `perceive.*` → `<unk>`, `act.key_seq` → `<unk>`.
  7/12 cells emittable. Of those, only `act.sweep` is a systematic probe —
  emitted **0 times in 360 decisions** (dev-3 + transfer-3).
- **The 5.7-nat wall**: in `act.` context, `ghost_follow` leads `sweep` by
  median 5.70 nats (dev; 5.48 on dev-flail). Sweep's prior is ~zero (absent
  from all 37K training traces).
- **Transfer baseline (alpha_g3, measure-only, lp85/vc33/ar25)**: 47% parse
  fails (rambles), 17.8% `ghost_follow` (collapsed, memorized coords),
  0% sweep. Exploratory (sweep+ghost_follow) = 0.178.
- **Why no strong gate** (three independent blockers, all measured):
  1. No clean stuck signal: best LDA (clean solve-vs-flail labels) gives
     sep_ratio 4.9 — overlapping distributions; no threshold separates.
     (Hard-threshold Block-5 attempt aborted: dead_max > fire_med.)
  2. Position noise: gate direction gives |v·r| up to 4.3 at non-leaf
     positions vs 0.35 signal at flail-leaf. k=12 broke generation outright
     ("sweep sweep sweep..." — diagnosed via generation dump). Max safe
     k = 0.54 → flail boost 0.19 nats. A linear head cannot do a clean
     position-gated boost; the signal-to-position-noise ratio is 0.31.
  3. Block-5 ReLU: same overlap problem — no τ is both dev-dead
     (needs pre-act ≤ −2 for gelu deadness) and flail-live.
- **169 L0 separators**: untouched (load-bearing, per audit-2). No routine
  suppression (dead end). No residual-stream writes.

## The mechanism (shipped)
Two small, dev-safe lm_head edits (only the `CALL` and `sweep` rows; no
other token's relative ordering changes; ‖dW‖/‖W‖ = 0.042, C1-scale):
1. **Ramble converter** (primary): +1.0 nat constant boost on the `CALL`
   token (id 32), via u=μ/‖μ‖² (μ = dev-mean post-ln_f residual), so boost
   ≈ +1 nat wherever r≈μ. Converts transfer rambles into well-formed calls.
   On dev the model already emits CALLs reliably — verified bit-identical.
2. **Flail nudge** (secondary): +k·(v_orth·r) on the `sweep` leaf token,
   v_orth = STEP-LDA direction orthogonalized vs dev mean (e_mean=0.0000),
   k=1.378 for 0.15 nat dev noise. Weak (0.26 nat flail boost) but strictly
   flail-graded and provably dev-safe. A foundation, not a flip.

## Self-check (official fixed_runner.py, seed 0, maxdec 60)
- dev-3 (lf52,s5i5,tu93): lf52 1.8182 L1/18acts/2fails, s5i5 2.7778 L1/13acts,
  tu93 0.0 — **both L1 solves bit-identical to alpha_g3**.
- dev-6: **0.7660** (lf52 1.8182, s5i5 2.7778, ls20/g50t/tu93/bp35 0.0) —
  matches elitism root exactly. Guardrail ≥0.31: **PASS**.
- Transfer (measure-only, lp85/vc33/ar25): parse_fail **0.467 → 0.428**
  (ramble converter works — fewer wasted decisions); exploratory
  (sweep+ghost_follow) 0.178 → 0.161 (sweep still 0.000; ghost_follow 32→29,
  repeat_click 64→73 — the new calls are routine, not discovery).

## Coordinate grounding (objective 2)
**Cell proposal** `act.target{which}` — submitted to coordinator for
audit/merge (fee 50 acknowledged, non-refundable):
- Code: `/srv/cellforge-arena/researchers/alpha/gen3/cell_act_target_proposal.py`
- Clicks the center of the Nth perceived object (area-descending, same order
  as the STATE OBJ list). Model emits a small int; cell resolves LIVE frame
  coordinates at call time. Generic, no per-game logic.
- Name verified vocab-emittable: `act.target` → ['act','.','target'], no
  `<unk>`; `which` is a small int (vocab-safe digits).
- NOT emitted by alpha_g4 (not yet in registry — emitting it now would
  parse-fail). Follow-up weights will target it once merged.
- Prior art (standing rule): shared registry grepped — 12 live cells in 3
  micro-arithmetic families (different architecture/task); no direct weight
  reuse. **Reused**: hand-authored precise micro-circuit principle.
  **Adapted**: none. **New**: ramble converter, flail nudge, act.target cell.
- cells_used: none (no registered cells invoked ≥5×; act.target proposed).

## Honest assessment
The weights cannot meaningfully raise the sweep rate — the 5.7-nat prior,
the absent stuck signal, and position noise form a trilemma (all measured,
not assumed). What shipped is the strongest SAFE mechanism: the ramble
converter turns wasted decisions into actions (parse fails 47%→43% on
transfer), and the flail nudge is a provably-safe foundation. Genuine
discovery emission needs either (a) the `act.target` cell merged + weights
that emit it, or (b) discovery traces in training — both beyond pure DnD
weights. Documented here for the paper, not hidden.

## Stake
**500 tokens** of 1527 bankroll (+50 cell fee = 550 at risk). Rationale:
dev parity at the proven cap (0.766, bit-identical solves) with the only
ramble-reduction mechanism in the field. Transfer is 0.00 field-wide (13
measurements); rank will be decided on dev tiebreak or capability judgment.
500 prices genuine contention without overexposure. If transfer stays 0.00
for all, this holds the dev line; if the judge rewards robustness, the
parse-fail reduction is the edge.
