# ALPHA-G3 — "The Cap and the Wire" (gen2-gen2)

## Lineage
`bred_g1_a25.pt` (elitism root: 0.25·alpha-g2 + 0.75·gamma-g2, dev-6 0.766%, two
independent evals) → + C1 (CALL-template hardening, lm_head, zero grad) →
`alpha_g3.pt`. Pure tensors `{cfg, state}`, 1,378,816 params, plain GPT
(4L/4H/d128/v2209). No RMT, breed-compatible. **Zero gradient steps.**
Base never retrained; dnd1 groups untouched.

Label map (this document): DEV-1..DEV-6 = the six dev games in judge order;
TX-A..TX-C = the three transfer games in the shared audit (measure-only).

## Audit (mine, grounded in Goodfire "Neural Networks Are Already Clean")
STEP 0: read `/srv/cellforge-arena/gen2/GOODFIRE_BRIEF.md`. Verified the shared
reference audit on the elitism root rather than duplicating it, and ran targeted
probes for my authoring directions.

- **Goodfire prediction CONFIRMED (independent check):** every audited transfer
  game matches the dominant dev routine (DEV-2's circuit) at cosine 0.96–0.99 by
  L3; top-200 neuron overlap transfer↔dominant-dev = 168/200, exceeding dev↔dev
  overlap (85/200). Transfer failure is **wrong-circuit firing (pattern
  collapse)**, not silence or dithering. The call logs agree: TX games are
  dominated by DEV-2's routine calls.
- **169/2048 MLP neurons separate novel-vs-seen** (effect sizes to 6.39, strongest
  at L0/L1, washed by L3) but nothing downstream reads them — a routing failure,
  not a detection failure. Zero dead neurons: capacity fully used, diffuse coding.
- **Malformed CALLs jump ~5%→~50% on novel states** (reference audit); on DEV-2 the
  species are: no-CALL rambles (model hallucinates new episodes instead of acting),
  64-token truncation of long CALLs, and genuine JSON confusion.
- **d_call direction verified:** the 'emitting CALL JSON' residual direction
  (synthetic prompts, no game data) separates cleanly from mid-think
  (separation 3.45σ). The dnd1 gating pattern transfers to the bred root.

## Authoring rounds (all DnD, zero grad)
- **R1 — novelty-tag wire (REVERTED, paper negative result):** implemented the
  directive "wire novelty separators to a distinct output branch / L2→L3 skip":
  differential tag `G·r·(a218−0.5·a6−0.5·a473)` written at L1 c_proj (G=20),
  centered by construction (dev pooled tag ≈ −0.04..−0.17, TX ≈ +0.38..+0.45).
  Result: **catastrophic** — 60/60 parse fails on DEV-1 and DEV-2 (dev 0.00).
  Post-mortem: the separators' **per-position** max|act| is 0.6–1.5, 50–100× their
  pooled-mean differences (0.01–0.02). The novelty signal exists only in
  game-pooled means; a linear per-position gate on it has no SNR — per-position
  tag perturbations (up to ~30 in a random direction) swamp the residual stream.
  Lesson: pooled-mean separability ≠ per-decision wiriability. N1 fully ablated.
- **R2 — C1 CALL-template hardening (KEPT):** +0.4 nats on structural tokens
  (`CALL { } " cell args : ,`), +0.25 on canonical cell-name heads, all gated on
  the measured d_call (state-independent). lm_head ||dW||/||W|| = 0.037.
  Result: both L1 solves preserved bit-for-bit (DEV-1 1.8182, DEV-2 2.7778),
  dev-3 = 0.766. Dev-neutral; parse-fail species on DEV-2 are confusion/truncation,
  not template sloppiness, so C1 doesn't move them.
- **R3 — thinking thermostat (measured, NOT submitted):** +1.0·(d_call−d_think)
  on the CALL token: suppresses CALL early in thought, boosts it at think-end.
  Result: DEV-2 steps 225→130 (−42%) but **RHAE flat** and parse fails 31→40.
  Led directly to the scorer finding below. Withheld from submission: higher
  parse-fail rate, zero fitness value, strictly more risk than R2.

## Key finding: dev-6 is mathematically capped at 0.766 (no new levels)
From `rhae_scorer.py`: `level_score = min(1, (human/agent)²)` (quadratic, capped),
`game_RHAE = 100·Σ(l·s_l)/(n(n+1)/2)` (level-weighted). Both solved levels are
**at cap** (DEV-1: 1.0 → 1.8182%; DEV-2: 1.0 → 2.7778%; the % differs only by
level count). **No efficiency or parse-reliability delta can raise dev-6.**
Improvement requires NEW level completions (L2+ on solved games, or L1 on a new
game). Training corpora contain **zero LVL2+ traces in 37,738 docs** — L2 is as
novel to this model as transfer. This reframes dev work: the tiebreak is won by
eliciting unseen routines, not by polishing solved ones.

## Self-check (dev-6, seed 0, maxdec 60, budget 3000, fixed_runner.py)
- DEV-1: 1.8182% RHAE, L1 solved (at cap)
- DEV-2: 2.7778% RHAE, L1 solved (at cap)
- DEV-3..DEV-6: 0.0 (unchanged from root)
- **dev-6 mean = 0.7660 ≥ 0.31 guardrail PASS** (matches elitism root exactly)
- Transfer: not tuned, not touched; judge measures post-staking. Honest
  expectation: 0.00 (no correct novel circuit exists to elicit; the N1/R3 work
  confirms the barrier is behavioral, not parametric).

## What I'd do with another iteration
Elicit an unseen routine: the only uncapped lever. Candidates: (a) L2 persistence
nudge — the model solves L1 in 13 actions then flails for 47 decisions; (b) the
reference audit's "suppress the dominant routine's default-on attractor" via a
**pooled-timescale** gate (the per-position attempt failed; a gate integrating
over ~10 decisions could recover the pooled SNR).

## Stake
**500 tokens** of 1527 bankroll. Rationale: transfer is 0.00 for the field
(curve 0.00 ×8+); rank will be decided on the dev tiebreak. I match the elitism
root at the proven dev cap (0.766) with the lowest-risk delta in the field
(single small lm_head edit, both solves bit-preserved). Gamma holds the gen-1
dev best (0.77); beta is on elimination warning and must gamble. 500 prices a
genuine 2nd-place contention without risking the bankroll on a tiebreak I can't
control.

## Addendum: novelty-switch architectural attempt (v1/v2, NOT submitted)
Under Sano's architectural-growth amendment I attempted the directed
novelty-switch as pure tensors (n_layer 4→5): L1 separator-evidence write +
new Block 4 with uniform-attention pooling head (causal mean over evidence
dims) + 1000× amplification + lm_head gate. Two failures, both instructive:
- v1: lm_head gate α=50 applied post-ln_f, where dims are unit-scale —
  ±25-nat logit swings, model broken (60/60 parse fails). Lesson: calibrate
  gates to post-normalization scale.
- v2: uncancelled natural baseline in evidence dims (+0.006 mean ×1000 =
  +6.0) drowned the systematic evidence (±0.003 ×1000 = ±3, measured
  differential only −0.4, wrong sign). Lesson: pooled evidence needs
  differential baseline cancellation; natural means dominate.
Both reverted. Submitted weights are r2 (C1-only) above. The switch remains
the right target for round 2 with differential read (dim pairs with matched
natural means identified: e.g. dims 12/74, Δμ=4e-5).
Prior art (standing rule): shared-cell-registry grepped — 12 live cells in
3 micro-arithmetic families (different architecture/task); no direct weight
reuse. Reused: the principle of hand-authored precise micro-circuits
(decimal-affine-borrow style). Adapted: none. New: C1 gating, thermostat,
switch architecture (attempted).
Cells used: none. Cell proposals: none (fee 50, dividend requires
above-mean transfer; transfer is 0.00 field-wide).
