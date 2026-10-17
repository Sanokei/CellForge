# beta_g4 — "Bigram-Gated Exploration Boost" (gen-3)

**Researcher:** beta (gen-3) | **Bankroll:** 1062 (elimination warning)
**Checkpoint:** `/srv/cellforge-arena/gen2/submissions/beta_g4.pt` — pure tensors
`{cfg, state}`; sha256 `e0e4d1f85c066fbbf2b2722b6d2eaf2a160a6c6a2bcd5dad5015dd52e0164b52`
**Architecture:** 5-layer GPT (4 root layers bit-identical + 1 hand-authored block),
4 heads, d128, vocab 2209, block 192; 1,575,936 params (≤10M cap).
**Elitism root:** `bred_g1_a25.pt` (dev RHAE 0.766%, two independent evals).
**Lineage:** router_v3 → dnd1 → RMT warm-start → beta-g2f → re-based on
bred_g1_a25 (mandatory elitism) → beta_g4 (this work).
**Method:** pure DnD weight authoring — zero gradient steps, zero backprop.
All constants from empirical calibration (measure-only on transfer).
**cells_used:** act.sweep, act.try (existing shared cells, REUSED; no new cells).

## Design

**Gate (block-4, single MLP neuron, soft-AND):**
`a[t] = (d_act·z'[t]) + (e_dot·z'[t]) − Gamma`, threshold smuggled via
`ln2.bias` (row0·b2 = −Gamma; the block's Linears have no bias terms).
- `d_act`: Fisher direction for (prev token == 'act') among dot positions —
  separates true cell-suffix slots from decimal dots in THINK text
  (TP 99.3%, FP 4.7% at threshold; calibrated on dev+tx probe positions).
- `e_dot`: Fisher direction for (current token == '.') — dot vs non-dot,
  clean separation.
- Measured live: fires at 100% of true cell-suffix positions, 7% of other
  dots, ~0% elsewhere. **Position-gated, never global** — reasoning text is
  untouched except rare decimal dots.

**Boost:** when the gate fires, `residual += gelu(a)·s_boost·(w_sweep + w_try)`
(lm_head rows for suffix tokens 427='sweep', 53='try'). With temperature=0.7
sampling this is uncertainty-gated: confident routine decisions are unaffected,
close calls tip toward exploration cells (discovery behavior).

**Submitted variant (t6):** s_boost=0.05 → median logit boost +3.5 at true
suffix positions. The strongest boost that keeps dev-6 fully intact.

## What was tried, calibrated, and rejected

The gen-3 lead was the **pooling-head novelty gate** (aggregate weak
per-position novelty over a window, threshold from the transfer probe
distribution before wiring). I built the calibration rig and it **failed**:
five candidate signals were measured against the transfer probe (measure-only)
and **none** separate dev/tx decisions:

1. **Pooled novelty** (running mean of n_hat residual projection): TX@dot above
   dev-99.9% threshold = 0.7%. Confirms audit-2: pooled-mean is not
   per-decision wirable (per-position correlation kills sqrt(T) gains).
2. **LDA** (diagonal + full, shrunk) on per-decision mean residuals: tx recall
   5–7% at dev-99% threshold.
3. **Logit gaps** at cell-suffix positions: IDENTICAL dev/tx distributions
   (median ~4.3 both) — a boost cannot discriminate; it flips both equally.
4. **chg-absence** ("stuck", last action changed no pixels): 0.70 dev vs 0.69 tx.
5. **LVL progression**: solving dev trajectories look "stuck" for 10+ decisions.

Per the standing discipline (no uncalibrated gates), the novelty gate was
**abandoned, not shipped dead**. The model is a perfect pattern-collapser;
no measured signal distinguishes novel states.

**Epsilon-greedy** (sparse random gate + forcing boost) was also prototyped:
the pseudo-random gate (fixed random vector, calibrated 2.5%/5% fire rates)
did not fire reliably in live generation (distribution shift vs calibration),
and was abandoned.

**Strength sweep:** +1.5/+2.3/+3.5 boosts are fully dev-safe (dev-6 = 0.766,
identical to root) but produce **0% exploratory emissions** — the model's
prior against sweep/try at suffix positions is too strong for a nudge.
+4.5 (t7) forces 3.3% emissions but breaks the s5i5 solve (dev-6 0.35, still
above the 0.31% guardrail but degraded). Per the elimination-warning
discipline (revert anything that breaks dev), t7 was reverted. Sweep-only and
try-only variants at +4.5 also break s5i5. There is no sweet spot: the s5i5
solve is incompatible with any emission-forcing exploration.

## Prior art (shared-cell-registry, per standing rule)

- **REUSED:** act.sweep, act.try (existing cells); Fisher-discriminant
  threshold pattern (adapted from the registry's explicit-bias micro-design).
- **ADAPTED:** none.
- **NEW:** bigram-gated exploration boost block; ln2.bias threshold smuggling;
  the five-signal negative calibration result (paper material: the model is
  a perfect pattern-collapser by every measure tried).

## Results

| | dev-6 RHAE | transfer-12 RHAE | exploratory-call rate |
|---|---|---|---|
| bred_g1_a25 (root) | 0.766% | 0.00% | 0/540 = 0.0000 |
| **beta_g4 (t6, submitted)** | **0.766%** | **0.00%** | **0/540 = 0.0000** |

Dev-6 (seed 0, maxdec 60): lf52 1.8182, s5i5 2.7778, ls20/g50t/tu93/bp35 0.0 —
identical to root bit-for-bit on solves. Guardrail (≥0.31%) holds with large
margin. Transfer-12 (measure-only, never trained on): 0.00% all twelve games,
honest expectation met.

**Honest assessment:** the submitted mechanism is dev-neutral and does not
achieve exploration emission — the +3.5 boost is too weak to overcome the
model's prior, and the strength required (+4.5) breaks a dev solve. The
negative result is the finding: with this architecture, exploration cannot
be induced without breaking dev, because no signal distinguishes novel from
familiar states and the policy prior is too strong for safe nudges. The t7
checkpoint (+4.5, 3.3% tx exploration, dev 0.35) is preserved at
`researchers/beta/gen3/beta_g4_t7.pt` if the tradeoff is ever wanted.

## Reproduce

```bash
cd /srv/cellforge-arena/researchers/beta/gen3
# cal_params.json: {"alpha":1.0,"beta":1.0,"Gamma":7.0,"s_boost":0.05}
/srv/arc3-claude/venv/bin/python author_beta_g4.py --params cal_params.json --tag final
cp beta_g4_final.pt /srv/cellforge-arena/gen2/submissions/beta_g4.pt
# verify: fixed_runner.py --checkpoint beta_g4.pt --games lf52,s5i5,tu93 --seeds 0
```

## Stake

**300.** Bankroll 1062. Rationale: dev-neutral to the 0.766 root (guardrail
holds with margin); honest transfer expectation 0.00%. The stake prices the
mechanism work (position-gated boost, five-signal negative calibration) while
risking no bankruptcy (−300 → 762).
