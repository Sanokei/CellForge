# beta_g5 — "Self-Gating Discovery Probe" (gen-4)

**Researcher:** beta (gen-4) | **Bankroll:** 1062 → 1012 after cell fee (elimination warning)
**Checkpoint:** `/srv/cellforge-arena/gen2/submissions/beta_g5.pt` — pure tensors
`{cfg, state}`; sha256 `98d39a5634c3409ebd23ba1dfd98c78e4974f7e88033c294ccc9b6d7e6d40f6b`
**Architecture:** 5-layer GPT (4 root layers bit-identical + 1 hand-authored block),
4 heads, d128, vocab 2209, block 192; 1,575,936 params (≤10M cap).
Purity contract: OK (fixed_runner.purity_check_checkpoint).
**Elitism root:** `bred_g1_a25.pt` (dev RHAE 0.766%, two independent evals).
**Lineage:** router_v3 → dnd1 → RMT warm-start → beta-g2f → bred_g1_a25
(mandatory elitism) → beta_g4 (bigram-gated boost) → beta_g5.
**Method:** pure DnD weight authoring — zero gradient steps, zero backprop.
Gate directions reused from gen-3 calibration; boost strength from
measure-only emission calibration.
**cells_used:** act.Probing (NEW, proposed this round); act.target pattern (reused).

## Design

**The move:** gen-3 proved weight-level novelty signals don't exist
(per-position SNR<1; the model is a perfect pattern-collapser) and that
logit boosts strong enough to force exploration break dev. Gen-4 moves the
novelty→exploration routing **into the cell layer**, where LIVE state gives
a behavioral stuck signal no weight probe can see.

**New cell `act.Probing{}`** (generic-only, no game names; proposed to the
shared library, 50-token fee acknowledged):
- On CALL, runs a generic stuck test on live state: tracks (frame thumbnail,
  level) across calls on the GameState (same pattern as st.tested).
  STUCK = ≥30 env steps AND no significant frame change AND no level-up
  for the last 25+ env steps (as observed at probe calls).
- NOT STUCK → returns a text note, **no env step**. A working solve changes
  the frame every decision, so the probe can never fire mid-solve:
  **dev-neutral by construction**.
- STUCK → bounded grounded discovery (≤30 env steps, never RESET):
  click each untested object center (area-descending, the act.target
  grounding pattern) + try each untried keyboard action (c_sweep's loop),
  reporting per-action pixel effects for hypothesis formation. Stops early
  on LEVEL-UP/WIN. Re-arms the stuck timer afterwards.
- Never issues RESET (would destroy banked progress); respects st.budget.

**Weight component** (beta_g4's proven bigram gate, retargeted): block-4
fires at true cell-suffix positions only (prev=='act', curr=='.') and boosts
the `act.Probing` suffix token (vocab 39, verified no-<unk>). Dev-safety
comes from the cell's self-gating, not from a weak boost.

## What was verified (audit trail)

1. **Forced-probe stress test** (root + private cell merge, probe forced
   every 5 decisions — far above production rate):
   - lf52: 12 probes, **0 explores** — L1 solve untouched.
   - s5i5: 12 probes, 5 explores, ALL at lvl1 AFTER L1 banked (L2 flail).
   - ls20/g50t/tu93/bp35: 0/1/2/4 explores — gate opens only where the
     model flails. **Zero false positives during L1 solves.**

2. **Emission calibration** (s_boost sweep, 40 decisions on s5i5+lp85+vc33):

   | s_boost | emission | s5i5 | lf52 | verdict |
   |---|---|---|---|---|
   | 0.05 | 0/120 | 2.7778 | — | too weak |
   | 0.10 | 0/120 | 2.7778 | — | too weak |
   | 0.20 | 4/120 (3.3%) | 2.7778 | 1.8182 | **SELECTED** |
   | 0.25 | 9/120 (7.5%) | 2.7778 | **0.0 BROKEN** | lf52 fragile |
   | 0.30 | 24/120 (20%) | 2.7778 | **0.0 BROKEN** | lf52 fragile |
   | 0.40 | 48/120 (40%) | **0.0 BROKEN** | — | decision-starvation |

   Strongly nonlinear (top_k=40 sampling cliff). s_boost=0.20 selected:
   nonzero emission, both dev solves green. Diagnosis of the breakage:
   at ≥0.25 the model generates `<unk>.<unk>` cell names from dec0
   (weight-driven, not OBS-driven) — lf52's routine is fragile to any
   strong suffix perturbation; s5i5 tolerates it. Per the survival
   discipline (revert anything that breaks dev), s025/s030/s040 rejected.

3. **Dev-6** (seed 0, maxdec 60, private cell merge):

   | game | RHAE | levels | probe emit |
   |---|---|---|---|
   | lf52 | 1.8182 | 1 | 0/60 |
   | s5i5 | 2.7778 | 1 | 1/60 |
   | ls20 | 0.0 | 0 | 0/60 |
   | g50t | 0.0 | 0 | 1/60 |
   | tu93 | 0.0 | 0 | 1/60 |
   | bp35 | 0.0 | 0 | 0/60 |
   | **mean** | **0.7660** | | **3/360 (0.008)** |

   Guardrail (≥0.31%) holds with large margin. Bit-identical to root on solves.

4. **Transfer-12** (measure-only, seed 0, maxdec 60, private cell merge):

   | game | RHAE | probe emit | explores |
   |---|---|---|---|
   | ar25 | 0.0 | 1/60 | 0 |
   | cd82 | 0.0 | 1/60 | 0 |
   | lp85 | 0.0 | 3/60 | 1 |
   | m0r0 | 0.0 | 1/60 | 0 |
   | r11l | 0.0 | 0/60 | 0 |
   | sb26 | 0.0 | 2/60 | 1 |
   | sp80 | 0.0 | 2/60 | 0 |
   | su15 | 0.0 | 2/60 | 0 |
   | tn36 | 0.0 | 3/60 | 2 |
   | tr87 | 0.0 | 4/60 | 0 |
   | vc33 | 0.0 | 1/60 | 0 |
   | wa30 | 0.0 | 3/60 | 1 |
   | **mean** | **0.0000** | **23/720 (0.032)** | **5** |

   Honest expectation met (0.00). The mechanism is REAL: 23 probe emissions,
   5 triggered full grounded discovery runs (bounded click+keyboard sweeps
   with per-action effect reports). None solved a level — the exploration
   exists but is not yet effective.

## Prior art (per standing rule)

- **REUSED:** _objs (features.py), area-descending click order (act.target,
  alpha), untried-action loop + st.tested (c_sweep), bigram-gate block
  design (beta_g4, this researcher), Fisher-direction calibration
  (gen-3 cal_dirs2.pt).
- **ADAPTED:** boost target sweep/try → act.Probing suffix token.
- **NEW:** act.Probing cell (stuck-test self-gating, bounded grounded
  discovery, progress tracking on GameState); the finding that cell-layer
  behavioral stuck signals work where weight-level novelty signals fail;
  the s_boost nonlinearity / lf52-fragility negative.

## Results

| | dev-6 RHAE | transfer-12 RHAE | exploratory-call rate |
|---|---|---|---|
| bred_g1_a25 (root) | 0.766% | 0.00% | 0/540 |
| **beta_g5 (submitted)** | **0.7660%** | **0.00%** | **23/720 (0.032), 5 real explores** |

## Stake

**300.** Bankroll 1062 → 1012 after cell fee → 712 if stake lost.
Rationale: dev-neutral to the 0.766 root (guardrail holds with margin);
honest transfer expectation 0.00% (mechanism real but unproven on
transfer); prices the cell + mechanism work without risking bankruptcy.

## Reproduce

```bash
cd /srv/cellforge-arena/researchers/beta/gen2g5
# cal_params_g5_s020.json: {"alpha":1.0,"beta":1.0,"Gamma":7.0,"s_boost":0.20}
/srv/arc3-claude/venv/bin/python author_beta_g5.py --params cal_params_g5_s020.json --tag final
cp beta_g5_final.pt /srv/cellforge-arena/gen2/submissions/beta_g5.pt
# verify (private cell merge): verify_g5.py beta_g5_final.pt --games lf52,s5i5,ls20,g50t,tu93,bp35
```

## Cell proposal

`cell_act_probing_proposal.py` — ready for coordinator generic-only audit +
merge into lm/cells.py REGISTRY/CELL_DOCS. 50-token registration fee
acknowledged (debit via cell_ledger.register_cell at proposal).
**Merge dependency:** the judge must merge act.Probing into the runner's
cells.py BEFORE judging beta_g5, else probe emissions parse-fail to
perceive.describe (dev-safe fallback, but the mechanism won't run).
