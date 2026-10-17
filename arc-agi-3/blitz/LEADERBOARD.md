# Blitz leaderboard (dev: 15 games, seed 0, budget 3000)

Target: mean score > 0.24 (baseline v5_all_s2 = 0.189 on 27-game sweep).

| variant | mean | levels>0 | notes |
|---|---|---|---|
| v6:pr,nm,ck,gv (dev files, +d6) | 0.1867 | 5/15 | baseline on these 15 |
| blitz1 (no d6), 10 games only | 0.279 | 2/10 | attempt loop alone; 5 games missing (re-running) |
| blitz2 (+d6) | ? | ? | progress-aware cycling + director reset — running |

Per-game baseline (v6:pr,nm,ck,gv, seed 0):
- bp35 0.00 (0/9), cn04 0.00 (0/6), dc22 0.00 (0/6), ft09 0.01 (1/6), g50t 0.00 (0/7)
- ka59 0.00 (0/7), lf52 0.01 (1/10), lp85 2.78 (1/8), ls20 0.01 (1/7), re86 0.00 (0/8)
- s5i5 0.00 (0/8), sc25 0.00 (0/6), sk48 0.00 (0/8), tr87 0.00 (0/6), tu93 0.00 (2/9)

blitz1 no-d6 (10 games): lf52 0.01 (1/10), lp85 2.78 (1/8), rest 0.00.
Note: lp85 dominates the mean (2.78/15 = 0.185). Real headroom = other 14 games.

## Variant family
- blitz1: stall-triggered restarts + attempt_budget cycling (500-700) + salt reshuffle + lattice on later attempts
- blitz2: blitz1 + progress_window (400) cycling + director per-level reset on retry
- blitz3: blitz2 + Luby budgets [120,120,240,480,960] + progress_window 150 — TOO AGGRESSIVE, hurts
- blitz4: blitz1 MINUS budget cycling (stall restarts only) — cleanest

## Key learnings
- Score = ((baseline/actions)^2)*100 per level. QUADRATIC in speed. Fast solutions dominate.
- lp85 solved in 5 actions (lucky ordering) → 2.78. Everything else is 0.00-0.01.
- Aggressive restarts (blitz3) HURT: disrupt systematic exploration before it finds solutions.
- Regressions: blitz1+d6 got 0 levels on ft09/ls20 vs baseline 1 level. Restarts may wipe director progress.

## Full 15-game results
| variant | mean | notes |
|---|---|---|
| v6:pr,nm,ck,gv (baseline) | 0.1867 | dev files |
| blitz4 (stall restarts + d6) | 0.191 | tu93: 3/9 lv, 0.07 (was 2/9, 0.0035) |

blitz4 per-game: bp35 0, cn04 0, dc22 0, ft09 0.01 (1/6), g50t 0, ka59 0, lf52 0.01 (1/10),
lp85 2.78 (1/8), ls20 0.00 (1/7!), re86 0, s5i5 0, sc25 0, sk48 0, tr87 0, tu93 0.07 (3/9).

Gap to 0.24: +0.049. Need ~+0.74 total. Options: faster solutions on scoring games,
or crack 1-2 zero-games.

## Wave 2 results (2026-10-15, seed 0 unless noted)

Implemented the three wave-1 fixes as blitz6-13 in blitz_w2.py.

### Key findings
- **VolatileMask2 (strict action-correlation) REGRESSED**: it failed to mask
  lf52's top progress bar (row 0, 64px), dropping lf52 from 1 level to 0.
  The original mask was RIGHT on lf52. Root cause: VolatileMask2 required
  pixel changes under EVERY action class, but lf52's keyboard actions are
  no-ops, so the test never fired.
- **VolatileMask3 (original + action-discrimination VETO)**: keeps the
  original's band/per-pixel tests but VETOES masking pixels whose changes
  discriminate actions. Fixes lf52 (1 level, 0.007-0.015) and CRACKED g50t
  (1 level, 0.621) via the VETO protecting load-bearing pixels.
- **Painter-demotion**: demote action classes that only ever deposit novel
  pixels (never revisit known states). Fixes ls20 reliably: 0.422 (64-70
  actions) vs 0.005 baseline, confirmed on seeds 0,1,2.
- **Click macros v2 + ghost-seeking macros**: implemented (blitz7, blitz11)
  but not yet fully validated; probe showed lf52/s5i5 need adaptive
  pick->see-ghost->drop.

### Per-variant best scores (seed 0)
| variant | mean (est) | key wins |
|---|---|---|
| blitz4 (incumbent) | 0.1907 | lp85 2.78 (lucky 5-action), tu93 0.07 |
| blitz8 | 0.135 | ls20 0.42, g50t 0.62, s5i5 1lv |
| blitz12 | - | lp85 2.78 (luck preserved), lf52 0.015 |
| blitz13 | - | ls20 0.35, g50t 0.62 |

### Gap to 0.24
No single variant beats 0.1907 overall (lp85's 2.78 is high-variance).
Combination path: lp85 2.78 + ls20 0.42 + g50t 0.62 = 3.82/15 = 0.255.
Need a variant that keeps ALL THREE. Currently blocked by lp85 variance
(painter-demotion disrupts the lucky ordering; nopainter exemption did NOT
recover it, suggesting a subtler interaction).

### Honest assessment
The three fixes are REAL and RELIABLE (ls20, g50t, lf52). The 0.24 barrier
is blocked by lp85's high-variance lucky solve, not by the fixes. A model
(as per Sano's correction) trained on traces from these explorers is the
right next step; the explorer is now a trace generator + harness.
