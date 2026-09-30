# Grokking-style delayed generalization in gradient-free NCA evolution

PRE-REGISTERED (written before the full sweep was run; only ~20k-eval pilots on task "swap"/"life" seed 0 had been seen).

## Criterion
Per run (task, seed, arm): let e_fit = first evaluation count at which the current best individual reproduces ALL demo outputs exactly.
"Delayed generalization" (DG) = all of:
 1. e_fit exists and e_fit <= 30,000 (leaves >= 3x more budget after fitting);
 2. held-out exactness at e_fit (extra-20 grid exact-match fraction) < 0.5;
 3. at the final budget (1e5 evals) the best still fits the demos exactly and extra-20 exact fraction >= 0.5 AND >= (value at e_fit) + 0.3.
Arm C (stop at first fit) = the held-out value at e_fit of arm A (same RNG stream, so the trajectory is identical up to e_fit;
C is computed from A's fit snapshot rather than re-run). Comparison of interest: extra20 at the final budget for A/B/D vs C ("does continuing help").
Also report: demo-fit rate, median e_fit, held-out exact vs eval count, parameter L1 and nonzero count.
Wilson 95% intervals for proportions. Unit of inference is the run; 12 tasks x 3 seeds = 36 runs per arm, 2 of the 12 tasks are
non-local controls an NCA (radius-3 receptive field, 3 steps) should not solve; tasks/seeds are not independent draws from a population.

## Setup
NCA: 10 channels, 4 hidden, 3 steps, 414 params, seeded at zero (identity), as nca_evo.py. (mu=6 + lambda=24) selection, Gaussian mutation
with 1/5-rule-like step-size adaptation, selection only (no gradients, no fitness-weighted averaging). 6x6 grids; 2 demo pairs, held-out = 21 fresh grids of the same rule
(1 "held-out pair" + 20 extra; all 21 pooled are `extra20` for criterion purposes = the 20 extras; held-out pair reported as `held`).
Arms: A plain, B parsimony (fitness = MSE + lam*mean|w|, lam in {0.001,0.004,0.016}, plus 25% of offspring are
"sparsify" mutants that zero the smallest 10% of nonzero weights of a parent; accepted only if penalised fitness survives selection), D random search (seed + N(0,sigma), sigma in {.02,.05,.15}, keep top mu).
Budget 1e5 evaluations (not 3e5: compute), checkpoints 1e3,3e3,1e4,3e4,1e5 (first generation boundary >= count).
Tasks: swap, cycle, count(neighbour-count recolour), border, hole, shift, life, dilate, interior, edge; controls CTRL_transpose, CTRL_majority.
Code: grok.py, analyze.py. Results: results.jsonl (one run per line), tables.md.

## Commands
    python grok.py 100000 3 results.jsonl      # 2 worker processes
    python analyze.py results.jsonl > tables.md
