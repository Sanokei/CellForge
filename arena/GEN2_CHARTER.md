# GEN-2 CHARTER — Transfer-Selection Researcher Loop
Ratified 2026-10-16 ~04:30 EDT by the orchestrator under Sano's directive (taper waived for this loop).

## Why gen-2 exists
Five flat attempts (dnd1, blackboard, first-contact/v5, world-model, RMT) all scored
0.00% transfer RHAE. Isolated mechanisms fight each other (full-LR finetuning caused
catastrophic forgetting; 3x OBS loss distorted call generation; unfocused memory went
unused). Gen-2 replaces fixed attempts with the evolutionary loop itself, with
TRANSFER as the fitness function and the full kit as the starting gene pool.

## Fitness
- PRIMARY: transfer RHAE over TRANSFER-12
  (ar25, cd82, lp85, m0r0, r11l, sb26, sp80, su15, tn36, tr87, vc33, wa30),
  seed 0, maxdec 60, budget 3000, fixed_runner.py, measure-only, no retries.
- GUARDRAIL: dev-6 RHAE (lf52, s5i5, ls20, g50t, tu93, bp35 — same protocol)
  must stay >= 0.31% (v3 baseline). Forgetting is PENALIZED:
  fitness = transfer_rhae * min(1.0, dev6_rhae / 0.31).
- Selection rank: (transfer_rhae desc, dev6_rhae desc).

## Architecture (fixed)
- Base: GPT 4L/4H/d128, vocab 2209 (router_v3 cfg). RMT variant: same + 8 memory
  tokens, positions 0-7, text at 8-, wpe 192->200 (gpt_rmt.py). Runner auto-detects.
- Submissions: PURE TENSORS ONLY — .pt {cfg, state} or LoRA-deltas JSON + declared
  base. No Python, ever. Violations = disqualification.

## Starting gene pool (mandatory elitism — all submissions descend from these)
- Base weights: lm/router_v3.pt (+ dnd1 deltas lm/dnd1_deltas.json: decisiveness,
  parse reliability — 0.69% dev)
- RMT: lm/gpt_rmt.py, lm/train_rmt.py, router_rmt.pt lineage
- Corpora: lm/traces_fc.jsonl (discovery) + traces_w2.jsonl + traces_grammar.jsonl
  (answers — rehearsal prevents forgetting)
- World model: lm/train_world.py (aux OBS-prediction loss)

## Encoded diagnosis (hard constraints on authoring)
1. Finetune LR <= 1e-4. Full-LR (3e-4) finetuning caused catastrophic forgetting.
2. ALWAYS mixed corpus: discovery + answers. Discovery-only diets forget.
3. World-model aux loss weight <= 1x. 3x distorted call generation (parse fails exploded).
4. RMT memory needs a FORCING loss (predict OBS from memory tokens) or it goes unused.
5. Discovery->solve traces beat pure-poking traces (fc corpus: ~1 solve / 5720 docs).

## Researchers (DnD weight-authors)
- alpha-g2 (bankroll 1467), beta-g2 (1462), gamma-g2 (1311). Consecutive-bottom25
  counters RESET for gen-2 (fitness function changed eras).

## Staking
- Researchers declare stakes BEFORE eval (recorded in ledger). Staking closes before
  the judge runs transfer evals.
- Payouts: rank 1 -> 3x; score >= mean (rank>1) -> 1.1x; score < mean -> -50%;
  bottom quartile (rank > 0.75n) -> -100%.
- Elimination: bankruptcy OR bottom-quartile twice running. Successors MUST start
  from bred champion weights (mandatory elitism), lineage recorded.

## Loop
AUTHOR (researchers author weight variants) -> COMPETE (judge runs dev-6 + transfer-12
via fixed_runner) -> SELECT (fitness rank) -> BREED (Sakana-style tensor recombination
of top-2: same-arch interpolation alpha in {0.25, 0.5, 0.75} per-tensor or per-layer
crossover; cross-arch merges deferred). Iterate generations. Stop only when transfer
moves sustainably or Sano calls it.

## Standing anti-cheat
- held_v6.jsonl SEALED (assistant/orchestrator only, final scoring).
- Transfer-12: measure-only. Never trained on, never dissected, never opened by researchers.
- No game names in prompts, traces, deltas, or notes.
- Token/compute metering harness-side. Deterministic evals.

## Compute discipline (cellforge-1: 2 vCPU, no GPU)
- OMP_NUM_THREADS=1 on all torch jobs. Max 2 concurrent heavy jobs; check `uptime`
  first. Trainings sequenced via /srv/arc3-claude/v6run/model/gen2/TRAIN_LOCK.

## Reporting
- Every generation: attempt->transfer curve, (transfer, dev6, fitness) per researcher,
  stakes/payouts/bankrolls, breeding recipe + parent lineage.
- Real numbers pushed to CellForge Live dashboard by the orchestrator.
