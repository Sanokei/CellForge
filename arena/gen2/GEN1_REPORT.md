# GEN-2 GENERATION 1 — Judge Report
Date: 2026-10-16 ~10:30 UTC. Charter: /srv/cellforge-arena/GEN2_CHARTER.md

## Compete (judge-measured, fixed_runner.py, seed 0, maxdec 60, budget 3000)
Dev-6 self-checks (researcher-run, specified protocol, accepted as deterministic):

| Researcher | Strategy | Transfer-12 RHAE | Dev-6 RHAE | Guardrail | Fitness |
|---|---|---|---|---|---|
| gamma-g2 | Authored LoRA v2, 0 grad steps | **0.00%** | 0.77% | PASS | 0.00 |
| alpha-g2 | KL-anchored mixed finetune, lr 1e-4 | **0.00%** | 0.766% | PASS | 0.00 |
| beta-g2 | RMT + memory-forcing loss, frozen text path | **0.00%** | 0.46% | PASS | 0.00 |

Fitness = transfer × min(1, dev6/0.31). All guardrails passed; no forgetting penalty
applied. Transfer curve now 0.00 ×8 (dnd1, blackboard, v5, v3world, v5world, RMT,
gen2-alpha, gen2-beta, gen2-gamma).

## Select
Rank by (transfer, dev6): 1. gamma, 2. alpha, 3. beta.

## Staking settlement
| Researcher | Bankroll in | Stake | Payout | Balance out |
|---|---|---|---|---|
| gamma-g2 | 1311 | 400 | 1200 (3×, rank 1) | **2111** |
| alpha-g2 | 1467 | 600 | 660 (1.1×, =mean) | **1527** |
| beta-g2 | 1462 | 400 | 0 (−100%, bottom quartile) | **1062** |

Beta: 1 consecutive bottom-quartile (counters reset for gen-2 era). No eliminations.

## Breed
Top-2 (gamma + alpha, same base arch) → Sakana-style tensor interpolation:
- bred_g1_a50.pt — 0.5·alpha + 0.5·gamma
- bred_g1_a25.pt — 0.25·alpha + 0.75·gamma
Lineage: /srv/cellforge-arena/gen2/bred/lineage.json. Children unevaluated;
gen-2 successors start from these (mandatory elitism).

## Researcher findings (paper material)
- ALPHA: dev solves are knife-edge deterministic — base_dnd1 reproduces L1 solves
  bit-identically, yet every unconstrained finetune (even 50 steps @1e-5, even
  answers-only) killed them 6/6. It isn't the fc corpus that's poisonous; any
  unconstrained weight drift flips fragile early decisions. KL anchor
  (loss = LM + 5·KL(p‖p_base) on answer batches, dnd1 groups frozen) is what
  makes rehearsal work.
- BETA: RMT finetuning per se destroys dev at 1.4M params even at lr 1e-4 with
  rehearsal (5 variants v1–v5 all 0.00% dev-6, as did gen-1's router_rmt.pt).
  Freezing the transformer and training only mem_init + isolated probe preserved
  dev-6 (0.46%) and made memory measurably predictive (probe loss 7.7→6.0) —
  but transfer stayed 0.00%.
- GAMMA: pure DnD authoring (zero gradients) beat the dnd1 control on dev-6
  (0.77 vs 0.75) with fewer parse fails on 4/6 games. Negative result: static
  LoRA gates on lm_head can't be position-specific — a novelty-gated probe
  design regressed dev (0.46%) and was discarded; only large-projection
  natural-context gating is safe there.

## Honest assessment
The loop ran cleanly end-to-end (author→compete→select→breed, staking settled,
lineage recorded) but transfer did not move: 0.00% ×8. The selection pressure
is real but the fitness landscape is flat at zero — nothing yet to select FOR
on transfer. Dev-6 improved (gamma 0.77 > dnd1 0.69), so the loop selects on
the dev tiebreak. Next iteration needs a transfer-positive variant to exist;
otherwise the loop is breeding dev improvements under a transfer banner.
