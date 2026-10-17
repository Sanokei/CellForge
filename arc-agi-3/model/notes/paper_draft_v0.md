# CellForge: Author-Then-Evolve Tiny Agents for ARC-AGI-3 — Draft v0 (2026-10-16)

## Title

CellForge: Breeding Tiny Models by Hand-Authored Weights and Competitive Selection
for Interactive Reasoning Benchmarks

## Abstract

We present CellForge, a research program that pursues capable tiny models for
ARC-AGI-3 by inverting the usual pipeline: instead of gradient descent on large
models, a frontier language model directly authors the weights of tiny models
(1–2M parameters), and an evolutionary loop — author → stake → compete → payout →
eliminate → breed — selects and recombines them under competitive pressure.
Perception is organized as a blackboard: capability cells bid passively for the
router's attention each step, with bids evolving under selection. We adopt the
official RHAE metric (arXiv 2603.24621 §4.1) exactly and report honest numbers:
our gen-0 champion scores 0.77% dev RHAE and 0.00% first-contact transfer RHAE
on 12 unseen games; our 1.38M-parameter neural router scores 0.31% dev and 0.00%
transfer. The gap between dev and transfer is the finding: hand-authored
game-specific knowledge exhibits total bimodal transfer failure, confirming the
ARC-AGI-3 report's warning in our own numbers. We frame this work for the
community leaderboard — harness research the report explicitly values — and
release our scorer, transfer split, and methods.

## 1. Introduction: our lane

The ARC-AGI-3 technical report is explicit: the official leaderboard never
reports public-set scores, and "evaluating on it is emphatically not a valid
measure of progress towards AGI." The official track uses private games, no
harness, one fixed system prompt. Our lane is the **community leaderboard**,
which the report explicitly values: harness research is "economically valuable"
in its own right. This paper is a harness-research paper. We claim no AGI
progress; we claim a method for building tiny models that act, and an honest
measurement of where that method stands.

The core bet: **author-then-evolve**. "Using gradient descent is kind of
cheating" — it drifts from the method and is not our differentiator. The
literature now supports the alternative: prompt-conditioned parameter generation
("Drag-and-Drop LLMs," arXiv:2506.16406) beats the strongest gradient-trained
LoRAs by up to 30% on unseen reasoning benchmarks at 12,000× lower cost, with
the authors concluding that "prompt-conditioned parameter generation is a viable
alternative to gradient-based adaptation." Hypernetworks turn interaction data
into weight modulations ("Infinite-Parameter LLMs," arXiv:2609.18842),
amortizing compute and freeing the context window. We scope backprop to optional
local refinement — the Baldwin effect — and put authoring and selection at the
center.

## 2. Method

**2.1 The loop.** Each generation: researchers AUTHOR agents (hand-written Python
or LLM-authored weights, never gradient-trained as the primary step), STAKE
token bankrolls on their own entries before evaluation (staking closes before
eval seeds are drawn), COMPETE on fresh seeds under the true RHAE metric,
receive PAYOUTS by rank (1st: 3×, top 25%: 1.5×, above mean: 1.1×, below mean:
−50%, bottom 25%: −100%), face ELIMINATION on bankruptcy or two consecutive
bottom-25% finishes, and are REPLACED by successors required to breed from the
champion's cells (mandatory elitism). Tokens buy compute (1 per game-seed
rollout); researchers set per-game think-token budgets. All scores, bankrolls,
and stakes are public; champion weights are public.

**2.2 The blackboard.** Perception is passive: every capability cell implements
`observe(state) → (annotation, bid ∈ [0,1])`, run every step at zero action cost.
Top-k bids become `INTAKE` lines in the router's prompt — peripheral vision, not
function calls. Bids start heuristic (salience rules authored per cell) and
evolve: we weight bids by measured usefulness (which intakes precede level-ups
and efficiency gains). This is Hearsay-II's blackboard meets Global Workspace
Theory — specialists competing for the router's attention — with learned bids as
the evolved glue. Lineage: the Oct-1 cell registry (3 typed families,
input-type gating as hard 0/1 salience, hash-bound inheritance receipts) is the
prior art; the blackboard is the same thesis with the competition switched on.

**2.3 The router.** A 1.38M-parameter decoder-only transformer (4 layers, 4
heads, d128) trained by distillation on teacher traces: English reasoning →
cell-call routing → level completion. The DnD-style weight author (frontier LLM
conditioned on probe findings + cell descriptions + current weights, directly
authoring LoRA deltas, zero backprop) is raced against the distilled router on
transfer RHAE to decide the program's direction.

**2.4 Anti-cheat.** The held-out set (`held_v6.jsonl`) is sealed — final scoring
only. Eval seeds are drawn after staking closes. The transfer split (12 games
never in traces, training, probes, or selection) is measure-only: never tuned
on, never trained on. Load-bearing tests throughout: remove or scramble a
component (an intake cell, the memory tokens) and performance must degrade, or
the component is theater.

## 3. Results

All numbers are true RHAE per the paper spec (cap 1.0, second-best-human
baselines, level-weighted environments, mean across envs), verified by
independent hand computation.

| System | Dev RHAE | Transfer RHAE (12 unseen) |
|---|---|---|
| Champion alpha-g0 (scripted) | 0.77% (6 games) / 0.18% (25 games) | **0.00%** (0 levels) |
| router_v3 (1.38M params, neural) | 0.31% (6 games) | **0.00%** (0 levels) |

Notable: the tiny router solved lf52 L1 in exactly 32 actions — the
second-best-human count — a genuine neural reasoning-to-routing solve. On s5i5
L1 it used 166 wasteful actions vs the champion's 13 (RHAE punishes action count
quadratically: 0.04% vs 2.78%), which is the efficiency gap distillation must
close. Parse failures ran 20–50% per game on the LM — its call grammar is
itself overfit.

The headline is the transfer column: **total bimodal transfer failure** in both
systems. This confirms the ARC-AGI-3 report's own warning (their Duke harness:
97.1% on a seen variant, 0% on unseen) in our numbers. Dev-set RHAE is
secondary; transfer RHAE is the progress metric, reported every generation.

## 4. Lineage

HyperNEAT (Stanley 2009: N-dimensional substrates, CPPN(x1,y1,z1,x2,y2,z2)=w) →
HyperNetworks (Ha 2016: genotype→phenotype, "reminiscent of HyperNEAT") →
Sakana evolutionary model merges (evolved recipes, sub-gradient memory
footprint) → prompt-to-weights (arXiv:2506.16406) → CellForge. One family tree:
evolution and authoring over weight space, not gradient descent. Our paper sits
in it.

## 5. Limitations (stated openly)

- Transfer RHAE is zero. Everything that works is game-specific.
- The blackboard's bids are heuristic; the learned-glue loop is built, not yet validated.
- The DnD weight-author race is in progress; the program's direction is undecided.
- No GPU: all training is CPU-bound (<10M params), which constrains iteration speed.
- 25 dev games is a small world; the transfer split is one-shot (eval-seen after first contact).

## 6. Future work

- **3D weight topology**: learnable neuron coordinates, inverse-distance weights,
  wiring-cost penalty (lineage: arXiv:2506.13410, arXiv:2606.14975). Timeboxed
  experiment vs baseline on transfer RHAE.
- **RMT memory tokens** with the zeroing load-bearing test ("memory dependence" metric).
- **Decoder-only vision**: patch projection + end-to-end resampler, visual tokens
  in-stream, one loss — no reconstruction objective.
- **Jev-Mem split**: bid spikes as wake-up signals; LM runs at decision points only.

---

## DECISION RECORD — tonight's ship candidate (2026-10-16)

**Ship candidate: the gen-0 champion lineage (alpha-g0 / gamma-g1 scripted agent).**
Why: it is the only system with nonzero RHAE on anything (0.77% dev, seed-stable
solves on lf52/s5i5 L1), it is fully offline (pure Python, O(1) decisions, ~100
think tokens/game, no model weights to bundle), and its wall-clock cost fits the
9h Kaggle budget with margin. The neural router (0.31% dev, 0.00% transfer,
20–50% parse failures, slow CPU inference) is the research bet, not the entry.
If Track A (LLM-authored weights) or Track B (blackboard intake) posts a
transfer-RHAE win within days, the candidate changes; until then, the entry
ships as the best evolved scripted agent, and the paper carries the honest
0.00% transfer numbers as the problem statement.
