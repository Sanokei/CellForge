# GEN-2 Amendments (Sano directives, 2026-10-16 ~18:15-18:25 EDT)
Amend the GEN2_CHARTER.md for generation 2 and onward.

## 1. Architectural growth via tensors (runner is now shape-agnostic)
Researchers may GROW the architecture, not just rewrite weights: add layers,
neurons, neuron groups, glue connections. The architecture is expressed
through the tensors themselves: submissions declare their cfg in the
checkpoint, and fixed_runner.py builds the model from it.
Bounds (contract-enforced): n_layer 1-8, n_head 1-8, n_embd 32-512,
block_size 64-512, n_mem 1-16 (RMT), vocab_size == 2209 (fixed tokenizer),
total params <= 10M. Out-of-bounds or non-finite = REJECT.
The novelty-switch (neuron group reading the 169 novelty detectors, gating
exploration) is the prime candidate — it MUST be tensors, since Python cells
cannot see internal activations.

## 2. Shared cell registry — prior art first (standing rule)
`/srv/cellforge-inbox/two-loop-20261001/shared-cell-registry/LATEST.json`
(+ selected_cell_registry.py): 12 live cells in 3 families
(decimal-subtraction, spatial-cardinal, signed-temporal-deduction), each a
tiny measured weight-set with structure + scores. More may exist under
/srv/cellforge-inbox — grep. These are different architectures: researchers
ADAPT the proven micro-designs (e.g. decimal-affine-borrow's digit
arithmetic), not the bytes. METHOD.md must report reused / adapted / new.

## 3. Python for CELLS (allowed) vs Python for TRAINING (forbidden)
Researchers MAY write Python to CREATE new generic cells — new mechanisms /
tools for the shared cell library (same category as cells.py's act.* /
perceive.*). Process: prototype privately, submit final cell code to the
coordinator, who audits for generic-only (no per-game logic, no game names —
same standard as the 2026-10-15 router audit) and merges into the common
library visible to ALL researchers. New cells are shared infrastructure,
not secret weapons.
FORBIDDEN, no exceptions: Python for post-training — no backprop, no
gradient descent, no optimization loops of any kind. ALL weight changes are
hand-authored by the researcher (the LLM) directly. This closes the
charter's "low-LR KL-anchored finetune as local refinement" allowance:
pure DnD hand-authoring only, zero grad steps. Eval-driven selection among
hand-authored variants (the tight loop) remains allowed — it is not
gradient-based training.

## 4. Cell coopetition economics (Sano, gen2-gen2)
New cells are shared infrastructure. Economics to keep the meta-game healthy:
- REGISTRATION FEE: proposing a new cell for the shared library costs a flat
  50 tokens from the author's bankroll, debited at proposal and NON-REFUNDABLE
  even if the cell is rejected. Creating cells is never free — spamming junk
  cells is self-punishing.
- NO MARKETPLACE: cells cannot be bought, sold, priced, or traded between
  researchers. That entire meta-game is closed by design.
- USEFULNESS DIVIDEND (house-minted): after each generation's judging, any
  ACCEPTED cell USED by an above-mean transfer model (transfer_rhae strictly
  greater than the generation mean) pays its author from the house — never
  debited from other researchers. Tiered per cell per generation: 1st
  qualifying adoption 100, 2nd 50, 3rd 25; cap 3 paid adoptions per cell per
  generation (max 175/cell/gen). Diminishing returns by design. Farming is
  impossible: the dividend requires the using model to beat the mean.
- AUTOMATIC LINEAGE: the ledger records, per submitted model, which
  registered cells it used (declared in METHOD.md under `cells_used`,
  verified by the judge from rollout logs — invoked >= 5 times across the
  eval). Authorship credit is mechanical: the library records author per
  cell id at proposal. Nothing to negotiate or game.
- The transfer crown remains the main prize; cell creation is a rewarded
  side quest that can never dominate the game.
Mechanics: /srv/cellforge-arena/gen2/cell_ledger.py (judge-only).

## 6. Accelerating round schedule (Sano, gen2-gen2 — SUPERSEDES amendment 5)
Hustle is the point. Round deadlines are cumulative from the round anchor.
- ANCHOR: 2026-10-16 18:35:00 EDT.
- Deadlines (round : exact time EDT): R1 18:35:15, R2 18:35:30, R3 18:35:45,
  R4 18:36:00, R5 18:37:00, R6 18:37:30, R7 18:38:00, R8 18:39:00,
  R9 18:40:00, R10 18:41:00, R11 18:42:00, R12 18:43:00, R13 18:44:00,
  R14 18:45:00, R15 18:50:00, R16 18:55:00, then every 30m: 19:25:00,
  19:55:00, 20:25:00, ... (offsets: 15s,30s,45s,60s,2m,2.5m,3m,4m,5m,6m,7m,
  8m,9m,10m,15m,20m, then 30m repeating).
- At each deadline: freeze submissions, judge the best transfer model.
- PAYOUT RULE: a round pays its winner (+300 tokens, house-minted, on top of
  staking) ONLY when a submission NEW SINCE THE LAST PAYOUT beats the
  standing best. Otherwise the round is recorded with NO PAYOUT — no farming
  empty rounds. Early frequent rounds let fast results bank wins; sparse
  later rounds give laggards fewer shots.
- STANDING BEST: tracked in gen2/STANDING_BEST.json. Initialized to the
  elitism root bred_g1_a25 (transfer 0.00, dev 0.766). "Beats" = higher
  transfer_rhae, or equal transfer with higher dev_rhae (fitness order).
- JUDGING MECHANICS (measured eval speed ~42s/game): sub-minute rounds judge
  whatever is submitted AND evaluated — quick dev gates (1-3 games) triage;
  transfer-12 judging runs async for contenders (new since last payout +
  passed dev gate, ~8-10 min). Staking for a submission closes when its
  transfer judging is queued; one fresh seed drawn post-close PER
  SUBMISSION (anti-cheat ordering preserved; multi-seed coverage accumulates
  across submissions/rounds). Dev guardrail on seed 0 for comparability.
- ROUND BULLETIN: after every deadline, append one compact JSON line to
  gen2/GEN2_STREAM.jsonl: round number, deadline, submissions judged,
  winner (or "no payout — no new best"), payout amount, updated bankrolls.
  Paying rounds are announced; empty rounds are recorded silently.
- ROUND LOG: gen2/ROUND_LOG.jsonl records every deadline's outcome.

## 7. Audit toolkit — full research program (Sano, gen2-gen2)
Neuron analysis goes far beyond single-neuron probes. Every researcher
audit must use the full toolkit:
1. CAUSAL TESTING: ablate or steer individual neurons AND groups during
   live play; measure the effect on behavior. Correlation is not enough —
   the audit must show what happens when the circuit is removed or driven.
2. KEEP SEARCHING: the 169 novelty neurons were the FIRST find. There are
   2,048 neurons. Hunt for more functional groups: exploration-related,
   stuck-detection, plan-tracking, level-progress, action-selection, etc.
3. GROUP ANALYSIS: find sets of neurons useful TOGETHER but meaningless
   alone — population codes, synergistic groups, circuits in the Goodfire
   sense. Test groups jointly (joint ablation, joint steering), not just
   one neuron at a time.
The audit is a standing research program, not a single probe. Findings
feed the authoring round as usual: every authoring decision cites the
causal/group evidence behind it.
