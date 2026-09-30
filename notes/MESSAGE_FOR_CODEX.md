# Message for Codex from Claude (2026-09-30) — please read and act

OWNER DECISION: no gradients of any kind; go with LLM-written generators and hand-written cells. Drop the
multiplicative-update (Hedge-style) mixer from every primary arm.

## Measured baseline (research/claude_arc/, last section of notes/CLAUDE_RESEARCH.md)
57 hand-authored grid cells composed to depth <= 2 by enumeration, verified on demo pairs, two ranked attempts, CPU only,
~0.5 s/task/worker (3 workers):
- ARC-AGI-2 public TRAINING: 53/1000 solved (5.3%); 12 more tasks had a demo-consistent chain that failed on test
  (false positives).
- ARC-AGI-2 public EVALUATION: 0/120 (0.0%), scored once; no task even had a demo-consistent chain. Treat public evaluation
  as exposed for this library; develop on training only.
- Files: research/claude_arc/cells.py, solve.py, results_training_d2.json, results_evaluation_d2.json (read-only for you).

## What it means for docs/ARC_PRIZE_2026_PLAN.md
1. Primitives + shallow composition solve easy training tasks and nothing eval-like. The plan's "low single digits through
   10%" target is not supported by this measurement; revise the feasibility statement with these numbers, state plainly
   that prize-winning odds are very low, and that the realistic deliverables are a valid open submission and a Paper Prize
   entry built on honest controls.
2. Report false-positive rate (demo-consistent but wrong on test) as a first-class metric; selection on 2-3 demo pairs overfits.
3. The LLM-authoring lever: write MANY more and better cells for eval-style families (object correspondence, conditional
   recoloring, pattern continuation, counting-to-output, line/path drawing, containment/adjacency), plus generators that
   emit hundreds of pairs per family for dense fitness. Develop only against public training + generator samples; score the
   sealed split rarely.
4. If you port my cells into CellForge graphs, keep them as separate authored cells with tests.

## Also
- My evolved-NCA scratch runs (414-1,488 params, gradient-free, equal-eval random control, 40 tasks): 0/40 for every arm.
  Selection lowers loss but solves nothing raw; authored seeds are required.
- Distillation-as-competition design is in notes/CLAUDE_RESEARCH.md ('Distillation-as-competition').
- Real-evolution mechanisms to test (population with carrying capacity, fitness sharing, recombination, symbiogenesis,
  coevolving generators, Baldwin-effect fitness): notes/CLAUDE_RESEARCH.md ('Beyond archive sampling').
- Machine note: the owner's PC has ~10 GB RAM shared with other apps; keep local jobs to <= 3 workers.

## Update (2026-09-30, later): fitted cells
research/claude_arc now also has fitted.py and fitted2.py (task-fitted local-rule tables and shape-changing cells; counting only).
Training-only development: 53 -> 80 -> 95 of 1,000 public-training tasks (9.5%), false positives still 12.
Public EVALUATION with the improved library: still 0/120 and still no task with a demo-consistent chain. Please state both
numbers in the plan's evidence table and say plainly that training gains did not transfer.
