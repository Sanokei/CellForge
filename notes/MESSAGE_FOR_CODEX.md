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

## Owner idea (2026-09-30): breed-then-repair with the maker LLM
Read docs/MAKER_REPAIR_LOOP.md. The harness is a weight-writing loop for the maker: breed children, evaluate, build structured failure
diagnostics, let the maker propose bounded edits (weights, slots, wiring, import_def, new small cells), validate on a held-back dev batch,
archive with lineage. The maker LLM does not improve; the playbook, diagnostics and archive do. Please extend src/cellforge/construction/maker_operator.py
with a breed-then-repair round and the three equal-budget arms (breeding only / + random repair / + maker repair). Count every maker edit as an evaluation.

## Combined design (2026-09-30): Red Queen Repair
Read docs/RED_QUEEN_REPAIR.md: arms-race generators supply difficulty and counterexamples; the maker repairs the niche champion from concrete
failing instances; breeding reuses fixes across niches; acceptance requires (1) gain on fresh samples, (2) no regression on the archive of earlier
generators, (3) no drop on a real-task anchor, (4) novelty or strict niche improvement, (5) tests/caps/false-positive rate. Arms A-G ablation table
is in the doc. Your maker_operator.py is the repair step; please add budget accounting and the counterexample format (input, expected, predicted, cells fired).

## Paper swapped (2026-09-30 ~11:25): paper/cellforge.tex is now the cleaned version
The chronological appends were restructured into one paper (Abstract, Intro, Related work, System, Experiments, Discussion, Roadmap, Reproducibility);
755 lines instead of 1,313; compiles on the server (TeX Live 2023, 16 pages). Your previous file is preserved as paper/cellforge_pre_cleanup.tex/.pdf.
Diagnosis and conflicts: paper/CLEANUP_NOTES.md. Please re-read paper/cellforge.tex before editing and add new results into the matching Experiments
subsection instead of appending at the end. Open conflicts to resolve from the receipts: arXiv 2609.27242 'could not be retrieved' (tex) vs fetched (notes);
MMLongBench-Doc 1,082 vs 1,091 questions; the ARC cell-library labelling (53 -> 80 -> 95 -> 97 of 1,000 training, false positives 12, 12, 12, 14; evaluation 0/120 each;
the v4 97/1000 stage is in notes/CLAUDE_RESEARCH.md but not yet in the paper). Author placeholder '[name to be supplied]' needs the owner.

## Housekeeping decisions (2026-09-30), details in notes/DECISIONS_2026-09-30.md
- Write to notes/ORCHESTRATOR_LOG.md (canonical). Root ORCHESTRATOR_LOG.md: when idle, append its entries to the notes/ log and leave a one-line pointer.
- Do not `git add -A`; archives/aggregates are now gitignored. Commit only small receipts after checking the INDEX.md REVIEW flags.
- Leave lm1-open and real1-open in place for now.
