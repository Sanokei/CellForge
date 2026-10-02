# ONE STANDARD (owner decision, 2026-09-30) — supersedes every other genome format

Everything in this project runs under this loop. Anything not under it is either a baseline or archived.

1. **The genome IS the weights.** A genome is a set of literal weight tensors for a cell (or a group of cells) — not a list of slot names
   chosen from a fixed gene library. Reference implementation: `WeightGenome` in `src/cellforge/construction/niche_archive.py`
   (branch worktree-agent-abb53cbb3041918a9), which wraps literal tensors, a template name, a content hash and lineage.
2. **Founders are written by the LLM.** The LLM authors the initial weights for each cell/niche directly (from a task description and
   examples), as literal tensors. Hand-coded founder *libraries* are not the population; they are at most baselines.
3. **Mutation is written by the LLM.** The LLM reads concrete failing examples (input, expected, predicted, cells that fired, pre-activations)
   and writes literal weight edits. The random-edit arm is the control, given the same edit vocabulary and the same budget.
4. **The router is part of the genome.** A joint genome = cell weights + router weights (literal tensors that choose which cell(s) handle an input).
   The LLM writes founder routers and router mutations from counterexamples (wrong-cell firings), exactly like cell weights; random router edits are the control;
   router and cells are mutated and crossed over together and compete together. No gradients. Without a router, merged single-skill cells interfere (a flip cell fires
   on every task), so crossover cannot help; a run whose router can be replaced by a uniform router with no accuracy loss does not count.
5. **Crossover** swaps or merges cell weight groups between parents (including across benchmark niches ARC-1/2/3).
6. **Competition**: within a niche (champion vs challenger on fresh generated tasks) and across niches; Red Queen acceptance
   (fresh gain, no regression on replay archive, no drop on real-task anchor, novelty or strict improvement, caps/false-positive rate).
7. **Every proposed edit is charged** to the budget, including invalid and neutral ones. Budget is equal across arms.
8. **Same model across ARC-AGI-1/2/3**; benchmarks are niches; ARC-1 and ARC-2 are merged by task id (no separate ARC-1 score).
9. **No gradients, no API calls unless the owner supplies a maker route.** The offline/human-supervised LLM maker uses the
   file outbox/inbox protocol in `scripts/run_three_arm.py`.

## What the existing pieces are
- `research/claude_arc/` cell libraries, Codex's 11-gene allele population (`runs/*/ledger.jsonl`), and the synthetic-gene runs are BASELINES
  (hand-authored or selection-only). They exist to show what authored weights must beat. They are not the project.
- The toy (`runs/three-arm-v1/`): LLM mutation 1.00 vs random ~0.62 held-out, 3 seeds, 4 niches, 37-parameter template. This is the
  standard's proof of concept. Its weakness is scale: one template, one trivial skill.

## The next milestone (single one)
Scale the standard: many weight-cell templates (real tensors: conv kernels over one-hot colors, color maps, gates, selectors),
LLM-authored founders for each niche, LLM mutation from counterexamples, competition + crossover, on generated ARC-style tasks,
with the random-weight and scripted controls. Report how many distinct weight genomes exist and their accuracy over generations.

## Required two-level architecture gate (owner clarification, 2026-10-01)
Individual literal-weight cells compete within typed structural/skill niches and preserve historical opponents. Cells earn cooperation through measured development competence; circuits/MoEs consume selected live cells. Whole MoEs maintain a separate lineage and admission archive. Ensemble-only or slot-only evaluation does not satisfy this requirement. Structural/interface repairs must preserve inherited-versus-authored numeric provenance and be charged.
Required command: `python -m unittest tests.test_one_standard tests.test_standard_semantic tests.test_cell_population_contract tests.test_action_attribution tests.test_action_verification tests.test_adhesion tests.test_circuit_population tests.test_action_growth_integrity tests.test_action_phase_protocol tests.test_action_cell_continuation tests.test_experiment_manifest -v`.
The first independent-cell pilot is `runs/weights-cell-population-g1`: single seed, two authored cell proposals, one aligned crossover, three paired random controls. It is not a balanced three-arm/three-seed milestone and has no admitted children. Its STANDARD.json records those limitations; do not relabel it as a completed scientific milestone.

## Independent action-population replay
Use python -m cellforge.construction.action_verification runs/weights-action-population-corrected-g3 to independently recompute recorded weights, matched operator budgets, live-parent selection, birth/retirement/archive transitions and literal joint fixture effects. JSON schema remains cellforge-verified-action-trial/1; integrated receipts add kind=independent-population. Missing/stale/rehashed fabricated evidence fails closed. runtime_acceptance is scoped mechanism acceptance; scientific_acceptance and scientific_headline_allowed remain false until the full architecture/scientific gates pass. Preserve failed historical runs and do not substitute a positive fixture metric for ARC transfer. Corrected g3 repeats existing frozen reporting solely for control correction; it is not fresh held-out evidence.
