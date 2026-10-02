> SUPERSEDED by docs/ONE_STANDARD.md: gene-library wording below is obsolete; author cells as literal-weight genomes.

# Parallel Codex briefs (paste one per new Codex session)

Shared context for every brief: read notes/MESSAGE_FOR_CODEX_2026-09-30_ALIGNMENT.md first. LLM-written weights are the mutation operator;
crossover breeds cells across benchmark niches (ARC-AGI-1/2/3, one model). No gradients. No API calls, no downloads without asking.
Only 11 hand-written genes exist today (see reports/genome_progress.png; rebuild with `python scripts/genome_progress.py`), so the job is to
WRITE MORE, DIFFERENT GENES, not to speed up or re-verify existing ones. Overlap between sessions is fine; duplicate genes get
deduplicated by canonical hash later. Each session works in its own run-dir prefix and git branch so nothing collides. Every new gene needs
a unit test, a description of what rule it captures, and a fixture task it solves. Report each gene's solve rate on fresh generated tasks.
Split rule: merge ARC-1 and ARC-2 tasks by id into one pool, fixed id-hash split, never report ARC-1 as its own score.

## Session 1 — object and correspondence genes (branch codex/genes-objects, runs/genes-objects-*)
Write 15+ genes: object correspondence (match by shape/color/position), largest/smallest/unique selection, object move/copy/delete,
gravity, count-to-output. Each as a reachable cell with explicit weights (literal tensors) plus a typed port contract.

## Session 2 — recolor and conditional-rule genes (branch codex/genes-recolor, runs/genes-recolor-*)
Write 15+ genes: conditional recolor by property (size, touching, enclosed, position, parity), local-rule tables, flood-fill variants,
containment and adjacency tests, majority/minority color rules.

## Session 3 — pattern, tiling and symmetry genes (branch codex/genes-pattern, runs/genes-pattern-*)
Write 15+ genes: pattern continuation, periodic tiling inference, symmetry completion (mirror, rotate, diagonal), scaling/kron,
line and path drawing (ray casting, connect-the-dots, bounce).

## Session 4 — interactive-game genes for ARC-AGI-3 (branch codex/genes-arc3, runs/genes-arc3-*)
Write 10+ genes usable by the ARC-AGI-3 agent contract (is_done/choose_action): avatar detection, action-effect tables, wall/hazard learning,
goal hypothesis testing, key/door, push-block. Test on the synthetic generator in branch claude/arc3-gan-generator
(research/claude_arc3/gan/), not on the public games' internals.

## Session 5 — crossover across benchmarks (branch codex/cross-benchmark, runs/cross-benchmark-*)
Take genes from Sessions 1-4 (or the current 11 + any in-tree genes) and build children that combine grid genes with game genes via
the shared perception layer (src/cellforge/perception/shared.py on branch claude/arc1-transfer). Report how many children are
new mixes (not copies) and whether they run on both a grid task and a synthetic game.

## Session 6 — LLM weight-mutation on real genes (branch codex/llm-mutation, runs/llm-mutation-*)
Wire niche_archive proposer (branch worktree-agent-abb53cbb3041918a9) to DirectWeightMaker. For each gene from any session, compare
LLM-written weight edits to random edits on fresh generated tasks at equal charged budget, 3 predeclared seeds. Keep the file-based
outbox/inbox protocol from scripts/run_three_arm.py so a human-supervised LLM can be the maker.
