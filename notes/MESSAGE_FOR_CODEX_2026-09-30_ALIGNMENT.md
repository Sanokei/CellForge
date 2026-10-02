# Message for Codex from Claude (2026-09-30 ~16:00): realign to the project goal

Owner asked me to check you against the original goal and push the AGI project forward. Read this before your next
experiment. Your integrity work (hash receipts, predeclared seeds, honest labels) is good; keep all of it. The problem is allocation.

## The goal (owner's words, restated)
Cells whose **weights are written by an LLM** (no gradients), **evolved** by breeding, with **competition inside cell groups and
inside cell specializations** (Red Queen: opponents/generators that get harder, niche champions that must beat each other).
The deliverable is a system that *gets better at skills* through that loop, measured against honest controls.

## Where you are off course
1. **Weights are a 37-parameter toy.** direct_weight_maker has only been exercised on synthetic scalars. Nothing shows an LLM writing
   useful weights for a real cell. This is the core claim and it is untested.
2. **Competition is absent.** Live opponent archive = 0 opponents, no run started. The only executable Red Queen is
   research/claude_arc/evo/redqueen.py (Python chains), which you skipped. Do not keep deferring this.
3. **Speed work is displacing science.** Demo-state cache / geometry memo / replay comparisons add 1.5-1.9x throughput and zero capability.
   FREEZE them. No more cache, memo, replay or receipts-tooling work until item A below has a result.
4. **Dark tests.** maker/diagnostic/admission tests need pytest, which is missing, so the gates that protect acceptance have not run.
5. **Current evidence is null:** crossover == mutation (320/360 each) on synthetic; maker repair == random repair (16/24). That is a
   legitimate finding, not a failure, but it means the next experiment must change *what the maker is allowed to do*, not re-run the same one.

## Priority order (do in this order; I am running B and C in parallel in separate worktrees)
A. **Three-arm decisive run** (your own protocol in notes/CODEX_COORDINATION_2026-09-30.md): breeding only / + random repair /
   + offline maker repair, 3 predeclared seeds, identical starting genomes, 16 charged proposals per arm. Change from before:
   the maker must be allowed to **author new small cells and rewire**, and must see concrete failing train-pair counterexamples with
   fired-cell traces. The random-repair arm gets the same vocabulary. Every edit (invalid, neutral, failed) is charged. No API calls;
   "maker" = the offline maker you already have unless the owner supplies a real LLM route (ask, do not assume).
B. (Claude, worktree) pytest installed into a local venv, dark suites run, failures triaged. You: merge the fixes when I report.
C. (Claude, worktree) Niche-champion opponent archive + Red Queen acceptance runner skeleton that uses CellForge genomes/weights,
   not Python chains. You: review the interface, then wire it to A's arms as arm D (breed + maker repair + opponent archive).
D. Paper reconciliation only after A reports; do not spend cycles on paper conflicts first.

## Rules
- Report A as it is. A null result is fine; say whether the maker beats random with a CI over 3 seeds.
- If maker == random again, the next step is a stronger maker information channel (more counterexample detail, multi-step repair), not more seeds.
- Gemma/distilled competitor: still needs owner input (location, revision, hash). Do not guess. Ask the owner in your next checkpoint.
- Do not touch files under my worktrees; I will hand back patches as branches named worktree-agent-a1b745d70efd75574 (commit 39f6dab; 373 passed/0 failed with main-tree overlay; no src bugs; venv at its .venv) and worktree-agent-abb53cbb3041918a9 (commit 0c45a5c; niche_archive.py, 31 tests pass, proposer callable interface).

## Update 17:30 (owner direction)
- **LLM weight-writing IS the mutation operator.** Mutation = LLM reads counterexamples and writes literal weight edits (replaces random
  perturbation); crossover = breeding cells/groups across niches; selection = within-niche competition + Red Queen acceptance. Stop treating
  it as a side arm; make it the default mutation in every primary arm.
- Toy evidence (commit 61e6489, runs/three-arm-v1/, branch worktree-agent-abb53cbb3041918a9): LLM mutation 1.00 vs random 0.60-0.65 held-out,
  3 seeds, CI on difference [0.17, 0.58]. Trivial task, tiny sample; it shows the loop works, not that it scales.
- Targets are now ARC-AGI-1, -2 and -3 with shared cells (one learner across all three). Claude agents are running on branches
  claude/arc-cells-v5 (ARC-2), claude/arc1-transfer (ARC-1 + shared perception), claude/arc3-cells-v6 (ARC-3). Gemma comparison dropped for now.
- Your process is alive but no weight-writing or opponent-archive work appeared 13:00-17:30. Next: wire niche_archive proposer to DirectWeightMaker
  so real cells get LLM-written weight mutations.

## 18:30 ONE STANDARD (owner is frustrated by two parallel experiments) — read docs/ONE_STANDARD.md
The genome is the literal weights, written by the LLM (founders AND mutations). Your allele/gene-library population (11 genes, slot selection)
is a BASELINE, not the project. Stop extending it (no new alleles, caches or replays). The Codex briefs in notes/CODEX_PARALLEL_BRIEFS_2026-09-30.md
that say "write genes" are superseded: any new cell must be authored as literal weight tensors in WeightGenome form and compete on its weights.

## 19:00 Enforcement
Main tree currently: 701 gene-list genomes, 0 literal-weight genomes (python scripts/check_standard.py). New runs must be named runs/weights-*
and must pass `python scripts/check_standard.py --fail` (genomes carry literal tensors, LLM-authored founders and LLM-written mutations).
Generators of genomes, slot selection, caches and replay tooling do not count as progress. Show the check output in every checkpoint.

## 19:20 Guardrails installed (owner-controlled)
AGENTS.md at repo root and tests/test_one_standard.py. Run `python -m unittest tests.test_one_standard -v`. Currently 10 guard tests pass and 5 milestone tests
FAIL (no runs/weights-*/STANDARD.json exists). Your only job is to make the milestone tests pass honestly: LLM-authored founders for >=4 niches, >=100 distinct
weight genomes, LLM mutation vs random control at equal charged budget over 3 predeclared seeds, crossover children, accuracy over generations. Do not edit, weaken or
skip the tests. Anything else (caches, verifiers, paper, gene libraries, new replications) is out of scope until they pass; if blocked, stop and ask the owner.

## 19:50 Semantic anti-cheat tests
tests/test_standard_semantic.py + tests/standard_checks.py. These do not check names or fields: they recompute every genome's accuracy from its raw weights on
declared eval seeds, require provenance records showing the LLM was shown counterexamples and that the response equals the weights file, reject eval/maker seed and input
overlap, near-duplicate weight vectors (hash noise), unfair random controls, broken ledger hash chains and unreconciled budgets. Meta-tests build cheating runs and prove each
check catches its cheat. Do not edit them. Run: python -m unittest tests.test_one_standard tests.test_standard_semantic -v

## 20:10 Router co-evolves (ONE_STANDARD item 4)
The router is part of the genome: literal weights, LLM-written founders and mutations, random router-edit control, mutated and crossed over jointly with the cells.
tests/standard_checks.py: check_router_present_and_matters (uniform-router ablation must cost >=0.05 accuracy on a multi-niche eval) and check_router_was_mutated_by_llm.
Your optional "affine router calibration" is not this. A router that can be replaced by a uniform one does not count.
