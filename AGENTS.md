# AGENTS.md — read before doing anything in this repo

> **Owner returned, 2026-10-02:** the owner explicitly authorized source
> commit/push/merge, archival of the two legacy pilots, and Codex resumption on
> research that targets reusable capability. ARC-3 direction was delegated to
> a subagent. See the owner-decisions section in `LAB.md` for the bounded shared
> policy experiment. This supersedes the owner-away stand-down for that scope.
> Closed lane campaigns stay closed; all provenance, controls, budget limits and
> unmodified test/floor requirements remain. ARC-AGI-2 evaluation access remains
> excluded until the owner explicitly authorizes it.

> ## ORCHESTRATOR REGIME (2026-10-02, owner handed full control of the lab to the head-center orchestrator, Claude) — READ FIRST
> The owner is away and cannot answer questions. The orchestrator decides. Until `LAB.md` says otherwise:
> 1. **Stand down.** Do NOT start, resume or "continue" any lane, niche, arena, tournament, hard-lane generation, Red Queen, genome or infrastructure work. No maker
>    calls, no `codex exec` on the server, no new run dirs. All 20+ lanes are closed (see `LAB.md`). Do not edit `HARD_LANES.json`, tests or floors.
> 2. **Why:** the weekly ChatGPT limit is ONE shared pool for this desktop app and the server maker route (measured: 57% used on 2026-10-02 03:10Z, resets
>    2026-10-06 ~21:33Z). It is the lab's real budget. The orchestrator meters it (`scripts/maker.py`, server `BUDGET.jsonl`) and stops at 85%.
> 3. **Single source of truth is `LAB.md`** (state, tracks, numbers, commands). Read it before touching anything. If you have something to say, write
>    `notes/CODEX_TO_HEAD_<date>.md` and stop; do not start work to "stay busy". Rule 6 below (stop and ask) is now the default posture.
> 4. Rule 5 (the ARC goal test) and rule 7 (hard lanes) below are superseded by `LAB.md`: the goal file is green with the structure-bank score, and lane credits are met.

The ONLY project goal: genomes ARE literal weight tensors, written by an LLM (founders AND mutations), competing on those weights
(within and across niches), crossed over, under equal charged budgets, against random-weight controls. One model across ARC-AGI-1/2/3.
Full spec: `docs/ONE_STANDARD.md`. Owner decision, 2026-09-30.

## Hard rules
1. Do NOT write genome generators, gene libraries, slot/allele selection, caches, replay tooling, verifiers or paper edits as "progress".
   Those are baselines or infrastructure. They do not count until a weights run exists that shows the loop working.
2. Every new experiment lives in `runs/weights-<name>/` and ships `STANDARD.json` (schema in `tests/test_one_standard.py`).
3. Before claiming anything, run `python -m unittest tests.test_one_standard tests.test_standard_semantic -v` and paste the result. The semantic tests RECOMPUTE accuracy from the raw weights, verify LLM provenance (prompt, counterexamples shown, response == weights), seed separation, near-duplicate weights, random-control fairness, ledger hash chain and budget reconciliation. Faking a field does not pass them. If the milestone tests fail, say so and
   work on the failing milestone next. Do not skip, weaken or delete these tests; changing them needs the owner.
4. Report nulls honestly. No API calls, no downloads, no gradients without asking the owner.
5. DONE MEANS `python -m unittest tests.test_arc_goal -v` IS GREEN, not "the loop ran". It requires `runs/weights-*/ARC_SCORE.json`
   (contract in `tests/arc_goal_checks.py`): ARC-AGI task accuracy RECOMPUTED from the final weights (test outputs stripped), beating
   zero/random-weight controls, a per-iteration history that improves, and `ARC_FLOOR.json` (owner-controlled, never edit). Never end
   a turn, report "complete" or idle compute while this file is red; launch the next generation instead. Do not edit these tests,
   the floor, or route around them (baseline cells, answer lookups, evaluators that ignore the weights). ARC data: set `ARC_ROOT`
   (no downloads without asking the owner).
6. If you have nothing to do for the current milestone, stop and ask; do not invent side projects.
7. HARD LANES (head-center directive, owner request 2026-10-01; read `notes/HEAD_TO_CODEX_2026-10-01.md`). A lane the maker already
   solves gets ONE generation, never more: re-running it only produces neutral rewrites. Hard lanes (ARC, multistep reasoning, tool use)
   must keep generating changed literal weights from failing cases for at least 8 credited generations even with no gain; difficulty is
   not a reason to stop, wait for a declaration, or switch to infrastructure. `python scripts/lane_snapshot.py && python -m unittest
   tests.test_hard_lane_focus -v` enforces this; do not edit `HARD_LANES.json`, `scripts/lane_snapshot.py` or the `*hard_lane*` tests.
   Ask for lane registration in `notes/CODEX_TO_HEAD_<date>.md`.
8. HANDOFF (2026-10-01): a working skill-bank loop exists (held-out ARC 1 -> 57 of 579 (report split 29 of 302), 52 niches defined, maker-written). Read
   `notes/HANDOFF_TO_CODEX_2026-10-01_HEAD.md` before doing anything: section 3 lists the rules that keep the numbers honest, section 5 is your task list.
9. CELL WARS (owner request 2026-10-02; season 4 is a head-to-head knockout tournament, `scripts/tournament.py`, board `TOURNAMENT_BOARD.md`): the makers now compete. `scripts/arena.py` runs several independent maker houses on the same niche, scores
   them on hidden referee tasks against best-of-24 random-tensor luck, credits ARC selection-split kills by ablation, and writes
   `runs/weights-head-arena1/ARENA_BOARD.md` (houses, workshop duel, bounty board, rigged tables, structure bounties). Points are recomputed from saved
   tensors (`python scripts/arena.py verify <run>`; `python -m unittest tests.test_arena -v`); nothing self-reported counts. To play: add a niche
   additively in `scripts/niches.py`, name it in `notes/CODEX_TO_HEAD_<date>.md`, read the board, take a bounty. Rules 1-6 are unchanged: the board
   rewards weights that generalise, not infrastructure. Rules and caveats: `docs/ARENA.md`.
10. MAKER-PROPOSED STRUCTURES (owner request 2026-10-02, "do all three"): houses may write the structure as restricted Python plus its literal tensors
   (`scripts/structcode.py`, `structlib.py`, `struct_pilot.py`, `struct_solve.py`; rules, safety, acceptance and limits in `docs/STRUCTURE_PROPOSALS.md`).
   Never run model-written code outside the sandbox. A proposal counts only if the zeroed and random tensor controls fail. The ARC report split is run ONCE per
   protocol and its outcomes are never used to tune a prompt, an acceptance rule or a retry. `runs/weights-head-skillbank-final/` ships `ARC_SCORE.json`
   (report split) and `tests.test_arc_goal` is green on the server with `ARC_ROOT=/srv/cellforge-private/arcagi2/data`; the 1% floor in `ARC_FLOOR.json` is far
   below "beat ARC-AGI-1" and is the owner's to raise.
