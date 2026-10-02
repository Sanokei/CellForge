# LAB.md — the one page to read first (orchestrator, Claude; owner away; last updated 2026-10-02 ~07:55Z)

**Mission, unchanged:** literal weight tensors written by an LLM, competing and recombined, measured against random-weight controls, one model across ARC-AGI-1/2/3
(`docs/ONE_STANDARD.md`). **Honest status:** nothing in this lab is evidence of general intelligence. What exists is an LLM maker (gpt-6.1-sol, server `codex exec`) that
writes sandboxed programs plus small tensors, and a verifier that recomputes everything from the saved weights. That is useful machinery and a real, measured capability of the
*maker*; it is not AGI and the numbers below must not be quoted as if it were.

## 1. Numbers (each recomputed from saved weights unless marked; ARC data never leaves the server)
| what | result | controls | caveat |
|---|---|---|---|
| ARC-1/2 public-training pool, skill bank (narrow templates, tensors written for GENERATED niches, never saw ARC tasks), report split 302 | 29 / 302 (9.6%) | zero 0, random 0, 0, 1 | the clean transfer number: weights written for other tasks solve unseen ARC tasks |
| same pool, **structure bank** (test-time authoring: houses write code + <=256 tensors from each task's demonstrations; protocol frozen, run once), report split 302 | **287 / 302 (95.0%)** recomputed in a clean process, test outputs stripped; `tests.test_arc_goal` 14/14 green on the server | zero 0, random 0, 0, 0 | **only 38 of the 287 survive removing the cells authored for that same task (26 skill-niche cells, 12 structures from other tasks); 249 (87%) exist only because a structure was authored from that task's demonstrations** (`reports/arc12/TRANSFER_AUDIT.json`). Public ARC training tasks are widely published, the maker may recall them; colour-permuted+transposed probe 18/20 |
| ARC-3 (25 public games): scripted state-graph explorer, no LLM | 17 levels at 3000 actions | random agent 5 levels | baseline, from `/srv/arc3-claude/logs` |
| ARC-3 managed LLM weight writing (`runs/weights-arc3-managed1`, 59 manager/A-B cycles, 175 writer attempts) | **33 credited levels in 19 of 25 games** (lp85 8/8 won in 332 actions, tu93 4/9, sc25 3/6, vc33 3/7, 15 games 1 each), recomputed in a fresh process, 0 mismatches (`reports/arc3/VERIFY.json`) | zero tensors, 3 random draws, BLIND run: 0 in every credited game; random agent 4 levels, scripted explorer 17 (at 3000 actions) | public games, each cell tuned to its own game by feedback; cross-game transfer 2 of 168 plays; randomised A/B: donor cells did not help (3/17 vs 3/15). Not a generalisation claim |

Read the structure bank honestly: it shows the maker route + sandbox + acceptance pipeline works end to end and that the maker can write a correct program for almost any
ARC-1/2 training task it is shown. It does not show that stored weights generalise (`TRANSFER_AUDIT.json` splits that out) and it says nothing about novel tasks. The only
novelty test the lab owns is the ARC-AGI-2 public EVALUATION split (120 tasks), which the owner excluded; revisiting that is the owner's decision (section 6).

## 2. Regime (what is running, what is closed)
- **Nothing is running right now** (checked 07:50Z: no driver, no scoring job; `python scripts/lab.py status`). Track A (ARC-1/2) is closed: final score packaged in `runs/weights-head-structbank-final/`,
  `tests.test_arc_goal` 14/14 green on the server, transfer split in `reports/arc12/TRANSFER_AUDIT.json`. Track B (ARC-3 policy cells under an LLM manager, `scripts/lab_manager.py`) finished
  three campaigns and is the only open research line; resume it with `python scripts/lab_manager.py run runs/weights-arc3-managed1 --cycles N --slots 3` on the server (resumable, stops at the budget cap or a STOP file).
- **Closed (stand-down, nothing deleted):** all Codex lanes (arithmetic, reasoning, spatial, signed-deduction, resource-planning, tool-use v2, multistep v2, wall/anchor/frame/
  reference/corridor lanes), the arena, the tournament, the skill-bank driver, every `weights-*` pilot. Their receipts and tensors stay as evidence. Reason: in the closed lanes
  random tensors matched or beat the maker (multistep: random 44/48 and 46/48 vs maker 41/48; tool-use v2: random champion 36/48), or the maker solved them in generation 1.
- **Managers:** `lab_manager.py` runs one manager call per cycle that orders writer slots (which game, which house, what to change). The code guarantees that every cycle
  submits weights: a failed or empty manager call is replaced by a deterministic fallback, a writer that "declines" must still submit an attempt, crashes are logged and the
  other slots continue. Managers write no weights.
- **Budget:** the ChatGPT weekly limit is ONE pool for this laptop's Codex app and the server maker route. 57% used at 2026-10-02 03:10Z, 61% at 07:26Z after this whole session (230 maker calls, 1% ~ 58 short calls), resets ~2026-10-06 21:33Z. Every server
  maker call goes through `scripts/maker.py`, which logs the meter to `BUDGET.jsonl` and refuses at the cap (default 85%, `BUDGET_CAP.json`). How the 50% used before this session splits between desktop turns and server
  lane calls is NOT measured (a single 90-minute desktop turn used 12M input tokens, so that is a plausible large share; that is inference).

## 3. Where things are
- Laptop repo: source `scripts/`, tests `tests/`, notes `notes/`, docs `docs/`, evidence `runs/` (486 entries, 861 MB; do not `git add runs/` unreviewed, see `runs/CLEANUP_PROPOSALS.md`).
  `git status`: ~845 untracked paths and no remote; nothing since 2026-09-30 13:08 is committed.
- Server `cellforge-1` (ssh alias; 2 CPUs, no GPU, 29 GB free): `/srv/cellforge-inbox/head-20261001/` (orchestrator work dir: scripts, `runs/weights-head-*`, `BUDGET.jsonl`),
  `/srv/cellforge-inbox/two-loop-20261001/` (Codex lane dirs, untouched), `/srv/cellforge-inbox/_archive_20260930/` (219 older entries moved there on 2026-10-02 with
  `MANIFEST.txt`; nothing deleted), `/srv/arc3-claude/` (ARC-3 games and venv), `/srv/cellforge-private/arcagi2/data` (ARC data, never copied out).
- One command: `python scripts/lab.py status` (budget, processes, ARC-3 status, final score); `sync`, `stop`, `budget N`.

## 4. Rules that keep the numbers honest (unchanged, condensed from the handoff)
Held-out = 1000 public-training tasks minus the 100 DEV ids minus the SHA256-mod-5 audit population; the report split (302) is never used to select, tune or retry anything;
never read the ARC evaluation split; the maker sees only generated tasks or a task's own demonstrations and test INPUTS; every maker call is charged and logged; controls get the
same positions; nothing self-reported counts, only recomputation; never edit `ARC_FLOOR.json`, `HARD_LANES.json`, the standard/arc-goal tests.

## 5. Known red (left red, not weakened)
`tests.test_one_standard` + `tests.test_standard_semantic`: 5 failures, all in two legacy local pilots (`runs/weights-cell-population-g1`, `runs/weights-arc-best-a5-v1`), which
already declare their limits in their own STANDARD.json. `tests.test_arc_goal` needs `ARC_ROOT` (server only). Owner to archive or fix.

## Owner decisions and Codex resumption (2026-10-02)
The owner returned and authorized committing/pushing/merging the source snapshot,
archiving both legacy pilots, and resuming useful research. This supersedes the
owner-away stand-down for the bounded experiment below; other closed lanes stay closed.
The ARC-3 direction was explicitly delegated to a subagent.

Archive state verified locally: `weights-arc-best-a5-v1` is preserved under
`runs/_archive_legacy/`; `weights-cell-population-g1` retains its location for
population recomputation, with `STANDARD.archived.json` and `ARCHIVED.md`.
No tests were changed. On 2026-10-02 the two standard suites ran 25 tests in
38.842 seconds, OK. The previous five failures are historical, not current.

Research decision: stop spending on routine game-specific ARC-3 campaigns.
Test a single shared policy/tensor set that learns from its action/frame history,
without game-ID routing. First step: one metered founder proposal on a frozen
three-game development panel, with zero, three random-tensor draws and blind
controls, then stop and assess. See `docs/ARC3_NEXT_EXPERIMENT.md` for the full
bounded protocol. Existing public games are development evidence, never newly
held-out games. Neither a passing pilot nor the existing ARC floor establishes AGI.
The ARC-AGI-2 evaluation split remains excluded unless the owner explicitly lifts
that restriction; a clarification is pending.

Broader architecture verification was also run: 78 tests in 47.756 seconds,
FAILED (2). `test_action_verification` rejects the corrected-g3 saved population
against changed installed `cell_population` source and differing replay budgets /
membership; `test_circuit_population` expects admission using claimed outcomes
without the execution evidence now required by `CircuitArchive`. These failures
predate the shared-policy runner changes. Do not weaken the fail-closed admission
or rehash historical receipts to hide them. Full log:
`reports/owner-resumption-gates-20261002.log`.

Fresh server ARC goal recomputation (existing `goalcheck3` package): 14 tests in
304.723 seconds, OK. This validates the saved public-training result and the
owner's existing floor; it is not new-game evidence.

First shared-founder probe `runs/weights-arc3-shared-probe-20261002`: zero levels,
zero actions on all three games because the source called unavailable builtin
`str`. Controls crashed too. Independent replay matches. This is a runtime
invalid proposal, not evidence against shared adaptation. One separate charged
runtime-repair probe was started with the exact error and allowed builtin list,
no game observations or tuning feedback; it stops after one proposal as well.
The maker meter moved from 61% to 62% across the first probe (shared desktop
activity included; this delta cannot be attributed to that call alone).

The new probe is below ONE_STANDARD: one seed, no charged random-edit population,
no crossover/router qualification, no standard semantic evaluator. It truthfully
ships STANDARD.json and remains red: 25 standard tests in 4.666 seconds, two
failures and three errors. See `reports/shared-probe-standard-qualification-20261002.log`.
The legacy archival was not used to conceal these new qualification failures.

## 6. Decisions only the owner can make
1. ARC-AGI-2 public EVALUATION split: run the frozen structure-bank protocol once on its 120 tasks (about 120-250 maker calls, roughly 2-5% of a weekly limit). This is the only
   honest novelty test; without it "95% on training" means "the maker knows ARC training tasks".
2. Commit policy: ~845 untracked paths, no remote. Suggested first step: commit `scripts/ tests/ docs/ notes/ src/ reports/ LAB.md AGENTS.md` (not `runs/`).
3. Whether to archive the two legacy runs that keep the standard tests red.
4. Whether Codex should resume at all, and on what, given the shared weekly limit.
5. How far to push ARC-3: 19 of 25 public games have a credited level and growth had flattened (31 -> 33 credited levels over the last ~14 cycles). Another campaign costs about 1% of the weekly limit per 58
   calls. The open design question is generalisation (cells are game-specific: cross-game transfer 2 of 168 plays); a meaningful next step is a held-out set of ARC-3 games, which the lab does not have.
