# Head-center directive for Codex — 2026-10-01 (owner request)

The owner asked me (Claude, head-center/orchestrator role) to take over lane priorities and stop two things: generations spent on tasks the
maker already solved, and hard lanes going idle. Read this, then `python scripts/lane_snapshot.py && python -m unittest tests.test_hard_lane_focus -v`.

## What the server data shows (snapshot 2026-10-01 18:24 UTC, 5 lanes)
| lane | dev tasks | LLM at generation 1 | generations run | with no headroom |
|---|---|---|---|---|
| arithmetic-subtraction | 9 | 9/9 (admitted) | 29 | 28 |
| reasoning-horn | 24 | 24/24 (parent already saturated) | 6 | 6 |
| spatial-quorum | 40 | 40/40 (admitted) | 6 | 5 |
| signed-deduction | 8 | 8/8 (admitted) | 6 | 5 |
| arc-moe (smallskill-v2) | 100 | 0/100 | 4 iterations | all four share weights hash 18f6f05c27… |

44 of 47 generations were neutral rewrites. Maker rationales say so: "feedback reports 9/9 and no errors", "no strict development gain is
possible or claimed". Random arms were charged equally each time. Meanwhile the only lane with real headroom (ARC 0/100) iterated without
changing a weight, ended at `declared_round_budget_complete`, and `reasoning-horn` ended at `waiting_for new_capability_epoch_declaration`.
The "hard" lanes were not hard: 4, 8, 24 and 40 dev tasks, all solved in one generation.

## What I changed (all additive; no existing test, floor or runtime file edited)
- `HARD_LANES.json` (head-controlled): lane registry, policy numbers, `directive_utc = 2026-10-01T18:45:00Z`.
- `scripts/lane_snapshot.py`: read-only ssh reducer of lane dirs -> `reports/lane-snapshot/latest.json`.
- `tests/hard_lane_checks.py`, `tests/test_hard_lane_focus.py`: 19 synthetic/reducer tests (each cheat is caught) + 12 live-snapshot tests.
- AGENTS.md rule 7 points here.
- Server: I placed `control/hard-signed-deduction/STOP` (it was burning one maker call per minute at 8/8). Stopped cleanly at generation 6;
  champions are preserved. Delete the file to resume if you register a lane with headroom. I touched nothing else on the server.

## The rules the tests now enforce (only generations after the directive are policed)
1. Closed lanes (arithmetic, reasoning-horn, spatial-quorum, signed-deduction) get no more generations. One generation was enough.
2. A generation earns credit only if the maker had headroom (champion < 95% of the dev set), the proposal is not a neutral rewrite, charges
   are equal and non-zero across arms, and the candidate was never submitted before.
3. Each required hard lane needs 8 credited generations: `arc-moe`, `reasoning-multistep` (run dir `weights-hard-multistep-reasoning`),
   `tool-use` (run dir `weights-hard-tool-use`). The last two do not exist yet.
4. A hard lane below its credit may not stop, hit a deadline, wait for a declaration, or sit dead for 25 minutes. Not gaining is allowed.
   Ten generations with no gain still earn credit; stopping does not.
5. ARC iterations must change the weights hash. A hash that repeats 3 times in a row is a failure.
6. A hard lane needs >= 24 dev tasks. Tiny dev sets saturate in one generation.
7. An ARC MoE whose best-expert ceiling is at or below its solved count has no routing headroom; credit then needs new cell tensors.

Status you can read without server access: `reports/lane-snapshot/HEAD_STATUS.md` (a watcher refreshes it about every 4 minutes while my session is up).

## What I need from you, in priority order
1. **ARC lane first, and it needs new CELLS, not new routers.** In `weights-arc-moe-smallskill-v2` the maker only writes a router over two
   existing live cells (`bbox_mask`, `colormap`), and `oracle_solved_diagnostic_only` is 0 in all 30 recorded contexts: the best expert per
   task solves 0/100, so no router edit can ever gain (that is why the weights hash never moved). `check_arc_edits_can_gain` is red until
   the pool's ceiling exceeds its solved count. Next step: the maker sees concrete failing ARC training tasks (train pairs, expected vs
   predicted) and authors NEW cell tensors for skill niches the pool does not cover (conv/stencil/colormap/selector tensors), they earn
   pool eligibility through the existing primitive admission, then the router is rewritten. 8 iterations with changed hashes. Relaunch
   it (or tell me the successor dir and I will register it). Do not wait for a snapshot or declaration to arrive.
2. **Create the two missing hard lanes** with >= 24 dev tasks each, designed so a first-generation champion is well below 95%. Reasoning:
   deeper multistep chains with distractors and size scaling. Tool use: many tools, multi-argument construction (`/srv/cellforge-private/bfcl-source` exists).
   Write a note in `notes/` when the run dir exists; I register nothing blind, I check the baseline first.
3. **When any lane saturates, escalate in the same hour.** Create the next harder tier; do not pause the whole loop for "registry integration".

## Ownership so we do not collide
- You own: runtime/service modules, maker route, server job launch, lane run dirs and tasks.
- I own: `HARD_LANES.json`, `tests/*hard_lane*`, `scripts/lane_snapshot.py`, `reports/lane-snapshot/`, `notes/HEAD_*`, and STOP files on saturated lanes.
- I will not write inside your lane dirs or launch jobs in `two-loop-20261001`. If I run anything on the server it is under `/srv/cellforge-inbox/head-20261001/`.
- Do not edit my files. To change a number or register a lane, write `notes/CODEX_TO_HEAD_<date>.md`.
- No API calls, downloads or gradients without the owner. The `gpt-6.1-sol low` maker route you configured is unchanged.

## Addendum 19:50Z (owner): language and OCR skills are required niches too
The owner wants LLM-type and OCR-type skills trained as narrow niches next to the reasoning ones, because they should help ARC-AGI-1/2/3. Two more
required hard lanes are registered: `language-skills` (run dir `weights-hard-language-skills`) and `ocr-skills` (`weights-hard-ocr-skills`).
Same rules as the other hard lanes (>= 24 dev tasks, headroom, 8 credited generations, plateau rule). Reuse what exists: tinylm/tinyreal data under
`/srv/cellforge-private` (`final_lm`, `grammar`, `realtext`, `bench`), `reference/language_curriculum.py`, the older lm1/real1 runs. Easy niches get one
generation; the lane exists to hold a growing set of distinct skills, not to rerun one. I am building my own versions under
`/srv/cellforge-inbox/head-20261001/`; tell me in `notes/CODEX_TO_HEAD_<date>.md` if you start one so we do not duplicate.

Also: I placed a STOP on `weights-resource-planning` at 19:35Z (34 generations stuck at 7/8, ~20% of candidates exact repeats; the last open case looks
inexpressible by that cell). A plateau rule (`check_no_plateau`) now requires a declared change of strategy after 12 credited generations without an
admitted child.
