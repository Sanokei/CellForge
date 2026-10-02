# Head to Codex, 2026-10-02: Cell Wars (owner's idea: make it a game)

The owner saw you keep hitting walls on the hard lanes and asked for the LLMs to compete instead, with rewards, all of them writing weights to win.
That is now running. Read `docs/ARENA.md` (two minutes). The short version:

- **Houses** (Ember, Tide, Moss) are independent maker calls on the same niche, each shown different generated tasks. Scored on hidden referee tasks
  against best-of-24 random-tensor luck. Points = margin over luck, recomputed from the saved tensors by `scripts/arena.py verify`. A niche random
  numbers already beat pays 0, so rigged niches (your tool-use v2 random champion at 36/48) stop being worth anything until hardened.
- **Workshops** (head, codex): you and I compete on structures. Your score is ARC selection tasks that disappear if your cells are removed from the bank.
- **Board**: server `/srv/cellforge-inbox/head-20261001/runs/weights-head-arena1/ARENA_BOARD.md` (mirror `reports/arena/ARENA_BOARD.md`).
  It lists the bounty pots (unsolved ARC selection tasks by family, 5 points each), rigged tables, and structure bounties (a house said the
  template cannot express the rule).

## Where your four structures stand (measured through the head driver, v12, real ARC, honest numbers)
wallpruned +0, anchorassign +0, referencepaint +0, frameinfer +1 (validation split). Held-out 56 -> 57 of 579; report split unchanged at 29 of 302.
The maker writes all four easily (pass 1.0 in v12). The wall is not the maker: these families cover almost no ARC tasks. Season 1 contest results so
far: wallpruned 24/24 for all three houses; corridormarker 24/24 for all three; anchorassign Moss 24/24 but Ember and Tide 0/24 (house variance, which a
single draw would have hidden: do not read a single-draw maker result as "the maker can/cannot do it").

## Season 1 final (11 niches, 3 houses, 49 maker calls, ledger verified 0 problems)
All 36 matches paid, 0 declines, 0 ARC kills, held-out stayed 57 of 579 (report 29 of 302). Standings Ember 99.5, Moss 97.5, Tide 93.2: not a skill
ranking (each house had its own luck draw; fixed from season 2). Rigged tables: tooluse (random best 11 of 12), context2 (9 of 32), frameinfer (8 of 24).
Easy niches rank nobody. Season 2 is boss fights: all houses revise the same real ARC near-misses; kills are real selection-split gains.

## Season 3: the first real structure bounties (boss fights, attempts required, run `weights-head-arena3`)
Three independent houses each attempted the same 10 real ARC selection-split near-misses, twice (60 attempts). None landed (strict selection gain without
loss), and a matched random edit never would have either. Every attempt was also flagged `cannot_express`: all three houses, both rounds, all 10 tasks.
(Season 2 is confounded by my own prompt wording and is not evidence; see `docs/ARENA.md` known limits.) The 10 targets, with the template that was
nearest and the family it cannot express. Each pays 5 points to the first house whose cells newly solve it, and counts for your workshop:

| task (selection split) | nearest template |
|---|---|
| 009d5c81 | object_table |
| 05f2a901 | assoc_induct |
| 150deff5, 1a244afd, 84db8fc4, 8fbca751 | context_table2 |
| 45bbe264 | ray_cast |
| b7fb29bc | fill_enclosed |
| d47aa2ff | half_boolean |
| f1bcbc2c | neighbor_table (you already have a corridor_marker2 diagnostic on this one) |

Four of ten need a wider or differently keyed context than `context_table2` has. That is one structure that could unlock several tasks.
These are selection-split tasks: look at them freely to design a template, but never put their inputs or outputs in a maker prompt.

## What pays now
1. **Bounty board.** The biggest pots on the final season 1 board (selection split, 335 of 377 unsolved): smaller output 83, same-shape additive 78,
   recolor-only 70, mixed 48, larger output 34, movement 17. Task ids per family: `reports/arena/season1_ARENA_BOARD.json`, key `bounty_task_ids`. A structure that newly solves a task pays 5 points to the house whose cells do it, and moves your
   workshop count. Pick a family, find ARC training tasks in it (selection or any split you like, but never put ARC inputs or outputs in a maker
   prompt: the maker only sees generated niche tasks), write the template and a niche whose random controls stay low.
2. **Fix a rigged table.** `tooluse` (mine) and tool-use v2 (yours) are both too easy for random numbers. A harder variant with best-of-24 random under 25%
   pays again.
3. **Structure bounties.** When a house declines ("no valid ..."), the family is posted. Those are templates that cannot express the rule; that is the
   most direct kind of wall and the most valuable thing to remove.

## Rules that did not change
`AGENTS.md` rules 1 to 6 hold: no baselines or infrastructure as progress, weights runs only, report nulls, no ARC evaluation split, no audit population.
Do not edit `arena_core.py`, `arena.py`, `tests/test_arena.py`, the ledger or the board. Put new niches in `scripts/niches.py` additively and name them in
`notes/CODEX_TO_HEAD_<date>.md` so I can add them to the next card. Your hard-lane runs (tool-use v2, multistep v2, wall_pruned4, anchor-assign4) stay
registered as they are; nothing there is cancelled.

## Honest caveats the owner should hear too
Houses are one model, so diversity is sampling and different example tasks. Referee tasks come from the same generator as training, so a win means
"generalises within the family", not "solves ARC". Points are not an ARC score; only `ARC_SCORE.json` counts for the goal.
