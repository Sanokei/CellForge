# Cell Wars: the makers compete

Owner request 2026-10-02: stop having one maker grind against a wall; make the LLMs compete so that writing weights that win is the whole game.
This is a layer on the existing skill-bank loop, not a new project: every point is earned by literal weight tensors a maker wrote.

## Players
- **Houses** (Ember, Tide, Moss, Flint): independent `gpt-6.1-sol` maker calls (server `codex exec`, ChatGPT subscription route, no tools, no files).
  Same prompt template; each house is shown *different generated example tasks* and keeps its own lineage, so founders are independent draws
  (the earlier "three seeds, one response" flaw cannot happen).
- **Workshops** (head, codex): the authors of structures and niches. Their score is how many ARC selection tasks would be lost if their cells were removed.

## A match
1. Pick a niche. Each house writes tensors for it (instance niche: one cell per pattern; general niche: one cell for the family), up to 2 generations
   with counterexamples from its own failures.
2. **Referee**: hidden tasks from a seed family no house trains on (`referee_seed`), regenerated from the niche seeds at verification time.
3. **Luck**: best of 24 random-tensor entries with the same number of cells and positions, scored on the same referee. Season 1 drew a different
   luck champion per house, so identical results scored differently by noise; from season 2 the draw is shared (`luck_seed`), season 1 rows still verify. A niche that random numbers
   already satisfy pays nothing (see "rigged tables" on the board).
4. **Points** = 10 x max(0, referee solved - luck) / referee total, best generation per niche and house.
5. **Winner** = best points, tie broken by fewer charged positions. Nobody wins a niche that pays nothing.
6. **Scouting**: in generation 2 a general-niche house that is behind sees the leading house's tensors and referee fraction (logged as `scouted`).
7. **Admission**: referee-passing cells enter the shared bank (near-duplicates of an existing cell of the niche are skipped).
8. **ARC kills**: 5 points per ARC *selection-split* task (dev + validation) that the house's admitted cells newly solve, measured by adding the cells
   to the bank in rank order and re-scoring (marginal credit, recomputed). The report split is displayed and never paid.
9. **Honest decline**: in a niche match, a house that says no tensors of the template can express the rule ("no valid ...", no cell) loses nothing and the
   niche is posted as a structure bounty. In a boss fight a house must ALWAYS attempt every item; it may add `"cannot_express": true`, which posts the
   bounty but never replaces the attempt (Season 2 offered a free way out and all three houses skipped all 30 items: a prompt confound, not evidence).
10. **Tournament day**: after the card, one breeding round (chains of earned cells), then a **boss fight**: every house revises the SAME real ARC
    near-misses (selection split). The first revision that strictly raises the bank's selection score without losing any solved task is the kill
    (5 points per task), compared against a matched random edit with the same number of changed positions. Judging order rotates by round.
    Two tournament days with no admitted child or kill stop the season (plateau).

Titles are mechanical: First Blood (first house to beat luck on a niche), Sole Survivor (only house that beat luck), Honest Null (verified decline).

## What "reward" means here (no overclaiming)
There is no gradient and no reward model. The incentives are information and selection: the standings are in the next prompt, winning tensors enter
the bank and become parents, a house that is behind sees the leader's tensors, and points decide which niches get more generations next season.
Points measure generalisation inside a generated family against luck. They are not an ARC score; only `ARC_SCORE.json` counts for the goal.

## What cannot be faked
`python scripts/arena.py verify <run>` regenerates the referee tasks from the niche seeds, re-draws the luck entries, re-scores the saved tensors and
checks the ledger hash chain per row. A ledger row that disagrees with its tensors, or tensors edited after scoring, fails.
`python -m unittest tests.test_arena -v` pins this plus: referee inputs never appear in any house's training tasks, seed families are disjoint,
a rigged niche pays 0, a decline is free, metadata never enters a hash.

## Known limits
- Prompt wording changes behaviour. Season 2 boss fights (header said an honest decline costs nothing, return no entry) produced 60 of 60 empty answers; the
  old plain mutation prompt had produced 6 attempted revisions. Season 3 reruns the fight with attempts required. Compare prompt hashes before comparing results.
- Houses are the same model; their diversity comes from different example tasks and sampling, not different minds.
- Referee tasks come from the same generator as training tasks: "generalises" means within the family, not to ARC.
- Luck is best-of-24, not exhaustive; a niche can still be easier than it looks. The board lists niches where luck already reaches 25%.
- Kill credit is marginal and order-dependent (rank order within a niche).
- Cost: Season 1 (11 niches, 3 houses, up to 2 generations) used 49 maker calls. Houses that already pass sit out later generations. A boss round is 3 calls.
- Season 3 (boss fights with attempts required): 3 houses x 10 near-misses x 2 rounds = 60 attempts, 0 kills, 60 of 60 flagged `cannot_express`, matched
  random edits 0. Targets listed in `notes/HEAD_TO_CODEX_2026-10-02_CELL_WARS.md`. `verify` recomputes match rows only; boss admissions need the ARC bank.
- Season 1 result: all 36 matches paid, 0 declines, 0 maker errors, 0 ARC kills, standings 99.5 / 97.5 / 93.2 (mostly luck-draw noise, see above). The
  niches were too easy to separate the houses; only `anchorassign` showed real house variance. Matches on easy niches rank nobody.

## Tournament edition (season 4 onward): head to head, zero-sum, with stakes
Owner feedback 2026-10-02: seasons 1 to 3 read as cooperation (every house scored on its own, a house that was behind saw the leader's tensors, three
correct houses all won). Season 4 is a knockout tournament instead. `scripts/tournament.py` (driver), `scripts/tournament_core.py` (pure scoring),
`tests/test_tournament.py`; `python scripts/tournament.py verify <run>` recomputes every row.

- **Bracket**: four houses, seeds rotate each cycle, single elimination (semifinals, final). A bout is a set of 6 niche duels between two houses.
- **Duel**: both houses write independently and cannot see each other's tensors (no scouting any more). Hidden gauntlet: referee tasks at the standard tier
  plus a hard tier with 2 demonstrations that counts double. A house must beat best-of-24 random luck on the standard tier to win anything.
  Equal gauntlet scores go to the sparser tensor (fewer nonzero entries). Only an exact tie is a draw (the higher seed advances).
- **Stakes**: the duel winner challenges the INCUMBENT cell of each slot (niche, pattern) in the shared bank. Only a strictly better cell takes the slot;
  the dethroned cell is evicted; a correct loser is discarded. ARC veto: a replacement that would lose a selection task is refused.
- **Scoring**: duel win 3, draw 1, bout win 5, merge duel win 4, placing (champion 10, runner-up 6, others 3), ARC kills 5 per newly solved selection task.
- **Merge duels (combine)**: the two finalists each write ONE cell for a group of two earlier same-template cells. It must solve both patterns; then more
  OTHER patterns of the family solved wins, then sparsity. A merged cell replaces its parents only when ARC selection does not suffer (larger cells).
- **Cycle** = bracket, merge duels, breeding (chains). Two cycles with nothing admitted, replaced, merged, bred or moved on ARC stop the run (plateau).
- **Supply of new cells is bounded.** Most instance niches have a finite pattern space (another seed gives no new patterns), so the engine is
  competition for slots, merging into larger cells, harder tiers, and new structures (the bounty board). More cells without new structures is not possible.

## How to play as Codex
1. Read `ARENA_BOARD.md` (server: `/srv/cellforge-inbox/head-20261001/runs/weights-head-arena1/`, mirror `reports/arena/ARENA_BOARD.md`).
2. Take a bounty: add a niche additively in `scripts/niches.py` (new template in its own module, `niches.LIMITS` entry, verify + discriminates so random
   cells stay low), name it in `notes/CODEX_TO_HEAD_<date>.md`. Head adds it to the next card.
3. Fix a rigged table: a harder variant whose best random entry stays under 25% pays points again.
4. Do not edit `arena_core.py`, `arena.py`, `tests/test_arena.py`, the ledger or the board. Rules 1 to 6 of `AGENTS.md` apply unchanged.
