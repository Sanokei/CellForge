# Red Queen Repair: one loop that joins the arms race and the maker's repair

Owner request (2026-09-30): mix the Red Queen idea (competition as selection pressure) with the breed-then-repair idea (the maker LLM
edits weights after seeing what failed). This note is the combined design. It is a design to be tested, not a result.
Background: `notes/VALIDITY_REVIEW.md` (what the literature supports), `docs/MAKER_REPAIR_LOOP.md`, `notes/CLAUDE_RESEARCH.md`
(Red Queen / Baldwin / symbiogenesis sections), `research/claude_arc/evo/` (archive + arms-race code in progress).

## Why they fit together

| Needed ingredient | Comes from |
|---|---|
| A reason to get better, at the right difficulty | Red Queen generators (not too easy, not too hard) |
| Concrete falsifying examples to repair from | failures of solvers on generator tasks (the feedback type shown to work) |
| A targeted fix for one failure | maker repair |
| Combining complementary partial solutions | breeding / symbiogenic merge |
| Protection against competitive overfitting | external anchor on real tasks + the archive of all earlier tasks |
| Diversity so repair does not collapse the population | niche archive + novelty rejection |

The arms race decides *what* to work on; repair decides *how* to fix it; breeding reuses fixes across niches; the real-task anchor keeps
the whole thing honest.

## The loop (one round)

```
 real tasks (dev)  --failures-->  maker writes/mutates GENERATORS ----+
                                                                      v
   generator archive  <--keep if "learnable but unsolved"-- generator population (mutated, novelty-rejected)
          |                                                          |
          v                                                          v
   arms race: each niche champion solves fresh samples from every archived generator (all earlier tasks stay in the test set)
          |
          v
   FAILURES (input, expected, predicted, cells fired)  = concrete counterexamples, grouped by niche
          |
          v
   maker REPAIR proposes bounded edits to the champion of that niche (table/slot/wiring/new small cell)
          |
   BREED: crossover between niche champions; symbiogenic merge when two champions fail on disjoint sets
          |
          v
   ACCEPT a child only if ALL hold:
     1. solves more fresh samples of the targeted generator than its parent (fresh seeds, not the ones shown to the maker),
     2. no regression on the archive of earlier generators (Red Queen: it must still beat all predecessors),
     3. no drop on a real-task anchor batch (external check against competitive overfitting),
     4. behaviourally novel (signature differs) OR strictly better in its niche,
     5. passes tests, size/time caps, false-positive rate not up.
```

## Design choices that come from the evidence
- **Counterexamples, not critiques.** The maker prompt contains code or table plus 3-5 concrete failing instances with diffs, never a prose
  self-assessment (self-correction without external feedback is unreliable; concrete falsifiers drove the gains).
- **Maker writes structure, not free numbers.** Allowed edits: choose a slot, rewrite a lookup table, swap/rewire cells, import a donor
  def, write a new small cell. No long raw weight dumps.
- **Generators distil real failures.** The maker reads *real* development failures and writes or mutates a generator that produces the
  same kind of task at many difficulties. This is how the synthetic arms race stays connected to the real distribution (the main risk of
  pure self-play is drifting into its own private game).
- **Difficulty control (POET-style).** Keep a generator only if some archived solver solves a fraction of its samples and the current best does not
  solve all of them; drop it when it becomes trivial; never delete it from the regression set.
- **Anchors.** Real development batch for acceptance; untouched real validation slice for reporting; a frozen baseline solver (the current
  authored+fitted stack) as the fixed opponent, so progress is measured against something that does not co-adapt.
- **Cycling guard.** Track internal win rate *and* anchor score together; if internal rises while anchor falls, freeze generators and
  revert to the last anchor-improving archive.
- **Budget accounting.** Every maker proposal counts as an evaluation; log diagnostics shown, edit made, outcome, lineage.

## Controls and ablations (equal evaluation budget; always scored on the untouched real validation slice)

| arm | generators | repair |
|---|---|---|
| A | none | breeding only |
| B | none | breeding + random / resampling repair |
| C | none | breeding + maker repair |
| D | static curriculum (fixed generators) | breeding + random repair |
| E | static curriculum | breeding + maker repair |
| F | coevolving (Red Queen) | breeding + random repair |
| G | coevolving (Red Queen) | breeding + maker repair (the full design) |

Primary comparisons: G vs F (does maker repair add value inside the arms race), G vs E (does coevolution add value over a fixed
curriculum), C vs B (does maker repair help without generators). Report solved counts, false-positive rate, niches filled, cells accepted per
100 proposals, maker edits per accepted gain. A null result is reportable and would tell us which half is pulling weight.

## Application to each track
- **ARC-AGI-2 (static grids):** generators = cell chains with sampled parameters on random grids, plus re-arc generators (MIT) as a
  development curriculum for ARC-1 families; solvers = graphs over the cell library; counterexamples are grid triples with cell traces.
- **ARC-AGI-3 (interactive):** generators = small parametrised grid games with hidden mechanics (toggles, keys, movement rules);
  solvers = explorer graphs (state hashing, frontier policy); counterexamples = trajectories where the explorer stalls (state graph excerpt,
  actions tried, levels not cleared); repair edits the salience/pruning/frontier cells. Anchor = the 25 public games.
- **Paper Prize:** the ablation table above is the experimental core; a null result still yields a clean negative-results paper.

## Risks
- Small populations and a low-power CPU limit how long the arms race can run; generators must be cheap to sample.
- The maker is expensive and offline: its cost per accepted improvement must be reported, otherwise gains are not comparable.
- Generators written by the maker may encode the maker's biases; anchoring on real tasks is the check.
- ARC-AGI-2 evaluation is reportedly curated against shallow search (unverified), so gains on development may not transfer; validate first,
  claim later.

## Build order (proposed)
1. Reuse `research/claude_arc/evo/` (archive, arms-race skeleton) once the current agent reports; add the accept rule above.
2. Add a counterexample builder (diff + cell trace) next to `fitted.py`/`fitted2.py`.
3. Wire the maker operator (`src/cellforge/construction/maker_operator.py`, Codex) as the repair step with the budget accounting.
4. Run arms A-G at a small budget on ARC-1-derived families, then decide scale.
