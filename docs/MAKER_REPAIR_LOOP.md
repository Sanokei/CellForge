# Breed-then-repair: the harness as a weight-writing loop for the maker LLM

Owner idea (2026-09-30): the harness exists so that an LLM (the *maker*) can write weights, look at test results, and keep
iterating. Breeding produces children; afterwards the maker inspects what the children got wrong and repairs them by editing
weights and wiring. The maker "slowly gets better at writing weights" through the loop.

This is an offline authoring loop only. The final artifact is a pure CellForge graph with no LLM at runtime. The design
matches what `src/cellforge/construction/maker_operator.py` and `docs/MAKER_OPERATOR.md` already started; this note states the
loop, its guard rails and the measurements that would show it helps.

## The loop

```
seed cells (authored)  ->  breed (import_def crossover, recombination, symbiogenic merge)
        ^                                   |
        |                                   v
   archive (niches,                 evaluate children on development tasks
   novelty-rejected)                        |
        ^                                   v
        |                     per-task failure diagnostics (structured)
        |                                   |
        |                                   v
        +------ accept repaired child <---- maker LLM proposes edits (weights, slots, wiring, import_def)
                 only if it improves dev score on a
                 fresh batch AND passes validation
```

1. **Breed.** Produce children from archive parents (typed-slot crossover, `import_def` transfer, merge of two viable graphs).
2. **Evaluate.** Score every child on the development tasks. Keep per-task outputs, demo reconstruction errors, invalid outputs.
3. **Diagnose.** Convert failures into compact structured diagnostics: which tasks fail, predicted vs expected grids (as small
   diffs), which cells fired, which routing branch was taken, false-positive cases (demo-consistent but wrong).
4. **Repair.** The maker receives the diagnostics and proposes a bounded edit set. Allowed edits are small, typed and reversible:
   set or scale a parameter slot, rewrite a lookup table, swap a cell in a slot, rewire a port, import a donor def. The maker may
   also write a *new* small authored cell. It may not change the scorer, the data, or the sealed splits.
5. **Validate.** Re-run tests and the development batch; accept only if the score improves on a held-back development batch
   (not the batch that produced the diagnostics) and false positives do not rise disproportionately.
6. **Archive.** Keep the repaired child and its parent, the diagnostics shown and the edit made (lineage and receipts in the
   ledger). Reject behavioural duplicates (novelty rejection); keep the best per niche.

## What "gets better at writing weights" can and cannot mean

The maker LLM's own parameters do not change. What improves is the *harness around it*:

- a growing **playbook** of edits that worked, with the diagnostic pattern that triggered each (retrieved into later prompts),
- better **diagnostics** and edit tools (this is where a Darwin-Gödel-style step applies: the harness code itself is evolved and
  validated on held-out tasks),
- a larger, more diverse **archive** of cells to edit.

Claims must be phrased that way. Improvement shows up as a rising repair success rate over rounds, not as a model improving.

## Guard rails (non-negotiable)

- **Exposure accounting.** The maker sees development results, so development scores become exposure. Count every maker edit as an
  evaluation, log the diagnostics shown, and score the sealed split rarely and only on frozen artifacts.
- **No test leakage.** Diagnostics are built from development tasks only. Public evaluation tasks are never shown to the maker.
- **Equal-budget controls.** Compare (a) breeding only, (b) breeding + random local mutation repair, (c) breeding + maker repair,
  each with the same number of evaluations. Report (c)-(b), not (c) against nothing.
- **Bounded edits.** Edit size caps, rollback on regression, a limit on rounds, and a time cap per round.
- **Purity.** No gradients. Fitting by exact consistency/counting on demonstration pairs is allowed and labelled `fitted`.

## Measurements that would show it works

- repair success rate per round (child improved on the held-back batch) for maker vs random-repair control,
- dev and validation solved counts and false-positive rate after k rounds,
- archive coverage (niches filled) and cells accepted per 100 maker proposals,
- cost: maker tokens or edit count per accepted improvement (so the gain can be weighed against effort).

A negative result (maker repair no better than random repair at equal evaluations) is a legitimate finding and should be reported.

## Fit with the rest of the plan

Shared perception cells and the router are described in `docs/SHARED_PERCEPTION_MOE.md` (in progress). Diversity archive and Red-Queen
arms race: `research/claude_arc/evo/` (in progress). ARC-AGI-3 explorer: `research/claude_arc3/` (in progress).
