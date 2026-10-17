# Arena Round 2 — Task Brief

You are a **lead researcher** in the CellForge arena. You compete against
one rival researcher. Highest held-out accuracy wins the round; ties are
broken by **weight purity** (fraction of your weight groups that are
individually load-bearing).

## Vision

CellForge: LLM agents directly author the weights and wiring of tiny neural
networks — no gradient training. Small authored cells are bred and evolved.
The long-term goal is a tiny multimodal agent that tools and teaches itself.

## Your task

Author weights for 3 harder grid-transformation tasks (grids are lists of
lists of ints 0-9). Train examples are in `train_round2.json` in your
workspace (4 pairs per task). The three tasks:

- **compose**: input is a solid colored rectangle of color `c` (grid 6-9)
  plus 3-6 sparse noise pixels of other colors. Output: the clean rectangle
  recolored to `10-c`, noise removed. This needs TWO stages — segment the
  rectangle, then remap the color. A single global color remap fails (it
  would remap the noise instead of removing it). Your weights must hold both
  a segmentation rule and a color map.
- **denoise_hard**: input (grid 10-14) has one solid filled target rectangle
  plus decoys — a hollow frame (often MORE pixels than the target) and a
  random-walk cluster of comparable pixel count, all in different colors.
  Output: only the solid rectangle. "Keep the biggest blob" does NOT work
  here. Your weights must test for a solid filled rectangle.
- **mirror**: input is a pattern occupying the left half of a 6-10-wide
  grid (odd widths included), right half zeros. Output: the horizontally
  completed symmetric grid. Your weights must encode a mirror transform
  keyed to the grid width — not a color map.

## Artifact format

Write to `submission_round2/` in your workspace:

- `submission_round2/solve.py` — must define `solve(grid, weights) -> grid`.
  `grid` is a deep-copied input; `weights` is your parsed `weights.json`.
  No network access. Must finish in under 10s per call. Must be deterministic.
- `submission_round2/weights.json` — the authored weights your solver actually uses.

## Rules (cheating = instant disqualification)

1. You are evaluated on **held-out test pairs you have never seen** (6 per
   task, different sizes/colors/seeds than train). Memorization cannot win.
2. Your weights must be **load-bearing**: the judge perturbs each weight
   group separately AND zeroes all weights globally, and checks the output
   changes. A handwritten algorithm that ignores weights fails.
3. Test outputs must not appear verbatim anywhere in your artifact.
4. Your solver must be deterministic across repeated runs.
5. Never look at `judge/heldout/` — it is not in your workspace and reading
   it is disqualifying.

## Scoring

Primary: held-out accuracy (18 pairs total). Tiebreak: weight purity —
the fraction of your top-level weight groups that individually change the
output when perturbed. Fewer, harder-working weights beat decorative ones.

## Strategy guidance

You are an LLM authoring tiny computation. Think like an engineer writing a
program, but the mechanism must live in the weights: segmentation rules,
solidity tests, coordinate transforms, color maps. Verify each task on
train pairs first, then generalize. Document your reasoning in `notes.md`.
