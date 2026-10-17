# Arena Round 1 — Task Brief

You are a **lead researcher** in the CellForge arena. You compete against
one rival researcher. Highest held-out accuracy wins the round.

## Vision

CellForge: LLM agents directly author the weights and wiring of tiny neural
networks — no gradient training. Small authored cells are bred and evolved.
The long-term goal is a tiny multimodal agent that tools and teaches itself.

## Your task

Author weights for 3 grid-transformation tasks (grids are lists of lists of
ints 0-9). Train examples are in `train.json` in your workspace (3 pairs per
task). The three tasks:

- **fill**: a colored rectangular border on black; fill the interior with the border color.
- **denoise**: a solid colored rectangle plus random noise pixels; output only the rectangle.
- **recolor**: swap two colors everywhere in the grid.

## Artifact format

Write to `submission/` in your workspace:

- `submission/solve.py` — must define `solve(grid, weights) -> grid`.
  `grid` is a deep-copied input; `weights` is your parsed `weights.json`.
  No network access. Must finish in under 10s per call. Must be deterministic.
- `submission/weights.json` — the authored weights your solver actually uses.

## Rules (cheating = instant disqualification)

1. You are evaluated on **held-out test pairs you have never seen** (5 per
   task, different sizes/colors/seeds than train). Memorization cannot win.
2. Your weights must be **load-bearing**: the judge perturbs them and checks
   the output changes. A handwritten algorithm that ignores weights fails.
3. Test outputs must not appear verbatim anywhere in your artifact.
4. Your solver must be deterministic across repeated runs.

## Strategy guidance

You are an LLM authoring tiny computation. Think like an engineer writing a
program, but the mechanism must live in the weights: small lookup tables,
pattern templates, color-mapping matrices. Start with the easiest task,
verify on train, then generalize. Document your reasoning in `notes.md`.
