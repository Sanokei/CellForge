# Maker guide addendum: mnist1d/1 (`maker-guide-clf/1`)

Read `docs/MAKER_GUIDE.md` first (graph IR, ops, proposal format, hard rules).
This addendum replaces the task-specific parts.

## Task

MNIST-1D: classify 40-point 1-D signals into 10 classes. Each class is a
characteristic shape (a template of 12 points) that has been padded, randomly
**shifted**, scaled, sheared and corrupted with correlated + white noise, then
resampled to 40 points. Reference numbers from the literature for gradient-trained
models on ~4000 examples: logistic ~32%, small MLP ~68%, 1-D CNN ~94%, human ~96%.
Chance is 10%.

### Interface

Entry `predict` (single entry bundle, no encode/step):
* input `x`: `[40]` float signal per example (values roughly in ±4);
* output `logits`: `[10]`. Prediction = argmax (ties → lowest class).

Nothing else enters the model (no labels, no position metadata).

### Data you may look at

Under `runs/m1d1/dev/` (run `m1d1`; other runs have the same files):
* `class_samples.png` — 6 signals for each class (look at it!).
* `dev_train.npz` (`x` [4000,40], `y` [4000]) — signals and labels you may load and
  view, plot or inspect yourself.
* `dev_eval.npz` — 1000 signals scored by `dev-eval`.

### Rules specific to this task

* All hard rules in `docs/MAKER_GUIDE.md` apply: **you choose every value**; no code that
  fits weights, computes templates/means/statistics from the signals, runs least
  squares, gradient steps, or searches candidate values against a metric.
  Reading plots and numbers yourself and typing your own templates/filters/thresholds
  is the intended method. Helper scripts may only expand values/formulas *you typed*.
* **Do not read the `mnist1d` Python package source or data** (site-packages/mnist1d):
  it contains the true templates; using them would bypass the experiment.
  Do not try to reproduce the generator's parameters from memory of its paper either;
  work from the dev signals you can see.
* `cellforge dev-eval --run runs/m1d1 --task <id> --proposal <file>` reports accuracy,
  per-class accuracy, confusions and wrong examples, plus (if activated) plugin
  diagnostics. Budget: 25 calls; `--validate-only` is free.

## Useful ops for shift-tolerant recognition

The class is defined by a shape at an unknown position, so think about
shift tolerance: slide a filter along the signal (stack `op:shift` copies or
`op:slice` windows → `op:concat`; a `cell` applies the same weights at every
window because it acts on the last axis), apply a nonlinearity, then max-pool
over positions (`op:max`, axis 0) to get one score per class. Differences
(`x[t+1]-x[t]`), cumulative sums, absolute values and normalization by signal
scale (e.g. dividing by a max/abs-mean) are also available as ops.
The profile is `open/1` unless the task says otherwise (any nesting, routing optional).

## Deliverables

`proposal.json`, `notes.md` (what you built, key numbers and why, dev-eval results
by number, what failed, next ideas; state which knowledge came from looking at dev
signals vs general knowledge), `observations.json`, `scripts/`. Final message:
final dev accuracy, dev-evals used, architecture summary. Be truthful.
