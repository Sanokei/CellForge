# Grokking: what the research says and what it means for CellForge (2026-09-30)

Owner question: do we need to hit overfitting to start grokking, and is our rising training-set score progress toward it?
Sources are abstracts/search summaries unless noted.

## What grokking is
- Power et al. 2022 (arXiv 2201.02177): on small algorithmic datasets, test accuracy jumps long after training accuracy hit ~100%.
- Mechanism (Nanda et al. 2023 progress measures; Varma et al. 2023, arXiv 2309.02390 'circuit efficiency'): a memorising circuit forms first;
  a slower-to-learn but more parameter-efficient generalising circuit forms and grows; weight decay (a pressure toward low norm / efficiency) then
  cleans up the memoriser. Grokking needs: (1) a learner that keeps changing after it has fit the data, (2) a pressure that favours the more
  efficient/simple solution (weight decay is the reliable trigger), (3) a task that has a generalising solution simpler than memorisation, and
  (4) enough data: there is a critical dataset size below which generalisation does not appear (Varma et al.; 'Critical Data Size of Language Models
  from a Grokking Perspective', arXiv 2401.10463). Grokking can vanish with larger datasets without regularisation.
- It is not specific to gradient descent or neural nets: Recursive Feature Machines show grokking-like emergence on modular arithmetic (arXiv 2407.20199).
  A survey-style note frames the generalising solution as algorithmically simpler (Occam) than the memorised one (arXiv 2603.29262).
- ARC link: the ARC-AGI-2 Technical Report (arXiv 2603.06590) lists 'grokking' among steps that raised a 200M-parameter LongT5 solution from ~7.08% to
  ~19.86% (and to 27.08% semi-private with later additions), using long training on ~2.3M synthetic tasks. Gradient-trained, GPU, not our setting.

## Do we "need to overfit first"?
Only in a specific sense: grokking happens *after* a model has fit (memorised) the training set, but fitting is not sufficient and it is not progress by itself.
It needs the continued pressure toward the simpler generalising solution. Fitting without that pressure just overfits.

## Does our rising training score count as progress toward grokking?
No, not as it stands. Our numbers (97 -> 131/1000 training solves, evaluation 0/120) come from ADDING hand-written cells: that is capacity growth evaluated
on the training set. Nothing is being trained past a fit, and there is no cleanup pressure. The false positives (demo-consistent but wrong on the test input;
12 -> 14 -> 18) are the closest analogue of memorisation: rules that fit the demonstrations without the underlying concept.
What would make the analogy real is an MDL / parsimony 'weight decay' on the library and on evolved graphs:
- prune cells that explain few tasks, merge near-duplicate cells into one parameterised cell, reward short programs and small graphs (DreamCoder's
  compression 'sleep' phase and CompressARC's MDL objective are the same idea),
- keep evolving after the training set is fit, with a size/complexity penalty, and watch held-out behaviour for a delayed jump,
- stay above a critical data size: per-task we only have 2-3 demonstration pairs, so within-task grokking is data-starved by construction; across tasks
  (1,000 training tasks, plus generators) there is more data to support a generalising library.

## Testable predictions (gradient-free)
1. Within-task: evolve a tiny NCA on 2 demonstration pairs with and without a parsimony penalty, hold out the third pair; if grokking-like cleanup occurs,
   held-out accuracy should rise late, after demo loss reaches zero, only in the penalised arm.
2. Library-level: measure validation-bucket solves and false positives before/after a compression step (merge/prune); expect fewer false positives and
   equal or better validation solves if compression acts like cleanup.
3. Negative control: same compute, no penalty.
A null result (no delayed generalisation, penalty only reduces size) is a legitimate finding.
