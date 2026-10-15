# weights-head-skillbank-final

The head-center skill bank as scored for the goal test (`ARC_SCORE.json`, contract in `tests/arc_goal_checks.py`).

- Weights: `FINAL_WEIGHTS.json` = 586 cells (code template + literal tensors written by gpt-6.1-sol low via the server `codex exec`, ChatGPT subscription route; no gradients) plus a router.
- Evaluator: `arc_eval.py` (+ `skillcore.py` and the template modules next to it). It is self-contained: no PYTHONPATH, no external package.
- Scored set: the REPORT split (302 public-training tasks never used to select, breed, mutate or admit anything). The 100 DEV tasks, the audit population and the evaluation split are excluded.
- Result: 29 of 302 (9.6%); zero weights 0; random weights 0, 0, 1; history 0 -> 29 over 13 recomputed iterations.
- Data is NOT in the repo. Recompute on the server: `ARC_ROOT=/srv/cellforge-private/arcagi2/data python -m unittest tests.test_arc_goal -v`
  with this directory under `runs/`.
- Provenance of every maker call (prompt hash, raw response, exit record) lives on the server in
  `/srv/cellforge-inbox/head-20261001/runs/weights-head-{skillbank-v*,arena*,tourney1}/AUTHOR-*`. This run does NOT ship `STANDARD.json`: the head driver
  predates that schema, so `tests.test_one_standard` does not enumerate it (it only enumerates runs with `STANDARD.json`).
- Produced by `scripts/make_arc_score.py`.
