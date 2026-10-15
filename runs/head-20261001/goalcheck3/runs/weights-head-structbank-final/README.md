# weights-head-structbank-final

The skill bank PLUS 280 maker-written structures, scored for the goal test (`ARC_SCORE.json`, contract in `tests/arc_goal_checks.py`).

**This is not evolved weights.** Each structure is restricted Python written by gpt-6.1-sol low for ONE task from that task's demonstrations; its tensors hold 2 to 34
constants (colours, steps, periods). Read it as "an LLM writing a program per task, with a weight wrapper", not as the project's thesis. See `docs/STRUCTURE_PROPOSALS.md`.

- Scored set: the REPORT split (302 public-training tasks never used to select, breed, mutate or admit anything). DEV, audit population and evaluation split excluded.
- Result: 287 of 302 (95.0%); zero weights 0; random weights 0, 0, 0; history 0 -> 287 over 14 recomputed iterations. The owner's seven checks pass in a clean process.
- Protocol (frozen before this split was touched, run once): the maker saw only each task's demonstration pairs and test INPUTS. A structure was accepted only if it
  reproduced every demonstration leave-one-out, FAILED with zeroed tensors and three random tensor draws, and used at most 256 tensor entries. Tasks where an existing
  cell already reproduced every demonstration were not authored. No decision read a test output (`STRUCT_RESULTS.jsonl` `killed` is a measurement made afterwards).
- CAVEATS: (1) test-time authoring: a structure scored on a task was written from that same task's demonstrations; see `TRANSFER.json` on the server for how much the
  library solves when a task's own structure is removed. (2) These are public training tasks; the model may recall familiar rules (a colour-permuted, transposed probe
  on the selection split still solved 18 of 20 attempts). (3) The ARC evaluation split was never read, so novelty is unmeasured.
- Safety: all structure code runs only through `structcode.py` (static whitelist, separate worker, no network, resource limits, timeout breaker in `structlib.py`).
- Data is NOT in the repo. Recompute on the server: `ARC_ROOT=/srv/cellforge-private/arcagi2/data python -m unittest tests.test_arc_goal -v` with this directory under `runs/`.
- Provenance of every maker call (prompt hash, raw response, exit record): server `/srv/cellforge-inbox/head-20261001/runs/weights-head-structbank1/AUTHOR-*`.
- Produced by `scripts/struct_solve.py` then `scripts/make_arc_score.py`. This run does NOT ship `STANDARD.json` (the head driver predates that schema).
