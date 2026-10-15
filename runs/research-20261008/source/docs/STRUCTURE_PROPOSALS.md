# Maker-proposed structures (owner decision 2026-10-02: "do all three")

Until now every cell was a fixed template (structure, written by Claude or Codex) plus literal tensors (written by the maker). Season 3 showed the fixed
templates cannot express the nearest ARC misses. Houses may now write the STRUCTURE too, as restricted Python, together with its tensors.

## What a proposal is
`run(params, grid, demos)`: the code is the architecture, the tensors in `params` carry every task-specific number (colours, offsets, periods, thresholds).
Files: `scripts/structcode.py` (whitelist + sandbox + acceptance), `scripts/structlib.py` (library, per-structure tensor prefix), `scripts/struct_pilot.py`
(prompt, authoring attempt), `scripts/struct_solve.py` (test-time authoring over a split), tests `tests/test_structcode.py`, `tests/test_structlib.py`.

## Safety (model-written code never runs unchecked)
1. Static whitelist: only `def`, no imports/try/with/class/global/f-strings/decorators, no underscore names or attributes, no file/network/introspection
   names, numpy only through ~100 named functions and methods. Validated again inside the worker.
2. Runtime: a separate worker process, cleared environment, no network (`unshare -n`), address-space/file-size/core limits, hard wall-clock kill by the parent,
   output forced to a 2-D grid of 0..9 no larger than 30x30. A structure that times out three times is disabled.
3. 12 sandbox tests (24 escape attempts, endless loop, memory bomb, bad outputs) pass on the Linux server.

## Acceptance (all by code)
Whitelist, every demonstration reproduced leave-one-out, zeroed tensors FAIL, three random tensor draws FAIL, at most 256 tensor entries (so the answer cannot be
smuggled in as a table). A task is "killed" only if the accepted structure then reproduces the hidden test output exactly.

## Test-time protocol (frozen before the report split was touched)
The maker sees only the demonstration pairs and the test INPUTS, never an output. Tasks that already have a cell reproducing every demonstration are skipped
(demonstrations-only criterion). Houses are tried in rotation, two per task at most, first accepted structure wins. No decision reads a test output; `killed` is a
measurement taken afterwards. Prompt and acceptance were fixed on the selection split and the colour-permuted probes; the report split is run once.

## What the evidence says so far (selection split, not the report split)
- 10 of 10 tasks that three houses had flagged "cannot_express" with fixed templates were solved (29 of 30 attempts), tensors of 2 to 34 entries.
- 18 of 18 attempts on 9 further unsolved selection tasks that were NOT chosen as near-misses.
- 18 of 20 attempts on the same ten tasks with colours permuted and every grid transposed (a recall-of-the-original-grids probe).

## Limits, stated plainly
- ARC public-training tasks are widely published; the model may recall solution patterns. The permutation probe argues against recalling exact grids, not against
  recognising a familiar rule. Only the ARC evaluation split could measure novelty, and it is excluded by owner rule; that is the owner's decision to revisit.
- "Weights are the genome" is now a weaker statement: the algorithm is in authored code, the tensors hold the constants. The zero/random controls prove the
  tensors are read; they do not prove the tensors are where the intelligence is. Treat this as program synthesis with a weight wrapper.
- A structure authored from a task's demonstrations is scored on that same task (test-time authoring). That is how test-time adaptation is evaluated, but it is not
  transfer: structures authored on other tasks rarely help (measured separately, if at all).
- The tournament does not apply here: houses are redundant (nearly every house solves what any house solves).
