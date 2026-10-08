# Fresh local baseline: operational result, 2026-10-06

An explicit fresh initialization path now completes a local cell phase without
the missing server artifacts. The final saved baseline, zero-weight control and
three independently seeded random-weight controls each score **0/4 search and
0/4 development-selection tasks**, with **zero runtime errors** in independent
saved-evaluator replay. No model-worker calls were made. This establishes local
operational execution only, not a useful founder, learning gain, qualified
ONE_STANDARD experiment, held-out benchmark result or chatbot capability.

## Source and input isolation

The parent explicitly authorized fresh initialization and downloading official
public training tasks. Inputs come from [ARC Prize's ARC-AGI-2 repository](https://github.com/arcprize/ARC-AGI-2)
at commit `f3283f727488ad98fe575ea6a5ac981e4a188e49`; its
[Apache-2.0 license](https://github.com/arcprize/ARC-AGI-2/blob/f3283f727488ad98fe575ea6a5ac981e4a188e49/LICENSE)
is preserved byte-for-byte in `data/arcagi2-local-20261006/LICENSE`.

Only eight training JSON files were fetched, in 4.11 seconds. File bytes match
their pinned upstream git-blob hashes and their recorded SHA256 hashes.
`SOURCE.json` freezes source, license hash, task IDs, content hashes and split
selection rule. Selection used filenames alone, before task content was read:
exclude the existing audit hash population; rank by SHA256 of a declared prefix
and ID; take four search IDs, then four `val:`-hash-even selection IDs and two
`val:`-hash-odd report IDs. This is a NEW small development panel, not the absent
historical DEV100 or a recreation of its reported score. Existing frozen runs
and partitions were not edited.

The two report-task IDs are reserved but their files were never downloaded or
loaded. No ARC evaluation files were requested or accessed. Public training data
must not be described as unseen to the maker or as an independent benchmark.

## Implementation

- `scripts/fresh_inputs.py`: reproducible fresh random literals through existing
  `skillcore.make_cell`/typed-bank constructors; explicit manifest path/hash;
  source, license and task-byte verification; disjoint bounded task panels;
  report-ID reservation without report reads. No network or maker invocation.
- `scripts/head_driver.py`: opt-in `fresh_inputs` / `--fresh-inputs` path uses the
  new cells as its immutable initial prefix, copies provenance, and emits an
  evaluator that does not import a fabricated `pop1_core.py`. Legacy defaults
  remain unchanged. Fresh mode requires an explicit real evolution policy and
  cannot combine with historical seed files or `--seed-from`.
- `scripts/evolution_epochs.py`: optional `fresh_inputs` plan field binds local
  inputs and validates them before campaign creation or allowance reservation.
  Default plans still make no calls; changing a frozen plan still fails closed.
- Four isolated checks in `tests/test_fresh_inputs.py` cover absent report
  files, task tampering before reservation, changed baseline provenance and
  partition overlap. Result: **4 tests in 0.295 seconds, OK**.

A real execution exposed a scope-hash bug: the driver hashed task IDs in supplied
order while the manifest canonicalized them. The driver now sorts partition IDs
for that hash, matching the existing manifest contract. No acceptance rule,
protected test or floor was weakened.

## Preserved attempts and final artifacts

All attempts used one numerical thread, below-normal process priority, a
120-second subprocess timeout, disabled workers, zero call allowance, zero
breeding pairs and zero mutations. None auto-started a subsequent phase.

1. `runs/weights-local-fresh-20261006`: completed its bounded body, but completion
   publication failed on the scope-hash ordering mismatch. Its incomplete
   reservation and log remain intact. Elapsed 0.985 seconds; no completion claim.
2. `runs/weights-local-fresh-20261006-fix1`: scope fix completed in 1.093 seconds.
   Independent replay rejected the tiling baseline as valid evaluation evidence:
   the saved cell threw on one rectangular task; three random arms had four more
   runtime errors. These are recorded as null outcomes, not ordinary wrong
   answers. The driver had counted them as failures; its score alone was
   insufficient. Saved artifacts and `OPERATIONAL_REPLAY.json` preserve this.
3. `runs/weights-local-fresh-rectangular-20261006`: separate, documented runtime
   repair replaces the partial-domain tiling template with existing
   `assoc_induct`, retaining `glyph_recolor`, the same random seed
   `2026100601`, task panel and zero-call protocol. Neither template is an oracle.
   Phase 0 completed in **1.093 seconds**; independent replay took **1.078 seconds**.
   All five arms execute successfully and score zero on both panels.

The final phase directory is
`runs/weights-local-fresh-rectangular-20261006/phases/weights-phase-0000-cells/`.
It contains `STANDARD.json` with `scientific_acceptance:false`,
`FRESH_BASELINE.json`, `DATA_SOURCE.json`, `FINAL_WEIGHTS.json`, `SKILL_BANK.json`,
the copied evaluator/runtime, frozen manifest/journal, state,
`PHASE_CHECKPOINT.json`, `PHASE_COMPLETION.json` and `OPERATIONAL_REPLAY.json`.
It contains no `parent.json`, `pop1_core.py` or maker AUTHOR directory.

Final checkpoint hash:
`ac7d7d0ba84fe047de7a3eb0a36116a124e21290e69570eb5ec6f9819691066c`.
Completion hash:
`02532b1c745de19e0e941a07aa96392765012c0e50ba1eaa6e355f557304bd71`.

Independent replay verified manifest/receipt consistency and every saved runtime
hash, then imported the saved evaluator. Predictions received demonstration
pairs and query INPUTS only. Adding poisoned query outputs did not change them.
Shape-preserving zero and full random overrides used every cell/router position;
random seeds were `2026100651`, `2026100652`, `2026100653`. Saved predictions match
the checkpoint outcomes. Report files remain absent. The replay record keeps
runtime errors distinct from wrong answers.

## Reproducibility and remaining bounds

Use the bundled Python documented in `LOCAL_READINESS_20261006.md`, with
`PYTHONDONTWRITEBYTECODE=1` and the three numerical thread limits set to 1.
The exact normalized plan is
`data/arcagi2-local-20261006/ZERO_WORKER_PLAN_RECTANGULAR.json`.
Its explicit fresh manifest has SHA256
`e9f1774ea2b05f60314c8dd99a2553e5c8d5aea3652cd2fc924381ae6ccdd8df`.
Preview is read-only:

```text
python scripts/evolution_epochs.py runs/weights-local-fresh-rectangular-20261006 --config data/arcagi2-local-20261006/ZERO_WORKER_PLAN_RECTANGULAR.json
```

Do not replay completed phase 0 or silently change its frozen plan. A separately
authorized experiment needs its own directory and budget. No four-call worker
batch has been launched: the parent's pending call-limit decision and accounting
work remain separate. The 62,000-credit budget was not converted into tokens or
percentage quotas, and no hard credit-cap enforcement is claimed.

The original server artifacts remain absent. Standard/ARC-goal suites were not
rerun as a broad campaign, and their known qualification limitations are not
resolved by these checks. The small input tests initially hit a Windows sandbox
permission issue with Python's `TemporaryDirectory(0700)`; they now use ordinary
inherited-permission temporary directories with guarded cleanup, without changing
ACLs or credentials. Four inaccessible empty temporary directories from that
failed test setup were left untouched. No dependency installation, server work,
security-setting change, commit, push or deployment occurred.

Logs: `fresh-local-input-tests-20261006.log`,
`fresh-local-phase-20261006.log`, `fresh-local-phase-fix1-20261006.log`,
`fresh-local-replay-invalid-20261006.log`,
`fresh-local-phase-rectangular-20261006.log`,
`fresh-local-replay-rectangular-20261006.log` in `reports/`.
