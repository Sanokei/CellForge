# Control execution must be valid before ablation counts

Base commit: `02beaf18251fca5ed1f1eb55e92f00a3e3820e88`.
This is an integrity correction, not a learned-weight capability result.

## Defect and correction

`structcode.evaluate` previously kept only the hit count from each control's
`fit` result. A zero/random control that crashed could therefore count as an
imperfect control and support `controls_ok=True`.

The evaluator now preserves every arm's `fit` and `errors`, including random
seed attribution, in `control_results`. It also emits flattened `control_errors`.
Acceptance requires all four arms to execute without errors as well as score
below perfect demonstration fit. Existing `zero_fit` and max `random_fit`
numeric fields retain their meanings. Static/candidate `errors` remain separate.
No control is resampled, clipped, dropped, or repaired. Seeds 11/12/13 and
uniform[-9,9] are unchanged. Query scoring behavior is unchanged.

`struct_pilot.feedback_of` now reports `control_execution_errors` rather than
claiming that invalid controls prove tensors do not matter. Feedback requires
no extra execution and reads no query data.

The owner specifically authorized correcting the existing raw-LUT regression
test. That fixture fits its candidate demonstrations/query, but all three random
arms produce invalid grid values. It must be rejected. A separate new modulo-10
fixture is total over the existing control domain and demonstrates legitimate
positive acceptance. No production template or qualification test was changed.

## Verification

Cloud Python 3.12.14 / NumPy 2.3.5. Source was materialized through exact-commit
GitHub reads and checked against Git blob identities; this is a partial source
snapshot, not a full Git clone. All scripts/src/tests and governing files were
present; historical run artifacts were retrieved only as needed.

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest tests.test_structcode tests.test_structcode_control_errors tests.test_struct_pilot_feedback -v
Ran 28 tests in 2.487s
OK
```

Independent rerun: 28 tests, OK in 2.451s. Coverage includes real sandbox
positive/negative controls, all four arms and four error categories, partial
fit plus errors, untruncated diagnostics, perfect-control rejection, unchanged
seed/parameter/leave-one-out behavior, invalid-source/shape early exit,
query-free feedback and fake-author admission rejection. `git diff --check`
passed. Existing sandbox process/pipe cleanup emitted nonfatal ResourceWarnings;
this patch does not change process lifecycle behavior.

Full focused output: `cloud-control-integrity-20261008/final-focused.log`.

## Qualification remains red

All 19 tracked `STANDARD.json` files were retrieved unchanged at the base
commit (16 top-level run manifests are discovered by these gates).

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 60s python -m unittest tests.test_one_standard tests.test_standard_semantic -v
Ran 25 tests in 1.681s
FAILED (failures=7, errors=3)
```

Seven failures: charged-budget reconciliation, seed count, increasing accuracy,
crossover, founder/niche coverage, mutation-over-random, distinct-genome count.
Three semantic errors are `KeyError: 'evaluator'` on the frozen incomplete
manifests, not permission errors or a claim that missing source code was fixed.
These manifests contain empty founder/genome lists; no missing referenced tensor
file explains those failures. The receipts were not changed to make gates green.
See `cloud-control-integrity-20261008/required-standard-with-manifests.log`.

The historical skill-niche/hard-lane command ran 69 tests and failed 19
real-run checks because the partial snapshot lacks a lane snapshot. No closed
lane was reopened or fresh lane result manufactured. Additional related
regressions ran 47 tests with seven errors from absent historical composition
fixture artifacts; these are not full-suite passes.

The first standard check before retrieving manifests had eight missing-run
failures. That diagnostic is superseded by the complete-manifest result above.
The first focused run, before the authorized test correction, had one raw-LUT
expectation failure; the final 28-test result above is authoritative for this
patch. No historical scientific receipt was rescored or overwritten.

No author call, new campaign, benchmark download/access, floor change, protected
test change, or STOP override occurred. Full project qualification and a shared
learned model across ARC-1/2/3 remain unestablished.
