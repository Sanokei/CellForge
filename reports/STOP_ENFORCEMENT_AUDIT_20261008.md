# STOP and proposal-accounting audit

Base source snapshot: `02beaf18251fca5ed1f1eb55e92f00a3e3820e88`.
The cloud checkout is an exact-blob partial snapshot, not a full Git clone.
This is an offline source/regression audit, not a weights experiment or a claim
of improved model capability.

## Reproduced defect

A `maker.ask` exception after dispatch can coexist with an attempt-specific,
valid meter receipt. `WeightWorkers._attempt` then recorded a finished attempt
with `submitted=None`, `submission_uncertain=True`, and `metered=True`, but did
not halt. `_load`/construction checked only unfinished attempts. Consequently,
both continued use and reopening the pool admitted further work despite the
unresolved submission. A meter proves usage attribution, not settled submission.

Before correction, a temporary fake-maker reproduction printed:

```
uncertain receipt: {'finished': True, 'submitted': None,
                   'submission_uncertain': True, 'metered': True,
                   'charged_positions': 1}
halt: None
in-process later proposals: 1
resumed later proposals: 1 total fake calls: 3
```

The initial eight-test audit failed specifically on the missing safety halt;
the seven other checks passed. No actual maker process was invoked.

## Minimal correction

Four added source lines in `scripts/weight_workers.py`:

- Durably halt with `uncertain_submission` when a dispatched attempt finishes
  with unknown submission state, before recording completion.
- On reopening a preexisting ledger, halt if any finished attempt has unknown
  submission state or the explicit uncertainty flag.

Reservations remain charged. No receipt is rewritten, retried, replaced, or
released as a valid proposal. Existing pending-reservation, STOP, timeout,
budget, lease, and accounting semantics remain in place. Already running remote
work cannot be cancelled by this cooperative STOP mechanism; the patch does not
claim otherwise. Reconciliation is still a separate authorized operation.

## Verified scope

The ten new tests in `tests/test_stop_enforcement_audit.py` exercise:

1. STOP before reservation, including a dangling marker, with zero calls.
2. STOP after reservation, before worker dispatch, preserving its charge and
   the durable halt after marker removal and reopening.
3. STOP immediately after the dispatch receipt, with no maker call.
4. Pending reservation reopening without replacement.
5. A metered but uncertain maker exception blocking both current and resumed work.
6. Reopening a synthetic pre-fix finished-uncertain ledger without altering it.
7. Invalid and identical valid proposals each consuming their own position charge.
8. Explicit disabled driver policy refusing the legacy backend; disabled pool
   making no calls.
9. Manifest reopening rejecting changed source hash, task-content hash, and task
   population.
10. Driver resume rejecting reordered router experts.

Existing `tests.test_native_parallel_batch` contributes seven unchanged tests,
including full charging of invalid and duplicate native submissions, repeat-slot
rejection, saved-response tamper rejection, frozen context provenance, STOP, and
rectangular-safe random controls. The native batch script itself has no model
dispatcher; its import/evaluation STOP check is in `native_proposal_import.stop`.
Its slot directory is created exclusively, preventing import overwrite/retry.

Source inspection also confirms `head_driver` binds all script digests, saved
runtime dependencies, search/development task contents, and the ordered router
roster before resumed worker construction. `evolution_epochs` rejects uncertain
worker completions for phase publication, including metered ones. That downstream
check did not prevent the worker-pool defect in this report.

The direct batch04 terminal and checkpoint remain untouched: three submitted
calls, one cancelled/uncreated author slot, zero additional author allowance,
no replacement, and no batch05. No direct-batch author API was invoked or
simulated as a new campaign.

## Checks

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest tests.test_stop_enforcement_audit tests.test_native_parallel_batch -v
Ran 17 tests in 0.122s
OK

python -m py_compile scripts/weight_workers.py tests/test_stop_enforcement_audit.py
# Passed
```

Full focused output: `cloud-stop-audit-20261008/focused.log`.
No preexisting worker-specific unittest file was found in the materialized
snapshot; the pool checks above are additive. Driver tests use minimal fixtures
for the selected boundaries, not a complete live-data driver integration run.

No author/model call, experiment, external data access, protected test edit,
floor change, closed-run modification, or STOP override occurred. Full project
qualification remains red/incomplete as documented separately; these focused
safety tests do not establish ONE_STANDARD compliance or scientific acceptance.
