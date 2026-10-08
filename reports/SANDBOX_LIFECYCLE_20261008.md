# Bounded sandbox worker cleanup

The control-integrity tests exposed existing subprocess/pipe ResourceWarnings:
`Sandbox.close` killed the worker but did not wait for it or close its streams.
This follow-up is operational reliability, not scientific capability progress.

The parent now reaps the process with a finite wait and closes stdin. The pump
closes stdout in `finally` and always signals end-of-stream, even when stream
read/close raises. The reader thread is joined with a finite timeout. Closing an
already-closed sandbox is harmless. Expected broken-pipe errors during stdin
cleanup do not interrupt teardown.

If process reaping times out, ownership is retained and the sandbox records
that it is closing. Later calls first retry cleanup and report
`worker cleanup pending` instead of writing to a closed pipe or starting a
second worker. A worker that exited between calls is also cleaned before it is
replaced. The source whitelist, process environment, network-isolation attempt,
memory/file-size limits, scoring, tensors and call timeout remain unchanged.

## Tests and review

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -W error::ResourceWarning -m unittest tests.test_structcode_lifecycle tests.test_structcode tests.test_structcode_control_errors tests.test_struct_pilot_feedback -v
Ran 39 tests in 2.528s
OK
```

The independent reviewer ran the same 39 tests in 2.535s, also OK without
warnings, and checked the source and exceptional cleanup paths. The 11 additive
lifecycle tests include real worker stream/process release, already-exited
cleanup, idempotence, bounded reap timeout and recovery, pending-call refusal,
broken-pipe cleanup, and reader read/close exceptions. Full output:
`cloud-control-integrity-20261008/lifecycle-focused.log`.

Limits: reader joins are bounded; a pathological stuck reader is not proved to
terminate by this change. Concurrent use of one Sandbox from multiple callers
is not promised. No model-author calls, experiment runs, historical replay,
protected-test edits, threshold changes or security-boundary changes occurred.
The full standard remains red; see `CONTROL_ERROR_INTEGRITY_20261008.md`.
