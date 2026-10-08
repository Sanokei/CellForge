# Separate A/B founder adapter: code preparation only

## Status

The adapter and focused tests are implemented. No real execution directory,
reservation, consumed attempt, model call, candidate prediction or matched control
evaluation has been created. All stateful tests use temporary fixtures; evaluator
execution is mocked. One temporary unit test constructs synthetic seven-value
literals solely to check the seed/order contract and never applies them to a panel.

The already frozen preparation contract remains unchanged:
`f4f89d4eaeec4adbd50f4f8bd42c5e35ad56cf535396032f36f516a5b93a09f4`.
The earlier terminal author tranche, phase1 runtime, templates, receipts and
protected tests are unchanged. This supplement does not authorize execution.

## Implementation boundary

`scripts/nca_prompt_comparison_intake.py` is separate from the old mutation intake.
It reuses the hash-verified historical strict literal parser, observed-zero-tool
HOST_AUDIT validator, numeric fingerprinting, frozen runtime loader, exact tensor
injection and comparison helpers. It never invokes the old reservation, mutation,
second-random or dispatch path. There is no model-dispatch implementation.

A distinct future execution directory carries one fsync-backed hash-chained
JSONL ledger. The finite transition sequence is:

1. Initialize under new exact approval; reserve A
2. Record A submission intent and full 1,488-position charges per accounting arm
   before any possible external dispatch
3. Capture A's original bytes, validate host/native receipts and numeric response
4. Run A and its seed-2026100803 matched random in bounded child processes
5. Only after valid, complete A/control results, reserve/consume/capture/evaluate
   an independent B with seed 2026100804

Both roles are fresh founders with no parent tensor and no cross-arm feedback.
At most two author invocations and 2,976 positions per accounting arm are allowed.
Invalid/refused/audit-failed/uncertain consumed attempts retain their charges and
terminally stop. No control is constructed after terminal STOP; charged positions
and actual control computation remain separate. A timeout blocks B. Duplicate
reservations, submission, capture and evaluation are rejected. Partial ledger or
artifact transactions cannot be silently reset or repaired; a missing original
completion stays unresolved until terminal handling, never a relaunch.

All mutating transitions and workers check STOP with `lexists`, including dangling
STOP symlinks. Status and explicit terminal recording remain available when the
execution STOP marker is present. Repository and state paths reject symlinks and
are resolved before protected-root containment checks, including `..` components.

Fresh route snapshots bind packet/schema inputs, new independent sessions, no
prior weights or control/cross-arm feedback, exact gpt-6.1-sol/low settings,
120-second and 2 GiB author limits, no retries, and observed-zero-tool provenance.
The same empty input-only role directory may be reused by separate ephemeral
invocations; independent sessions and absence of prior artifacts/context must be
verified by the parent. Preventive tool isolation is not claimed.

Quota snapshots must be authoritative, ordinaryUsageAllowed=true, no more than
300 seconds old, in the same declared period/reset, and below the 85% shared STOP.
The parent supplies the maximum usage across all relevant authoritative windows;
an incomplete/unmetered attestation is not valid permission. Both reservation and
submission recheck freshness. Snapshot/config hashes and exact packet-specific
parent approvals are recorded. Period/reset drift blocks this comparison.

## Actual V5 route bindings and receipt evidence

The adapter supplement binds these existing locally staged artifacts under
`../cellforge_route_review/`:

- Reviewed capture source `prospective_native_capture_v5.py`:
  `5d822fff874cbd84856474ed275912a43936da3af881dff63c33e1d89539c43b`
- A config, 8,833 bytes:
  `720dd9f88aa0af7022ea9235a83886bbadaf69d0db7a04dc2d1a9d390c83ceaf`
- B config, 8,833 bytes:
  `77c49e86ebb18fa7437ef50aaf5e3289c97082ccc753c0b7d15ecd6c49f065c8`

The original V5 validator owns full native-envelope validation. The new adapter
does not duplicate native trace parsing. It validates the existing HOST_AUDIT
contract and exact original V5 result/NATIVE_USAGE bytes, line-hash correspondence,
all five native usage counters, EOFs, empty stderr, timeout/RSS limits and raw
trace hashes. The parent must still verify actual prompt/schema/command receipts
and that the V5 result belongs to the recorded submission. Parent attestation is
not a cryptographic platform receipt. V5's server file writes remain non-fsync;
the new cloud ledger and capture copies are fsync-backed. Missing/crashed host
receipts never authorize relaunch.

Candidate bytes are durable before parsing. Exact submitted audit, native-usage
and host-result bytes are also saved before parsing, including invalid data, under
`restricted_raw_evidence/`. These are hash-bound separately in the ledger and
are **not publishable artifacts**. Do not attach or publish that directory.
Unexpected strings and private reasoning are not copied into ordinary receipts
or errors. The original complete server trace remains restricted outside this
adapter's artifacts.

## Evaluation and interpretation

Future workers use unchanged phase1 float64 inference, exact literal injection,
17-demo coverage/order, leave-one-out context and resource bounds. The adapter
checks each process result and summarizes saved rows with the original helper.
Fixed identity, neighbor-smoothing and zero results are read unchanged, never
reevaluated here. Exact RNG is Python `random.Random(seed).choice`, the common
vocabulary, sorted tensor names and C/row-major traversal; the exact Python runtime
version is bound by the execution ledger before dispatch.

The final B receipt reports descriptive B−A exact demos, secondary task/error
summaries, numerical identity equivalence and panel-local prediction equivalence.
Positive feasibility separately requires strict exact-demo superiority over the
best fixed reference and every consumed matched random, preserving reference
successes. At B completion, A is reassessed against both random controls so that
a provisional A-only label cannot survive a stronger B-matched random. B>A alone
is not feasibility or capability gain. One pair has no causal/generalization claim.

Only temporary fixtures have exercised the integration. The bounded runtime
worker and real V5 handoff have not run for this comparison. Fresh-process
independent replay remains required before any scientific result claim, as in
the frozen preparation. This code-preparation stage does not constitute runtime,
model-route or ONE_STANDARD qualification.

## Commands and API

All commands below are future operational interfaces, not authorization. Each
non-freeze command requires `--repo`, the exact `--supplement` file and externally
retained `--supplement-sha256`, and a new separate `--state` directory. The planned
supplement is frozen outside the preparation at
`reports/nca-prompt-comparison-integration-20261008/ADAPTER_SUPPLEMENT.json`.

The read-only CLI verification below was also run successfully. Its proposed
execution path remained absent before and afterward:

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/nca_prompt_comparison_intake.py verify \
  --repo . \
  --supplement reports/nca-prompt-comparison-integration-20261008/ADAPTER_SUPPLEMENT.json \
  --supplement-sha256 72682bb5fc3643a77c3715ee5c657596d4027f48865bd3550ecc811558990412 \
  --state runs/weights-nca-prompt-comparison-execution-20261008
```

```text
python scripts/nca_prompt_comparison_intake.py verify [common flags]
python scripts/nca_prompt_comparison_intake.py initialize [common flags] --approval A_APPROVAL.json
python scripts/nca_prompt_comparison_intake.py reserve [common flags] --arm A --approval A_APPROVAL.json
python scripts/nca_prompt_comparison_intake.py submit [common flags] --arm A --host-reference EXACT_HOST_REFERENCE --route A_ROUTE.json --quota CURRENT_QUOTA.json
python scripts/nca_prompt_comparison_intake.py capture [common flags] --arm A --raw FINAL_ORIGINAL.bin --audit HOST_AUDIT_ORIGINAL.json --native-usage NATIVE_USAGE.json --host-result result.json
python scripts/nca_prompt_comparison_intake.py evaluate [common flags] --arm A
python scripts/nca_prompt_comparison_intake.py status [common flags]
python scripts/nca_prompt_comparison_intake.py stop [common flags] --arm A --reason SPECIFIC_REASON --host-reference ORIGINAL_RECEIPT_REFERENCE
```

Only after A's complete valid evaluation can the corresponding B commands run.
`submit` writes consumption; it does not dispatch. The external parent-controlled
host route must dispatch at most once only after the fsynced submission receipt.
`worker` is an internal evaluation-child entry point, not an author interface.

Python entry points are `Adapter(...)`, `initialize`, `reserve`, `submit`,
`capture`, `evaluate`, `status` and `stop`. `capture` requires exact original byte
strings for candidate, submitted audit, native usage and V5 result. Approval and
route/quota object contracts are explicit in `approval_check`, `route_check` and
`quota_check`; complete temporary fixtures are in the new test module. No
production approval or quota receipt is synthesized by the adapter.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_nca_prompt_comparison_prepare tests.test_nca_prompt_comparison_intake -v
Ran 58 tests in 3.785s
OK
```

This is 30 unchanged frozen-preparation tests and 28 new adapter tests. The latter
cover finite order/charges, independent founders, duplicate actions, pending
reservations, partial capture/hash-chain/evaluation failures, invalid terminal
accounting, no post-STOP random construction, stale/85%/ordinaryUsageAllowed and
period drift, input isolation, real V5 config hashes, exact receipt-byte retention,
private-evidence separation, STOP/dangling STOP, path/repository traversal,
artifact tampering, relative repository paths, seed ordering, fsync and
frozen-preparation preservation. A relative-path bug was caught by the final
pre-freeze CLI check, corrected with one normalization line, and covered by the
new regression before the successful exclusive freeze.

The broad standard suites were not rerun under this preparation-only scope.
The prior terminal-tranche receipt remains 25 tests, seven failures and three
errors; its command, timestamp and hash are in
`NCA_PROMPT_COMPARISON_PREPARATION_20261008.md`. No qualification claim is made.

Final source: 42,623 bytes,
`fa770e3ad95158d37490a3653efc6ddcf787121c82759d4650925f55eb86c0aa`.
Tests: 25,462 bytes,
`53a5e270af1a2f13d8a72bdb0ebb5d6dbc2353d21a20dacc8ee118196832671c`.
Frozen supplement: 2,455 bytes,
`72682bb5fc3643a77c3715ee5c657596d4027f48865bd3550ecc811558990412`.
Exclusive creation and read-only verification succeeded. The supplement records
execution approval as false and real reservations as zero. No real initialization
or reservation was performed.
