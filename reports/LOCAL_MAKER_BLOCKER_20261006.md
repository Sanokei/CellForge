# Initial authorized maker batch: stopped before dispatch

The owner authorized continuing model work within the existing 62,000-credit
ceiling; the parent bounded the first batch to at most two short calls,
concurrency one, before review. The first dispatch attempt is now terminal:
the unchanged worker guard durably halted with `budget_unavailable`.

**Zero calls reserved, zero model invocations, zero proposals released.** No
weight/evaluation result exists for this batch. Token usage and credit usage
remain unknown/null, not fabricated zero totals for the ongoing Codex session.

## Concrete artifacts

New, separate run: `runs/weights-local-maker-initial-20261006/`.
It contains its explicit fresh baseline/source, frozen evolution policy/manifest,
`STANDARD.json` marked scientifically unqualified, `BATCH_REQUEST.json`, the
search-only `PLANNED_PROMPT.txt`, `BATCH_RECEIPT.json`, state/status, and the
hash-chained `WORKER_PROPOSALS.jsonl` halt record. No maker AUTHOR directory was
created, and no BUDGET row or response was fabricated.

Policy: `gpt-6-luna`, low reasoning, maximum two calls, one proposal per prompt,
parallelism one, 120-second per-call limit. The existing ChatGPT-authenticated
CLI executable was configured through process-local `CODEX_BIN`; ledger/cap
locations were process-local paths in the new run. No stored credentials or
security configuration changed. The existing default 85% quota guard was not
raised or disabled. The entire dispatch attempt took 0.25 seconds, inside a
150-second supervisor limit.

The prompt binds one task from the frozen `search_fitness` partition to
`assoc_induct` and requests literal tensors from demonstration pairs only. No
selection outputs, report tasks or historical LLM provenance were provided.
Previous successful and rejected baseline artifacts were left untouched.

## Supported tooling diagnosis

`codex-cli 0.160.0 login status` again reported `Logged in using ChatGPT`.
This confirms the configured login, not remaining credits or model availability.

Only official, read-only account RPCs were attempted; none created a model turn:

1. In the filesystem sandbox, `app-server --stdio` exited because its local
   SQLite state runtime under `.codex` could not initialize (`Access is denied`).
2. A normally permitted standalone invocation removed that startup permission
   failure, but the account request did not return within its deadline.
3. `app-server proxy` with normal runtime access exited because the existing
   app-server control socket was unavailable. No daemon was started to replace it.
4. One final standalone probe confirmed initialization succeeded, then
   `account/rateLimits/read` timed out after a bounded 20-second response wait
   (24.938 seconds including startup/cleanup). It returned no quota or balance.

No credentials, session secrets, API keys or raw authenticated endpoints were
read as a workaround. The transient probe processes exited; no persistent
service was intentionally started. Existing auth/STOP/meter guards were preserved.

The immediate blocker is a **current usable quota receipt**: `maker.last_used()`
returns `(None, None)`, while `WeightWorkers._budget` requires finite usage and
a future reset below the configured cap before reserving a call. The worker
correctly refused to dispatch. Consequently `gpt-6-luna` serving availability,
per-call token receipts and optional estimated-credit fields remain untested.
No hard 62,000-credit enforcement claim is possible from the current tooling.

Proceed only after the supported account route returns a usable current meter
and the receipt path is validated. Preserve this halted run; do not clear its
halt or invent a baseline to retry it. Existing authorization does not need to
be requested again merely to perform the same bounded first batch once this
operational blocker is resolved.

Evidence: `reports/local-maker-initial-20261006.log` and the run's
`BATCH_RECEIPT.json` / `WORKER_PROPOSALS.jsonl`.

## Failure-memory interface for the separate owner

For host-verified grid failures, preserve: manifest/evaluator/input-scope hashes,
`partition="search_fitness"`, task ID, candidate ID, template, tensor hash,
demonstration index, input, expected output, actual output, and status distinguishing
incorrect prediction from runtime-invalid evidence. Runtime-invalid records carry
an error type; expected/actual grids must not be invented after an exception.
Attach the originating worker receipt when one exists; invalid or refused
proposals must remain charged attempts rather than successful candidate records.

A bounded prompt-feedback function should accept an authoritative search-task
allowlist/scope and validated host records, enforce record/byte limits, and emit
only whitelisted search fields. Reject selection/report IDs, mixed scopes and
maker self-reported scores. No failure-memory integration was made in this frozen
run, and this grid evidence is not evidence of language or social understanding.
