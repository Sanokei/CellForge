# Third audit repair: owned failed-unit collection

REPAIRED FOR INDEPENDENT RE-REVIEW ONLY. Migration/research dispatch remain held.
The frozen scientific protocol, PLAN and nine science identities are unchanged.
No new ARC3 study, paid API or paid compute was launched.

## Focused change

Systemd.stop still verifies that the owned job tree is empty. A separate
Systemd.collect operation removes only verified owned empty failed-unit metadata
with systemctl reset-failed <exact fixed unit>. It requires the durable owner
token and matching invocation, rechecks them immediately before mutation, and
proves LoadState=not-found plus an empty fixed cgroup before reuse. Unknown status,
wrong owner, missing/mismatched invocation or populated/unreadable cgroup blocks
collection. Inactive transient metadata is only waited on, never reset as failed.

The controller now follows this durable order:
1. Stop and verify the owned tree is empty; preserve raw failed-unit status.
2. Persist the immutable TERMINAL receipt, including original failure effect and
   backoff deadline.
3. Collect the owned empty transient unit, then persist COLLECTION.json binding
   the terminal receipt, owner/invocation and verified absence.
4. Replay/apply the terminal effect exactly once and only then clear active.

Recovery of an existing receipt repeats verification/collection without creating
a new outcome, deadline or failure count. A crash before/after reset-failed or
after collection retains the original receipt and accounting. Active is never
cleared while failed metadata still prevents reuse. Success/cancellation behavior
and global failure/backoff/circuit rules are otherwise unchanged.

The former Fake.stop backend incorrectly made stopped failed units absent. It
now retains loaded/failed metadata until the real inherited collect method runs;
reset-failed commands are simulated explicitly. This lets the offline tests
reproduce the verified integration failure.

## Independent regressions

collection-regressions.log:11 focused independent regressions pass, including
failure→original backoff→next distinct condition; crashes after receipt,
immediately before/after reset, and after collection before accounting; third
failure circuit; unknown/unowned/invocation/cgroup refusal; and owner replacement
at the collection checkpoint.

focused-tests-collection.log:75 invocations/66 distinct regressions pass (nine
controller cases repeat through an inherited fixture class).
standard-contract-collection.log:seven unchanged contract tests and unchanged
hash/ledger/charge checks pass. No protected qualification test is altered.

## Actual Linux integration

LINUX_COLLECTION_PROBES.json and LINUX_COLLECTION_RUN.log record actual v2 main,
Systemd and Controller executions on the existing server. These were isolated
operational fixtures: a detached exit2/sleeper worker, no research admission,
no ARC game/inference, an isolated unit/cgroup/dispatcher lock, and runtime
tightened to40seconds. The helper comparator always returns False and is used
only for failed fixture jobs; it never supplies scientific results.

Baseline and three real SIGKILL cases (before reset, after reset, after collection)
all retained the failed loaded-unit status in the original receipt, collected
the unit, preserved count1/deadline/receipt bytes, waited the actual30-second
backoff without editing it, and launched exactly one different condition.
Each had eight pending jobs after the first failure and two launch attempts
after progress. The next fixture was cancelled before any third launch.
Detached descendants and cgroups were dead/empty.

A real SIGKILL after reset on the third failure recovered count3, kept the
circuit closed and launched no next condition. Actual retained-unit refusal
checks rejected wrong owners and invocations. Unknown read failure was injected
only into the probe read transport so the shared real bus was not disrupted;
it caused zero mutation calls and left the retained unit unchanged. Correct
owned collection then proved absence.

No external/manual reset-failed cleanup was used to make these cases progress.
All six probe units were collected by v2 code, verified absent/empty, and only
the generated fixture directory was removed. The evidence binds the exact
tested supervisor/common source hashes. Prior CPU/timeout/OOM containment
evidence remains in the bundle and was not rerun for this metadata-only change.

## Preserved evidence and next step

CHECKPOINT.json verifies the old watcher remains paused_exhausted, job PID0,
nine completed outcomes, and unchanged first-study code, plan, receipts/results
and20 protected STOPs. Previous sealed reviews are preserved in workspace
review-history/. This bundle contains no production approval, admission,
dispatch marker or service installation.

The science plan ID remains
b409293e233bcd7d027b88b34506a867d32c0b68ffe043e95a5b62ed7bc54dfb.
Full scientific qualification remains RED/UNQUALIFIED; no supervised accuracy,
model provenance timestamps or missing population/lineage/router milestone
is fabricated. Independent review of the new operational manifest is still
required before a reviewed migration.

