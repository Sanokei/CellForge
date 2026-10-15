# Repairs for independent re-review

1. Unknown status and ownership
supervisor.Systemd.show checks transport success and all properties.
empty rejects unverified/incomplete status, checks the fixed cgroup even when
ControlGroup is blank, and treats unreadable/missing population evidence as
unknown. owned checks Description token and durable InvocationID. Unowned jobs
are not stopped or replaced. StatusSafety plus controller fixtures cover these.

2. Exactly-once failure accounting
common.make_terminal writes an immutable ordered effect; replay_terminal checks
its digest/sequence and uses receipt IDs to apply it once. Receipt first, registry
second. Original terminal time anchors backoff. Crash after the third failed
receipt recovers count3 and blocks the next job; replay after state commit does
not recount. ReceiptReplay and ControllerCrashPoints cover both boundaries.

3. Science identity and shared consumption
science_identity hashes normalized actual AST/tensors/control multiset and
data/packages/seed/action/scoring/reset bindings. It excludes packaging hashes,
provenance packets, timestamps, labels, paths and source comments/docstrings.
MANIFEST separately binds operational bytes. Registry spans approved plans.
Legacy import verifies original config/result/source hashes and preserves
original receipt hashes without edits. Duplicates queued or consumed anywhere
are blocked. Study cap9 is validated and enforced on every reservation, including
pending cancellation. New plans cannot reset failure count/backoff.

4. Standard schema/discovery
New studies use canonical existing lab runs/weights-transition-<plan hash>/,
with jobs under the study. The proposed STANDARD supplies both contract and
importable evaluator boundary, full required base fields and extended
provenance/ledger fields. Seven unchanged contract tests and three unchanged
artifact checks pass in the declared preview root. No qualification is claimed.
The interactive adapter intentionally raises Unsupported rather than fabricating
classification tasks/accuracy. Existing milestone/semantic gates stay red.

5. Evaluator provenance
The next bundle preserves deployed guarded E source hash958e922e... exactly.
EVALUATOR_PROVENANCE documents four extra null-observation guard lines versus
historical head. The earlier phrase byte-identical evaluator was only true
against the chosen research source, not historical head. Old receipts are intact.

6. Next-study protocol
New state-conditioned effect memory includes bounded observed successor
lookahead and eight-decision delayed public-progress credit. Retained controls:
random, zero, action_off, memory_off, blind, frozen spatial. Added context_off
for exact aggregate memory and a matched new scripted frontier comparator.
All primary/random masks and support ranges match, with fixed spatial/gates.
Construction charges and replay charges are distinct. Initial raw states and
observed legal masks are captured; partial/time-limited jobs are incomplete.
Acceptance requires same founder across at least2/3 seeds, actual completed-level
gain over every comparator on a multi-action game, and no spatial-reference
regression. lp85-only gains cannot qualify action selection. Model calls remain0.

7. STOP and evidence hardening
Explicit service stop writes persistent STOP and verifies the owned job tree;
crash ExecStop does not synthesize STOP. The controller independently checks
all27 result records rather than trusting a success summary. Tests cover absent
results, matched complete results, explicit STOP and crash adoption.

Not established by these local fixtures:
Deployment of v2, actual new ARC outcomes,
full standard qualification, successful quota adapter or autonomous scientific
queue refill. The first deployed runner's resource/crash tests remain historical
evidence only. Independent re-review and reviewed migration are outstanding.


Second independent re-review blockers and actual Linux results are documented
in REREVIEW_HANDOFF.md. Version-stable field encoding, active cancellation and
zero-comparator failures were independently reproduced and repaired. Linux v2
lifecycle/resource behavior is now tested with isolated operational fixtures;
scientific qualification and migration are still not established.

Latest operational repair and evidence:COLLECTION_HANDOFF.md documents owned
failed-unit collection before terminal accounting/active clearance. The frozen
science plan is unchanged;75 focused test invocations and the actual Linux
failure/backoff/next-condition and collection crash/refusal cases pass. Migration
and research dispatch remain held; full qualification remains red.
