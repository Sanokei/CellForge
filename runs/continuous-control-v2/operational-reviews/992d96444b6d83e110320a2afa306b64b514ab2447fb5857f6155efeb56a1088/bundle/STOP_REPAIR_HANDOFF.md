# Stop during launch repair: independent operational review handoff

PREPARED ONLY. No production replacement, research dispatch, resume, new scientific admission or external model call occurred during this repair. Start here; the original README/COLLECTION_HANDOFF and other copied review documents describe the preceding approved seal and are preserved history.

## Exact scope

The only changed file from the preceding 188-file manifest is supervisor.py. Original supervisor SHA256: 1863b422bba669d75936b7635bc90195d60cdd064be7ecbdfbb650e16b49bc35. Repaired supervisor SHA256: 9493ef435131c7f504a87ac470c02902d40eb6390ed00fb3173e1e4889b5480e. REPAIR.diff is the exact unified diff against the prior source. The new observe_dispatch.py is a separate read-only operator tool.

PLAN canonical ID remains b409293e233bcd7d027b88b34506a867d32c0b68ffe043e95a5b62ed7bc54dfb; PLAN bytes, all nine fingerprints, policy, evaluator, tensors, controls, game data/version bindings, charges, seeds, acceptance and limits are unchanged. common.py, worker.py and all protected qualification tests are unchanged. The original archive is preserved verbatim at prior-seal/approved-review.tgz, SHA256 230e3c69490463f73f845c2ab11d40960f4ce4ffe8a52bffbf4ca6175da00052. BASE_MANIFEST.json is its original manifest. Migration evidence is preserved under migration-evidence/.

## Repair behavior

Controller.tick rechecks halt/STOP/dispatch removal after gates, immediately before reservation while holding REGISTRY.lock. Such an early stop leaves the condition pending and the attempt count unchanged. After durable intent and just before the launch call it rechecks all stop/cancel gates; cancellation then consumes exactly the already-reserved attempt without launching. A stop during the actual RPC persists/adopts the exact owner invocation and immediately cancels it through the unchanged durable terminal/collection/accounting path. A lost RPC acknowledgement never guesses ownership: unknown state keeps active; verified owned or absent state can reconcile cancellation.

The --stop path creates STOP once using exclusive creation, file fsync and POSIX directory fsync. It never overwrites existing marker bytes. It calls Controller.tick, which shares the registry lock with launch admission, then verifies durable active=None plus verified fixed-unit absence and an empty tree. It cannot use the former unlocked idle snapshot halfway through launch. The main loop drains before a controlled halt return and retains the global dispatcher lock while unknown ownership/transport/collection requires retry. No successful graceful exit is claimed without the proof.

Uncatchable SIGKILL or an external forced termination cannot execute a receipt handler at the instant of death. Durable intent/token and invocation when already committed remain for recovery; the actual Linux tests verify STOP-driven recovery, original invocation reconciliation and no repeat launch. The job's original systemd resource/runtime containment is unchanged. The repair does not claim to make arbitrary SIGKILL synchronous or bypass unavailable ownership evidence.

Systemd stop/ownership/cgroup/failed-unit collection checks are unchanged. The order remains terminal receipt persisted, verified collection and COLLECTION.json persisted, then accounting/active clearance. The original sequence, backoff deadline, global failure circuit, consumed identities and cap survive. Pending individual cancellation still consumes the finite cap, as before.

## Operator observation repair

observe_dispatch.py only reads atomic registry snapshots, unit/cgroup status and actual resource policy. A running claim requires matching token/invocation, rechecked unit identity across the resource read, stable registry bytes, and original 1CPU/1GiB/600seconds/64tasks/private-network settings. An owned process under STOP is labeled owned_running_stop_requested, not a continuing dispatch success. Missing invocation is reconciliation; unknown status or replacement ownership blocks a running claim.

Default usage is one snapshot. A bounded observation slice such as --watch-seconds 45 returns observation_pending when it expires. It never creates/removes STOP or DISPATCH_ENABLED, stops a unit, cancels a condition, edits registry, or imposes a new worker deadline. Use PYTHONDONTWRITEBYTECODE=1 when executing it. The real worker RuntimeMaxSec=600 and genuine stop/cancel/failure gates determine job termination. The former 15-second operator deadline and its automatic rollback are not reused.

READONLY_PRODUCTION_OBSERVER.json records this exact tool observing the held production queue. Expiry returned observation_pending, with root registry and both new STOP hashes identical before/after.

## Verification

- preserved-regressions.log: all 75 prior offline test invocations pass (66 distinct cases; nine inherited controller cases repeat through the existing collection fixture).
- stop-boundaries.log: 20 new test invocations pass, including the four original failed reproductions and parameterized gates/intent/RPC/crash boundaries.
- observer-final-tests.log: six final observer cases pass, including one additional identity-recheck/STOP case; five repeat earlier observer cases. Across these logs there are 21 distinct new offline cases. The combined suite now has 96 invocations/87 distinct cases if run once.
- repair-standard-contract.log: seven unchanged contract tests pass; hash integrity, ledger chain and budget reconciliation pass. Full scientific semantic/milestone qualification remains RED/UNQUALIFIED.
- LINUX_STOP_LIFECYCLE.json and stop-linux-run.log: eleven real Linux cases pass: STOP/--stop, SIGTERM and dispatch removal during gates; concurrent --stop and SIGTERM during actual launch; real SIGKILL at gates, durable intent, before RPC, committed launch, receipt and collection boundaries. The wrapper uses the exact repaired Controller/main/Systemd with real POSIX registry locks, signals and systemd cgroups. Only gates are bypassed for fixtures; worker.py is replaced by detached sleep-only workers. All descendant PIDs die and unit/cgroup absence is proved before graceful return or after recovery.
- LINUX_RPC_CRASH.json and rpc-crash-linux-run.log: an additional real SIGKILL after the worker spawns but before invocation commit leaves launch_intent with token and no invocation. Recovery matches the actual owner, cancels the original invocation, persists receipt/collection once and launches no next condition.

There were zero research admissions, ARC plays or model calls in these operational fixtures. Isolated runtime was tightened to 40 seconds; production remains 600 seconds. All twelve fixture units/cgroups are absent and both generated fixture directories are removed. Every hashed production file is identical before/after, including the admitted immutable bundle, registry, old code/receipts/results, 20 historical STOPs and both added STOP markers. The first eleven-case driver is preserved as stop_fixture_driver_11cases.py; the extra RPC-crash hook is in stop_fixture_driver.py. Both exact payload streams are included and bind their respective report source hashes. The intermediate stop-boundaries-initial.log records a local missing-file invocation before fixture files were saved; the completed verification is in the later logs above.

To reproduce offline: python -m unittest test_review test_rereview test_collection test_stop_during_launch test_stop_lifecycle -v; then python check_standard_contract.py. The Linux payloads create isolated sleep-only units and are not research runners; use only within the same explicit operational fixture authorization. Independently verify their hashes and cleanup evidence rather than assuming a printed PASS proves the source.

## Production hold and proposed resume

Current production remains paused_stop, 0 completed new jobs, 1 cancelled, 8 pending, active=None, attempted1/cap9, sequence1, consecutive_failures0, dispatch absent, both watchers stopped/disabled. Cancelled fingerprint 15daf18a8e92f551a4c04fee1d85fbe22ccba1f20b620219e1c751e8da18d686 stays globally consumed. Its 13/27 partial plays remain preserved and incomplete, never replayed/requeued or treated as a completed comparison. No scientific follow-on is inferred from them.

The new control STOP hash is fb8860d6f4e469a92092c2901f8450e5bcc1b05e0d1f8a8f52a063b6842c237b; the old control STOP hash is 6d20f7a21219d86f8b32e2ffc3ddd22ebf8f0a02cd3bf7c3fd1676aacd7b0190. All 20 earlier STOP markers remain byte-identical too.

RESUME_PROPOSAL.json enumerates only the eight eligible conditions, original global accounting and staged procedure. Existing root STOP remains an unconditional launch prohibition in this repair. A repair review PASS alone cannot authorize removing, ignoring, reinterpreting or relocating around that marker. Therefore this proposal remains held: a specific owner/parent STOP-handling decision and review of any new resume mechanism are required before actual resume can be concrete. No such override is implemented or used. Operational deployment, if separately authorized after exact-seal review, must be staged paused, preserve/back up the runtime, bind the new source to its own operational audit, and leave the immutable admitted scientific bundle intact. It must not re-admit the same plan, reset attempted/sequence/failures, or replay the cancelled condition.

## Review request

Independently verify this exact operational seal and diff, rerun the offline cases, check Linux source hashes/real lifecycle records and live cleanup, and verify the held production registry/STOP/old evidence. Assess whether controlled stop returns only after durable cancellation/collection, including unknown/lost-ack and pre-invocation crash recovery, while all previous guarantees remain. Return PASS/FAIL and exact blockers for this operational repair only. Do not deploy, remove/reinterpret STOPs, resume or dispatch research during review. Scientific qualification remains false and the resume policy decision remains open.