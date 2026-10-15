# CellForge durable queue handoff — 2026-10-11

Owner scope: existing server should perform continuous sequential CellForge work,
with bounded jobs, no paid API charges, preserved cancelled/closed experiments,
and explicit pause when no valid work exists. No other project was touched.

## Final operating state

Deployment: /srv/cellforge-inbox/continuous-spatial-20261011
Supervisor: cellforge-continuous.service, enabled, active, PID 577307 after one
intentional supervisor-only crash/restart. Research job unit:
cellforge-continuous-job.service, inactive/dead with empty cgroup.
STATE.json is paused_exhausted: all nine preregistered jobs completed, zero pending,
zero active, zero consecutive failures. No further dispatch or automatic expansion.
The durable service survives SSH/chat loss and reboot but never repeats completed
combinations. A new recorded hypothesis/protocol is needed for another study.

Study: three explicit assistant-authored 16-position tensor hypotheses x seeds
0,2,3. Public-development games ar25/lp85/sc25, 1500 actions each, same game-file
hashes, seed, runtime and scoring function for every hypothesis/control pair.
All six arms reported: authored, matched random, zero, spatial-off, action-off,
blind. Exactly 16 positions charged to authored and matched random per job.
No remote maker calls, no paid API calls, no gradients, no downloads, no excluded
evaluation access. Existing legacy maker.py was not used because it reads session
rollouts. Desktop quota was read (0% used) but is NOT asserted as server quota.
Any future writer queue needs a supported fresh account RPC and existing 85% cap.

## Reproducible result and limitations

All nine experiments completed all 18 comparisons with no runtime errors and
the same 1500-action limits. COMPARISON_AUDIT.json verifies the actual seed,
budget, scoring-source and complete-arm bindings. A timeout/partial comparison
is explicitly incomplete and cannot support a complete-arm capability comparison.
Raw results, first-400 action traces, final-frame hashes, weights, seeds, source
hashes and six development-file hashes are retained. Offline recordings were
declared disposable scratch, not evaluation evidence.

Every authored experiment solved the same ONE lp85 development level; ar25 and
sc25 stayed at zero. Every action-off ablation ALSO solved that level.
All zero, spatial-off and blind arms solved zero.
Matched random solved one in rare-region/seed0 and stagnation/seed3, zero in the
other seven comparisons. Do NOT aggregate repeated successes as nine distinct
levels or novel-game transfer. No action-weight contribution, population learning,
crossover advantage, general capability or AGI result is established.
The spatial mechanism is a versioned generic heuristic baseline. No admission,
promotion, source/runtime/template change to any historical experiment occurred.

First terminal:
843960219d8c2f0fd75e72c5ede7b1dbb3367833655e2b572cbe8fca5734fd89
rare_region_exploration, seed0, completed 2026-10-11T15:33:01.013171+00:00.
18 plays; authored1/random1/zero0/spatial-off0/action-off1/blind0 levels.
62.878874 CPU seconds, memory peak170807296 bytes, job_tree_empty=true.
Result SHA256 d7408548bb7dcbe5e5cbbf3bbc333f2187733f17adb3edb72b3d306060bbfa3a.

Second terminal:
5432a9b480fec2fc982bbc53490d74b2a2ecebb57d65566c4d9319f94570ec8c
boundary_exploration, seed0, completed 2026-10-11T15:34:10.448429+00:00.
18 plays; authored1/random0/zero0/spatial-off0/action-off1/blind0 levels.
65.510269 CPU seconds, memory peak111771648 bytes, job_tree_empty=true.
Result SHA256 824a6509cff0bdb86286153ed436f02f97b7b388ff77b22b1664fbf92a93bcb5.
The supervisor advanced to distinct third fingerprint a23e158fc99a5fc2...,
then consumed the remaining different hypotheses/seeds exactly once.

## Execution safety and acceptance evidence

Each ENTIRE job tree shares CPUQuota=100%, CPUAffinity=0, library threads=1,
MemoryMax=1073741824, MemorySwapMax=0, RuntimeMaxSec=600, TasksMax=64,
KillMode=control-group, SendSIGKILL=yes, OOMPolicy=kill.
Jobs additionally have PrivateNetwork, ProtectHome, ProtectSystem=strict and
InaccessiblePaths=/srv/cellforge-private. No credentials or network/security
configuration was changed.
Disk floor5GiB, per-run64MiB and total256MiB artifact caps, individual-file16MiB
limit, rotated1MiB event logs. Full study artifacts were about1.3MiB before bundle.
No raw unrestricted worker log grows indefinitely.

Global flock plus ONE fixed transient job unit prevents duplicate launch.
STATE intent is fsynced before service creation. Terminal receipts are atomic,
fsynced, and written only after complete-tree stop/emptiness verification.
Restart adopts the existing invocation or harvests its terminal outcome;
unowned/orphan units block new launch. Incomplete launch intents become interrupted
without replay. Failed jobs are not silently retried; 30/60/120-second backoff and
a three-failure circuit breaker apply. Global STOP and individual cancellation
override restart. Existing 20 protected STOP files are hash-checked every tick.

Acceptance tests:14/14 passed, including duplicate flock, live-job adoption,
orphan blocking, STOP persistence, per-job cancellation, receipt/state crash
recovery, backoff/circuit, disk floor, queue exhaustion, fingerprint duplicates,
remote-call rejection and interrupted launch handling.
Actual cgroup probes: two detached descendants shared0.99916 CPU seconds per wall
second. A10-second timeout killed the entire tree. Two600MiB descendants triggered
the1GiB aggregate OOM gate and both died; cgroup empty in both cases.
Actual duplicate supervisor invocation was refused.
An intentional main-process SIGKILL recovered job invocation
82a1cf0b343e489091306bcc9bf84064, and it completed with the SAME invocation and
exactly ONE launched event. New supervisor started15:33:55, before that job's
15:34:10 terminal receipt. It then advanced normally.
See GUARD_EVIDENCE.json and RESTART_COMPLETED_EVIDENCE.json.

## Mandatory test/gate interpretation — no label-based exemption

Read AGENTS.md rule8, notes/HANDOFF_TO_CODEX_2026-10-01_HEAD.md, the newer LAB,
ARC3_NEXT_EXPERIMENT.md and CODEX_TENSOR_EFFECT_RESULT_20261002.md before planning.

AGENTS.md rule3 says: "Before claiming anything, run" the two standard suites,
and "If the milestone tests fail, say so and work on the failing milestone next."
It forbids skipping/weakening/deleting tests. It does not command a global ban on
all bounded baseline measurement after any historical qualification failure.
The head handoff similarly requires running/pasting niche/focus tests.

The newer shared-pilot protocol explicitly directs an undersized measurement:
"The truthful undersized STANDARD.json deliberately leaves standard contract/
scientific requirements unmet; report the resulting red tests."
It says not to claim the full standard is met. This is positive permission for
a bounded development pilot, not an exemption invented after failures.
Our PROTOCOL.md, zero-call allowance and qualified=false scope were frozen BEFORE
the tests and first launch. No completed run was renamed to hide a failed gate.

ONE_STANDARD.md says anything not under the full loop is a baseline or archived.
It also says runtime_acceptance is scoped mechanism acceptance, while
"scientific_acceptance and scientific_headline_allowed remain false until the
full architecture/scientific gates pass."
These ARE mandatory gates for admission/full scientific claims. They remain
unsatisfied here. This new isolated study is baseline development evidence,
not completion of the project's primary evolutionary milestone.

Required niche/focus run initially:69 tests, one stale-snapshot failure.
The actual unmodified lane reducer refreshed the report from live server files,
preserving the old report as latest.pre-continuous-20261011.json.
All69 niche/focus tests then passed. No HARD_LANES, tests, floors or closed lane
contents were changed.

Combined niche/focus + standard run:
Ran94 tests in9.715s; FAILED(failures=3, errors=3).
Exact failures:
- tests.test_one_standard.ContractOfEveryWeightsRun.test_equal_charged_budget_every_arm_and_seed:
  AssertionError:0 !=1, old weights-arc3-shared-probe-20261002/STANDARD.json,
  arm=random seed=0 charged0.
- tests.test_one_standard.ContractOfEveryWeightsRun.test_seeds_predeclared_and_enough:
  AssertionError:1 not greater than or equal to3, same old shared probe.
- tests.test_one_standard.RepoGuards.test_no_new_gene_list_runs_after_cutoff:
  23 existing legacy gene-list/selection records in the migration snapshot;
  exact offender list is retained in required-tests-fresh.log.
Exact errors, all KeyError:'evaluator' at tests/standard_checks.py:74:
- tests.test_standard_semantic.RealRuns.test_every_semantic_check_passes_on_every_real_run
- tests.test_standard_semantic.RealRuns.test_recomputed_llm_beats_random_on_a_real_run
- tests.test_standard_semantic.RealRuns.test_router_exists_matters_and_was_llm_mutated_on_a_real_run

These arise in the preserved historical evidence, not a silently altered test.
They prevent a full standard/admission claim. The new baseline also deliberately
does not establish the full population/router/crossover architecture and cannot
be promoted merely because its process ran. Full architecture command was also
attempted; initial old-installed-package import failures and source-path retry
are both retained in architecture-required-tests.log / architecture-source-tests.log.
No fixture, receipt, source hash or old evidence was edited to manufacture green.

## Control and next work

Stop research and the durable service:
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes cellforge-1 "systemctl stop cellforge-continuous.service"
ExecStop writes a persistent STOP in the NEW deployment root and kills the job
tree. A subsequent boot/start observes that STOP and stays paused.
For one experiment, create cancel/<full fingerprint> inside this new root.
Never remove protected historical STOP markers or revive closed/cancelled runs.

Status:
ssh cellforge-1 "cat /srv/cellforge-inbox/continuous-spatial-20261011/STATE.json"
Review files: supervisor.py, worker.py, policy_v2.py, prepare.py,
test_supervisor.py, guard_tests.py, audit_results.py, PROTOCOL.md, PLAN.json,
cellforge-continuous.service, EVENTS.jsonl, COMPARISON_AUDIT.json and all receipts.
Next meaningful work must be separately recorded, target the missing useful
capability rather than repeat this matrix, and preserve current evidence and gates.

## Full architecture gate terminal result

With the snapshot's own src on PYTHONPATH, the full ONE_STANDARD architecture
command ran78 tests in7.227s: FAILED(failures=5, errors=3).
It retained the same three standard failures/three missing-evaluator errors,
plus:
- tests.test_action_verification.TestPopulationVerification.test_population_replay_reconstructs_selection_and_budgets
- tests.test_circuit_population.TestCircuitPopulation.test_invalid_and_equivalent_proposals_are_charged_without_lineage_growth

These two additional architecture failures remain visible in
architecture-source-tests.log. The initial import-failure log is also preserved.
No source, tests, historical receipts or floors were repaired/rehash-ed to claim
qualification. Full scientific/admission acceptance is still blocked.

