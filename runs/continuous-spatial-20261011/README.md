# CellForge continuous queue — review package

Installed root: /srv/cellforge-inbox/continuous-spatial-20261011
Service: cellforge-continuous.service (enabled, active).
Final queue: paused_exhausted;9/9 complete, zero pending/active.
This is a durable sequential runner, not an infinite repeated-work loop.
A separately recorded next hypothesis/protocol is required.

Read CODEX_TO_HEAD_2026-10-11.md for exact gate interpretation, results and controls.
PLAN.json freezes hypotheses, full scientific fingerprints, code/data hashes and
protected STOPs. COMPARISON_AUDIT.json reports ALL complete/incomplete comparisons.
Each run has immutable CONFIG/START/ARMS/RESULTS/SUMMARY/TERMINAL artifacts.

Operational checks:14 supervisor tests passed. Actual CPU, timeout and aggregate
OOM tree probes passed. Duplicate launch was refused. Main crash recovered the
same job invocation exactly once. All9 completed receipts certify empty trees.
Limits:1 CPU,1GiB,600s, no swap/network/private validation access;5GiB disk floor,
64MiB/run and256MiB total caps; failure backoff and circuit breaker.

Scientific results: all authored policies solved the same one lp85 development
level; ar25/sc25 zero. Every action-off ablation tied authored performance.
Matched random solved one in2/9 runs, all blind/spatial-off/zero scored zero.
No action-weight benefit, new-game transfer, population learning, admission,
or general-capability claim. This is baseline development evidence.
Zero remote model/API calls or paid API charges.
69 niche/focus tests passed after a real snapshot refresh;94 combined tests
retained3 failures/3 errors in historical standard qualification evidence.
Full unmodified logs are included.

Stop:
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes cellforge-1 "systemctl stop cellforge-continuous.service"
This writes a persistent STOP and kills the complete research tree.
Do not remove historical STOPs or replay consumed fingerprints.

