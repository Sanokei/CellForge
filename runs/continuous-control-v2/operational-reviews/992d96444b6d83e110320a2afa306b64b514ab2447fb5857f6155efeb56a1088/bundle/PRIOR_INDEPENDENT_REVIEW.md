**PASS for the supervisor stop-race repair. Research resume remains HOLD.**

This finding applies specifically to seal **`a70f343e9da351c19c93ba0839c1cc0418e6923c7844c2995c84fa7cd82213ca`**. All 236 manifest files and archive contents match, including after testing. Prior seal approval was not substituted.

- [Launch checks](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/stop-race-repair-review-20261011/supervisor.py:325) recheck STOP, halt and dispatch around reservation and RPC. Early stops leave conditions unattempted; cancellation after intent consumes exactly one attempt. A raced launch records its invocation before cancellation.
- [Stop verification](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/stop-race-repair-review-20261011/supervisor.py:192) shares the registry lock and requires active clearance, unit absence and an empty tree. [Graceful halt](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/stop-race-repair-review-20261011/supervisor.py:365) retries unknown reconciliation instead of claiming success. Existing STOP bytes are preserved.
- Independently reran **96 offline tests**, all passing, plus seven unchanged contract tests and hash/ledger/budget checks. Both Linux payloads match the repaired source. Their twelve recorded cases verify cancellation and crash recovery, including reconciliation of the original worker before invocation commit.
- Live inspection at **2026-10-12 17:01:24 UTC** confirmed all twelve probe units/cgroups and both fixture directories absent. All 239 protected hashes matched. Production remained **0 completed, 1 cancelled, 8 pending**, attempted **1/cap 9**, sequence 1, no active job, dispatch absent and both watchers disabled. The cancelled receipt binds the preserved configuration and 13 partial results; its fingerprint remains consumed.

Two **nonblocking observer defects** were independently reproduced:

- [The running label](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/stop-race-repair-review-20261011/observe_dispatch.py:29) can retain stale STOP status when STOP arrives during the resource read.
- [Duration validation](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/stop-race-repair-review-20261011/observe_dispatch.py:38) accepts NaN, preventing watch expiry. Reject non-finite durations and recheck markers before treating this tool as a strict verification gate.

The science plan and **1 CPU / 1 GiB / 600-second / zero-API** limits are unchanged; qualification remains red. Existing STOP still prohibits resume, and this repair implements no resume mechanism.

No deployment, research launch, dispatch change or STOP change was performed.