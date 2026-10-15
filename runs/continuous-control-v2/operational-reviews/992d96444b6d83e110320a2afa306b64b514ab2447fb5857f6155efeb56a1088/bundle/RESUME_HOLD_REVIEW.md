**HOLD archive-and-rearm and research resume for exact seal `9d658de47abd45555bb26cc2504f14014ff8526f041a1d445cb76b298648cd48`.**

I independently reproduced three blocking ownership races in disposable Linux fixtures:

- **Foreign dispatch marker overwritten.** After the check at [line 163](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/resume-control-review-20261012/resume_operation.py:163), a foreign marker appeared at the release boundary. The atomic write replaced it, and activation returned `DISPATCH_AUTHORIZED_WATCHER_STILL_STOPPED`.
- **Foreign replacement deleted during rollback.** The [read-then-unlink operation](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/resume-control-review-20261012/resume_operation.py:169) deleted a foreign replacement introduced after the ownership read. The new STOP survived, but the foreign marker did not.
- **Archive collision overwritten.** A foreign `ACKNOWLEDGED_STOP` appeared immediately before [POSIX rename](/C:/Users/wkeif/Documents/Codex/2026-10-10/task-8/resume-control-review-20261012/resume_operation.py:137). Rename overwrote it, and stage returned `REARMED_STOPPED_DISPATCH_DISABLED`.

These violate the transaction’s preservation guarantees. Use atomic publication that refuses an existing foreign destination, and rollback that cannot delete a replacement between ownership verification and removal. Add regressions at those mutation boundaries, then reseal.

The other checks pass:

- Observer fixes pass, including marker changes and non-finite duration rejection.
- Independently reran all **16 Linux tests**, including real SIGKILL recovery and lock exclusion; all passed. Local tests passed with two POSIX skips. Seven unchanged contract tests and hash/ledger/budget checks passed.
- All 253 manifest files and archive contents match. The science plan is unchanged, and the request matches the live paused registry, consumed cancellation, eight pending fingerprints and cap **9 including the cancelled attempt**.
- At **18:16:37 UTC**, all **245 protected production hashes** and the operative STOP’s identity remained unchanged. Production remained held, with no active job or dispatch.

**No production transition is approved under this seal.** Read-only preparation and observer use pass review; archive/rearm, activation and watcher start remain held. Limits remain **1 CPU, 1 GiB, 600 seconds/job, zero API calls**, with scientific qualification false.

No deployment or research resume was performed.