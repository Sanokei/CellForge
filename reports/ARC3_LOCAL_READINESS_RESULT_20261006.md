# ARC-3 local readiness result, 2026-10-06

The repository-local runtime is installed and can execute the publisher's public
SDK test fixture offline. Actual ARC-3 benchmark-game readiness remains blocked
by missing authorized game assets. No approval request is pending.

## Installed environment

`.venv-arc3/Scripts/python.exe` uses Python 3.12.14 with `arc-agi==0.9.9`,
`arcengine==0.9.3`, NumPy 2.5.3 and Pydantic 2.13.5. Installation used official
public PyPI wheels in this isolated environment. `pip --isolated check` reports:

```text
No broken requirements found.
```

Exact resolved versions: [requirements](arc3-local-requirements-20261006.txt).
Package URLs/hashes and resolver metadata:
[installation report](arc3-local-pip-install-20261006.json),
[installation log](arc3-local-pip-install-20261006.log).
The initial venv bootstrap encountered a temporary-file permission denial;
normal tool approval allowed repository-local ensurepip and installation.
No global runtime, credentials, ACLs or security configuration were changed.
The venv has its own ignore file and is not a source deliverable.

## Public asset and bounded action check

Official repository `arcprize/ARC-AGI`, pinned commit
`f12822c4d550121c35a275008d964afbbed47d2f`, supplies a small SDK fixture:
`test_environment_files/game1/bt11.py` and its metadata. Those files and the MIT
license were downloaded directly from unauthenticated raw GitHub URLs. All
three Git blob hashes match the official tree; source SHA-256 hashes and exact
URLs are in [MANIFEST.json](../data/arc3-official-sdk-fixture-20261006/MANIFEST.json).
The three downloaded files total 3,895 bytes. This is **a public SDK test
fixture, not an ARC-3 benchmark game, evaluation asset or benchmark score**.

The [frozen protocol](arc3-local-readiness-20261006/protocol.json) allowed one
automatic reset, at most 16 additional actions, a 30-second external process
limit, seed 0, and immediate stopping on WIN/GAME_OVER. The action queue is JSON
data: alternating 3 and 4. There was no model-authored policy, learning,
adaptation, retry, reset after loss or campaign. The trusted publisher engine
and fixture were the only game implementation executed.

Observed result: reset plus actions **[3, 4, 3, 4]**, five observations, final
**GAME_OVER**, **zero levels**, elapsed **4.281 seconds**. Every observation has
its full 64x64 frame. Evidence:

- [Full action/observation trace](arc3-local-readiness-20261006/attempt01/trace.jsonl)
  (44,264 bytes; SHA-256
  `85745dc36c66795afc28eaea8f6cb54e9ac7cbbb10023f69c82471a129d3aad9`).
- [Original process receipt](arc3-local-readiness-20261006/attempt01/process.json),
  [original result](arc3-local-readiness-20261006/attempt01/result.json),
  [stdout](arc3-local-readiness-20261006/attempt01/stdout.log),
  [stderr](arc3-local-readiness-20261006/attempt01/stderr.log).
- [Independent saved-trace validation](arc3-local-readiness-20261006/verification.json)
  checked hash chain, source hashes, action availability, frozen queue, terminal
  stopping, action limits, full frames and echoed action inputs. It did not
  replay the game or change the failed process result.

## Strict diagnostic failure, preserved

The original check exits **1** because its final zero-audit-event assertion
failed. During import, urllib3 attempted to construct an IPv6 socket to test
local loopback support; the child audit hook denied construction. The engine
still ran the complete four-action trace. No guard was relaxed, no socket was
allowed, and the game was not rerun to obtain a green result.

A separate [import-only probe](arc3-local-readiness-20261006/import_probe.py)
reproduced exactly one blocked `socket.__new__` event, with the stack at
`urllib3/util/connection.py:126`, `_has_ipv6`. Both official modules imported
successfully; zero game instances/actions were created. See
[probe output](arc3-local-readiness-20261006/import-probe.stdout.log) and
[process receipt](arc3-local-readiness-20261006/import-probe-process.json)
(exit 0, 1.641 seconds). The first diagnostic remains failed; this narrower
check explains the failure rather than replacing it.

Game execution used the default restricted execution sandbox, explicit
`OperationMode.OFFLINE`, an explicit local asset directory, a clean child
environment without credentials, disabled dotenv loading, and a CPython audit
hook denying every socket event and subprocess launch. The only observed socket
event was blocked before construction. This hook is a safeguard for trusted
code, **not a security sandbox for arbitrary Python or native libraries**.

## Remaining boundaries and blockers

1. Actual public benchmark games such as ar25/lp85/sc25 are absent locally.
   The official SDK's normal downloader requests an API key (including an
   anonymous-key route); neither was used. Inspected official ARC-AGI,
   ARC-AGI-3-Agents and ARC-AGI-3-Kaggle-Starter repositories did not supply those
   source assets. This is a limit of the inspected routes, not proof that no
   official credential-free distribution can exist. A documented official
   direct distribution or explicitly authorized supplied files is needed.
2. A bounded-memory sandbox for arbitrary authored policy execution on Windows
   is still unestablished. Installing the engine does not resolve it. Keep any
   future authorized data-only baseline separate from policy-tensor execution.
3. The strict readiness harness has the preserved blocked-import-probe failure
   above. Engine imports and traced offline actions worked; do not report the
   whole harness as green.
4. No paid/model calls, server/Hetzner access, credentials, private evaluation,
   broad campaign, dispatcher integration, active-run source edit, commit or
   deployment occurred. Historical full-standard failures remain unchanged;
   no qualification claim or broad-suite rerun was made for this setup.

Official references: [toolkit overview](https://docs.arcprize.org/toolkit/overview),
[offline operation](https://docs.arcprize.org/local-vs-online),
[published package](https://pypi.org/project/arc-agi/0.9.9/), and
[pinned public fixture](https://github.com/arcprize/ARC-AGI/blob/f12822c4d550121c35a275008d964afbbed47d2f/test_environment_files/game1/bt11.py).
