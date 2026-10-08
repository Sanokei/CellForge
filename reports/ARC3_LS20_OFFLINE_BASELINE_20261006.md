# Public ls20 offline baseline, 2026-10-06

**The genuine public game runs locally.** One bounded, frozen action-data
baseline on **ls20-9607627b** completed in **3.984 seconds**: **129 moves plus
one initialization reset**, final **GAME_OVER**, **0/7 levels**, local SDK
score **0.0**. The process exited 0. This is an operational baseline, not a
learned policy, model-authored tensor result, held-out test or benchmark-wide
score. No second game run or model call occurred.

## Provenance and license

The user personally ran the prepared downloader and obtained this one public
game from the official service. This session verified both original files
against that saved receipt; it did not read `.env`, a key, or make any network
request. Receipt provenance is user-attested official HTTPS acquisition plus
matching local file hashes, not an independent publisher signature.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| metadata.json | 125 | `b511ec4f453d3d0435773a2ae4eeb22c912badc105ff686eecc0b2cc70169fe4` |
| ls20.py | 105,874 | `298c810da2850d557c95d92a2cbd846df29a45d7134e20888617bedf5dafcd92` |

The source itself carries **MIT License**, copyright 2026 ARC Prize Foundation.
Its complete leading notice is preserved in
[GAME_LICENSE.txt](arc3-ls20-offline-baseline-20261006/GAME_LICENSE.txt); the
original game file is unchanged. Inspection exposed only metadata, leading
license comments and import-module names (`arcengine`, `math`, `numpy`,
`typing`), not implementation logic for solving the game.
See [asset inspection](arc3-ls20-offline-baseline-20261006/asset-inspection.json)
and [user download receipt](../data/arc3-public-ls20/DOWNLOAD.json).

## Frozen run and execution boundary

[Protocol](arc3-ls20-offline-baseline-20261006/protocol.json): engine seed 0;
149 pre-generated, uniform choices from action IDs 1–4 using queue seed
20261006; one automatic initialization reset; at most 150 total engine calls;
30-second external child limit. The action queue was saved before gameplay.
It allowed no feedback selection, substitute action, retry or post-loss reset.
The game reached GAME_OVER before exhausting the queue.

The [host harness](arc3-ls20-offline-baseline-20261006/check.py) ran installed
official `arc-agi 0.9.9` / `arcengine 0.9.3` with explicit OFFLINE mode and the
explicit local asset directory. It consumed action JSON data; no authored
policy source was loaded or executed. The official publisher game implementation
is trusted engine content, not a candidate policy.

The child inherited no credential environment. Dotenv loading was disabled
before toolkit import. The default restricted execution sandbox and an audit
hook denying socket events and subprocesses stayed active throughout. The
known urllib3 import-time IPv6 socket construction was blocked and recorded
separately; **zero denied network/process events occurred after imports**.
No import probe was allowed through or security setting changed. This is a
trusted-code safeguard, not a bounded-memory sandbox for arbitrary Python.

The SDK created and closed local bookkeeping in OFFLINE mode. Its saved
`api_key` is null. No online scorecard, submission or API action occurred.

## Evidence and validation

- [Full action/observation trace](arc3-ls20-offline-baseline-20261006/attempt01/trace.jsonl):
  **130 observations, 140 complete 64x64 frame layers**, 1,274,931 bytes.
  SHA-256: `46544293a36ffa80022d9c41d84cf3c3a524408df2705447f92c7033da17b768`.
- [Process receipt](arc3-ls20-offline-baseline-20261006/attempt01/process.json),
  [result](arc3-ls20-offline-baseline-20261006/attempt01/result.json),
  [stdout](arc3-ls20-offline-baseline-20261006/attempt01/stdout.log),
  [stderr](arc3-ls20-offline-baseline-20261006/attempt01/stderr.log).
- [Local scorecard](arc3-ls20-offline-baseline-20261006/attempt01/LOCAL_SCORECARD.json):
  129 actions, zero scored resets, zero levels, score 0.0. The SDK excludes
  the initialization reset from these counters; our resource accounting
  includes it, giving 130 engine calls.
- [Independent saved-trace validation](arc3-ls20-offline-baseline-20261006/verification.json):
  checked the hash chain, every complete frame, echoed actions, prior legal
  action set, exact frozen queue, terminal stopping, limits, score counters,
  initial-observation identity and unchanged source/protocol/runner hashes.
  This was a data check, not an engine replay or new game run.

The trace contains 131 distinct frame layers. No consecutive complete frame
lists are identical; this is only a pixel-level statistic and does not prove
that each move made useful progress. All levels remain unsolved.

## Initial observation and subsequent action-data interface

Initial state: **NOT_FINISHED**, levels completed **0**, win levels **7**;
available actions **[1, 2, 3, 4]**. Each is a simple action accepting empty data.
Useful policy-facing inputs are limited to observed frames, state, legal actions
and action history:

- [Initial numeric observation](arc3-ls20-offline-baseline-20261006/attempt01/INITIAL_OBSERVATION.json)
- [Initial frame PNG](arc3-ls20-offline-baseline-20261006/attempt01/INITIAL_FRAME.png)
  — exact observed pixels enlarged with nearest-neighbor using the installed
  official renderer's palette; no game-source-derived annotation.
- [Official action schemas](arc3-ls20-offline-baseline-20261006/attempt01/ACTION_SCHEMA.json)

For a later bounded LLM action-data trial, a reply can be constrained to
`{"id": <one of the latest available action IDs>, "data": {}}`.
That interface is sufficient here; no code, tool names, file paths or credentials
belong in the policy output. Action meanings should be learned from observations
and permitted action history, not the implementation. The initial observation
is a saved record, not a currently live session; a new trial would need its own
explicit finite budget and fresh initialization. No LLM policy run was started.

The game's metadata supplies seven human action baselines:
`[22, 123, 73, 84, 96, 192, 186]`. The local SDK's RHAE computation yields 0.0
because no level completed. This score pertains only to this bounded public
game run; it is not a platform leaderboard submission or evidence of novelty.

No active driver, maker, failure journal, standard test or floor was changed.
Existing broad qualification failures remain unchanged; no full suites were
rerun for this runtime check. Arbitrary authored-policy sandboxing remains a
separate unresolved capability.
