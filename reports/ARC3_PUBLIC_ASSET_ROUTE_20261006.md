# Genuine ARC-3 public-game acquisition: stopped at API-key prerequisite

The follow-up search did not establish an official credential-free downloadable
bundle for a genuine public game. No additional game was downloaded or run, and
no key was issued, read, logged or transmitted. The installed runtime and prior
SDK-fixture evidence remain unchanged.

## Smallest documented route

The official docs identify `ls20`, `ft09` and `vc33` as example public games and
state that three games are available anonymously. Anonymous access still uses
an API key: the toolkit obtains one automatically when no registered key is
provided. This requires no signed-in account according to the documented flow,
but **does require key issuance and transmission**, which the current delegation
explicitly requires us to stop before.

Sources: [available games](https://docs.arcprize.org/available-games),
[toolkit quickstart](https://docs.arcprize.org/toolkit/overview),
[local versus online](https://docs.arcprize.org/local-vs-online).

Installed `arc-agi==0.9.9` confirms the concrete sequence in
`.venv-arc3/Lib/site-packages/arc_agi/base.py`:

1. `GET https://three.arcprize.org/api/games/anonkey` obtains an anonymous key.
2. Metadata requests use `X-Api-Key` at `/api/games` or `/api/games/ls20`.
3. After resolving the exact version, source retrieval uses `X-Api-Key` at
   `/api/games/ls20-<version>/source`.
4. Once acquired, metadata/source can be used by `OperationMode.OFFLINE` in a
   separate network-denied process. Offline execution does not acquire assets.

These endpoints were read from source, **not invoked**. Source SHA-256:
`42973618f223096d6269422352980e85bdf6b2b51dcb42e7390f96164b0f994f`.
The SDK's anonymous-key helper logs the key at INFO; any future authorized
acquisition must keep that value out of logs and receipts.

Required next decision: explicitly authorize anonymous ARC-key issuance and
transmission to the official service for one public demonstration game's
metadata/source, or provide an official local game bundle with provenance and
license. A registered key, login, paid/model service and broader campaign are
not needed for the documented anonymous route. Availability of a particular
version is unconfirmed until authorized metadata retrieval.

## Other routes checked

- Installed wheel manifests for `arc-agi 0.9.9` and `arcengine 0.9.3` contain no
  `environment_files`, game metadata, or sources for ls20/ft09/vc33/ar25/lp85/sc25.
  Package installation therefore does not remove the asset prerequisite.
- The previously saved official ARC-AGI Git tree supplies only SDK fixtures;
  ARC-AGI-3-Agents and ARC-AGI-3-Kaggle-Starter trees do not supply the game assets.
  Existing receipts: `arc3-official-tree-20261006.json`,
  `arc3-official-agents-tree-20261006.json`,
  `arc3-official-starter-tree-20261006.json`.
- The official Kaggle kit's documented competition workflow requires a Kaggle
  account, rules acceptance and token. Its **local** runner actually uses
  `Arcade(operation_mode=OperationMode.NORMAL)` and thus the same ARC key-based
  downloader; it is not an alternate game bundle and does not require Kaggle
  credentials merely for that local route. See the
  [prerequisites](https://docs.arcprize.org/arc-prize-2026) and
  [pinned local runner](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter/blob/eeb1535404f321d280a8f9194bbc1d7aca5f05fc/scripts/play_local.py).
- The [official benchmarking repository](https://github.com/arcprize/arc-agi-3-benchmarking)
  also instructs obtaining an ARC API key. It was not installed or executed.
- The [engine repository](https://github.com/arcprize/ARCEngine) documents the
  engine under MIT, not a benchmark-game distribution. Additional engine-tree
  and SDK-release listings were unavailable through the web reader. No claim
  is made that every possible official distribution has been exhaustively ruled out.

## Game identity, license and score semantics

Candidate public game ID: **ls20**. Acquired version: **none**. A version string
in a starter-code example is not authoritative current metadata. Game-specific
source/license: **not acquired or verified**. The SDK/engine MIT license must
not be assumed to license separately served game assets. No consent was accepted.

No genuine-game score or action trace exists from this follow-up. The prior
`bt11-fd9df0622a1a` result remains strictly an SDK-fixture diagnostic.

For a future authorized local check, preserve the exact version, source hashes,
license/terms evidence, all frames/actions, raw levels completed, action count
and stop reason. Keep the requested ceiling at 150 actions and 30 seconds;
count the initialization reset separately and stop at terminal state. No
authored Python policy is needed for a frozen data-only action queue.

Raw levels completed are not the official benchmark score. Official Relative
Human Action Efficiency squares the ratio of baseline actions to agent actions
for completed levels, caps the per-level score at 1.15, weights levels by their
1-based index, applies a completion-dependent game cap, then averages game
scores. One bounded public-game diagnostic would not be a benchmark-wide score,
held-out result, learned-policy result or leaderboard submission.
Source: [official scoring methodology](https://docs.arcprize.org/methodology).

No gameplay, package installation, policy execution, network-guard change,
active-source edit, full-suite rerun, server access or model call occurred in
this follow-up. The Windows sandbox blocker for arbitrary authored policy
execution remains separate from trusted-engine/data-only readiness.
