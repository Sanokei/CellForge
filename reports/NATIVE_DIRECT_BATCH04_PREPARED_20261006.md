# Direct ARC batch04 prepared, 2026-10-06

Four label-free direct-prediction packets are ready. Zero authors were launched
and zero query outcomes were inspected or scored during preparation. After this
batch, stop for the requested capability/weights research checkpoint; no batch05
or larger campaign is authorized or prepared automatically.

Run: `runs/weights-native-arc-direct-batch04-20261006/`.
Data: `data/arc-training-batch04-20261006/`.

Seed `202610060404` was frozen before new task contents. Reused the official
training manifests, dataset pins and base exposure inventory from batch03,
augmented with current selected-task metadata and downloaded task filenames.
The four batch03 IDs were added to exclusions. The final eligible pools were
160 ARC-AGI IDs and 201 ARC-AGI-2-only IDs. The predeclared SHA256 ordering selected:

| Dataset | Selected IDs | Commit |
| --- | --- | --- |
| ARC-AGI | ed36ccf7, ef135b50 | 399030444e0ab0cc8b4e199870fb20b863846f34 |
| ARC-AGI-2 | e7639916, 712bf12e | f3283f727488ad98fe575ea6a5ac981e4a188e49 |

All selections are absent from the prior inventory and all previously selected
task IDs. All 391 shared version IDs remain excluded from the ARC-AGI-2 pool.
Selection was independently reproduced from the frozen seed before delivery.
Historical remote/nontext and model pretraining exposure are unverified: say
**newly selected**, not proven never seen or held-out.

Only these four official pinned public-training JSONs were downloaded. Their Git
blob hashes matched the manifests. The two existing pinned Apache-2.0 LICENSE
files were verified and reused, with source/hash references preserved. No new
metadata download, evaluation/private access, credentials or server work occurred.

Each `AUTHOR_<id>.json` contains demonstrations and one query INPUT. Reuse
`COMMON_PROMPT.txt`: one isolated tools-free native Codex author per task, four
calls maximum, strict data-only digit-row responses. Freeze raw text/provenance
before host-only exact scoring and independent replay. No query feedback or
retries. Model identity and credit usage must remain unknown unless supplied.

The previous scorer was preserved for the same minimal task/data rebinding used
in batch03. No framework, new template or unrelated broad test suite was added.
The full standard remains red/incomplete; no weight-learning or benchmark-score
claim follows from this direct baseline.

Seed receipt: `1a8ab8a52303ac3e1bf04681d3df11e195efc7040cf09bac25297647ebcf8c48`.

Selection: `d9a4505114b626494df9e6cd1e6a276c82d5d19b179422c1e62db091768501b8`.

Protocol: `bdeb759ad176ffec86802c5f349aee38fc931323b192b848f21ef32b791e5c6d`.
