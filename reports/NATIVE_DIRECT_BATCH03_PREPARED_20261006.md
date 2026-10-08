# Newly selected official training batch, 2026-10-06

Four direct-prediction packets are prepared, with zero author calls and zero query
scores. The metadata retry passed automatic approval review after the parent
provided the owner's ARC continuation/training-storm statements. The subsequent
four-task/two-license download also passed review. Earlier blocked receipts are
preserved and linked by `RESUMPTION.json`; no workaround was used.

Run: `runs/weights-native-arc-direct-batch03-20261006/`.
Data: `data/arc-training-batch03-20261006/`.

## Selection and provenance

Seed `202610060303` was frozen before official metadata or new task contents were
read. Selection ranks eligible IDs by SHA256 of `seed:dataset:task_id`, taking
the first two per dataset. It was independently reproduced before packet delivery.

| Dataset | Official pinned commit | Training IDs | Known-record exclusions | Eligible after exclusions | Selected |
| --- | --- | ---: | ---: | ---: | --- |
| ARC-AGI | 399030444e0ab0cc8b4e199870fb20b863846f34 | 400 | 238 | 162 | 46442a0e, c8cbb738 |
| ARC-AGI-2 | f3283f727488ad98fe575ea6a5ac981e4a188e49 | 1000 | 644 | 203 | 984d8a3e, 0c786b71 |

The two training manifests share 391 IDs. All shared IDs were additionally
excluded from the ARC-AGI-2 candidate pool, giving four unique task IDs. Official
training Git-tree metadata was used; no evaluation tree or task contents were
read for selection. Each selected task download matched its pinned Git blob.
Both repositories' LICENSE files were downloaded and identified as Apache-2.0;
their exact hashes are retained in data `SOURCE.json`.

The local exposure inventory scanned 1,613 text artifacts totaling 171,444,454
bytes, across runs/archives, reports, notes, docs, scripts, tests and data. Its
9,874 candidate eight-hex tokens were conservatively intersected with official
training IDs. All eight earlier local training IDs and both reserved report IDs
were covered. No file changed during the scan. The 308 nontext paths and remote
history were not fully inventoried; model pretraining exposure is also unknown.
Therefore call these tasks **newly selected and absent from scanned local
records**, not never seen or proven disjoint from all historical exposure.

Sources are the official pinned repositories:
[ARC-AGI training](https://github.com/fchollet/ARC-AGI/tree/399030444e0ab0cc8b4e199870fb20b863846f34/data/training)
and [ARC-AGI-2 training](https://github.com/arcprize/ARC-AGI-2/tree/f3283f727488ad98fe575ea6a5ac981e4a188e49/data/training).

## Bounded author/scoring contract

Each `AUTHOR_<id>.json` contains demonstrations and query INPUTS only. All four
tasks have one query. Prepend `COMMON_PROMPT.txt`; one isolated tools-free native
Codex author per task, four calls maximum. No query-label feedback, selection
among responses, retries or automatic continuation. Record exact raw text,
backend/reasoning/thread/turn when available, and unknown credit usage honestly.

Freeze responses before host-only exact comparison and independent replay, using
the current data-only scorer with the new task/data bindings. Identity remains a
weak baseline; no benchmark score, held-out transfer or literal-weight claim is
authorized by these results. No authored program execution or new framework is
needed. Existing red/incomplete standard status carries forward without unrelated
full-suite reruns. No protected floor/test, credential, remote job or ARC-3 runtime
was changed.

Seed receipt: `82c8311a41a9cf622b18a2bf9918accef454d696c640462cd982216201ac6c1e`.

Selection: `921cd69afab5e81223c312a121bd55ac30026d568685aeb60d7be50b10e490b9`.

Protocol: `45283aa5504dc47797a5dcb67eb4565c8b3f18313d00022e0ca66b1a11793a8b`.
