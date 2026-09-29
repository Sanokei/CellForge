# Evidence-backed visual memory — CellForge OCR v3

**Status:** Implementation specification, not a working memory service or evidence of OCR improvement. Revision 3.0, September 29, 2026. Read together with `AGENT_HANDOFF.md`.

## 1. Purpose and boundaries

Give the construction-time orchestrator and maker agents a persistent, queryable research memory. Preserve the original visual evidence and the exact machine-readable experiment records so an agent can inspect what was tried, under which conditions, what happened, and what remains uncertain. Do not rely on a single repeatedly rewritten conversation summary.

This memory supports the people and LLM agents constructing the neural model. It is **not** an embedding table, hidden state, or retrieval dependency inside the primary exported OCR network. The primary inference process still receives pixels and emits tokens using its constructed numerical graph. No development image archive, label lookup, pretrained visual retriever, or author LLM participates in that inference.

Keep three related but separate structures:

* The **neural graph** contains executable cells, recursive bundles, parameters, state, and routing.
* The **evidence graph** links experiments, patches, model/component versions, artifacts, observations, claims, and decisions.
* The **retrieval index** is a replaceable search view over permission-appropriate evidence. Losing or changing it must not destroy source records.

The orchestrator can reorganize memory views, summaries, clusters, and retrieval policies. Those operations do not alter historical measurements or force the memory graph to mirror the neural hierarchy. Hypotheses and latent component functions need not have human-defined skill names.

## 2. What is borrowed from the literature

**VisRAG** retrieves document-page images using a vision-language embedding model and supplies retrieved images to a vision-language generator, avoiding an OCR-first document representation. Its retriever is trained. Adapt the principle of retaining original visual sources, not a claim that every record should become an image. [R3]

**VisRAG 2.0 / EVisRAG** records per-image evidence before synthesizing an answer. Its main method includes supervised and reinforcement training; Appendix A.4 also examines evidence-guided prompting without additional training. Adapt a concise observation/evidence/decision interface. Do not describe a prompted implementation as a reproduction of the trained method. [R4]

**VimRAG** uses a multimodal memory graph and allocates visual detail to selected evidence in iterative retrieval. Borrow graph-linked history and budgeted visual inspection as design ideas. Its full trained system is not required here. [R5]

None of these papers establishes that this memory design improves direct numerical neural construction. That is a CellForge hypothesis to test. External source statements and agent interpretations must stay distinguishable from local experiment measurements.

## 3. Evidence representation

Use an append-only event/receipt store and a content-addressed artifact store. Agents submit annotations; a separately controlled recorder attaches verified execution metadata and hashes. A caller-written field claiming `trusted=true` is not authentication. Content hashes establish identity/integrity, not truth or authorization.

Store native representations. Keep tensors as safe numerical arrays, parameters/graphs/patches as structured records, metrics as exact structured values with denominators, source documents as their originals where permitted, and images as original pixels. Optional renders, charts, or overlays are derived views. Never require reading a screenshot to recover a number already available in JSON.

For a development experiment, capture:

| Field group | Required information |
|---|---|
| Identity and origin | Stable event ID, run/branch/task IDs, producer identity, event type, timestamp, source receipt, access namespace, fixture/live classification. |
| Comparison context | Baseline/candidate model hashes, execution backend, harness, task/tokenization, dataset manifest, seeds, evaluation recipe, selection and memory policy versions. |
| Neural scope | Component definition hashes, invocation paths, affected dependency closure, patch ID, clone/share status, and lineage mapping for organizational refactors. |
| Visual evidence | Original development image references; optional crops linked to source hashes, dimensions and coordinate transforms; derived visualizations with rendering recipes. |
| Numerical evidence | Predictions, reference source strings where author access is permitted, per-example errors, exact aggregate counts, failures/timeouts, route traces, activation arrays or summaries linked to arrays. |
| Interpretation | Concise observations with source locators, hypotheses, counterevidence, scope of claims, uncertainty, and the next experiment or decision. |
| Outcome and cost | Proposed/evaluated/accepted/rejected/rolled-back status, actual results and retries, API/image-token/embedding usage, bytes, latency, and relevant construction costs. |

Do not store private model chain-of-thought. Store actions, observations, concise decision summaries, and evidence references. An explanation is not a measurement.

The included `evidence_packet.schema.json` is a starter contract for a single evidence packet, not a complete distributed event store. `examples/evidence_packet.json` is explicitly a synthetic structural fixture. Its numbers are invented test data, not observed OCR results.

## 4. Observations, claims, and negative results

Require separate fields for **what the source contains**, **what the agent infers**, and **what a controlled comparison establishes**. An agent reading a route plot may hypothesize that routing causes an error. The plot alone does not establish the cause; request a targeted intervention or comparison.

Support source-linked claim states such as proposed, measured-on-development, reproduced-under-specified-conditions, contradicted, superseded, and inconclusive. Avoid an unqualified global `verified` status. Multiple agents repeating one receipt do not constitute independent replications. Confidence values must declare their origin and are not calibrated probabilities unless tested as such.

Record failed edits and null results as well as successes. A failure should retain the task scope, versions, attempted change, budget, and outcome. It is evidence about that attempt, not proof that the entire approach cannot work. Conversely, a success on one font or one model version is not automatically transferable.

When new evidence conflicts with an old claim, append a linked correction/counterexample. Retrieve both when relevant. Do not erase the earlier event or let frequently cited summaries outrank their own contradictory source records without visibility.

## 5. Retrieval pipeline

Implement a shared query service with per-worker permissions and small task-local working sets. Apply authorization and split restrictions **before** indexing, retrieval, graph expansion, reranking, thumbnails, or cache lookup can expose disallowed material. Use physically separate final-evaluation storage, credentials and processes; prompt-only filtering is not a boundary.

A proposed retrieval sequence:

1. Form a query from the current task, failure observations, relevant component versions and constraints. Exact development case or patch IDs can bypass semantic search.
2. Search exact metadata/IDs and text records. When configured, also search image/page embeddings from a frozen, construction-only visual retriever.
3. Union candidates, expand relevant dependency/lineage/counterevidence links, deduplicate by source and experiment, and rerank under a context budget. Return provenance and scope, not just snippets.
4. Inspect selected originals or source-linked crops with a vision-capable author, asking for brief per-source observations and precise evidence references. Text-only authors may receive qualified summaries but must not claim direct visual inspection.
5. Assemble a compact evidence packet containing observations, applicable measurements, contradictions, uncertainty and a proposed action. Permit an explicit insufficient-evidence result.
6. Submit a model or harness proposal separately. The retrieval service never changes weights or executes code recovered from a document.

The visual adapter should support the actual documented capabilities of the chosen model. Original VisRAG establishes text-query/page-image retrieval; do not assume an image-to-image query interface is available without implementing and testing it. A maker can formulate a text query from a failing image; direct image queries are an optional verified adapter capability.

Start with local metadata and text search plus exact artifact fetch. Add a genuine frozen visual-retrieval adapter for the live visual-memory arm. A deterministic replay or filename match tests interfaces only and must never be labeled VisRAG-quality retrieval. Offline scaffolding need not download models or consume paid APIs.

Expose provider/model/checkpoint/processor revision, preprocessing, embedding shape, distance/similarity rule, retrieval configuration and index snapshot. Pin and review third-party code and model licenses; no unapproved dynamic remote code execution. Changing the embedding model or preprocessing produces a new index namespace or a full reindex. Do not compare incompatible embedding vectors or treat stored image embeddings as interchangeable with an author's internal image-token states.

## 6. Context and storage budgets

Original images and native arrays remain authoritative. Use previews and cheap metadata to shortlist material, then fetch selected sources at suitable detail. Crops must preserve source references and enough surrounding context; do not silently alter content or present an overlay as an original.

Provider image inputs may be resized or re-encoded on each call. Measure actual or explicitly estimated image-token and monetary cost for the chosen provider; changing image resolution is not guaranteed to reduce billed tokens linearly. Record estimates separately from provider-reported usage.

Active context can be compressed without rewriting durable evidence. Store source-linked summaries, not summary-of-summary chains with no recoverable originals. Under storage limits, an operator policy may archive or garbage-collect permitted data; create explicit tombstones and invalidate claims whose evidence is no longer available. Never claim retained evidence that cannot be fetched. Do not embed credentials, inaccessible final-case identifiers, or private raw reasoning in summaries or caches.

## 7. Dynamic architecture and memory lineage

A bundle's display name is not a permanent capability identifier. Refer to a component version by hash and to its invocation/dependencies explicitly. After a pure wrap/unwrap, record old/new mappings and the equivalence-test receipt. That permits retrieval across an organizational change without asserting that untested computational changes preserve behavior.

A shared component patch changes every dependent invocation. Flag affected current applicability, not the historical truth of earlier results. Preserve the original result at its original version and ask for revalidation where it matters. Alternative branches may keep conflicting results without one overwriting another.

The memory graph can have similarity and contradiction links without a tree-shaped hierarchy. Derivation/provenance links must have an auditable, non-circular source chain. Do not promote a hypothesis by citing its own derivative summary. Grouping records into a new memory cluster is organizational work, not fresh experimental evidence.

## 8. Evolvable memory harness

Allow harness-maker tasks to improve retrieval queries, reranking, chunk/record grouping, context selection, image preview/crop policy, deduplication, source-linked summaries, failure taxonomies, and proposed diagnostics. Memory-maker is an optional task specialization, not another mandatory rank.

Version these changes as part of the development harness and separately record `memory_policy_sha256` and `memory_snapshot_sha256` for each decision. Preserve the exact retrieved IDs and delivered representations so the decision context can be replayed. If a retrieval policy uses a model, record that model and its cost.

The protected part is small: authentic receipts, access controls, provenance, final-data isolation, resource limits and historical comparability. Agents cannot disable those by changing a memory policy. Legitimate corrections append or externally supersede records; they do not silently rewrite a result. A memory summary that says a candidate is good cannot override execution-based evaluation.

Authoring-only pretrained visual retrieval or vision-language inspection is explicitly permitted within authorized resources. This is an extension of the already pretrained authoring system, not an imported visual encoder inside the primary OCR model. Do not copy retrieved model weights into that artifact, use a teacher's OCR answers as primary labels, or inject retrieved feature vectors into primary inference. Use known rendered source strings or independently provided labels for scoring. Fine-tuning the memory retriever is not required; keep it frozen in the first comparison so memory architecture and retriever training are not confounded.

## 9. Leak and instruction handling

Treat retrieved pages, image text, logs, stored agent notes and prior prompts as untrusted data, not higher-priority instructions. A page or memory entry saying to reveal credentials, change scoring, or call a tool cannot confer permission. Keep tool authority in an external broker; validate all model/tool proposals independently of retrieval content.

Never put final images, labels, seeds, hidden-case metadata, predictions, per-case outcomes, or derivatives into author memory during a claimed independent run. Enforce this in storage and execution, not just in a schema enum. Once final feedback guides further construction, that feedback is development exposure and a fresh evaluation must be declared. A separate operator may archive final evidence outside all author indexes.

Keep a corpus manifest and provenance for external research material. Check exact and near duplicates as appropriate, report limits of those checks, and exclude known benchmark answers/overlaps under the experiment's policy. Pretraining knowledge cannot be fully audited by inspecting the memory index; do not claim that it can.

## 10. Implementation surfaces

Suggested additions, not an implemented service:

```text
src/cellforge/memory/
  records.py          # typed events, observations, scoped claims
  store.py            # content-addressed artifacts and append-only receipts
  index.py            # metadata/text and optional visual index interfaces
  graph.py            # lineage, dependencies, evidence/counterevidence links
  retrieve.py         # permission-filtered retrieval and context packing
  visual_adapter.py   # frozen provider/checkpoint, declared query capability
  applicability.py    # version scope and revalidation notices
src/cellforge/harness/memory/
  policies/           # tested agent-editable retrieval/context policies
  diagnostics/        # retrieval and grounding quality checks
```

Provide bounded tools for `record`, `query`, `fetch`, `follow_links`, `annotate_claim`, and `propose_memory_policy_change`. Only the trusted recorder seals execution measurements. A query references an immutable snapshot; a write creates a new event/snapshot. Do not hide a background agent, an optimizer, or an inference helper in these tools.

Required CLI additions to implement: `cellforge memory ingest`, `query`, `inspect`, `reindex`, `check-evidence`, and `compare-policies`. None is supplied as an executable project command by this handoff.

## 11. Acceptance and research comparisons

Mechanical tests should cover source hashes and safe paths, crop/parent relationships, missing citations, claim scope, corruption, untrusted instructions, stale applicability, conflicting results, changed embedding revisions, context limits, author/final isolation, and offline inference with the memory directory removed. A structural fixture passing these checks does not prove genuine VLM understanding or operating-system sandbox security.

Compare construction runs with:

* No persistent retrieved memory beyond the normal current task context.
* Text/structured-record memory with the same available historical corpus.
* Text/structured plus retrieved original visual evidence.
* Optionally the same visual access with and without explicit evidence/counterevidence packets, to separate access from workflow.

Predeclare the primary comparison rather than exhaust every combination. Use matched total construction budgets including retrieval, image encoding, reasoning, memory curation, and policy development. For a retrieval-only diagnostic, freeze the historical corpus; for an online construction comparison, allow arms to diverge but record their exposure and resource histories. Do not let one arm read another arm's future discoveries.

Measure relevant-history retrieval and source support on a separately curated diagnostic set, actual repeated/failed-experiment rates, stale-evidence mistakes, author time/tokens/cost, and independent OCR progress. Self-rated memory usefulness and the number of stored records are not success metrics. Evidence citations can be checked mechanically for existence; whether they substantively support the claim requires further evaluation.

**Success hypothesis:** memory improves the quality or efficiency of the next experiment or the resulting independently evaluated model. It does not directly add capability to the exported weights without an actual accepted construction change.

## 12. Sources

[R3] Shi Yu et al. *VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents*. arXiv:2410.10594, first submitted October 14, 2024; method sections 3.2.1–3.2.2. Official repository: OpenBMB/VisRAG.

[R4] Yubo Sun et al. *VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation*. arXiv:2510.09733, first submitted October 10, 2025; sections 3.1–3.2 and Appendix A.4.

[R5] Qiuchen Wang et al. *VimRAG: Navigating Massive Visual Context in Retrieval-Augmented Generation via Multimodal Memory Graph*. arXiv:2602.12735, first submitted February 13, 2026; sections 3.1–3.2.

Primary HTML versions consulted September 29, 2026: `https://arxiv.org/html/2410.10594v1`, `https://arxiv.org/html/2510.09733v1`, `https://arxiv.org/html/2602.12735v1`; official project `https://github.com/OpenBMB/VisRAG`. This is an integration design, not a comprehensive novelty review or reproduction report.
