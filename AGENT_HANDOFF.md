# CellForge OCR v3 — recursive architecture, adaptive development and visual memory

**Working name:** CellForge OCR. **Revision:** 3.0, September 29, 2026.  
**Status:** Implementation handoff, not an implemented OCR model or a demonstrated research result.  
**Supersedes:** The v1 and v2 handoff packages. Preserve v2 recursive composition and adaptive development; add evidence-backed authoring memory. Read `MEMORY.md` as a required part of this specification.

## 1. Build this project

Build a construction-time **orchestrator LLM that commissions maker agents to directly construct an independently executable image-to-text neural model**. The agents choose actual numerical weights, biases, connections, and architecture. They evaluate candidates and propose subsequent edits. There is no gradient-based fitting of the constructed artifact in the primary experiment.

The owner has refined the architecture: **do not impose a cells → circuits → experts ladder.** Use primitive neural cells and recursively composable bundles. A bundle can contain individual cells, other bundles, or any mixture of the two. Different paths can have different nesting depths. The orchestrator decides when to introduce, remove, combine, split, reuse, or bypass a bundle. It also decides whether and where conditional routing is useful. An “expert,” “encoder,” or “router” is a role of a subgraph, not a mandatory structural rank or a human-assigned skill category.

The owner has also refined the experiment: **the development harness must be improvable.** The system may improve diagnostics, development examples, curriculum, screening strategies, development selection policy, author prompts, scheduling strategy, and development tooling through versioned changes. A small, separately controlled evidence and execution boundary prevents those changes from quietly redefining success, leaking final labels, or exceeding authorized resources. Do not interpret this boundary as freezing the whole harness forever.

The owner also requests **VisRAG-inspired evidence-backed visual memory for the construction harness**. Preserve original visual artifacts alongside exact records; retrieve source-linked observations and earlier experiments for author agents. The memory organization and retrieval policy can evolve. This is authoring memory, not an inference dependency. See `MEMORY.md` for the required design, source distinctions, interfaces and tests.

The primary question is:

> Can LLM agents discover useful neural computations, their recursive organization, and better ways of developing them using evidence-backed memory and direct numerical and structural edits, producing an offline image-to-text artifact that improves on independent evaluation?

Neither a hand-built adder, a scripted recognition circuit, an increasing bundle count, nor a working agent scheduler establishes that hypothesis.

### Construction versus deployment

During construction, pretrained author LLMs may reason, write permitted tooling patches, create workers, inspect development failures, query evidence memory, and revise the graph. Authorized frozen visual retrievers and vision-language readers may assist these authors; their outputs/weights are not imported into the primary OCR artifact. During deployment, only the exported numerical graph and its minimal execution machinery run. No author LLM, memory retriever/store, OCR service, pretrained feature extractor, development data, or label lookup participates.

Architecture changes occur between candidate versions. Input-dependent routing and bounded state updates can occur during inference, but do not imply inference-time weight editing or author calls. Online self-modification is a separate experimental extension, not required or silently introduced here.

## 2. Preserve the mechanism

### 2.1 Direct construction

Every primary-model parameter or topology change after an explicitly recorded bootstrap must trace to an accepted LLM proposal. Permit explicit tensor assignment, sparse edits, exact copies, sharing, rewiring, and deterministic expansion of an author-specified pattern. Deterministic tools materialize the author's choices; they do not fit values or independently decide advantageous changes.

Do not introduce backpropagation, automatic differentiation for update selection, finite-difference gradient fitting, least-squares fitting, a hidden optimizer, or an automatic mutation/search engine into the primary path. Random or evolutionary search and gradient-trained controls are separate labeled comparison arms. No such arm may initialize or repair the primary candidate. A labeled zero or seeded-random bootstrap is permitted; do not run an unreported search over bootstrap seeds.

“Without gradient-based fitting” does not mean that no learning occurs in a broad sense. Feedback-guided parameter editing is optimization, and the author LLMs are pretrained. Do not claim otherwise. Distillation from a larger model is not required by this specification.

### 2.2 Pixels to tokens

The primary model starts with decoded pixels and public shape information. Fixed numerical normalization, padding, patch extraction, coordinates, and masks are permitted if documented and free of recognition logic. The constructed graph must supply any semantic visual processing.

Do not import pretrained encoders, embeddings, model weights, teacher-produced OCR targets, an OCR engine, external language-model predictions, dictionary correction, glyph matching, or per-image lookup. Synthetic labels may come from the source strings actually rendered. A known mathematical circuit can be a runtime unit test, not visual-discovery evidence.

### 2.3 Open architecture, bounded execution

Do not prescribe the final cell count, bundle count, expert count, hierarchy depth, bundles per rank, or equal component sizes. Do not create one expert per character by default. The orchestrator may discover shallow, deep, shared, dense, conditional, or heterogeneous structures.

External spending, memory, expanded-graph size, and execution limits remain mandatory. These are resource safeguards, not a preferred architecture. Never interpret “let the orchestrator decide” as authority to buy unlimited API calls, weaken isolation, or rewrite accounting.

## 3. Model representation: cells and recursively composable bundles

### 3.1 Primitive cell

A starting parameterized cell may use:

```text
h = activation(W1 @ x + b1)
y = W2 @ h + b2
slots = input_width*hidden_width + hidden_width
      + hidden_width*output_width + output_width
```

A 4→4→4 cell has 40 scalar weight/bias slots; a 4→4→2 cell has 30. These are arithmetic configurations, not capability claims. Record all weights, biases, numerical constants, activations, and initialization.

Default to a flexible profile with small cells encouraged but not mandatory. Provide a strict-microcell comparison profile capped at 40 parameter slots per parameterized primitive. The cap does not apply to bundles. It is not a proven limit on LLM ability and must not be the reason to reject larger experiments. Count any parameter-bearing adapter, route projection, edge coefficient, embedding, or output projection; helper functions cannot hide extra weights.

Cells may have latent, unnamed functions. Do not require each cell to solve a standalone task or match a human-defined operation.

### 3.2 One recursive composition rule

Use a component registry with primitive-cell definitions and bundle definitions. A bundle is a typed graph whose members reference either kind. All nesting levels use the same bundle representation and editing API.

For illustration only:

```text
A contains cell_1 and cell_2
B contains an instance of A, cell_3, and cell_4
C contains an instance of B, another instance of A, and cell_5
D contains C and a conditional branch to some other bundle
```

Containment does not specify connectivity; each definition includes explicit port bindings and output bindings. There is no requirement for all cells to be bundled first or for a bundle to “graduate” to an expert. A single new cell may attach to a large existing bundle. A route may select among bundles with different internal structures. Boundaries and role labels remain revisable.

The registry can be shared rather than a strict ownership tree. Several parent definitions may reference the same child definition. Parameter sharing must be explicit; repeated structure, equal values, and shared storage are different properties. Invocation state is separate per instance unless a particular shared-state mechanism is explicitly declared and validated. Updating a shared definition invalidates all dependent evaluations and caches.

### 3.3 Bundling is sometimes only refactoring

Support a **pure wrap** operation that encapsulates a selected legal subgraph behind ports without changing its numerical behavior. Support the inverse inline/unwrap operation. A wrapper may help delegation, reuse, or editing granularity without improving current predictions. Permit it on that basis and test behavioral equivalence; do not claim OCR progress from a new label or wrapper.

Distinguish pure wrapping from computational edits such as adding connections, changing weights, sharing previously independent weights, altering state flow, changing activation functions, or adding routing. Such changes need new model evaluation. Preserve the distinction in the ledger.

Bundles may be provisional. The orchestrator can undo boundaries, jointly edit across them, or compare alternative decompositions. “Freeze” is a version-management choice, not a permanent ban on touching successful components.

### 3.4 Interface contracts without prescribed semantics

Each component declares input/output shapes, data types, parameter references, state, and numerical range assumptions where known. Do not impose a semantic label for each feature. A shape match is necessary, not proof of compatibility. Evaluate combinations in the surrounding task, and test plausible input perturbations where useful.

Boundary-crossing wires must become explicit ports when wrapping. Arbitrary input/output arity and residual/skip paths are permitted within configured limits. Constructed adapters may bridge representations, but their coefficients count and their changes have provenance. Do not auto-fit adapters.

### 3.5 Reusable definitions are not unbounded execution recursion

Allow nested component references of any supported depth up to externally configured expansion and memory limits. Reject direct and indirect recursive template references that imply infinite expansion. Use a directed acyclic execution graph within one forward step. Recurrent behavior must use declared state updated between bounded steps.

Compile or interpret nested definitions with equivalent semantics. Flattening is an implementation option, not a requirement to discard the authoring graph. Guard against exponential instance expansion even when a registry is small. Sharing parameter storage does not make repeated execution or state free. Report registry size, expanded instances, executed operations, actual working memory, and serialized bytes separately.

### 3.6 Routing is optional and composable

Any bundle can emit scores consumed by a conditional dispatch primitive. A route can occur inside a bundle, between branches, or at the root. The orchestrator chooses where to put it, whether it is hierarchical, and what alternatives it selects. Do not require a single top-level “master router” or a preallocated expert bank.

Provide initial top-1 dispatch and a path to declared top-k/combination. Specify tie-breaking, invalid scores, branch interface checks, and actual lazy execution. A weighted sum of already executed branches is not sparse execution. All route-scoring weights and adapters belong to the constructed graph.

Use “MoE” for exports that actually contain conditional selection or combination among learned/constructed alternatives. An arbitrary recursively bundled graph is not automatically a transformer or an MoE. Preserve the original ambition by supporting attention and routed alternatives, not by attaching those names to any graph.

## 4. Evolvable development harness; independent evidence boundary

### 4.1 The distinction

The construction system has two evolving artifacts: the neural model and the development process used to improve it. Separate them from the external machinery used to establish what happened.

| Surface | Agent authority | Requirement |
|---|---|---|
| Model parameters, wiring, bundle boundaries, routing | Direct proposal and automated validation | Every accepted change is versioned and replayable. |
| Development curriculum, samples, diagnostics, failure clustering | May revise and extend | Log distributions, data versions, usage, and feedback exposure. |
| Development screening, branch selection, author prompts, worker strategy | May revise within budget | Version policy; preserve decisions and distinguish selection from capability evidence. |
| Development tool implementations and accelerated runtime | May submit code changes in a sandboxed branch | Test capabilities, equivalence where applicable, and isolation before activation. |
| Reference scoring definitions, task scope, audit runtime, final protocol | May propose changes but cannot self-certify them | External versioned approval; rerun affected comparisons under the corrected/new protocol. |
| Final test data, external permissions, spend ceilings, accounting truth, append-only record | Cannot modify or bypass | Controlled outside author agents' environment. |

Independence means actual separation of permissions, data, and records—not simply naming another author LLM “the judge.” Routine development improvements can pass automated checks; do not require human approval for every extra diagnostic or data generator setting.

### 4.2 What harness-maker agents may build

Examples include better activation summaries, targeted perturbation tests, error clusters, batch selection, curriculum stages, instrumented route traces, cache strategies, tool-output compression, and development candidate-ranking heuristics. These are proposals, not guaranteed improvements.

Construction tooling code may be edited in a constrained build environment. This differs from model edits, which remain declarative graph/tensor changes. Tooling patches are untrusted code until sandboxed checks pass. Do not give a harness worker API secrets, final labels, a writable reference evaluator, or unrestricted network access merely because it writes code. Audit both declared dependencies and executed behavior; a package-name denylist alone does not establish that a hidden fitting algorithm is absent.

The harness can help an LLM inspect a tensor or expand an exact pattern the LLM chose. It must not quietly turn a task into numerical fitting and then attribute the results to direct LLM edits.

### 4.3 Version the model and the measuring process independently

Record every result against:

```text
model_hash + development_harness_hash + execution_backend_hash
+ task/tokenization_protocol_hash + dataset_manifest_hash + seed
+ selection_policy_hash + memory_policy_hash + memory_snapshot_hash
+ retrieved_context_manifest_hash + resource_measurement_context
```

If a harness patch changes a diagnostic or metric implementation, rerun an unchanged candidate on the same examples before/after the change. If both model and harness change, evaluate old/new models under old/new harnesses when feasible, and always compare old/new exported models under the same reference evaluator and examples. Do not attribute an easier sampler or a changed denominator to model improvement.

For semantics-preserving runtime acceleration, require differential tests against the reference implementation plus timed benchmarks on identical workloads. Test outputs, state, dispatch, error behavior, and numerical tolerances. A faster incorrect implementation is not an accepted acceleration.

For a new curriculum or diagnostic, exact score equality is not expected. Report it as new development coverage. To claim improved construction efficiency, compare downstream progress under matched construction budgets on independent evaluation—not just the new diagnostic's own score.

### 4.4 Development acceptance can change; audit meaning cannot silently drift

The orchestrator may change how it selects exploration branches or prioritizes candidates. Preserve a separately evaluated reference best-so-far alongside its working candidate, and record why they differ. An external safety/resource violation cannot be overridden by a development selection policy.

Keep anchor metrics on a documented reference development distribution for longitudinal comparisons. All repeatedly returned anchor scores are development feedback, not final evidence. The reference distribution may be expanded or corrected through a versioned protocol change; re-evaluate comparison candidates and report separate result epochs. Do not freeze a known bug merely to preserve history, and do not silently overwrite history to hide the bug.

A final score is generated only after the candidate and protocol are frozen for that evaluation. If final feedback informs a subsequent candidate, the next claim requires a new untouched evaluation and disclosure of the prior exposure.

### 4.5 Evidence-backed visual memory

Implement `MEMORY.md` as a required construction-time subsystem. It separates the executable neural graph, durable experiment/evidence graph, and rebuildable retrieval index. Native arrays, exact metrics, graph edits, predictions, original images and source-linked crops remain inspectable; summaries are derived interpretations, not replacements for execution receipts.

Use a hybrid exact/metadata/text retrieval path with an optional configured frozen visual-retriever adapter. A live visual-memory claim requires actual image retrieval/inspection, not only replay or text lookup. Produce concise per-source observations, distinguish hypotheses from measured claims, retrieve relevant failures/counterevidence, and link resulting decisions to model or harness transactions. Preserve bundle lineage across refactors; do not equate similar images or renamed bundles with proven capability transfer.

Memory policies, indexing views and context compression are agent-improvable development surfaces. Version them and record snapshots plus the exact retrieved context. Source measurements, permissions, final-data isolation and resource authorization are not agent-editable. Construction-time pretrained vision is permitted for evidence retrieval/inspection only; it must not become an imported encoder, teacher-label source, or inference dependency in the primary neural model. No training of the retriever is needed for the initial memory comparison.

Final cases and their derivatives never enter author-accessible indexes, prompts, cached embeddings or summaries. External documents and stored notes are untrusted data, not tool instructions. Treat claims without available support as unsupported, not as valid because they have been repeated in memory.

## 5. Agent organization and transaction protocol

### 5.1 Orchestrator

The orchestrator decides what to build and what to learn from the next experiment. It can begin with a small unoptimized image-to-token graph and enlarge or restructure it as evidence warrants. It should see concise component cards, dependencies, provenance, failure summaries, costs, and versioned development measurements rather than every scalar on every turn.

Required architecture actions: create/edit cell; create/replace bundle definition; instantiate/reuse; wrap/unwrap; split/merge boundaries; copy/share/unshare parameters; rewire; insert/remove conditional routing; revise adapters and output paths; perform coordinated cross-boundary edits; retire components; request an ablation; stop. These actions are not all required on every run.

Required process actions: commission/close workers; define bounded subproblems; change development curriculum; commission a harness patch; request cross-version checks; change internal selection policy; request external protocol review; freeze/export a candidate. A deterministic scheduler enforces limits but must not choose the entire architecture behind the LLM's back.

### 5.2 Makers are scoped by subgraph or tool surface, not by rank

Do not hardcode “cell makers → circuit makers → expert makers.” A maker task targets any selected subgraph or bundle, at any depth. One worker may wrap several components; another may repair the same region numerically on an alternative branch. A worker may request a subtask through the orchestrator; it cannot mint new budget or unrestricted tool access.

A separate harness-maker role targets authorized development tools. Model and tooling changes use separate transaction types and permissions. Roles may evolve through logged prompt changes, but permissions are not encoded solely in editable prompts.

Tasks include base model/harness hashes, allowed IDs or paths, hypothesis, development context, interface constraints, evaluation requests, budget reservations, and expiry. Store concise rationales and structured actions, not a demand for private chain-of-thought.

### 5.3 Atomic edits and concurrency

Validate transactions before execution. Reject stale bases, missing references, invalid ports, non-finite values, infeasible allocations, unsafe paths, unauthorized operations, and resource violations. Either all coordinated changes and interface migrations commit, or none do. Explicitly rebase and re-evaluate conflicting proposals; do not silently combine tensor edits.

Shared-template changes require transitive invalidation. Distinguish a change to every shared use from clone-on-write of one instance. Bundle refactoring must preserve boundary wiring and per-instance state identity. Restore task states after crashes without duplicating accepted edits or paid calls.

Do not demand monotonic improvement from every local patch. Permit bounded exploration candidates and coordinated edits across components, while maintaining independently measured baselines.

### 5.4 Proposed loop

```text
load externally authorized task protocol, resource limits, and evidence runner
create versioned model, development harness, and branch archive
while resources remain and the orchestrator has not stopped:
    give orchestrator versioned evidence, component cards, failures, and costs
    orchestrator commissions model work, harness work, or both
    validate proposals against their exact bases and permissions
    sandbox and test harness changes before allowing tool activation
    apply model changes atomically; preserve author and parameter provenance
    evaluate comparable candidates on common examples under common scoring
    run cross-version checks when the measuring process changes
    record development-policy decisions separately from reference results
    update working branches and independently evaluated best-so-far
freeze selected model and reference protocol; export
independent evaluator runs untouched final cases without author access
report software status, live status, measured results, and unproven hypotheses
```

## 6. Image-to-text runtime and data

### 6.1 Initial interface, not a fixed internal pipeline

For a vertical slice, use configurable 32×128 grayscale images, normalize uint8 pixels by 255, and begin with fixed 2×2 patch extraction. These are engineering defaults, not constraints on final component organization. Log resizing/padding and prevent clipping from creating incorrect labels. The orchestrator can change model-internal patch aggregation or token processing through explicit graph edits. Input contract changes are versioned; never sneak recognition code into preprocessing.

Use a small character vocabulary with BOS/EOS/PAD and expand it through declared stages. No pretrained tokenizer/embedding is required. Record vocabulary versions. The model emits token logits, and a bounded driver selects tokens and stops on the model's EOS or a recorded limit. Generated prefix/state—not target text or target length—is the inference input after the image.

Fixed attention math is allowed as an audited operation with all parameterized projections constructed and counted. A non-attention graph is a valid early artifact, but is not a transformer. Dense and routed execution are both permissible hypotheses; a sparse variant must skip unselected branches in fact.

### 6.2 Synthetic development; independent final examples

Render known source strings with locally available, appropriately licensed fonts. Do not include system font files in this package or repository. Log font hashes, renderer settings, source strings, and split policies outside inference. No font IDs, filenames, random seeds, glyph boxes, source strings, or renderer metadata may leak into model inputs. Decode images before invoking the model.

The agent may adapt development data: lengths, spacing, perturbations, fonts available for development, class balance, and hard-case sampling. Preserve manifests and track which images or scores were exposed. Verify labels still describe the rendered content when transformations crop, occlude, or alter it. An adversarial example must not be mislabeled by a careless generator.

Separate single-character diagnostics from whole-image variable-length OCR with no supplied character boxes. Report unseen source strings, rendering variants, font families, and length shifts separately where supported. Do not invent an unseen-font result if only one font is available.

An implementation agent's ordinary filesystem cannot hold a secret test that the same agent can read. Final examples, labels, and seeds must be generated or provided by a separate operator/evaluator after artifact freeze. Repeated development pass/fail feedback is still development information. Include duplicate checks and disclose overlap policies.

## 7. Evidence, metrics, and comparisons

Use independently implemented free-running exact string match and character error rate (total Levenshtein edits / total reference characters) as initial reference metrics. Specify blank-reference behavior, normalization, permitted tokens, EOS policy, and all error denominators. CER may exceed 1 due to insertions. Timeouts, invalid outputs, NaN, and missing EOS are recorded failures, not dropped cases.

The agent may add diagnostic metrics or change development priorities. It cannot report a changed metric as the same historical series. Ground-truth-prefix negative log-likelihood is a diagnostic, not a free-running OCR success claim or teacher-model distillation.

Report unique parameter storage, expanded parameter uses, nonzero slots, cells, bundle definitions/instances, nesting and call depth, edges, active routes, executed work, graph/weight/export bytes, peak memory, latency on stated hardware, author tokens/calls, tooling-evolution cost, candidate-example passes including diagnostics, and any known charges. Bundle count alone is easy to inflate without a functional change; do not use it as capability evidence.

Minimum comparison plan:

* Initial artifact and one-shot author against iterative construction.
* Flat graph editing, fixed human hierarchy, and adaptive recursive bundling with comparable representation freedom and construction resources where feasible.
* Fixed development harness versus evolving development harness; account for tool-building calls and tests.
* Single author versus orchestrator/team; compare at matched total resources, not worker count alone.
* Random direct-edit baseline and, separately, conventional search or gradient-trained controls where affordable.
* Strict 40-slot cells versus flexible cells; shared versus unshared templates; local versus coordinated patches.
* No retrieved memory versus text/structured memory versus source-preserving visual evidence memory, under matched total authoring budgets; see `MEMORY.md`.

Do not imply that all combinations must be exhausted before delivering a vertical slice. Implement the comparison hooks and run a predeclared primary comparison within authorized resources. Clearly mark unrun arms. For repeatability claims, use independent construction runs, paired evaluation examples, and uncertainty intervals with correct independent sampling units. Do not treat thousands of near-identical renderings as thousands of independent documents.

The meaningful bundling outcome is improved construction efficiency, transfer/reuse, robustness, or held-out accuracy—not merely successful wrapping. The meaningful harness outcome is better independent progress or measured efficiency—not a higher editable development score.

## 8. Implementation plan

Use a transparent CPU numerical reference runtime first; acceleration is replaceable after equivalence testing. Python, a numerical-array package, an image renderer, schema validation, and an automated test runner are proposed tools, not mandatory library versions. Verify current official provider APIs during implementation and lock tested dependencies. Keep inference independent of author SDKs and development rendering dependencies.

Suggested layout:

```text
src/cellforge/
  ir/              # recursive registry, ports, references, versions
  runtime/         # cells, parameter-free ops, dispatch, state, decoder
  construction/    # orchestration, scoped tasks, transactions, branches
  harness/         # versioned development tools, curricula, policies
  memory/          # durable evidence, lineage, retrieval, source-grounded packets
  governance/      # capability checks, spend ledger, patch test broker
  reference/       # separately controlled scorer/interpreter interfaces
  providers/       # real configurable adapter; labeled replay adapter
  data/            # rendering and development manifests
  baselines/       # separate mechanisms, never hidden primary fallbacks
  reporting/       # provenance, model/harness histories, comparisons
  cli.py
```

An audit boundary cannot be implemented by a folder name alone. In live research runs, mount trusted reference files read-only, isolate private evaluation elsewhere, broker credentials, and use enforceable process capabilities. Offline fixtures test controls; they do not prove hostile-code isolation by themselves.

Implement at least one genuine live provider adapter and a replay adapter. Live calls need configured credentials and explicit positive bounded authorization. Missing credentials mean offline implementation can continue, not that replay may be relabeled live. Record provider-returned identifiers, settings, usage, retries, actions, and errors. Unknown costs are unknown, not zero.

Export a safe typed manifest, numeric arrays, vocabulary, pixel contract, state/route semantics, primitive version, hashes, and a model card. Reject unsafe deserialization, path traversal, oversized allocations, or arbitrary model callbacks. Preserve the authoring structure and provenance separately from the minimal inference payload. No keys, prompts, labels, or development cache in the export.

Required CLI families to implement—not commands already available here:

```text
cellforge doctor
cellforge smoke --provider replay
cellforge construct --config <run.yaml>
cellforge inspect --artifact <artifact> --view hierarchy|expanded
cellforge harness check --proposal <patch>
cellforge compare-versions --manifest <comparison>
cellforge export --run <run> --out <artifact>
cellforge predict --artifact <artifact> --image <image>
cellforge evaluate-final --artifact <artifact> --private-config <operator-config>
cellforge memory query --request <request.json>
cellforge memory check-evidence --packet <packet.json>
cellforge report --run <run>
```

## 9. Delivery gates

**G0 — Recursive runtime.** Cells and nested mixed bundles execute; references, sharing, wrapping/unwrapping, invalid recursion, state, counts, and offline export are tested. Fixtures can be arithmetic or arbitrary numerical graphs, but are engineering evidence only.

**G1 — Authoring loop.** A genuine orchestrator commissions a scoped maker, receives actual numerical/structural edits, evaluates them, returns feedback, and obtains another proposal. Replay exercises the same paths offline. Improvement is not required to establish plumbing; plumbing is not research success.

**G2 — Adaptive organization.** Logged LLM decisions create or remove bundle boundaries and perform cross-boundary edits. A wrapper can be accepted as refactoring without a false capability claim. Compare against a flat/fixed representation when conducting research.

**G3 — Harness evolution.** A harness-maker proposes a real development patch; it is tested, versioned, activated, and its impact separated from model change. Deliberately corrupt scoring/labels in a negative test and verify the independent reference result cannot be overwritten. Exercise legitimate reference-bug correction through the external versioning process.

**G3M — Evidence memory.** Durable visual/native sources, versioned retrieval, per-source observations and counterevidence feed a subsequent author task. Test source integrity, lineage, misleading notes, stale results, final isolation and offline export. Replay proves interfaces only; a genuine visual-retrieval adapter must be run before claiming live visual memory. Compare memory arms as specified in `MEMORY.md`.

**G4 — Visual discovery.** Evaluate single-character diagnostics and then whole-image variable-length text. Report whether live construction improves independently measured OCR under the registered comparison. No promised threshold or manufactured pass.

**G5 — Scaling.** Load hundreds of heterogeneous bundle instances, with mixed depths and shared definitions, under measured resources. Include actual conditional execution where claimed. A large fixture is a runtime scalability test, not discovered expertise.

Deliver working repository code, contracts, prompts, runnable commands, tests actually executed, provider/replay results, model/harness histories, exports that exist, and an honest report. Distinguish implementation, offline validation, live operation, visual improvement, and unproven scalability. Do not stop at a second planning document when implementing this handoff.

## 10. Context and limits

The revised mechanism is not just traditional architecture search: it seeks direct LLM authorship of both numerical parameters and architecture, plus a changeable development process. Recursive bundling is not evidence that these edits become easy or that the entire system will succeed.

One close architectural precedent is **Hierarchical Representations for Efficient Architecture Search** (Liu et al., 2017/2018). It describes smaller graph motifs used inside larger motifs. Its search/training setup does not establish direct LLM weight construction or an agent-improved harness. This source supports the composition concept, not CellForge results or a novelty claim. [R1]

**Evolving Deep Neural Networks** (Miikkulainen et al., 2017) presents CoDeepNEAT for automated topology/component/hyperparameter design. It is another architecture-search precedent, not evidence that the proposed direct-edit experiment works. [R2]

Earlier architecture sources retained from v2 (not re-reviewed for this memory revision):

[R1] `https://arxiv.org/html/1711.00436v2`  
[R2] `https://arxiv.org/abs/1703.00548`

For the VisRAG, EVisRAG and VimRAG source distinctions checked for v3, see `MEMORY.md`, sections 2 and 12.

Before any publication or novelty claim, perform a wider and current prior-art review. This handoff is an engineering/research proposal, not such a review.

**Core instruction: let the agents discover the abstractions and improve their development tools; keep the resulting claims independently checkable.**


---

# Embedded required companion: MEMORY.md

The standalone export includes the memory specification below. The ZIP also contains separate prompts, contracts, configuration and validation fixtures.

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
