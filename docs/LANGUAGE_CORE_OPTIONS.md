# Language core options — decision proposal, not measured scaling

Latest primary-arm correction: the adaptive Bayesian mixture uses a Hedge /
exponentiated-gradient-style multiplicative update. The owner prohibits gradients
of every kind and explicitly drops this mixer from every primary arm.
Historical equation-parity and side-reference receipts remain valid
records, not authorization. Static count-fitted seeds remain allowed. The active
four-arm engine must enforce this exclusion. Earlier mixer-archive plans below
are superseded where they imply primary inclusion.

Version `language-core-options/2`, September 29, 2026 (America/New_York).
Owner decision: reject gradient descent and the trained shared-core option.
The three-path review below is historical decision context, not the active plan.
The primary arm remains no-gradient. The owner now authorizes automatic
population-based numeric mutation and discrete fitness selection; this
supersedes historical authored-only/no-automated-search restrictions.
Latest owner decision also permits dev-training-fitted count primary seeds
with explicit corpus/split/method provenance; pure evolution remains distinct.
No option below authorizes spending, gradient methods, or establishes
an LLM, frontier superiority or AGI. Document benchmark execution is on hold
until a predeclared plain-text QA gate is passed; infrastructure diagnostics
do not satisfy that gate.

## Current evidence and bottleneck

`runs/unified-byte/artifact/MODEL_CARD.json`: frozen parent `f0b43dbe...`,
30,791 unique parameter slots (3,481 nonzero), eight routed experts, shared
259-token head and 225-dimensional state. Runtime receives 64 bytes, but inherited
experts use only eight projected bytes. The 387-byte compatibility excerpt scores
2.480344 bits/byte and 208 correct predictions; it is not representative language
evaluation. Ordinary prompts repeat `too`; a 64-context memory mutation worsened
loss to 2.512589 and both scored 0/4 novel bindings. Legacy delegated model
`82ebe5e5...` has normalized public MMLU/ARC-Challenge/HellaSwag scores
.241/.243/.241, n=1,000 each, with character filtering, not `mc-byte-loglik/1`.

Explicit OCR commands and unary controlled clauses now call specialists and
consume observations within authored syntax. That is useful component evidence,
but there is no broad plain-text QA or general instruction-following result.
UTF-8 coverage, JSON emission and faster execution remove plumbing barriers;
they do not supply general language competence. The present strategy is
uncompetitive on measured transfer tasks. This does **not** prove direct
weight authoring theoretically impossible; its scalable capability mechanism
has not been demonstrated.

Existing `src/cellforge/baselines/lm_baselines.py` trains an eight-context MLP
with NumPy Adam or n-gram controls and exports them to the IR; its narrow
character protocol is not a matched full-byte control. No Transformer or SSM
training/export control has been measured here. The bounded byte-Adam draft was
cancelled; no gradient job started. It is not current work.

An additional **separate local** founder curriculum baseline is now retained in
`runs/language-curriculum/FOUNDER_DECISION.json` and `founder-selection-v1.json`.
Eight authored development cases include 190 scored content bytes at
4.0663511910499714 bits/content byte (EOS excluded), with zero scoring failures.
Raw ordinary-language copy, recall and QA each score 0/2; all six generations
repeat ` too` to a 24-token cap, with `max_len` failures retained. Generation
uses compiled/state-only prefill; next-byte scoring is reference execution.
Elapsed time is 9.64137 seconds; three evaluator tests pass. This is not the
334-byte server calibration or an evolutionary generation and cannot support
a cross-corpus improvement comparison. The plain-text QA gate remains unmet.

## Active recommendation: discrete evolutionary construction

Owner rejects gradient descent. Evolve parameter and program/module populations,
with generation zero the frozen byte MoE `f0b43dbe...`. No trained trunk is
authorized. Keep its raw behavior as the anchor; do not replace failed language
answers with a host-side solver. The capability gap remains measured evidence,
not proof that evolutionary discovery is impossible or a guarantee of success.

Retain the rejected `long_context_memory` donor (2.512589 versus 2.480344
parent bits/byte) as an archived mutation source, not a promoted survivor.
Its failed metric and 0/4 binding answers remain attached to any reused module.

Executed pilot `language-evolution-minipilot/1` is complete; launch
receipt `runs/language-evolution-minipilot-v1-launch.json`. The bounded policy
below supersedes earlier proposed eight-slot populations. Terminal receipt:
`runs/evolution-language-minipilot-v1/result.json`, retained candidate graphs and
a verified 16-entry ledger. Elapsed152.67957s; peakRSS68,136KiB (~66.54MiB).
Twelve candidate evaluations produce two unique report-evaluated graphs.
Selection loss at generation0/1 is2.042730829833363 for both arms; generation2/3
is2.0422575408863954 for both, with identical winner `e4d1e5db...`.
Every generation/arm report loss is unchanged2.606790482283684, zero failures.
This is **no observed GA advantage and no report-only improvement**. Public
development report data are not unexposed final data. Winner QA was not evaluated
and no main-branch promotion occurred. Seven focused tests pass (root1.17s).

Generation-zero calibration is now measured, before any evolved generation:
`runs/evolution-language-gen0-v2.json` records founder `f0b43dbe...` on 334
content bytes, 2.042730829833363 bits/byte, zero failures, 9.81954 seconds on
the server under single-thread nice-15 execution. This is selection fitness on
public dev_train content (EOS excluded). A disjoint report-only pilot family is
reserved; report evaluations are zero and curves release only after generation/
winner freezes. Neither partition is unexposed final data or a multiple-choice score.
The earlier v1 selector requested whole stories of 150–256 bytes and found none,
failing before fitness. V2 revised bounds to 288–350 after inventory lengths
288–600, without using candidate outputs for the amendment. This small selected
corpus is different from the 387-byte compatibility probe; losses are not a
matched improvement comparison. The first engine completed under the policy
below; terminal results above show no evolutionary advantage. Calibration alone
is no evolutionary benefit.

- Existing two CPUs/4 GB only. One isolated nice/thread-one server worker;
  coordinator checked the live handle. PID38247, nice15, one thread, observed
  RSS57MB, 2GiB address-space limit and 600-second job timeout. No parallel
  CPU candidate jobs or new resources are added.
- Three generations, two logical islands (router and expert arrays), one
  proposal per island per generation: six GA and six founder-random evaluations,
  twelve total. Paired seeds make first-generation proposals numerically
  identical controls. Each proposal mutates eight nonzero coordinates of one
  active parameter array with Gaussian sigma `0.05*max(1, median abs active)`.
  Strict selection
  bits/byte improvement retains the child; otherwise retain parent. All graphs
  are archived, including failed candidates. Module crossover and island
  migration are not implemented in this first run.
- Numeric mutation operators: seeded Gaussian perturbations to selected array
  coordinates/blocks or graph-argument scalars, and compatible coordinate/block
  crossover. Select survivors by development fitness without forming derivatives
  or fitness-weighted update vectors. Evolved numbers are **fitness-selected**,
  not all hand-authored or untouched by data. Freeze distributions, magnitudes,
  seeds, mutation targets, bounds and cost gates in the pilot policy.
- Structural mutation operators: import an authored definition with `import_def`,
  reconnect typed edges, add/remove an expert, swap compatible cell/cluster
  modules, mutate graph-owned scalar values, change graph-owned
  router thresholds/state updates, or extend context mechanisms. Every proposal
  names exact base hash, changed definitions, numerical values and donor hashes.
  Shape/profile validation and state/reset/equivalence tests precede fitness.
- Crossover copies named compatible modules, preserving input/output contracts,
  shared-parameter identity and parent-to-child lineage. It is not averaging
  fitted weights or borrowing an answer-bearing artifact. An incompatible donor
  is rejected with evidence, not silently reshaped. Restoreability requires
  saved child graphs, source snapshots and append-only decisions.
- Selection uses measured development fitness only. No derivatives, optimizer,
  backpropagation, finite differences or fitness-weighted perturbation vectors.
  Evolution strategies that estimate a gradient by weighted perturbations are
  excluded even if branded evolutionary. Ordinary discrete GA proposal/evaluate/
  rank/retain does not estimate such a gradient. Automatic parameter fitness
  search and explicitly provenance-labeled count seeds are authorized; optimizer
  training and gradient estimators remain excluded. Count fitting is a distinct
  method, not to be relabeled entirely hand-authored parameter evolution.

Staged fitness is a proposal, to freeze before pilot results: first full-byte
next-byte prediction on shared author-visible corpus excerpts (bits/byte,
validity, EOS, failures); then copy/novel key-value recall with lengths outside
the inherited eight-byte window; then plain-text contextual QA with paraphrases,
unseen entities and absent-information controls. Do not promote on syntax or
one contrived question alone. Keep all earlier stages as regression checks,
including negative repetitive-output cases. A selection table reports all
metrics and resource failures rather than folding failures into a hidden scalar.
The 20-case QA gate above is still proposed, not frozen; independent broader QA
and final document-family evaluations remain required beyond pilot selection.

Run a matched discrete random-search control from the same generation-zero
graph, with the same allowed operators, candidate count, evaluator, cases,
runtime limits and author proposal budget. Random search does not retain
fitness-directed parents; GA does. Predeclare RNG seeds, tie rules and
mutation distribution. Preserve rejected candidates/donors and no-reuse/router/
missing/wrong-observation ablations. Report author tokens and evaluation compute
separately: parallel authors are not free model training or inference capability.

Root's first bounded parameter-GA versus founder-random pilot is completed.
The tool agent's larger v2 engine is future/unlaunched and must not be merged
with this run's receipts. Root-tested `import_def` breeding wrappers are ready
for a subsequent stage, not executed crossover evidence in v1.
Its small development selection-loss decrease is measured, but no evolutionary
advantage, report gain, QA capability, document result or frontier win is shown.
Document scoring remains held pending a plain-text QA gate. No new resources
or paid services are authorized by this plan. Keep the original full objective;
successful microtasks would only justify the next larger controlled experiment.

Next v2 scope is proposed, not launched: broader fixed-development context/state
dependency mutations, activation of initially zero parameter coordinates and
compatible module/parameter crossover. Keep fixed random-search resource controls,
all rejected donors and public v1 report exposure in lineage. Obtain a fresh
operator-owned report family and freeze it before selection; v1 report feedback
cannot become unexposed final evidence. Evaluate copy/recall/QA, not next-byte
fitness alone, before promoting language claims or resuming document scoring.

Active evolutionary cost model: measure each graph evaluation's elapsed time,
RSS, semantic MAC counts, cold/warm runtime startup and authoring effort on the
existing server. Total pilot time is `sum(candidate evaluation times) + validation
+ authoring/export overhead`; price is authorized existing-resource accounting,
not a new CPU/GPU purchase. The historical `6*P*D` gradient-training heuristic
above does not estimate discrete GA cost. Fit no performance scaling law before
three genuinely different authored populations have comparable receipts.


## Active critique and next pilot requirements

Throughput first: measure candidate evaluations/second for a fixed selection
workload, separating startup/validation/baseline/report time. Twelve candidates
over152.68s total includes baseline/report work; it is not a pure candidate rate.
Gen0's334bytes/9.81954s is about34 content bytes/second for that workload only.
No scalable training throughput or new speedup is claimed.

Proposed compiled scoring uses64-byte windows and safe upstream reuse. Verify
exact logits/state/loss and execution-stat semantics against reference across
changed inputs, recurrent states, parameters and routes. Cache guards must
cover exact state/input and graph/parameter identity; reuse invariant dependency
closures only. Retain cache failures. Profile pure candidate costs before making
speed claims.

Broaden operators to structured row/block mutation, expert-copy crossover,
router gates, context/state dependencies and typed `import_def` imports. Log
per-operator proposal validity, cost and acceptance. Self-adaptive probabilities
may respond to development acceptance under a frozen update rule, floors and
seeds; log every change, without fitness-weighted perturbation gradients.
This is fitness-selected search policy/parameters, not entirely hand-authored.

Proposed acceptance requires improvement on **two disjoint development batches**,
with earlier-stage regression and resource gates. Freeze batch manifests and
strict improvement/tie/failure rules before outcomes. These author-visible
batches are not independent final evidence; fresh report families remain
excluded from fitness and adaptive policy. Add copy/repetition byte fitness and
novel recall beyond eight-byte effective context, varying lengths and unrelated
prefixes. Failed memory donors remain archived, not promoted. No host-side
answer copier or language claim from copy-only success; document hold remains.

**Latest owner ruling supersedes the earlier control-only restriction:** fitted
count seeds may enter the primary as option3, with explicit fitted provenance.
Fit on dev_train/selection only, with corpus/split/method hashes; never report
or final data. Preserve the pure `f0b43dbe...` evolution lineage separately.
No gradients, backpropagation or gradient estimators are permitted in any arm.
Four matched arms: fitted seed+evolution, fitted seed+random, fitted seed alone,
and original founder+evolution. Compare bits/byte against evaluator calls/cost
on identical frozen known-development subsets, not final evidence. Throughput
and exact export/reference parity precede launch; adaptive policy excludes
report data. Root's preliminary bigram helper is tested but unused/superseded,
not a fitted run or seed. Current seed builder, four-arm engine and qualified
evaluator are in development; no new server evolution job is claimed.
V1's no-advantage/no-report-gain result remains unchanged.

New side research is retained unmodified in `notes/CLAUDE_RESEARCH.md`,
`scripts/claude_count_reference.py` and `scripts/claude_count_features.py`.
The scripts confirm per-document UTF-8 byte Witten-Bell interpolation and
dev_train fitting; the feature script selects mixture weights by grid search
on the last150 dev_train documents, not gradients. The side note reports order6
1.3816 bits/byte on all165 dev_eval documents/89,213bytes, versus founder2.5113
on only first24documents/12,894bytes. These unmatched scopes **do not establish
a matched 1.1-bit gain**. The feature study separately reports1.3909 countonly,
1.3090 wordcontext,1.2863 documentcache and1.2650 combined, after a different
fit/validation partition. These are side-note development results with scripts,
not independently reproduced frozen per-position receipts or QA evidence;
do not mix the1.3816 and1.3909 baselines. Reported batch timing is exploratory,
not a qualified candidate-evaluations/second comparison. New position-wise
fixtures/export parity are requested before launch.

Proposed current seed architecture: exact order6 count-reference export to the
IR as a fitted trunk, plus the inherited MoE residual in log-probability mixture
with zero residual contribution at generationzero, then parameter/program GA.
Freeze vocabulary/EOS/context/reset/probability-floor rules and prove every
per-position logit/probability matches the reference before claiming seed loss.
The zero mix must actually reproduce the fitted trunk; inherited residual
weights/routes and activation changes need their own lineage and ablations.
Export and selection are not yet a measured model or general-language result.

## Historical adaptive reference and exposure correction — primary inclusion suspended

The previously proposed order was exact seed/reference parity, four matched arms,
then an adaptive-mixer recipe archive and cache/word modules. Mixer inclusion is
now suspended under the latest ruling above; static-seed four-arm work continues.
No later component may
be reported as already exported or evaluated because a side reference works.
`scripts/claude_adaptive_mixer.py` and Claude side notes remain untouched. Notes
report1.2222 bits/byte for adaptive count mixing; this is a reference target,
not a certified IR match or independently replicated language capability result.
Preserve fit/validation/evaluation scope and recipe when transferring it.
The script resets Bayesian posterior mixture weights per document, updates from
past observed bytes and shrinks toward prior(0.8,0.15,0.05). It grid-selects
eta/decay on the last150 dev_train validation documents; notes identify
eta1.0/decay0.9, then one dev_eval measurement. This is a multiplicative-update
fitted mixer with development selection. The previous nongradient characterization
is superseded: the owner explicitly excludes Hedge/exponentiated-gradient-style
updates from every primary arm.

Maintain a recipe archive using measured development performance **and novelty**,
with failed recipes eligible as mutation/crossover donors only after validated
child reconstruction and fresh evaluation. Keep failure metrics attached;
archive membership is not promotion. Low-dimensional discrete recipe variation
must remain distinguishable from gradient estimation. ES/CMA-ES names alone do
not settle compliance: exclude fitness-weighted perturbation updates estimating
gradients; any uncertain optimizer variant needs an explicit method audit before
execution. No claimed breakthrough from literature abstracts or algorithm labels.

Exposure correction: earlier v1 report-only tiny story families came from the
public dev_train source. A full-dev_train count fit would consume those families
and violate a later report-only boundary. Exclude previously reserved/exposed
report families from the **new primary fit** and record fit/exclusion manifests.
The parity reference must use exactly the same fit partition and interpolation
method as the exported seed. Disclose corpus-size/method differences from
Claude's full-training or last150-validation side references; their published
numbers do not transfer automatically to the excluded-family seed. Current
corpus suffices for parity work; no new fit/run is claimed yet.

The owner ruled to keep the current corpus; additional corpus downloads are not authorized. Historical options ≤100MiB or1GiB were
requests, not approval. Check source, license, content suitability and isolation
before any download; no new corpus has been fetched under this plan. Preserve
current data/source hashes and do not silently change the four-arm workload.

## Historical three-path review (superseded)

| Path | Concrete experiment | What it answers | Authorization/status |
| --- | --- | --- | --- |
| 1. Trained shared language trunk + authored experts/router | Historical proposed Transformer/SSM trained trunk with authored specialists. | Learned substrate comparison proposed during review. | **Rejected by owner; inactive.** No gradient-trained core is authorized. |
| 2. Preserve no-gradient primary, measure matched trained control | Historical proposal for equal-size gradient comparison. | Quantifies capability gap under a different fitting method. | **Not the active recommendation; do not launch new gradient work under this decision.** Retain any already completed control evidence with its provenance. |
| 3. Three-size capability/cost study | After a working control, evaluate roughly 0.03M, 0.3M and 3M parameter cores with identical full-byte evaluation. Report loss versus data/compute, QA, specialist effects and cost. | Whether gains persist with scale and where budget is better spent. | Plan only; one bounded calibration job first. Larger training cannot be assumed affordable or useful. |

The following implementation considerations describe the rejected proposal,
not authorized work. For option 1, the review suggested a causal Transformer,
not two
large training projects at once. SSM is a separately budgeted alternative after
state/reset/equality export tests. Existing IR matmul/state primitives do not
automatically establish attention, normalization, gating or selective scans
correctly. Freeze operation semantics and verify training-framework/IR logits,
state/reset, masks and decoded bytes on novel prefixes before task evaluation.
Shared state and common logits must actually carry observations; a host script
choosing the correct expert is not learned routing. Pretrained imports, if ever
authorized, require an additional provenance arm and cannot be relabeled authored.

For option 2, primary size matching uses **total unique stored slots** with
active/nonzero/expanded uses reported, not zero padding to hit a count. A byte
MLP with context8/hidden13 has `(8*259 + 259 + 1)*13 + 259 = 30,575` slots,
within 1% of the 30,791-slot parent. This is a proposed bounded architecture,
not a run result. It lacks the parent's recurrent state; label that distinction.
Use the same 64-byte runtime input with explicitly disclosed effective contexts,
then add a stateful size-matched control if practical. A budget match does not
make an architecture-controlled causal claim. Include trigram/count controls
as separately labeled fitted baselines, never as inherited primary knowledge.

## Plain-text QA gate and isolation

The hold is a research priority, not completion of a smaller substitute goal.
Freeze a capability gate on answer-free synthetic/development fixtures before
results: natural-language requests that vary wording/order, unseen entities,
provided-fact retrieval, two-step relations, arithmetic and absent/contradictory
information; include non-tool questions and specialist-relevant questions.
Compare full system, language-only, missing/wrong observations and disabled tools.
Measure answer correctness, valid calls, observation dependence, abstention,
latency and UTF-8 failures. Do not reuse controlled grammar template self-tests
as general-language evidence. The coordinator proposes 20 novel short-context
QA cases with at least 16/20 coherent acceptable answers, valid UTF-8 and
improvement over a no-context control; chance must be defined for each format,
since free-text QA has no universal chance rate. Include wording outside
controlled commands. One ordinary question answered coherently is a necessary
minimum, not proof of generality; even passing 20 cases requires broader
independent QA. Numerical thresholds are **proposed, not frozen**; fix cases and
rules before outputs. This is a document resumption gate, not goal completion.
No document run resumes on a passing transport test or repeated `too` generation.

Corpus provenance/licensing, duplicate-document family split, encoding/newline
rules and length distribution must be versioned. Train only allowed dev_train;
use dev_eval for selection and reserve operator-owned unseen final families.
Do not train on official benchmark questions, choices, answers or held-out
document/OCR cases. Record author-LLM prior knowledge as an indirect prior.
Every exposed aggregate used to select architecture becomes development feedback.

## Historical trained-core scaling/cost proposal (inactive)

Proposed size bins: 30k, 300k, 3M unique slots, ±5% for the scaling arm (±1%
for the parent-matched first control). Freeze a single architecture family,
byte vocabulary and 256-byte training sequence before scaling; use three seeds
when calibration permits. Compare both fixed-data and fixed-compute slices;
equal epochs alone confound larger-model compute. No trained weights exist for
these bins yet and no loss curve can be extrapolated from the 387-byte excerpt.
Report micro-average bits/byte including a fixed EOS policy, document-cluster
uncertainty, total/active parameters, effective context, byte exposure, native
training steps, FLOPs estimates and measured elapsed/peak RSS separately.

[Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) report empirical loss
relationships with parameters, data and compute. [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556)
find balanced model/data scaling in their much larger trained-Transformer
regime. Neither validates a byte-level tiny-model fit or predicts AGI here.
[Gu and Dao (2023)](https://arxiv.org/abs/2312.00752) introduce selective SSM
sequence modeling; its efficiency claims do not certify our unbuilt IR export.
These primary papers were checked September 29, 2026; do not transplant their
token counts or optimal ratios directly to bytes.

Cost model `cpu-calibration/1`: existing server only, two CPUs/4 GB RAM;
no funded GPU assumed. Launch at most one bounded nice/thread-limited training
control after coordinator checks existing jobs. Use streaming minibatches rather
than materializing corpus one-hot tensors; reserve memory for other jobs.
Float32 weights+gradients+two Adam moments alone cost about `16*P` bytes
(0.48/4.8/48 MB at the three bins), excluding activations, state, optimizer copies,
runtime/IR export and interpreter arrays. This is a lower bound, not an RSS claim.
Float64 IR weights alone cost `8*P`; expansion/sharing overhead must be measured.

Before any size sweep, run ≤120-second calibration on answer-free development
text and record hardware, library versions, threads, batch/sequence length,
training bytes/second `r`, peak RSS, startup/export/evaluation time. For byte
exposure `D`, estimated time is `D/r + overhead`; price is
`hours * authorized hourly rate + storage/network`. No hourly price is asserted
or new capacity purchased. Conservative next-step feasibility uses the slowest
measured repeated throughput, not vendor peak FLOPs. Approximate dense training
compute `6*P*D` is only a bookkeeping heuristic; attention, SSM, recurrent and
small CPU overheads need actual profiling. Do not publish hypothetical dollar
costs or guarantees from unmeasured CPU throughput.


