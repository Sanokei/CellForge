# ARC Prize 2026: bounded offline entry plan

Verified September 29–30, 2026. This is a design and readiness plan, not a competition submission or a score prediction. No new external weights/data downloads or paid calls were made by these experiments. The active language evolution and four-arm controls remain a separate line.

## Current owner decision and measured feasibility

Handoff action check (September 30): `notes/MESSAGE_FOR_CODEX.md` has been
read in full, including the later maker-repair and Red Queen updates. The
primary artifact remains an authored CellForge graph; the maker may write
numeric weights, cells and wiring offline, without any gradient update.
Every proposed edit is charged, including invalid and neutral edits. The
maker's own parameters remain fixed.

The next decisive experiment is iterative breed-then-repair with the same
edit vocabulary and evaluation budget for breeding only, random repair and
maker repair. Acceptance requires fresh-generator gain, no per-task archive
or real-training-anchor regression, bounded resource use, and reported
demo-consistent false positives. Existing narrow synthetic repairs and new
gene admissions do not yet establish a maker advantage or native ARC gain.
Use training-only failure traces to author separate tested cells and dense
generators for object correspondence, conditional recoloring, continuation,
counting, paths and containment/adjacency. Do not revisit public evaluation.

Action-status check, September 30: reread the owner's message and independently
recounted all five saved baseline receipts without running a solver. The counts
and hashes below agree with those receipts. The no-gradient decision is binding;
the language-evolution entry point rejects known multiplicative/Hedge mixer
artifacts. Continue authored graph genes and original dense generators, with
equal-evaluation mutation/random controls and explicit false-positive reporting.
Claude's source, notes and receipts remain read-only. Public evaluation stays
exposed and excluded from further development; a valid Kaggle entry remains
unfinished, and no training score establishes hidden-set transfer.

Latest completed authored-gene measurement: separate factor-2 and factor-3
scaling graphs solve 2/789 and 1/789 expanded public-training development tasks,
respectively, with no demo-consistent wrong queries. Each has31 stored parameters;
native throughput is403.23/393.26 tasks/sec, whole-run peak RSS74.26 MiB.
Both solve300/300 original examples in their own synthetic family. The three
native successes were targeted from development demonstrations and lie outside
DEV200: this is coverage confirmation, not independent transfer or a new combined
portfolio score. See [scaling evidence](../runs/grid-repeat-evidence-v1/REPORT.md).

A separate graph-owned selector now preserves that three-task scaling union:
3/789 versus0/789 with demonstration adaptation disabled, and300/300 versus
100/300 mixed synthetic tasks. It passes ladder-micro/1 at64 parameters/state3;
native inference182.42 tasks/sec, whole-run RSS69.23 MiB. It does not yet include
the earlier object/color-association portfolio or establish breeding gains.
The first open-IR implementation's hierarchy failure and receipts are retained.
See [selector ablation](../runs/grid-scaling-selector-evidence-v2/REPORT.md).

The typed breeding bank additionally admits the two scaling genes: seven genes,
49 valid two-slot combinations,268 stored parameters/state1010. An actual
homologous crossover child preserves factor-2 and factor-3 parent behaviors in
compiled fixtures. This is construction validation, with no new native score or
population-efficacy measurement. See [bank inventory report](../runs/arc-grid-genes/scaling-two-slot-v2/REPORT.md).

The subsequent seven-gene population pilot found no advantage: crossover12/24,
mutation12/24, random14/24 synthetic report tasks, each six physical evaluations
and420 calls. All four crossover donor draws copied one parent, so this tiny run
did not evaluate complementary slot assembly. Ten report families were reused;
only scaling examples were newly seeded. No native query scoring occurred.
See [negative matched pilot](../runs/grid-scaling-population-matched-v1/REPORT.md).

With separate parent/proposal streams and16 predeclared proposals, crossover
assembled repeat2+repeat3, but all three arms tied14/24 on the same exposed
curriculum. Each used18 physical evaluations/1,260 calls; CPU costs differed.
This combined protocol change is not an isolated stream ablation or independent
transfer. See [larger matched study](../runs/grid-scaling-independent16-v1/REPORT.md).

An exhaustive offline diagnostic of all49 valid two-slot layouts found a best
single score14/24 on newly seeded original examples; the demo-selected graph
also scores14/24. Their query-oracle union is22/24, never a deployable model
score. This supports investigating slot/wiring capacity rather than increasing
search on the same tiny space. No native data were loaded. Half the scaling
fixtures are oversized-query fallback cases, limiting that family's evidence.
See [capacity diagnosis](../runs/grid-bank-capacity-v1/REPORT.md).

A seven-mutable-slot graph now produces110/120 original synthetic query solves
versus70/120 when only two specialist slots are active. Both have identical273
parameter values/state1515 and pass the hierarchy profile; active compute differs.
Expanded14.44 tasks/sec versus44.31 control, whole-run RSS100.65 MiB. All scaling
examples perform actual enlargement, with no fallback cases. Tiling remains0/10.
This is authored capacity utility, not evolutionary or native ARC transfer; the
old population runner still needs expanded-genome support.
See [active-slot ablation](../runs/grid-expanded-capacity-v1/REPORT.md).

Seven-slot homologous crossover, discrete mutation and canonical validation are
now implemented, with complementary frozen parents and four actual children.
Tests verify inheritance, lineage, neutral identity and contract rejection.
No offspring fitness/throughput was measured yet; connecting these operators to
matched population selection remains the next step.
See [expanded breeding inventory](../runs/arc-grid-genes/expanded-breeding-v2/REPORT.md).

Expanded population integration is now measured on original synthetic data:
seed772201 gives crossover22/24, mutation22/24, random20/24, each18 evaluations
and1,260 calls. Mutation ties crossover; one two-task gap is not general
superiority. Evolutionary winners run at about12 tasks/sec versus37 for random's
less complete winner. No native files were accessed. Replications are required
before promotion or scaling. See [matched seven-slot study](../runs/grid-expanded-population-v1/REPORT.md).

The September 30 instruction in [MESSAGE_FOR_CODEX.md](../notes/MESSAGE_FOR_CODEX.md)
supersedes the earlier pending mixer ruling: **no gradients of any kind**. Use
LLM-written original generators, hand-authored cells and discrete graph breeding.
The multiplicative-update/Hedge mixer is excluded from every primary arm; it is
no longer awaiting approval in this plan. Backpropagation, fitness-weighted gradient
estimators and optional test-time gradient adaptation are not authorized. Static
count-fitted language seeds retain their separate fitted provenance and earlier
dev_train-only permission; they do not silently become ARC cells.

The read-only Claude baseline has 57 authored grid cells and shallow composition
to depth at most two using a Python enumerator, with demonstration verification
and two ranked attempts. It is an external reference, **not a CellForge graph**:

| Receipt | Tasks solved | Tasks with a demo-consistent candidate | Demo-consistent but wrong on query |
|---|---:|---:|---:|
| Public training | 53/1,000 (5.3%) | 65/1,000 | 12/65 (18.5% of consistent tasks; 1.2% of all tasks) |
| Original public evaluation | 0/120 | 0/120 | Conditional rate unavailable: denominator zero |
| Training with fitted cells, v2 | 80/1,000 (8.0%) | 92/1,000 | 12/92 (13.0%; 1.2% of all tasks) |
| Training with fitted cells, v3 | 95/1,000 (9.5%) | 107/1,000 | 12/107 (11.2%; 1.2% of all tasks) |
| Improved public evaluation, v3 | 0/120 | 0/120 | Conditional rate unavailable: denominator zero |
| Training with fitted cells, v4 | 97/1,000 (9.7%) | 111/1,000 | 14/111 (12.6%; 1.4% of all tasks) |

Counts were recomputed from the read-only [training receipt](../research/claude_arc/results_training_d2.json)
and [evaluation receipt](../research/claude_arc/results_evaluation_d2.json).
SHA256: training `9cefd58f0bb4e8e94ea26e53b15d781d5b4d0c67674f77c45bb78c07487f108a`;
evaluation `eb4ec6bd15ed5efcbea531f7405d0784a5510752d8d813a2d39e1d7d0dae400b`.
The message reports roughly 0.5 seconds/task with three workers. Stored row times
average 1.527 seconds for training and 1.805 seconds for evaluation; these are
worker-level recorded durations, not our two-CPU device calibration. Peak RSS
and the exact source snapshot that generated the original `_d2` files are missing.
Current `solve.py` has additional local-rule fitting and changed output filenames,
so its present hash cannot certify the historical recipe. Keep all Claude files
read-only and do not quietly relabel these receipts as reproduced or matched
CellForge results.

The later message update is verified against the read-only v2/v3 receipts:
[training v2](../research/claude_arc/results_training_d2_v2.json),
[training v3](../research/claude_arc/results_training_d2_v3.json) and
[evaluation v3](../research/claude_arc/results_evaluation_d2_v3.json).
Their SHA256 values are respectively
`c8b98613ba3201048b98ecb2b33f55414abcfe4aa319e8df445598e37cd7b7d4`,
`6d0b5cc68870d00eba3ac6fb30365ea75771fb992edc0096ec3eb0d1d0485c4d` and
`b286f954e63c397a730976a71774ce277957c9c249583b48b310bcf3754784fe`.
These later variants include task-fitted count tables; they are not purely
hand-authored static cells. Training gains **53 → 80 → 95 did not transfer**.
The evaluation set has consequently been revisited by the co-researcher and is
exposed; "scored once" describes the original receipt, not the library's full history.
No solver was rerun to verify these aggregate counts.

The v4 training receipt was also recounted directly:
[training v4](../research/claude_arc/results_training_d2_v4.json), SHA256
`d2a8e539be4433c008881d62025a884b10d319c514cb04178393832acb272df1`.
It contains97 solved and14 demo-consistent wrong-query tasks. The handoff
reports continued0/120 evaluation; that later-stage assertion is not a newly
reproduced evaluation result here. No evaluation tasks or solver were loaded.

The previous "low single digits through 10%" development aspiration is withdrawn.
Training success did not transfer to the 120 evaluation tasks. On this evidence,
prize-winning odds are very low; no numeric odds or private-score prediction is
justified. The realistic near-term deliverables are a valid open CPU submission
and a Paper Prize entry with reproducible controls and negative evidence.
These deliverables do not replace the thread's broader superiority objective.

Public evaluation is now **exposed for this library** and must not be used for
cell/generator tuning or advertised as an untouched final split. Develop against
public training and original generator samples. Preserve the CellForge operator's
existing mod-five audit exclusion, while acknowledging that a future combined
library drawn from the fully evaluated 1,000-task training baseline cannot treat
that historical audit as independently unexposed. Freeze new artifacts/protocols
before any genuinely independent final access and score that split rarely.
Local jobs are limited to at most three workers; the working server stays at two
CPUs/4 GB RAM, with bounded nice-15 single-thread jobs.

## Verified deadlines, tracks and rules

The [official overview](https://arcprize.org/competitions/2026) lists November 2 for submissions, November 8 for papers and December 4 for results. The September 30 ARC-AGI-3 milestone is not a realistic target for a new implementation. September 30 through November 2 spans 34 calendar dates (33 elapsed days). Both [Kaggle ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) and [Kaggle ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) list October 26 entry/team-merger deadlines and 23:59 UTC deadlines unless specified otherwise.

| Track | Official advertised pool | Verified requirement |
|---|---:|---|
| [ARC-AGI-2](https://arcprize.org/competitions/2026/arc-agi-2) | $700,000 | Static grid reasoning; exactly two output attempts per test input; exact matching. |
| [ARC-AGI-3](https://arcprize.org/competitions/2026/arc-agi-3) | $850,000 | Interactive environments; exploration, modeling, goals and planning. |
| [Paper Prize](https://arcprize.org/competitions/2026/paper) | $450,000 | A corresponding working Kaggle entry is required; high leaderboard performance is not required for paper eligibility. |

Pools are not expected winnings. Prize distribution details differ between some ARC overview and Kaggle descriptions; confirm the governing track rules before claiming eligibility or a particular award.

The [general rules](https://arcprize.org/competitions/2026) require reproducible open solutions, release before official private scores, and submitter-authored code/methods under public-domain-style permissive terms such as CC0 or MIT-0. Third-party dependencies must permit public sharing. Kaggle evaluation has no internet, so inference must use locally attached publicly available artifacts, not remote model APIs. The Kaggle pages explicitly allow public external data, including pretrained models. This does not establish eligibility of every model/license: verify each pinned dependency and weight license.

## Compute facts and unresolved limits

These are primary-source search-indexed Kaggle overview statements; direct rules-page fetches were unavailable. Recheck the current notebook settings and governing rules before execution.

| Item | Evidence/status |
|---|---|
| ARC-AGI-2 runtime | CPU/GPU notebooks at most 12 hours; overview advertises L4x4, 96 GB aggregate GPU memory, restricted to this competition and no internet. [Kaggle source](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2). |
| ARC-AGI-3 runtime | CPU/GPU notebooks at most 9 hours; overview advertises RTX 6000 pool, machine type g4-standard-48, competition-only and no internet. [Kaggle source](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3). |
| Account allocation, weekly quota and host RAM | UNCONFIRMED. Advertised pool access is not a reservation for this owner. |
| ARC-AGI-2 private-set size | UNCONFIRMED for 2026; do not import a 2025 count. |
| ARC-AGI-3 action ceilings and exact score computation | UNCONFIRMED pending current SDK/methodology and competition-rule verification. No invented action allowance. |
| Paper formatting/word limits | UNCONFIRMED from governing Kaggle rules; do not copy third-party claims. |

Reported existing resources are local 10 GB RAM, no NVIDIA GPU, 3.5 GB free disk; server two CPUs/4 GB RAM. Those resources do not establish feasible pretrained inference. E4B's official model file is 16 GB; a theoretical four-bit 8B-weight payload is about 4 GB before overhead. Gemma 4 12B is 11.95B parameters, about 5.98 GB theoretical four-bit storage; actual quantized artifact size is unverified. See [SYSTEM12_DESIGN.md](SYSTEM12_DESIGN.md) and the [official E4B file](https://huggingface.co/google/gemma-4-E4B/blob/main/model.safetensors), [Gemma card](https://huggingface.co/google/gemma-4-31B). Kaggle owner-run jobs are a proposed deployment route, not an already available allocation. No download is authorized by this plan.

## Existing CellForge evidence

All numbers below are public-training development measurements, not competition-private scores. The associative run uses task-ID SHA256 modulo 5: zero reserved for audit, nonzero development, first 200 in hash order. Historical receipts state audit/public evaluation were untouched at that measurement; this plan does not independently re-audit subsequent access.

| Artifact/control | Recorded result | Frozen model hash |
|---|---|---|
| grid20 and fixed router | Both 0/200; zero invalid outputs | 5a5e8e210bc56159bf9b70212c9aced604ef6cae6494564c577c5075436cbf2e |
| Associative grid MoE | 3/200 tasks; 3/222 test inputs; zero invalid | 4f0a5ab4070c289d1b9b8c3b192efa4fa7e30cef2e295e1e58356a304661987b |
| Identity-color associative control | 0/200 | 458cd0eaabe1caba2fc02b8a6c73369498e2448d6bea29c09933900d498bf8f7 |

Receipts: [grid20 decision](../runs/grid20/DECISION.json), [associative decision](../runs/grid-associative/DECISION.json), [associative freeze](../runs/grid-associative/FREEZE.json), and respective development-results.jsonl files. Decision SHA256s: grid20 `d6febef67efd837a146cd9566a14a31dffbc55eea9349913f2343e284fbec73b`; associative `e4064e358cd0081d6bc8a080892d15fb5adc15120af2c3f8b8103171eda9944d`. The associative selected manifest hash is `4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4`. These narrow results justify a baseline, not an AGI claim or estimated winning probability.

## Pipeline A: authored CellForge graphs and offline breeding

Primary deployment budget is the existing two CPUs/4 GB RAM, no GPU. The solver is a small authored CellForge graph containing cells, bundles and wiring. Start the gene pool with the existing geometry20 and associative8 families, then author small object, symmetry, tiling, counting, gravity, flood, crop and selection cells as coverage gaps justify them. These prospective genes are not implemented capability claims. The graph itself owns routing, state updates, demonstration-conditioned reconstruction and task adaptation. The host transports grids, interprets the graph and measures results; it does not enumerate Python solutions or select a task's transformation using an external answer-solving algorithm.

Breed compatible cells/bundles using `import_def`, preserving explicit input/output/state contracts and namespace lineage. Mutate discrete graph wiring, authored constants and bounded recipe choices. Probe shapes/state/finite outputs before admission. Archive every parent, child, donor and failure with hashes, reproduction seeds and development scores. Evaluate each gene and each bred pair on the same 200 fixed nonzero-modulo-5 development tasks; select parents only from development evidence. Keep zero-modulo-5 audit tasks sealed until a selected graph and evaluation protocol are frozen. Compare archive breeding against matched random proposals and a no-crossover arm with equal development-evaluation budgets.

No CMA-ES/ES or gradient estimator is used in this primary arm. A discrete nondominated archive may retain candidates that trade accuracy against parameter count, graph size or CPU cost, with a predeclared ranking/selection policy. Novelty and archive retention do not establish capability. The earlier Python/DSL program-enumerator proposal was superseded by the owner's graph-first correction; no enumeration run was launched.

Use hard per-task CPU time, adaptation-step, graph-depth, node-count, array-size and process-memory limits. Record unique and expanded parameters, nonzero slots, definition/edge counts, graph bytes, CPU seconds per task, complete tasks per second, peak RSS, invalid outputs, timeouts and end-to-end notebook time as first-class metrics. Reject experiments that threaten existing language/count-parity jobs. First target is a valid internet-disabled Kaggle notebook around October 12; final target is November 2. No positive private-score target follows from current evidence. Package and measure against the 12-hour ARC-AGI-2 cap with a margin; packaging itself has not yet been validated on Kaggle.

False positives are first-class: record the number of tasks with demo-consistent
candidates, those solved on query inputs, and those demo-consistent but wrong.
Report both the conditional false-positive rate and its whole-task denominator.
A zero consistency denominator yields an unavailable conditional rate, not 0%.
Reconstruction of two or three demonstrations is not sufficient selection evidence;
dense original families provide hundreds of additional examples. Current
CellForge selectors use online association-conflict reconstruction proxies, not
a complete retrospective verification of every demo under final state, so do
not equate a zero proxy error with an exact demo-consistent candidate.

Prioritize authored genes and generators for object correspondence, conditional
recoloring, pattern continuation, counting-to-output, line/path drawing, and
containment/adjacency. Start from public-training concepts and validate each
original family on hundreds of independently seeded examples. If porting Claude's
read-only cells, implement each as a separate tested CellForge graph gene with
explicit donor/source provenance. Do not import its Python enumerator as the
primary runtime or embed query-answer lookups. Preserve the real-evolution
ablation plan and dense generator competition; simple synthetic family saturation
alone does not establish coverage of evaluation-style problems.

### Graph-owned two-output inference

For each task, transport only supplied demonstration pairs and test inputs. The graph reconstructs demonstration outputs through its authored cells and maintains bounded task-specific inference state. Graph-owned routing and confidence rank at most two output grids per test input. Inference-time demonstration reconstruction is permitted available evidence, not a benchmark score. Hidden test answers never enter the graph, adaptation state or host selection. Reset task state between tasks and validate ordering/reset invariance with synthetic fixtures.

Emit `submission.json` containing both `attempt_1` and `attempt_2` for every test input, original task IDs and original input ordering. Invalid outputs, exhausted adaptation budgets and timeouts receive a predeclared deterministic fallback grid and are counted as failures. The host performs format validation and serialization, not transformation inference. Freeze graph/interpreter, adaptation cap, routing/ranking, timeout, seed and fallback before independent audit. Symbolic runtime budget: graph startup + sum over tasks of (bounded demonstration-state updates + two output computations) + validation/serialization. Measure this budget on the intended CPU deployment; no model-size-based speed estimate is substituted.

### Optional pretrained side track

NanoJev's nominal 0.6B backbone and Gemma 4 E2B remain separate design-only comparisons, conditional on CPU compatibility, disk fit, peak RAM and latency measurements. NanoJev's game heads do not establish ARC skill. E2B's official text GGUF alone is 3.35 GB before runtime storage, so current 3.5 GB free disk is inadequate for a comfortable install. E4B/12B are possible future resource-dependent comparisons. No model download follows from this plan, and pretrained components do not replace the primary authored graph or supply its genes/donors.

## Former Pipeline B: test-time gradients rejected

The earlier conditional proposal is inactive under the owner's no-gradients-of-any-kind
decision. No test-time gradient experiment is authorized in either primary or
optional tracks. Historical pretrained references remain labeled as such; they
do not supply primary genes or change this decision.

### Small-model references are not CPU projections

The [TRM paper](https://arxiv.org/abs/2510.04871) reports a trained 7M-parameter network with 8% ARC-AGI-2 accuracy. The [official 2025 analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis) reports CompressARC at 76K parameters, roughly 4% on ARC-AGI-2 and 20–34% on ARC-AGI-1; its example task processing took about 20 minutes on an RTX 4070. The [author repository](https://github.com/iliao2345/CompressARC) explicitly uses GPU test-time training. These are learned/gradient-based reference results, not measurements or proposed performance of this authored CPU graph. The same official analysis records 2025's top competition result at 24% for $0.20/task. Different datasets, costs and methods prevent direct extrapolation to our entry.

## ARC-AGI-3: bounded world-model hypothesis

Use only observed frames, legal actions and documented feedback. The host records a bounded transition log; authored graph cells maintain object/state hypotheses, retrodict earlier frame transitions and choose bounded plans or informative actions. Implement these hypotheses in explicit cells/wiring rather than an external host planner. Penalize contradictions and retain alternatives under a frozen offline compute/action budget. Reset episode-specific state according to verified rules; persist only what the SDK explicitly allows. Distinguish logged evidence from inferred goals. Test reset correctness, legal actions, deterministic replay, memory limits, action count and score using official permitted environments before any competition claim. This is a hypothesis, not an existing agent.

## Milestones and paper evidence

| Dates | Deliverable/gate |
|---|---|
| September 30–October 4 | Verify governing rules, owner account eligibility/compute, artifact licensing; freeze split and baseline protocol. No rush submission for September 30. |
| October 5–11 | Author/freeze graph gene pool, import_def breeding and graph-owned adaptation; measure parameter/graph size, CPU time and peak RSS. |
| October 12–18 | First valid Kaggle notebook target around October 12; matched development ablations, failures/costs/random controls retained. ARC-AGI-2 CPU entry is primary. |
| October 19–25 | Freeze chosen recipe; audit once; package attached artifacts and internet-disabled notebook. Complete entry requirements before October 26. |
| October 26–November 1 | Reproducibility and deadline margin; check output completeness/runtime. No private-score tuning masquerading as independent evaluation. |
| November 2 | Owner-reviewed final code submission by verified deadline. |
| November 3–8 | Paper linked to working entry, release source/artifacts, document limitations and finish by paper deadline. |

The [paper track](https://arcprize.org/competitions/2026/paper) evaluates accuracy, universality, progress, theory, completeness and novelty. Preserve negative evidence, actual controls, lineage, data exposure and measured runtime. Keep existing public-training scores in internal baseline provenance; the official paper guidance says not to report training-set performance in the submitted results. Distinguish from-scratch, count-fitted and pretrained capability. A low score can still support a useful methodological paper, but eligibility is not an assurance of an award. No fabricated odds, new benchmark results or win claims are made here.


## Coexisting population mechanisms — proposed ablations

This proposal uses a living population, not only a historical archive. Start with
four coexisting graph individuals for three generations on the existing two-CPU,
4GB server; carrying-capacity alternatives4/8 require measured throughput/RSS
before launch. An archive retains lineage and failed/novel validated children,
but archived graphs are not simultaneously active organisms. These settings are
proposed, not a frozen run or resource-fit result.

| Mechanism | Concrete graph/population operation | Separate ablation against identical-seed mutation-only control |
|---|---|---|
| Mortality and niches | Track age; predeclare finite maximum age even for elites, replacement order and minimum niche representation. Per-development-task shared reward is1/nsolvers for each solver, zero for unsolved tasks; nsolvers counts current population only. | Mortality/sharing off; preserve equal proposal attempts and scoring access. |
| Contemporary sexual crossover | Select two currently living parents; homologous typed fixed slots exchange compatible graph cells/parameters while preserving state/interface invariants. | Mutation-only reproduction with same founder set and budget. |
| Horizontal transfer | A paired recombination event imports a bounded compatible module and replaces/prunes a recipient slot, with both parent hashes recorded. | No transfer; do not accumulate every donor forever. |
| Symbiogenesis and group selection | Name a merged child with explicit interface evolution and a prune rule; validate the joint group before selection, then measure its independent resource cost and task coverage. | Separate individuals/no merge, same evaluation budget. Prior MNIST child examples do not establish this mechanism on ARC. |
| Environment coevolution | Maintain bounded synthetic environment descriptors generated by transformations of allowed demonstration grids; target a predeclared difficulty band and diversity bins inspired by POET/MAP-Elites. | Fixed synthetic environments. Generators produce tasks, never Python answer solvers; synthetic scores cannot inflate native ARC scores. |
| Baldwin-style state adaptation | Fitness is measured after graph-owned demonstration-state updates, frozen on query; adaptation state resets by task and is not inherited as hidden answer data. | Plasticity disabled with identical genetic seeds and calls. |
| Evolvability credit | Record parent-to-child improvements and validated novelty over a fixed offspring window; select using predeclared offspring-credit rules in addition to direct performance. | Direct performance selection only; account for all extra offspring evaluations. |

Test one mechanism at a time. Match evaluated proposal count, invalid/no-change
attempt accounting, task access, seeds, scoring rules and CPU budgets. Report
any differing compute instead of hiding it behind equal generations. Do not
claim a combined seven-mechanism system was built or validated from these plans.
The native residue-zero audit remains outside maker input and synthetic-generator
selection; its historical reserved status is not a fresh access audit.

The completed baseline work is28 single graph genes, all pair coverage
upper bounds derived from aggregate/per-task development receipts on the exposed
200-task set, followed by three materialized pairs actually executed on those
same tasks. Pair-union bounds are not measured merged-graph performance. No
new competition, frontier, CPU-throughput or general-intelligence result follows.

## Completed graph coverage and separate reported scratch NCA experiments

`runs/arc-gene-coverage-v1/result.json` now records28 single graph genes on the
same exposed200development tasks. Three selected materialized pairs each solve
3/200, approximately51tasks/s and0.0195CPU seconds/task; whole-process peak RSS
87,384,064bytes is a conservative high-water bound, not incremental model RAM.
No breeding gain is demonstrated. Pair selection used exhaustive per-input union
upper bounds; those bounds are not scored graph children. These runs record zero
audit evaluations. Historical audit reservation is not a new access verification.

`runs/nca-gene/server-identity-t2-v1.json` records the CellForge1488parameter/T2
graph's30×30 synthetic identity probe:5/5 correct,93.23adapter calls/s,
0.0107seconds/call, peak RSS38,268,928bytes. Each adapter call contains demo/query
phases. This is a server execution measurement, not an ARC score or evolutionary
NCA result.

The following are **co-researcher-reported scratch results**, relayed by the
owner, not reproduced CellForge graph scores. Preserve Claude's research files
and scripts; do not merge throughput figures across implementations.

- Run1:1488parameters,16channels,8hidden,T10; Gaussian mutations with discrete
  mu+lambda selection,3000evaluations/task on first40same-shape ARC tasks smaller
  than14. Evolution0/40train/test, random0, identity0. On a synthetic6×6 swap,
  selected loss0.137→0.047 versus random0.137, but no solve after6000evaluations.
- Run2:414parameters,10channels,4hidden,T3; synthetic8000–14000evaluations took
  7–11seconds and fit training; held-out success1/3configurations. The reported
  ~1200evaluations/s applies to this scratch6×6 workload, not our graph/server.
- The later real40-task/8000evaluation run completed with 0/40 training and query
  solves for evolution, random perturbation and identity. Lower pixel loss did
  not become task success. The 313-task parallel follow-up stopped at 30 tasks,
  also with zero solves; it is not an unfinished live experiment.

Use this negative evidence to prioritize authored seed choices, wiring and small
shallow/light-tuning experiments, retaining equal-evaluation random/mutation
controls. Gaussian random perturbations followed by discrete selection are not
an estimated gradient when no fitness-weighted direction is computed. Cheap
synthetic fits alone do not establish ARC generalization or frontier capability.

### Small shallow CellForge follow-up, measured runtime only

The separate `construction/nca_shallow.py` artifact has414 stored parameters,
22microcells and3 shared local updates. Its residual skip preserves all10colors
at the authored identity seed. This differs from Claude's direct scratch NCA;
it is not a replication. The inlined gene passes `ladder-micro/1`. Two identical
expert wrappers support graph dispatch; they do not add distinct capabilities.
Three focused tests passed, including compiled/reference parity and parameter
count; the combined shallow/legacy suite passed8tests.

`runs/nca-shallow-v1/server-identity.json` records5/5 synthetic30x30 identity
calls,69.12calls/sec (0.01447seconds/call), and42,491,904bytes whole-process peak
RSS on the existing server with nice15 and one BLAS thread. Each call includes
demo/query phases. This has fewer parameters but is slower than the older T2
probe; iteration count, graph structure and dispatch differ. No ARC fitness,
test-time learning, evolutionary improvement or generalization was measured.
Do not scale the old1488parameter raw-weight experiment based on synthetic
identity speed. Next selection should prioritize authored cell choices and
typed wiring/light tuning, with a matched equal-evaluation random control and
separate frozen-winner development query scoring.

## Distillation as task competition — proposed graph breeding curriculum

Use dense deterministic synthetic input/output generators for each declared
family, with hundreds of cases per family only after freezing exact counts,
seed hashes and generation cost. The table is a proposed initial allocation,
not a dataset already generated or an experiment launched.

| Family | Proposed development cases | Fixed validation cases/seeds | Report separately |
|---|---:|---:|---|
| Geometry and recoloring | 200 | 100 disjoint seeds | Exact tasks/inputs solved, pixels, graph parameters, CPU/RSS |
| Object segmentation and bounding crops | 200 | 100 disjoint seeds | Same measures, empty/ambiguous-object failures |
| Local propagation and fill | 200 | 100 disjoint seeds | Same measures, length/size transfer and iteration caps |
| Counting and repeated structure | 200 | 100 disjoint seeds | Same measures, unseen count/spacing cases |

Public ARC training demonstrations and maker-authored CellForge cells supply
teacher knowledge with explicit provenance. A maker-authored generator/cell may
become a seed only after independent consistency checks; generator correctness
is not solver success. Keep synthetic validation seeds fixed and outside maker
selection. Real development remains the existing200 exposed tasks, reported in
its own row with its actual200task/222input denominator. Synthetic successes
must never increase the native ARC solve count. Transfer to hidden ARC is unproven.

Maintain bounded environment populations in a predeclared difficulty band,
inspired by POET. Task generators generate I/O environments, not host Python
solvers or answer-routing branches. Graph-owned demonstration features feed a
fast selector choosing CellForge cells; validation must show observation use,
and typed output decisions remain inside the graph. Offline discrete GA logs
solver/task lineage, proposals, failures, selection and resource use. No gradients,
fitness-weighted directions, maker API calls or model downloads are authorized.

Individually compare fixed generators against coevolved generators and selector
against no-selector at equal proposal evaluations, same seeds, synthetic/native
access and compute ceilings. Freeze family definitions, exact case counts,
difficulty thresholds, validation seeds, real-task denominator and stop budgets
before execution. Report per-family generation counts, exact solve rates, graph
slots/instances, CPU time and whole-process RSS; no combined mechanism claim
before these ablations run.

Proposed release: choose MIT-0 for original generator code and CC0 for original
synthetic examples only after checking dependency and source provenance.
Third-party/public ARC data cannot inherit these proposed licenses automatically.
External-data and Kaggle-rule compatibility remains pending verification; this
plan does not establish legal/competition eligibility.

Co-researcher correction: the completed414parameter/T3 scratch run at8000
proposals per task on40real tasks solved0training and0test tasks, with random0
andidentity0. Lower pixel loss is not a solved task. The newly reported313-task
scratch run on ten cores was initially reported pending. The latest read-only
Claude note corrects it to stopped at30/313, with zero solves; this is a partial
co-researcher report outside our two-core budget, not a CellForge benchmark.
Preserve Claude's original records unchanged.

### Curriculum measurement and selector constraints

The root is preparing six supported original synthetic families plus two
challenge families; the earlier four-row table is historical proposed allocation,
not the implementation manifest. Freeze actual family names/counts/configuration
and selection/validation seed hashes before scoring. Check duplicate I/O and
configuration overlap as well as disjoint random seeds; report challenge failures
without narrowing the declared denominator after results.

Native per-family transfer rates require a separate frozen task-family mapping
derived from demonstration pairs only, before query scoring. Record overlap,
ambiguous/unclassified tasks and each family's actual denominator. An aggregate
global200-task score cannot establish any family transfer rate. Keep synthetic
and native denominators separate in every measurement table.

The completed [dense curriculum comparison](../runs/grid-curriculum-baseline-v1/REPORT.md)
tests authored demonstration-conditioned selectors against constant routing.
There are eight original families, each with 200 selection and 100 validation
tasks. The eight-gene selector solves all five represented geometry/color
families; adding the crop gene covers a sixth. Tiling and fill remain unsolved
in this portfolio. All four selector/control graphs solve the same 3/200 native
development tasks. Native throughput is 58.47–70.95 tasks/sec; whole-process
peak RSS is 168.3 MiB. No evolved or amortized routing advantage is demonstrated.
These synthetic and native denominators remain separate. Original synthetic
examples/code have CC0-1.0/MIT-0 release labels; competition eligibility remains
unverified, and third-party ARC examples do not inherit those licenses.

## September 30 authored-gene transfer check

The entries below preserve experiment history. Earlier launch-time phrases such
as "pending" or "no benchmark yet" are superseded by their subsequent completed
measurement entries; they are not current job status.

The frozen [gene evidence report](../runs/grid-gene-evidence-v1/REPORT.md)
adds exact full-demo consistency auditing, rather than relying on routing error.
Local-fill and count-to-strip graphs each use 11 unique parameter slots and solve
300/300 original examples in their own family (200 selection, 100 validation).
Both solve **0/200 native dev tasks**. Counting has one demo-consistent wrong
query: 1/1 conditional false positives, 1/200 overall. Fill's conditional rate is
undefined because no native task is demo-consistent. The four existing selector
portfolios all remain at 3/200; no native improvement is demonstrated.

Native inference throughput is about 369 tasks/s for fill, 403 for count and
55–71 for the selector portfolios. Whole-process peak RSS is 65,680 KiB;
total run wall time is 60.74 seconds. Extra demonstration audits are timed
separately and must not be represented as normal deployment inference.

A separately tested largest-component graph prototype adds four-neighbor
label propagation and dynamic-background object selection. It has 11 parameter
slots but 1,340 gene nodes and an 810,000-element membership tensor: compute
is substantial despite few parameters. Its 64-step cap uses identity fallback
when convergence fails. Its completed measurement is recorded below.

### Largest-component prototype measured, September30
Frozen runs/largest4-evidence-v1:100/100 original two-rectangle tasks,1/200nativeDEV,zero native demo-consistent tasks. The single query success is not validated demo-conditioned transfer; overlap with portfolio3/200 unknown. Native13.58tasks/s,0.07362CPU s/task,peakRSS139.74MiB;30x30 calibration14.43tasks/s. No publiceval/audit/final/gradient access. Source and five-entry verified lineage preserved; no combined breeding or frontier gain. See runs/largest4-evidence-v1/REPORT.md.

### Matched one-slot wiring pilot completed
See runs/grid-maker-slot-v2/REPORT.md. Maker and random seed662021 chose the same shared fill gene: both accepted, demo accuracy0.75→0.875, fresh synthetic600→700/800, native3/200 unchanged. Identical candidates are not independent efficacy evidence. PeakRSS75032KiB,wall172.02s; oneproposal perarm,2400demo-pairfitness evaluations each. No gradients/publiceval/audit/final access. First sourcebundle import failure preserved; no scoring before retry.

### Conditional adjacency gene and living-population mechanics
Frozen adjacency gene32slots:200/200selection+100/100originalvalidation,0/200nativeDEV,zero native demo-consistent candidates. Native397.73tasks/s,RSS44.06MiB. Report runs/adjacency-evidence-v1/REPORT.md; no transfergain. LivingPopulation helper adds carryingcapacity, sharedtaskreward, age/mortality, contemporaryparents and full evaluated archive. Twelve helper/gene tests pass; no population mechanism efficacy experiment or recombination benchmark yet. No gradients/publicevaluation/audit/final access.

Two homologous typed mutable slots are implemented in scripts/build_grid_two_slot_genome.py. All16four-allele combinations retain195parameter slots and1010state capacity with a shared authored bank. Crossover inherits eachslot from a parent; discrete one-slot mutation logs no-change events. Frozen parents/child and import_def lineage: runs/arc-grid-genes/two-slot-genome-v1/INVENTORY.json. Two profile/parity/lineage tests pass; no efficacy measurement yet. This supplies an actual graph recombination operator for the next living-population matched ablation, not proof that recombination helps.

### Living two-slot population matched pilot: no advantage
Completed runs/grid-population-pilot-v2/REPORT.md:4proposalattempts/arm,crossover/mutation/random all140/180freshsynthetic and3/200native. Shared195slots/state1010;9unique functionalgenomes; unequal cachehits3/0/2mean expensive-eval/CPUbudgetsnotmatched despite equalattempts. Mortality/sharing active butnot separatelyablated. Native filecontent manifest absentfrompilotpolicy; do notclaimmatchedexternaldataset. PeakRSS73804KiB,wall150.57s;29entryledger verified. No gradients/publicevaluation/audit/final access or crossover/frontiergain.

### Offline graph JSON interface and corrected population protocol
OfflineCPU submission CLI implemented/tested:2synthetictasks/3queries server smoke0.14686s,RSS46.35MiB,no Kaggleentry/accuracyproof. See runs/arc-cpu-package/README.md. Sourceclosure bundler includes transitive siblingPython dependencies/perfilehashes; remote --help importpreflightpassed. Corrected run_grid_population_matched.py validates canonicalgenotypes (neutralmetadata doesnot resetage), independentlyscores2founders+4proposals EACHarm,no selectioncache/all demo calls,pins nativecontenthash+disjointsynthetic manifests beforepolicy. Launched immutable grid-population-matched-v1,session16777; no result yet. Conditionalstatev2 frozen1fb246...2authoredrankcellparams/state1002; no benchmarkyet. No gradients/publiceval/audit/final access.

### Corrected matched population pilot completed
Allarms6physicalselectionevaluations/3120predictorcalls,18total,zero executionfailures. Crossover120/180synthetic vs mutation/random140/180; all3/200native. Canonicalidentity,age,nocache,frozen+recheckednativehash/disjointsynthetic commitments implemented. Single4proposalpilot doesnot establishgeneral crossover disadvantage; no gain demonstrated. CPUbudgetdifferent despite matchedcallcounts. PeakRSS70388KiB,wall228.63s;36ledgerentriesverified. REPORT runs/grid-population-matched-v1/REPORT.md. Conditional-state comparison adapt/control launchedsession34727, pending; no gradients/publiceval/audit/final access.

### Conditional state adaptation measured
Two authoredrankcellparams/state1002,graph-owned1000adjacencyrules scored fromdemo counts.300/300 variable-color synthetic vs0/300 no-updatecontrol; native1/200 vs0/200, BOTHzero complete-demo-consistent candidates. No validatednativetransfer.117.47vs115.02native tasks/s,49.32/49.27MiBprocessRSS. Both12.03sexternalwall;source/data/artifacthashes andsixentryledgerseach verified. Report runs/conditional-state-evidence-v1/REPORT.md. No gradients/publicevaluation/audit/final access, no AGI/frontierclaim.

### Training-only failure evidence and serial wiring prototype
Train-onlydiagnostic API rejectsquery/reportfields;DEV200hashpinned,no queryscore/modelselection. Associative9:2/200consistent,57observedshape mismatches,141pixel/routingfailures. Conditionalstate:0consistent,57shape,143pixel/routingfailures. These observations do notprove missingprimitives.17.014swall/RSS60.28MiB,4entryledgerverified. Report runs/grid-train-diagnostics-v1/REPORT.md. Serial flattypedgene composition implemented with intermediate/finaldimensionmasking;195slots/state1010 unchanged,2tests pass; nobenchmarkyet. Next boundedlocal3x3context-state prototype underway,noperformanceclaims.

### Serial composition measured against constituent controls
Authoredadjacency→fill child300/300 originalsynthetic vs0/300 eitherconstituent. ALLnative3/200 unchanged;2consistent successes,zerofalsepositives.195slots/state1010 each;child57slotnodes/1650expandedallbranchIRnodes. Native46.88tasks/sec vs52.04/50.40controls,about55.4MiBRSS;extraCPU/graphsizecostdisclosed. Sixentryledgerseachverified,source/family/DEVhashesfrozen. Report runs/grid-serial-evidence-v1/REPORT.md. Explicitauthoredcomposition, NOTevolutionarydiscovery orARCtransfergain. Localcontextmemoryv1 frozena5b44...23params,state3602,4tests; unbenchmarked. No gradients/publicevaluation/audit/final access.

### Current owner-action status and context-cell result

The no-gradient decision is active in the language evolution entry point as well
as this plan: known multiplicative/Hedge mixer artifacts are rejected. Historical
mixer code is retained with `primary_no_gradient_eligible: false`. No new gradient
experiment is authorized. Claude's files and evaluation receipts remain read-only.

The frozen context comparison is now complete; see
[context evidence](../runs/grid-context-evidence-v1/REPORT.md). Both graphs have
23 stored parameters and graph-owned demonstration adaptation. Exact 3x3 memory
solves 24/100 synthetic validation tasks; adding unanimous per-color backoff
solves 57/100. Both have **43 demo-consistent wrong queries** on that validation
set. Native DEV200 scores are 1/200 and 3/200 respectively, with two
demo-consistent wrong tasks in each arm. No improved native coverage over the
existing 3/200 portfolio is established. Native throughput is 10.30 and 11.04
tasks/sec; peak process RSS is 126.68 and 126.28 MiB. Extra demo auditing is
timed separately from inference. Neither arm accesses public evaluation, audit,
or final data. These are authored ablations, not evolutionary discoveries.

Next cell development should target the owner's named families using original
dense generators and training-only failure diagnostics. Every port remains a
separate tested gene with donor provenance. Validate candidates on disjoint
development batches, count maker edits as evaluations, and compare discrete
mutation/crossover against equal-evaluation controls. Report false positives
alongside exact query accuracy: memorizing demonstrations is insufficient.
Kaggle validation and an actual submission remain unfinished owner-run steps.

### Discrete context-mask pilot completed: no gain
runs/grid-context-mask-pilot-v2/REPORT.md:12 physical evaluations per crossover/mutation/random arm (4shared founders+8proposals), zero accepted proposals. All retained authored crossmask186; selection1/24, fresh report0/24 despite95.83%pixels. Selection evals/sec0.684/0.676/0.654; CPU17.205/17.633/18.125s; wall59.39s;RSS123.54MiB.41ledgerentries independently verified after retrieval, external exit0. Identical winners are not independent quality evidence; no nativeARC/publiceval/audit/final access. Paper source updated, compilation unverified. Next diagnostic: direction selection and boundary coverage; no claimed evolutionary or frontier gain.

### Authored direction router measured: synthetic fix, native0
Independent scalar diagnosis on exposed mask selection matchedgraph:97wrongunknownkeys,94border,3interior,zero conflicts; all requestedsource neighbors inside. New graph owns fivehypothesis demo-error routing (identity+four neighborcopies),two rankingparams/nineopconstants/state6/207edges. Fresh synthetic200/200selection+100/100validation vs0/300 no-update; BOTHnative0/200,zero consistent (conditionalFPundefined).142.17/147.06native tasks/sec,0.00704/0.00680CPU s/task,RSS55.48/55.64MiB,wall9.92/9.81s. Two localtests passed; sixentryledgerseach independentlyverified. Explicitmakerproposal,not evolutionarydiscovery/maker-versus-random proof; authorcostunmeasured. Frozen source+gene/control hashes in runs/direction-router-evidence-v1-launch.json, report in runs/direction-router-evidence-v1/REPORT.md. No gradients/publiceval/audit/final/download/paidcalls. Nativecoverage unchanged; broader frontierobjective unproven. Paper source updated, compilationunverified.

### Train-only structural census: no newly explained tasks
Added census_grid_train_structure.py stricttrain API and three focused tests (passed). Frozen v1/v2 on fixedDEV200 assertedmanifest; no query scores.57shapechangingdemo tasks; all seven plainrules matchzero tasks. Sharedcolorassociation allows cropzero2/cropmajority1/identity2, all overlapping same two taskIDs alreadyassociative9democonsistent; tile/repeat2/3 allzero. Python diagnostic only, not runtime enumerator/graph coverage, not proof absentgenes; objectselection/compositions untested. V2 wall0.35s,RSS38.11MiB,exit0; twoledgereach locallyverified. Frozen archive6794c3939d8cff73e4be303907b993bdac32e08795603212fd92f6b112ee7c48, report runs/grid-structure-census-v2/REPORT.md. Prioritize richerobject/relational genes over redundantstandalonegeometry basedthisevidence. No gradients/publiceval/audit/final/download/paidcalls. Paper source updated, compilationunverified; broadgoal unproven.

### Object-association comparison completed: +1 exposed DEV task, ~6x slower
Child300/300originalsynthetic vscontrol0/300; native4/200,4/222inputs vs3/200,3/222; demo-consistent3/0wrong vs2/0wrong. Child9.82tasks/sec/.10180CPU s/task/RSS150.95MiB vs58.37/.01713/61.27MiB.174params/589opconstants/3004edges vs163/41/840; state1010 both. Capacity/CPU notmatched. Controlv1 metadataKeyError before scoring preserved; safegetlimitations repaired unchangedgraph in freshv3 (source0daceedfd77226273f22f8035042e3debade535c771e8efd8dcb606ede52759f), four scorer tests prove optional per-task recording keeps counts/calls. Observer confirms bf699163 newly solved andrank0demo-consistent child, controlfails; aggregatechild lacksfullpairedtaskrecords so no proof allregressionsabsent. Ledger6+6+3 locallyverified. Retainarchive donor, do notpromotedefaultbasedweakgain/highcost. Report runs/grid-object-association-evidence-v1/REPORT.md. No gradients/publiceval/audit/final/download/paidcalls; broadgoalunproven. Paper source updated, compilationunverified.

### T16 object donor completed; paired-control +1/no losses
T16same300/300synthetic, native4/200(4/222),3consistent/0wrong.12.49tasks/sec/.08006CPU s/task/RSS109.32MiB,wall190.87s/exit0; ledger6locallyverified. T64aggregate4/200 at9.82/.10180/150.95MiB; T16~21%lowerCPU and~28%lowerRSS, no repeatconfidence. Same174params/state1010; fewerops/edges; longcomponents mayregress. Newpairedreceipt validator rejects missing/manifests/population mismatch, invalidqueryoutcomes/aggregatedisagreement, threefocusedtests pass. Per-taskpairedcontrol gainedbf699163,lostnone,sharedthreepriornativeIDs; full T16vsT64 pairing unavailableaggregateonly64. Sourcebundle transitiveattachedbuilderstestadded,total4pass. RetainT16lower-costobjectdonor,nopromotedefault (~4.7xslowerthancontrol), no evolutionclaim. Report runs/object-association-t16-v2/REPORT.md and PAIRED_CONTROL.json. No gradients/publiceval/audit/final/download/paidcalls. Paper source updated, compilationunverified; goal remainsunproven.

### Five-gene object bank and complementary recombination measured
Generalized typedgenome/mutation/canonicalvalidator fromhardcodedfour to frozen available_genes, rejectbankmismatch. ImportedT16objectgene withlineage; all25two-slotcombinations type/profilevalid at206params/state1010. Eightgenome/populationtests pass. Frozencomplementaryparents each5/10originalsynthetic, explicithomologouschild10/10, base0; consistency5/5/10wrong0. Child14.12tasks/sec/objectparent14.37, nonobject73.40; wall7.20s/RSS113188KiB/exit0. Fiveledgerentrieslocallyverified. Source95edf35d6691dba4c74059339a5ba16583cb7a0550bbeb552b2e36346cfbfb55, inventory runs/arc-grid-genes/object-two-slot-v1/INVENTORY.json; report runs/grid-object-recombination-probe-v1/REPORT.md. Explicitbuildingblockretention,NOTpopulationevolutionadvantage ornativequeryscore. Nextmatchedarms needobjectfamilyselectionpressure; oldgeometry-onlyfitnessinadequate. No gradients/benchmarkfiles/audit/final/download/paidcalls. Paperupdated,compileunverified; broadergoalunproven.

### Object-bank matched population completed
18physicalgraphselectioneval; eacharm6eval/348predictorcalls, equaltrue,failed0. Crossover14/20freshsynthetic,4/200native (4/222inputs),3consistent/0wrong; mutation12/20,3/200,2/0; random14/20,3/200,2/0. Differentniches:crossobject+adj vsrandadj+bbox; FAMILY_SCORES.json provesstoredrecordfamilybreakdown. PAIRED_ARMS.json:crossgainsbf699163,lostnone vbothcontrols. CPUselection18.785/9.704/13.099s; evals/sec.319/.618/.458; native11.59/46.94/47.15tasks/sec; equalcallsnotCPU. wall155.77s/RSS140.66MiB/exit0. Ledger36independentlyverified;all12proposalhashes+winnerhashes+canonicalgenotypeschecked. No reportreuse. Earlyretrievalbeforedonefailed; polledsamelivehandle, no restart. Single4proposal/singleseedpilot, notgeneralcrossoveradvantage; randomtieandCPUcostnegativeevidence retained. Source618d07153341d3b0c287c699f3535ccb6a84ea7fbf6219e5302417a4018cef80, report runs/grid-object-population-matched-v1/REPORT.md. No gradients/publiceval/audit/final/download/paidcalls. Paperupdated,compileunverified; originalfrontiersuperioritygoal unproven.

### Three search-seed replications completed: crossover advantage not robust
Allpredeclaredseeds772041/51/61 on SAMEexposed20synthetic tasks:cross12/12/12,mutation14/14/14,random14/12/14. Means12/14/13.33, originalfavorablepilotexcluded. Eacharm18physicaleval/1044calls,total54eval;failures0,no nativequerydecode/scoring. CPUselection38.675/54.880/45.086s. Externalwall65.81/40.47/56.17,RSS142892/139120/142088KiB,exits0. Summarizercheckedpolicies/corpus/base/budgets/winnerhashes/scorearrays/three36entryledgers. All12crossproposals10or12demo solves; noneobject+adj. Bothfoundersageevictedgen3eachseed; living1atgen3, extinctgen4two; cumulativeeventduplicatesnotnewdeaths. Purecrossreachabilitylivingparentalleles vsbank-wide mutation/randomdifferent; sharedbanknotreachablediversitymatch. Earlymortalitycausalhypothesisnotproved. No scalingbasedfirstgain; preserveallnegativeevidence. Report runs/grid-object-replication-v1/REPORT.md, SUMMARY/CROSSOVER/MORTALITYdiagnostics. No gradients/publiceval/audit/final/download/paidcalls. Paperupdated,compileunverified; frontiersuperioritygoalunproven.

### Finite lifetime paired ablation completed: no crossover repair
Age3→8 samepredeclared3seeds:cross12→12all;mutation14→14,14→12,14→14;random14→14,12→12,14→14. Allpolicyfields identicalexceptmax_age/source_sha256; sourcebundledifferenceonlyrunnerageflag. Allsix36entryledgers/winnerhashes/task+bankmanifests/budgetsverified. Eacharm18physicaleval/1044calls percondition,failures0. Age8CPUselection48.841/50.972/44.863vsage3 38.675/54.880/45.086. Externalage8wall65.70/43.15/64.68,RSS147080/140668/144448KiB/exits0. No nativequerydecode/scoring; SAMEexposed20synthetic reporttasks notnewvalidation. Founderpreservationwithin4proposalhorizon (age8exceedsduration) failedtoimprovecross; don'tkeepage-tuningtinycurriculum. Needrichercoverage/reachablealleles,nomechanismcauseproof. Report runs/grid-object-age8-v1/REPORT.md and PAIRED_AGE.json; compare_object_lifetimes.py failsonanyotherpolicychange. No gradients/publiceval/audit/final/download/paidcalls. Paperupdated,compileunverified. Broadfrontiergoalactive/unproven.

### Expanded training-only census: new scaling gene targets, no queryscore
Canvas4erase/retainrules0/200 and0/789; containmentonlyidentity/colorassociationoverlaps(twoDEV200/threeexpanded), no newcoverage. Added --extended-dev diagnosticpartition allnonzeroSHA256mod5 publictraining=789,211residuez ero excluded, DEV200anchorassertedbeforeexpansion; fixed200benchmarkunchanged.17rulesexactunion7, associationunion13,258shapechanges. NewoutsideDEV200 repeat2tasks c59eb873/60c09cac, repeat3task9172f3a0 alldemosexact; usefulauthoredscalingtarget,nextbuildtypedgene nohiddenqueries. Sixdiagnostictestspass,extended+canvas+containmenttwoentryledgersverified. Extendedwall5.68s/RSS53596KiB/exit0,diagnosticcostonly. Sourceaafd378d92d2148311bb9ba9d2adad338d6cf4c560b80dba5d3f924a49e5fc98, extendedmanifest9b82cdec59033c4fe67ebd2f8b93ea2a495b90ec9bff5a0e4a5c851cd4291af9. Report runs/grid-extended-train-census-v1/REPORT.md. No queryscore/gradients/publiceval/audit/final/download/paidcalls; futurecellslabelexpandeddevelopmentexposure. Paperupdated,compileunverified; goalunproven.

### Expanded seven-slot replication: completed three-seed result
Three predeclared independent search/corpus seeds, each with 18 physical selection evaluations and 1260 predictor calls per arm, produced crossover 110/110/100, mutation 110/110/100, and random search 100/100/100 on three distinct 120-example synthetic report batches. Aggregates: 320/360, 320/360, 300/360. This supports a small selection-over-random signal on authored families, with no crossover advantage and no ARC evaluation-transfer claim. Each arm used 54 evaluations/3780 calls overall; selection CPU seconds were 288.323, 292.825, 318.189, so equal calls do not imply equal CPU cost. All 324 ledger entries and source/corpus/winner commitments were verified. Total wall time 1302.46 s; maximum whole-process peak RSS 172268 KiB. Crossover proposals 14-16 in the third seed were ExtinctPopulation placeholders, not successful offspring; they consumed their evaluation allocation. No native ARC, public evaluation, audit or final files were loaded. All report arms had zero demo-consistent-but-wrong examples; tiling remained unsolved in all arms. These generators are authored and are not independent evidence of general reasoning.

Authoritative receipts: runs/grid-expanded-replications-v1-SUMMARY.json and the three runs/grid-expanded-replication-772211/772221/772231 directories. The first favorable 24-example pilot is excluded. The geometry memo prototype passed output/state/reset/counter parity tests but is not enabled or benchmarked; no speedup is claimed.

The maker interface now exposes breed_then_repair_round: a frozen bred child, random repair and maker repair each consume one attempt, including unchanged edits; development train-pair counterexamples log input, expected, predicted and cells fired. It performs no model-service calls. Fresh development, archive and real-anchor acceptance gates remain caller responsibilities; no measured maker-repair improvement is claimed. Six focused maker/cache tests pass. Gradients, backprop, gradient estimators and multiplicative-update mixers remain excluded from primary arms. Frontier superiority remains unproven.

# Frozen repair admission measurement

Completed server run, source archive a8782793546bf288dec95cab0a33429270958ff3b6f1fe20009e94b5a7f767b4. Nice15, numerical threads1, timeout600. Eleven ledger entries, candidate hashes, per-task aggregates, matched population records and recomputed admission decisions verified locally. Public-training DEV200 only; no modulo-five audit, public evaluation or final data.

| Arm | Fresh synthetic /24 | Earlier-family archive /24 | Real ARC training anchor /200 | Anchor tasks/s |
|---|---:|---:|---:|---:|
| Baseline bred child |14|14|3|39.05|
| Random repair |16|16|4|10.00|
| Manual maker repair |16|16|3|33.53|

Zero demo-consistent-but-wrong tasks in every section/arm, and zero per-task losses versus baseline. Both repairs pass the scored fresh-gain and regression gates. Both remain unadmitted because caps/fixture flags fail closed: independent resource/fixture admission evidence was not attached. Whole-run wall152.985s, peak RSS142045184 bytes (~135.46 MiB), not per-model incremental memory. Consistency-audit cost is separate from reported query throughput. These are purposively selected, previously exposed one-edit pilot candidates, not independent replication or a maker advantage. Random repair alone adds one native anchor solve, while maker ties baseline there; synthetic gains do not establish improved ARC transfer. Retain this negative comparative evidence. No frontier comparison or AGI claim.

# Tiling repair native development anchor

Frozen inactive-donor control vs tile repair: both3/200 public-training DEV anchor, gains0/losses0, wrong-consistent0. Manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4. Four ledger entries, completed payload and paired task-record aggregates verified locally. Nice15, threads1, timeout600, server runtime terminal0.

Query throughput36.33vs32.24tasks/s; CPU0.02739vs0.02985s/task. Whole-run wall47.531s, peak RSS91668480bytes. One paired run, not a confidence interval. Repair added10/120 authored synthetic solves in the earlier experiment but adds no native anchor solve here and costs more compute. Keep tiling as an available donor; do not promote it as an ARC performance improvement. No archive acceptance, crossover benefit, independent evaluation transfer or frontier claim. Source archive c38c4999dbeab0c171a0319cd17e7de910f7ea6c7268e4cf212a5d235970d791; no gradients/public evaluation/audit/final access/downloads/paid calls.

# Modal crop plus graph-owned association/routing

import_def donor4a783d75... into expanded base; same304stored parameters in control and repair, same1515state. Atomic replacement of duplicate identity slot8 retains the other authored genes and fixed geometry branches. Frozen control9a404f66... and repaird96daa3e...; canonical/profile valid. Two reference/compiled fixtures pass nonzero-background crop plus demonstration-conditioned recoloring and repeated-call reset.

Server DEV200: both4/200, gains0/losses0, wrong-consistent0. Query12.23vs11.39tasks/s, CPU0.08014vs0.08725s/task. Whole-run138.616s, RSS115818496bytes; nice15/threads1/timeout600, exit0. Four-entry ledger/completed payload/paired outcomes verified. Sourcea26781a9be7234f903a84ace0831482c46d5b6649b88be4057cc7dcf18812061; native manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4. Runner's legacy tiling protocol label reused; hashes/arm names identify modal-crop comparison.

Standalone0/200 did not predict the portfolio4/200, but modal activation adds no solve and increases CPU. Retain as optional authored donor; do not promote on synthetic fixtures. No evolution advantage, archive acceptance or independent frontier comparison. No gradient/public-evaluation/audit/final/download/paid access. Need next gene hypotheses driven by train-pair failures rather than continued modal tuning on this anchor.

### Expanded operator attribution from committed receipts (2026-09-30)
Reverified all three108-entry ledgers and completed payloads. Across48proposals/arm: crossover5strict running-best demonstration gains, mutation5, random3 (10.42%,10.42%,6.25%). Crossover28duplicate canonical genotypes, mutation6, random0; cross3extinction failures, others0. Each proposal stream consumed3360predictor calls, excluding founder evaluations. Proposal evaluation CPU265.400/269.434/294.713s. Detailed per-attempt operator, prior best, gains, hash and duplication in runs/grid-expanded-operator-attribution-v1.json; reproducible script attribute_expanded_operators.py.
These are strict selection-record improvements, not archive acceptance, offspring adaptation or fresh-validation gains. Existing completed report comparison remains320/360cross=mutation vs300/360random, with no crossover-specific advantage. Duplicate burden motivates a future validity-preserving diversity operator at equal total proposal/evaluation budget; do not skip proposals without accounting for them. No new task data or hidden splits read, no gradients/downloads/paid calls. Frontier superiority remains unproven.

# New-gene development admission

Candidateb96a430e... admitted as a development tiling niche donor, not promoted as default solver. Freshseed3317000:16/24vsparent14; earlier-family seed3327000:16/24vs14; DEV200anchor3/200vs3. Per-task losses0, wrong-consistent0. Typed boundary fixtures and declared resource caps passed in the separately verified graph-bound receipt. Eleven score-ledger entries, graph hashes, per-task counts, false-positive denominators and recomputed admission verified. Archive ledger references exact score/cap receipt hashes; one admission entry verified. Prior rejected maker-attempt receipt remains unchanged.

Contextual fixed-bank random repair scored17/24fresh,16/24archive,4/200anchor. It lacks this run's attached cap/fixture receipt; that missing evidence is not evidence it is worse or unsafe. Different allowed edits prevent a matched maker comparison, and donor knowledge already existed. New tiling capability transfers across synthetic seeds but not to new real anchor solves. Run133.964s/RSS142245888bytes; nice15/threads1/timeout600/exit0. Native manifest remains fixedDEV200. Source293c2e6e0e762f21f98f2fb75e658175a958fde709186102ba06561752a75c64.

The inventory's EXPOSURE.json retains discarded draft seeds3237000/3247000; executed policy3317000/3327000 is authoritative. v1 upload was never run. One transient status SSH request failed, but the original job handle reported terminal0 and receipts were retrieved without restarting. No gradient/public evaluation/audit/final/download/paid access. No independent frontier superiority or AGI claim. Full predeclared equal-vocabulary three-seed maker/random repair experiment remains outstanding.

# Authored numeric gate MoE native comparison

Imported frozen two-round numeric donor1d36f411... through atomic import_slot_gene into the shared expanded MoE. Control carries identical imported parameters but leaves slot8identity. Both store[310, 310]parameter counts and1515state; only activation differs. Canonical/profile checked before freeze. Frozen hashesf318eadf...control andc3670057...activation.

Server DEV200:4/200vs4/200, gains0/losses0, wrong-consistent0. Query12.286vs11.261tasks/s; CPU0.081391vs0.088783s/task. Whole-run138.536s/RSS115077120bytes; nice15/threads1/timeout600, terminal0. Four ledger entries, completed payload, exact task aggregates and paired outcomes verified. Manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4; sourcebb0aaf8561ad90ce19e240dec27346bf58b8cda30d44f169b51aea1c241543ed. Legacy runner protocol label tiling-hgt-native-anchor/1 reused; artifact hashes identify numeric gate experiment.

The authored weights transfer100/100within the synthetic horizontal-neighbor family, but do not add a native anchor solve and incur more CPU. Keep the reusable cell as an optional donor; no default promotion or evidence that feedback improved the maker. Full iterative equal-vocabulary controls and failure-derived real-task capability additions remain outstanding. No gradients/runtime LLM/evaluation/audit/final/download/paid access; independent frontier superiority unproven.

Marker-mirror native paired result September30: frozen active5/200 versus inactive4/200; solegain2bcee788, nolosses, demo-consistent wrongqueries0both. Gain targeted from that task's train-only failures; authoring saw noquerylabels, but this is development coverage, not independent generalization or causal maker-over-random evidence. Four ledgerentries/completedpayload/per-task counts/gains/losses verified; manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4. Serverthroughputinactive11.5215/active10.6435tasks/sec, CPU.086783/.093944seconds/task; wall147.118s, peakRSS121581568bytes, externalexit0. Sourcea9e36f226e80ac0ffa7c92cd2e47d2aca3279c749c949bba8f81106fc0434e5e. Receipts runs/marker-mirror-native-v1; graph295storedparams/state1515. Retain as optional authored donor pending fresh-family/archive/cap admission and equal-budget repair controls. No evaluation/audit/final/download/paid/gradient access; frontier superiority unproven.

Marker eight-gene matched smoke test September30: runner now optionally includes original marker_mirror13thfamily; generator extracted to reference/marker_mirror_curriculum.py with coordinate oracle, no ARCquerylabels. Selection13tasks/37complete demo calls, report26freshtasks balanced2/family, manifests committed before winnerselection, fixedrank-consistentselection. Frozen source417e61b54aba1ef50372d2f762f0e0609dfc0d81ebccef9c2bfae7c627fcc651. Search773001, independentparent/proposalstreams, maxage8, fourproposals/arm plus2founders =6physicalevals/222calls each. Result crossover14/26, mutation14/26, random20/26; FP0all, executionfailures0. Crossoverwinner lostmirror niche; mutation retainsmirror2/2; randomretainsmirror2/2 pluscrop,adjacency,repeat2/3. Sparsefounders onlycrop/fill/mirror; crossover cannot create absent genes, whereas random sampleswholebank. Negative result at this tiny budget; no general randomsuperiority or crossoverbenefit claim. CPUselection6.586/6.646/13.835s, eval/sec.911/.903/.434; queryCPU/task.02511/.02724/.03203. ServerpeakRSS152559616bytes, externalexit0. Thirtysixledgerentries, completedpayload, corpora, allattempt/winner hashes, canonicalgenotypes, streams, equalcalls and per-familyscores verified by extended verifier. Fourteen relevant tests passed across prior13test batch plusnewbalancedcorpustest. Fetch session completed before final verification; earlier attempt duringfetch failed missingresult, retained as observation error without restarting. Runs/marker-population-matched-v1/result.json. No native/evaluation/audit/final/download/paid/gradient access. No promotion; frontier superiority unproven.

### Joint repair native regression and resource gates, September 30
The frozen joint graph scores4/200 versus3/200 for its same-bank inactive control, gaining2bcee788 with no losses against that control and zero demo-consistent wrong queries in both. Four-entry ledger, completed payload and per-task records verified; DEV200 manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4. Active28.856tasks/sec, .034654CPU seconds/task; inactive34.812tasks/sec, .028725CPU seconds/task. Whole run51.710s, peakRSS92,827,648bytes, externalexit0; sourcee2f3d16d87ed7c3768a965b6a3fa026217a5a4ffb0790e9639cabf95b29bc364.
Crucial cross-run comparison against saved earlier full-portfolio records on the identical manifest finds one gained task(2bcee788) and one lost task(bf699163), total4/200 unchanged. This is not an isolated paired treatment because banks differ. Inspection confirms tile activation displaced object_bank_largest4.gene at slot12. Therefore do not promote or admit under a no-regression rule. Retain this negative layout result and lineage for repair; inactive-control gain alone hid the displacement cost.
Sequential server boundary/cap run on frozen cad017c9a0956ba2a12b068a64f08a321ca42581519b4f7ba156b6f8d9ae2bfd passes reference/compiled finite-port parity at1and30grid boundaries for demo/query phases; graph363storedparameters, wall.386s, peakRSS84,082,688bytes, 3GiBaddress-space cap,1024parameter cap,60second wall cap, externalexit0. Graph hash/statistics/source hash and allfour cases checked against local artifact. Resource source0407c44ebc5070c4a96e916d80897b82cf452c761a444db9777344562b60e72e. Fixtures do not prove arbitrary correctness. Receipts runs/joint-repair-native-v1, ORIGINAL_PORTFOLIO_COMPARISON.json, and runs/joint-repair-caps-v1. No gradient/evaluation/audit/final/download/paid access; independent frontier superiority unproven.

### Charged repair of a bred joint layout, September 30
Constructed two compatible parents with complementary tile/mirror versus numeric genes; predeclared uniform crossover seed1 recreates the damaged joint layout. Concrete traced counterexamples use fresh synthetic development train pairs; author also knows the earlier DEV200 regression. Maker changes three slots: mirror to redundant identity slot8, restores object gene at12, moves tile to13. Random control makes three uniform slot/gene edits in the same ten-gene bank. Unrepaired baseline receives an unchanged charged evaluation. Each arm uses one physical evaluation and39 complete fixed-rank demo calls; report hashes frozen before query scoring. Receipts log the shown diagnostics, edits and exact parents; three maker-chain receipts and reproduction of the crossover/hash/canonical/report counts verified. All proposals remain unadmitted.
Fresh synthetic report: maker24/28, unrepaired22/28, random18/28; wrong demo-consistent queries0each. This is an exposed known-slot repair, not blind maker efficacy, learned maker improvement or a replicated random-control advantage. Maker authoring cost and local peakRSSnotmeasured; evaluation wall28.546s. Source96b7bc59ab1c4d31e08a1631230afbb51e24698b81002c99a750ef237ca3fba3; reports runs/joint-layout-repair-v1. Numeric generator had no family label in original corpus; post-hoc FAMILY_SCORES labels its final two rows horizontal_two_marker_gate without changing frozen receipts. Repaired graph still0/2repeat2 and0/2repeat3; it has not passed archive no-regression.
Frozen native comparison scores repaired5/200 versus unrepaired4/200, restoresbf699163 with no losses or false positives against that control. This ties the earlier mirror-only portfolio's5/200 rather than establishing a new best. Random repair was not scored natively. Four-entry ledger/per-task/hash comparison verified, same DEV200 manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4. CPU/query.088005s repaired vs.034793s unrepaired;11.362vs28.741tasks/sec, wholewall100.138s/RSS116330496bytes, externalexit0. Native sourcef7308094b59f36d9b509df5b6877b0cb7e6f7eb9d55b6d617de90d8fbd273c2e; receipts runs/joint-layout-native-v1. No gradient/evaluation/audit/final/download/paid access. No default promotion; archive/fresh/cap/replicated repair controls remain required, independent frontier superiority unproven.

### Nine-slot native regression and population interface, September 30
Frozen matched runtime pair sevenactive/nineactive both5/200DEV200, with no gained/lost tasks and zero demo-consistent wrongqueries. Four-entry ledger/completedpayload, frozen hashes and per-task recount verified. Manifest4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4; source74cd42513168513dd4609c9dfde81ceb6e001060ad32345c4a06554ea36c67b1. QueryCPU/task.093133/.102234seconds, throughput10.737/9.781tasks/sec, wholewall159.239s, peakRSS123219968bytes, externalexit0. Adding scaling genes preserves synthetic behaviors but yields no DEV200 benefit and approximately10% more query CPU here. No default promotion; expanded training anchors and matched evolutionary evidence still required.
Population random proposal helper now uses the recorded mutable slot count instead of hardcoded7; policy genotype wording follows it. Optional numeric family augments marker+original12families to14, with explicit family labels and39demo calls per one-example-family selection. Verifier reconstructs extra-family corpus commitments and family counts. Fourteen existing runner/mirror tests pass; a new nine-slot random child/canonical/full39call runtime fixture passes, with train-only14task selection and28task report. Existing seven-slot marker pilot re-verifies all36ledgerentries under current code. Nine-slot full population experiment is not yet measured. Receipts runs/nine-slot-native-v1, resource text and test source. No gradient/evaluation/audit/final/download/paid access; independent frontier superiority remains unproven.

### Nine-slot three-replication study completed, September 30
All runs terminal exit0. Crossover170/210 (60,50,60), mutation171/210 (55,56,60), random192/210 (65,66,61). Equal54 evaluations/4212 calls per arm, failures0/false positives0. All324 ledger entries, graph/corpus identities and random streams reverified. 210 distinct synthetic report tasks, fourteen families;365 stored parameters/1717 state; capacity4/maxage8; parity-verified opt-in demonstration cache. Selection CPU117.627/157.070/213.305seconds; maximum RSS178016256bytes. Source6239c8e8739ba505e3bc4830d40e3e6ffbabcf7914b6308f6f56203ec3ce483b.
Random leads every run under sparse complementary founders and whole-bank random proposals. No crossover advantage; this tests wiring choices, not numerical-weight evolution. No native/evaluation/audit/final/download/paid/gradient access. Native capacity comparison remains5/200 versus5/200. Next capability work needs failure-derived authored cells/weights with equal-budget repair controls; frontier superiority unproven, goal active. Plan runs/nine-slot-replications-v1-PLAN.json; aggregate runs/nine-slot-replications-v1-SUMMARY.json.
Paper ARC experiment, abstract and limits updated. Built-in compilation failed with platform error 'Unable to find standard directories for platform'; source preserved, compilation unverified.
