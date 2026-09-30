# What Small Authored, Fitted and Evolved Cell Libraries Can and Cannot Do on ARC-AGI-2 Under CPU and No-Gradient Constraints

Draft of 2026-09-30 for the ARC Prize 2026 Paper Prize. Status: DRAFT for owner review. Paper format, length and submission rules have not been verified against the governing pages; see Section 8.

## Abstract

We report a constrained attempt at ARC-AGI-2: no gradient-based training of any kind, CPU only, small components written by hand, fitted by counting on the demonstration pairs, or found by gradient-free evolution, and all code released. The main system is a library of 57 hand-written grid cells, composed to depth two by exhaustive enumeration, plus task-fitted lookup-table cells. On the 1,000 public training tasks it solves 53, 80, 95 and 97 tasks (5.3% to 9.7%) as cell families are added, at the cost of 12 to 14 demonstration-consistent but wrong answers. On the 120 public evaluation tasks it solves 0 at every stage, and no evaluation task even admits a demonstration-consistent program from the library. A statistical comparison of the two public splits shows evaluation grids are about twice as large, use more colours and objects, and admit a conflict-free local rule 3 times less often (22.4% vs 7.4% of same-shape tasks). Gradient-free evolution of tiny cellular automata solves 0 of 40 tasks in each arm, and in two small population studies (CellForge graph breeding and a byte-level language pilot) breeding does not beat its controls. We present these as negative results with their controls, discuss why the evaluation-style distribution defeats local and primitive rules, and list which ideas (object and relation cells, dense generators, amortised selection) we consider worth testing next. We make no competitive claim, no claim about general intelligence, and no estimate of private-set performance.

## 1. Motivation and constraints

The project behind this paper (CellForge) asks how far a system built from small, inspectable components can go when three constraints are imposed. First, inference and search must run on a CPU within the Kaggle budget (the plan in `docs/ARC_PRIZE_2026_PLAN.md` records a 12 hour notebook limit and no internet; these were read from search-indexed Kaggle overview text and have not been re-verified from the rules page). Second, no gradients of any kind are allowed: no backpropagation, no fitness-weighted gradient estimators such as evolution strategies, and no test-time gradient adaptation. Random mutation followed by discrete survivor selection is allowed (`docs/BREEDING.md`). Third, every component must be directly authored or fitted by counting, so that it can be read, hashed and released under a permissive licence.

These constraints are chosen for deployability and for research interest, not because they are expected to win. Published small-model results on ARC-AGI-2 rely on gradients: a 7M-parameter recursive model reported at 8% (TRM, arXiv 2510.04871) and a 76K-parameter model trained at test time reported at about 4% (CompressARC). Both figures are taken from `docs/ARC_PRIZE_2026_PLAN.md` and were not independently checked for this draft. Our question is narrower: what does the same budget buy without any gradient training, and where exactly does it stop?

The contribution is an honest measurement: a reproducible baseline, the controls that bound how much of any gain could be attributed to search, a diagnosis of why transfer to the evaluation-style distribution fails, and an account of what did not work.

## 2. Methods

### 2.1 Authored cells with depth-two composition

`research/claude_arc/cells.py` defines 57 grid-to-grid cells (the count was verified by importing the module). They cover geometry (flips, rotations, transpose), tiling and mirroring, cropping, object selection, hole filling, denoising, gravity, symmetry completion, boolean operations on grid halves, separator-block selection, and up- and down-scaling. `solve.py` enumerates all chains of one or two cells. A chain is kept only if its output on every demonstration input matches the demonstration output exactly, or matches after a single task-level colour map. That map is fitted by exact consistency (counting) and is rejected if it is the identity or conflicts. Candidate chains are deduplicated by their outputs on the demonstrations. The first two distinct prediction sets form the two ranked attempts, as required by the ARC-AGI-2 format. Scoring is exact match on all test inputs of a task. There is no language model at inference and no learned component. Recorded per-task wall time averages 1.5 to 1.8 s in the result files.

### 2.2 Task-fitted cells

`fitted.py` and `fitted2.py` add cells whose tables are set per task by counting on the demonstration pairs.

- Local-rule tables (same-shape tasks). The output cell colour is a lookup on a per-cell feature key: colour, four- and eight-neighbour counts, neighbour patches, connected-component size, rank or border status, row and column rays, parity, and row and column colour sets. A table is accepted only if it is conflict-free on the demonstrations, non-identity, and every key needed by the test input was seen.
- Colour-invariant "role" rules (added in the last stage). Neighbour relations are abstracted to background, same as centre, other, or outside, and the output token is either an absolute colour or "same as centre".
- Shape-changing cells: fractal Kronecker products, eight-orientation block tiling, and crop by criterion with rejection of ambiguous choices.

No gradients, sampling or language model are involved in any of these; acceptance is exact consistency.

### 2.3 CellForge graphs and breeding

CellForge represents a model as a graph of cells and wiring; the genome is the cells plus their wiring. Crossover uses a deterministic materialiser, `import_def`, which copies a definition and everything it references from another graph version under a namespace prefix (`docs/BREEDING.md`). Later versions use typed, fixed slots so that swapped parts are homologous and every child is valid by construction (the same design idea as the fixed-width genomes of arXiv 2609.27242). The CellForge ARC work is carried out by a separate agent and is reported here only through its receipts (Section 3.3). The Python enumerator of Section 2.1 is an external reference and is not a CellForge graph; the two should not be conflated.

### 2.4 Evolved tiny cellular automata (negative control)

A small shared per-cell network is iterated over the grid, in the style of Neural Cellular Automata for ARC (arXiv 2506.15746, which trains these with gradients). We instead evolve the parameters with a gradient-free (mu+lambda) Gaussian-mutation scheme seeded at the identity, against a control that perturbs the seed randomly with the same number of evaluations. Scoring is exact match on train and test outputs. The scripts for this experiment were kept in a scratch directory and are not in the repository (`notes/CLAUDE_RESEARCH.md`, evolved-NCA section), so this experiment is not reproducible from the repository as it stands.

## 3. Results

All tasks are from the public ARC-AGI-2 data. Development used the training split. The evaluation split was scored as a baseline and is now exposed for this library (Section 7). All tables below report single runs.

### 3.1 Cell library on public data

Verified by recounting the per-task result files in `research/claude_arc/` (fields `solved` and `n_sols`). "Consistent" means at least one candidate was consistent with all demonstrations; "false positive" means consistent but wrong on the test input.

| Stage | Result file | Training solved (of 1,000) | Consistent | False positives | Evaluation solved (of 120) | Evaluation consistent |
|---|---|---:|---:|---:|---:|---:|
| v1 authored cells only | `results_training_d2.json`, `results_evaluation_d2.json` | 53 (5.3%) | 65 | 12 | 0 | 0 |
| v2 + local-rule tables | `results_training_d2_v2.json` | 80 (8.0%) | 92 | 12 | not scored | not scored |
| v3 + shape-changing cells | `results_training_d2_v3.json`, `results_evaluation_d2_v3.json` | 95 (9.5%) | 107 | 12 | 0 | 0 |
| v4 + colour-invariant role rules | `results_training_d2_v4.json`, `results_evaluation_d2_v4.json` | 97 (9.7%) | 111 | 14 | 0 | 0 |

Wilson 95% intervals computed by hand from counts: 53/1000 is 4.1% to 6.9%; 97/1000 is 8.0% to 11.7%. For 0/120 the upper bound is 3.1%. The intervals describe task sampling only, not variation in the method, and the runs are single and deterministic.

In v3, fitted cells account for 40 of the 95 solves (tile_orient 9, n4count 5, ray 4, n8count 4, comp_rank 4, ray_pair 3, the rest 1 to 2 each), and the most common authored chains at v1 were identity plus colour map (4), mirror_4 (3) and symmetry fill (3); both breakdowns are from `notes/CLAUDE_RESEARCH.md` and were not recomputed. The demonstration-consistent but wrong rate is 12/65 (18.5%) at v1 and 14/111 (12.6%) at v4 (our arithmetic on the counts above). The increase from 12 to 14 at the last stage means the two extra solves were accompanied by two extra wrong answers.

The decisive observation is the evaluation column. The library solves 0 of 120 evaluation tasks at every stage, and for no evaluation task does any enumerated or fitted candidate even reproduce the demonstrations. Adding cells that raised training accuracy from 5.3% to 9.7% changed nothing on the evaluation split. Because every stage scored zero, we cannot say whether any component helps there.

### 3.2 Distribution-gap study

`research/claude_arc/gap_study.py` computes statistics over the task files only; no solutions were inspected and nothing was tuned on the result (`gap_study.json`).

| Statistic | Training (n=1,000) | Evaluation (n=120) |
|---|---:|---:|
| Same-shape tasks | 68.0% | 67.5% |
| Mean input cells | 182 | 373 |
| Mean colours per input | 3.9 | 5.3 |
| Mean objects per input | 4.7 | 5.7 |
| Mean train pairs per task | 3.2 | 3.0 |
| Mean test inputs per task | 1.08 | 1.39 |
| Same-shape tasks with a conflict-free local rule (best of 13 feature sets) | 22.4% | 7.4% |
| Mean best local-rule conflict rate | 5.9% | 5.8% |

The evaluation split has about twice as many cells per grid, more colours and objects, and a local-rule expressibility about three times lower. The fraction of same-shape tasks is almost identical, so the gap is not a matter of shape changes. The mean conflict rate is also nearly identical, which is a reminder that a low conflict rate does not mean a usable rule: a zero-conflict table is required to be accepted, and few evaluation tasks have one. The 7.4% figure is computed over a small number of same-shape evaluation tasks (67.5% of 120 is about 81 tasks; about 6 tasks), so it is imprecise. We did not compute an interval for it.

The colour-invariant rules of v4 were a direct test of one explanation, that absolute-colour keys prevent transfer. They did not help on the evaluation split (still 0/120, still no demonstration-consistent candidate). The failure is therefore not explained by colour keys. Our working conclusion, which we have not tested with an object-level library, is that evaluation-style rules are non-local and object-relational.

### 3.3 CellForge graph and breeding receipts

These numbers come from the CellForge agent's receipts as summarised in `docs/ARC_PRIZE_2026_PLAN.md` and were spot-checked against `runs/arc-gene-coverage-v1/result.json`. All are on a fixed development set of 200 public training tasks (SHA256 task-ID modulo 5 nonzero, first 200 in hash order; 222 test inputs). Synthetic denominators are separate and are never added to native counts.

| Artifact or control | Native dev result | Source |
|---|---|---|
| grid20 graph and fixed router | 0/200 | `runs/grid20/DECISION.json` (as cited in the plan; not opened by us) |
| Associative grid mixture of experts | 3/200 tasks, 3/222 inputs, 0 invalid | `runs/grid-associative/DECISION.json` (as cited; not opened) |
| Identity-colour associative control | 0/200 | as cited in the plan; not opened |
| 28 single graph genes, best single | 3/200 | `runs/arc-gene-coverage-v1/result.json` (27 of 28 genes solve at most 1, verified) |
| Three selected breeding pairs | 3/200 each (51.4 tasks/s for the first pair) | same file; no gain over best single |
| Object-association child (later study) | 4/200 vs 3/200 control, about 6x slower | plan, `runs/grid-object-association-evidence-v1/REPORT.md` (as cited) |

In the coverage study, 28 single genes were scored on the development tasks, 378 pairs were examined with an exhaustive per-input union upper bound, and three pairs were materialised and scored; each solved 3/200, the same as the best single gene (verified in `result.json`). The pair upper bound is not a measured child score. The plan also records several matched population pilots (crossover versus mutation versus random proposals) on small original synthetic curricula that ended in ties or in a crossover disadvantage (for example 12/24, 12/24, 14/24 and, in a later study with three seeds, 12, 14 and 13.3 mean solves for crossover, mutation and random). We cite these only as recorded in the plan and did not recompute them. Their authors describe them as single small pilots, not evidence against crossover in general. One later study found that crossover assembled complementary parents (child 10/10 original synthetic tasks where each parent got 5/10), which shows building-block retention, not a population advantage.

### 3.4 Evolved tiny cellular automata

Values are from `notes/CLAUDE_RESEARCH.md`. Setting: same-shape public training tasks under 14x14 (313 exist), first 40 used, seeded at identity, equal evaluations per arm.

| Configuration | Evaluations per task | Evolution (train / test) | Random perturbation | Identity |
|---|---:|---:|---:|---:|
| 1,488 parameters (16 channels, 8 hidden, 10 steps) | 3,000 | 0/40, 0/40 | 0/40 | 0/40 |
| 414 parameters (10 channels, 4 hidden, 3 steps) | 8,000 | 0/40, 0/40 | 0/40 | 0/40 |

The upper 95% Wilson bound for 0/40 is 8.8% (our arithmetic). Selection lowered pixel loss on several tasks (for example 0.021 versus 0.028 for random perturbation), but no task was solved. On a synthetic 6x6 colour swap, the 414-parameter model was solved in about 8,000 to 14,000 evaluations, and the test input generalised in one of three configurations, while the 1,488-parameter model reduced loss from 0.137 to 0.047 (random 0.137) without solving it. A parallel 313-task run was stopped at 30 tasks with no solves and is not counted. This is consistent with the unseeded genetic search that found no solutions in arXiv 2609.27242.

### 3.5 Byte-level language pilot

In a CellForge evolutionary pilot on a byte-level language model, genetic search and the equal-budget random arm both scored 2.042730829833363 selection bits per byte at generations zero and one, then 2.0422575408863954 at generations two and three, and selected the same winner (`runs/evolution-language-minipilot-v1/result.json` as cited in `paper/cellforge.tex`; we read the paper text, not the JSON). The report-only loss did not change. This is a tie, not an evolutionary advantage, on public development text. For context, a count-based Witten-Bell order-6 model reaches 1.38 bits per byte on the same public development evaluation (`notes/CLAUDE_RESEARCH.md`), which is outside the ARC scope of this paper and is included only to show how far an unseeded search is from a simple fitted seed.

### 3.6 Object and relation cells

Placeholder, to be filled. Another agent is adding object and relation cells to `research/claude_arc`; no results were present in `notes/CLAUDE_RESEARCH.md` at the time of writing.

| Stage | Training solved (of 1,000) | Consistent | False positives | Evaluation solved (of 120) | Evaluation consistent |
|---|---:|---:|---:|---:|---:|
| v5 object/relation cells | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED | TO BE FILLED |

## 4. Distribution-gap analysis

Section 3.2 gives the statistics. Three readings are consistent with the data, and we did not separate them. First, size: larger grids with more objects multiply the number of places a hand-written rule can fail and make exhaustive depth-two enumeration over whole-grid operations less likely to hit a composition. Second, locality: if only 7.4% of same-shape evaluation tasks admit a conflict-free local table, rule-table fitting has little to fit. Third, selection of the public splits: the public evaluation set was presumably assembled to be harder and less repetitive than the training set, which is a property of the benchmark, not of our library. Our evidence does not say which of these dominates. We note that any further statement about why evaluation tasks fail would require inspecting them, which we avoided so that the library is not tuned to the evaluation set.

## 5. What did not work and why (as far as we can tell)

1. Primitive and local rules do not express the evaluation-style structure. All four library stages score 0/120 and none finds a demonstration-consistent program there.
2. Unseeded evolution finds nothing. Evolved tiny automata solve 0/40 per arm, and the literature we rely on reports the same for an unseeded genetic search (Section 6).
3. Breeding near-identical genes adds nothing. In the CellForge gene study the best single gene and the best pairs tie at 3/200, and small matched populations tie with or trail their controls.
4. Selection on two or three demonstration pairs overfits. Demonstration-consistent but wrong answers appear at every stage (12 to 14 tasks), and the later CellForge studies report further demonstration-consistent wrong cases on synthetic validation tasks (43 in a context-memory ablation, per the plan). Matching two or three demonstrations is weak selection evidence.
5. Training gains did not transfer. Training accuracy roughly doubled with fitted cells while evaluation stayed at zero.

## 6. Related work

This section uses only sources recorded in `notes/CLAUDE_RESEARCH.md` and `docs/ARC_PRIZE_2026_PLAN.md`. "Abstract" means the abstract or fetched summary was read; "snippet" means only a search-result snippet was seen. No source was read in full for this draft.

- Genetic search seeded by an LLM (arXiv 2609.27242; fetched, abstract and HTML level). Programs are fixed-width gene lists so variation always yields valid programs; fitness is pixel match on train pairs. On the first 60 ARC-AGI-2 public tasks the LLM alone solves 2, the LLM-seeded genetic search 6, and the unseeded genetic search none. This supports our reading that seeds matter and motivates validity-preserving genomes.
- Neural cellular automata for ARC (Xu and Miikkulainen, arXiv 2506.15746; snippet-level figures, such as about 10k parameters solving 23 of 172 tasks, not confirmed from the PDF). Gradient-trained per task. We are not aware of a paper on gradient-free evolution of such cells per task; the note records that no such paper was found in a search, which is weaker than a literature review.
- Modular NEAT and CoDeepNEAT (Reisinger 2004; Miikkulainen 2017; cited from memory in the notes, not re-read). Co-evolution of modules and blueprints is the precedent for our cells-plus-wiring genome.
- ARC-AGI-2 technical report (arXiv 2603.06590; snippet). Test-time LoRA training of a 200M model on four L4 GPUs reached 27.08% on the semi-private set, and the report mentions an analysis that a large fraction of the private set was solved by at least one team, largely through program search. The latter claim is from a snippet and unverified.
- Tiny learned reference models. TRM at 7M parameters (arXiv 2510.04871) and CompressARC at 76K parameters, both gradient based and, for CompressARC, GPU test-time training (plan, citing the official 2025 analysis and author repository). These are context, not comparators: different methods, costs and datasets.
- Evolutionary model merging (arXiv 2403.13187), the Darwin Godel Machine (arXiv 2505.22954) and Infini-gram (arXiv 2401.17377) are recorded in the notes as abstract-level reads. They motivate recipe-level rather than raw-weight evolution, archive-based parent sampling and count-based models respectively. The Darwin Godel Machine uses a foundation model as its mutation operator, so it is not a from-scratch precedent.
- Vector Symbolic Algebras for ARC (arXiv 2511.08747; snippet) and a public CPU-only DSL-and-heuristic repository (quality unverified) are the closest in spirit to our CPU-only enumerator and are recorded only for completeness.

The ARC Prize 2025 analysis cited in the plan also names top competition results that are far above our evaluation numbers; we do not reproduce them here because we verified none of them independently.

## 7. Threats to validity

- Exposed evaluation set. The public evaluation split was scored, and was later revisited while developing later stages, so it is exposed for this library and cannot be presented as unseen data. Our numbers there mean "not solved", not "untouched".
- Training-set numbers are development numbers. The plan records official paper guidance that training-set performance should not be reported as a result. We report it only as development evidence for the gap analysis and ask the owner to decide how to present it.
- Single runs, no repeated seeds for the enumerative library (which is deterministic) or for the evolved-automata runs. Confidence intervals are Wilson intervals over task counts only.
- Author-written cells. The 57 cells and the fitted families were designed by the authors after looking at public training tasks, so training accuracy partly reflects that familiarity. The fitting code's feature sets were chosen by us.
- No private-set score exists and none is estimated. No claim is made about Kaggle leaderboard standing.
- Reproducibility gaps recorded in the plan: the exact source snapshot that produced the original `_d2` result files and the peak memory of those runs are missing, and the current `solve.py` has additional rules and different output file names, so its present hash does not certify the historical v1 recipe. The evolved-automata scripts are not in the repository.
- CellForge numbers come from a different agent's receipts. We spot-checked only `runs/arc-gene-coverage-v1/result.json`.
- Parts of the experiments ran concurrently on a shared 10 GB machine; this affects wall times only.

## 8. What might transfer, and reproducibility

### 8.1 Directions we consider worth testing (untested)

1. Object and relation cells (correspondence, containment, adjacency, counting to output), because the gap analysis and failure diagnostics point to object-relational rules. Results belong in Section 3.6.
2. Dense, generator-based fitness: hundreds of synthetic examples per family reduce the overfitting of selection on two or three demonstration pairs, provided generators emit tasks and never answer programs.
3. An amortised selector that maps demonstration features to the cells to try first, to cut search time within the CPU budget.
4. Baldwin-style fitness, measured after per-task adaptation of graph state from the demonstrations, with a plasticity-disabled control.

None is claimed to work. Each needs an equal-evaluation mutation-only or random control.

### 8.2 Reproducibility

Code and results are in the repository. Commands (run from `research/claude_arc/`; set the environment variable `ARC2` to a directory that contains `training/` and `evaluation/` subfolders of ARC-AGI-2 JSON task files; the default path in `solve.py` is machine-specific):

```
python solve.py training 2 8 3        # split, max depth, per-task time limit seconds, workers
python solve.py evaluation 2 8 3
python gap_study.py                   # writes gap_study.json
```

The argument order is from the `solve.py` source (`split depth tlimit workers [limit]`). The exact flags used to produce each result file were not recorded in the logs, and `run_v4.log` records `training depth<=2: solved 97/1000 (9.7%) in 563s; tasks with any demo-consistent chain: 111`, which matches the v4 file. Output file names in the current script differ from the historical ones and should be checked before rerunning.

Result files: `research/claude_arc/results_training_d2.json`, `results_training_d2_v2.json`, `results_training_d2_v3.json`, `results_training_d2_v4.json`, `results_evaluation_d2.json`, `results_evaluation_d2_v3.json`, `results_evaluation_d2_v4.json`, `gap_study.json`. CellForge receipts: `runs/arc-gene-coverage-v1/result.json`, `runs/grid20/DECISION.json`, `runs/grid-associative/DECISION.json`, `runs/grid-associative/FREEZE.json`, `runs/evolution-language-minipilot-v1/result.json`. Notes: `notes/CLAUDE_RESEARCH.md`, `docs/ARC_PRIZE_2026_PLAN.md`, `docs/BREEDING.md`. SHA256 of the original receipts recorded in the plan: training `9cefd58f...108`, evaluation `eb4ec6bd...400b`.

Still to be done by the owner before submission:

1. Create and validate the internet-disabled Kaggle notebook that runs the method and links to this paper. A working entry is required for the Paper Prize (plan). It does not exist yet.
2. Add license files: CC0 for original data and MIT-0 (or equivalent permissive) for code, after checking dependencies and that third-party ARC data does not inherit these licences.
3. Check the governing Paper Prize and Kaggle rules for format, length, deadline (plan lists November 8 for papers and November 2 for submissions, from the official overview) and whether training-set results may appear in the paper.
4. Fill Section 3.6 with the object and relation cell results, or delete it.
5. Decide how the exposed evaluation split and the early authored-cell development are disclosed.
6. Bring the evolved-automata scripts into the repository, or state that they are unavailable.

## Notes for the owner (delete before submission)

- Not verified: the byte-language tie value was checked only in `paper/cellforge.tex` text; TRM and CompressARC figures, the CellForge receipts for grid20, associative and object-association, the matched population pilot counts, and the 43 wrong-query figure come from the plan and were not recomputed.
- Section 3.1 breakdown of solving chains is from notes, not recomputed.
