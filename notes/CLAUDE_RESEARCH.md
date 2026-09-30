# Claude side-research (independent of Codex edits)

## Count-model reference vs founder (2026-09-29, local, tinyreal public data)
Protocol: fit on data/tinyreal/dev_train.json (2088 docs, 1.12 MB), score data/tinyreal/dev_eval.json
(165 docs, 89,213 bytes), byte-level, per-document context, no EOS in denominator.
Script: scripts/claude_count_reference.py (Witten-Bell interpolated, orders 0..N). No gradients.

| model | bits/byte |
|---|---:|
| order-1 counts | 3.2015 |
| order-2 | 2.3140 |
| order-3 | 1.7587 |
| order-4 | 1.5049 |
| order-6 | 1.3816 |
| xz / zlib on eval text | 2.070 / 2.296 |
| founder f0b43dbe (first 24 eval docs, 12,894 bytes, content bytes only) | 2.5113 |

Reading: the founder is worse than a trigram count model and worse than xz. A fitted count seed
is worth roughly 1.1 bits/byte over the founder; single-coordinate mutation cannot close that.
Caveats: founder scored on a 24-doc subset; count models scored on all 165 docs; both public dev
data (exposed). These are development numbers, not final results, and not MC accuracy.
Batched compiled-free reference scoring of the founder ran at ~5 ms/byte (24 docs batched) versus ~30 ms/byte
in the server single-document calibration, so batching alone gives ~6x evaluation throughput.

## Count-only structure features (scripts/claude_count_features.py)
Fit counts on dev_train[:-1 * 150], choose fixed linear-mixture weights on dev_train[-150:], score dev_eval once.
No gradients; mixture weights by grid search on the validation slice.

| mixture | eval bits/byte |
|---|---:|
| order-6 counts only | 1.3909 |
| + word-context model (prev word + current prefix) | 1.3090 |
| + in-document adaptive cache (order<=4) | 1.2863 |
| + both | 1.2650 |

Reading: word-level context (-0.08) and in-document adaptivity (-0.10) each help and stack (-0.126).
The in-document cache is the same capability the founder lacks (long-context reuse), so it is the
best-supported structural target for module evolution. Counts still plateau around 1.25-1.4 bpb;
this is a reference for what a fitted seed plus simple structure reaches, not a language-ability claim.
Development data only (exposed); one run, no repeated seeds.

## Gradient-free adaptive mixer (scripts/claude_adaptive_mixer.py)
Bayesian posterior weighting with forgetting, reset per document: w_i <- w_i * p_i(byte)^eta, normalize,
then shrink toward the prior weights with factor `decay`. No gradients. Prior (0.8,0.15,0.05).
Hyperparameters chosen on the dev_train validation slice; dev_eval scored once.
Fixed mixture 1.2650 bpb -> adaptive 1.2222 bpb (eta=1.0, decay=0.9). Best settings sit on the edge of the
grid (eta 1.0, decay 0.9), so more tuning may help. One run; development data.

## Literature notes (web search, 2026-09-29; abstract-level reading, not full-paper verification)
- Evolution Strategies at Scale (arXiv 2509.24372): backprop-free full-parameter ES fine-tuning of
  billion-parameter LLMs, competitive with PPO/GRPO on structured reasoning. Fine-tuning of an already
  gradient-pretrained model, and ES is a gradient *estimator*; check against the owner's no-gradient rule.
- EA4LLM (arXiv 2510.10603): claims evolutionary optimization from the pretraining stage for 0.5B-32B
  models; abstract gives no population size, compute or numbers. Treat as unverified until read in full.
- Sakana Evolutionary Model Merge (Nature MI, arXiv 2403.13187): CMA-ES over merge weights (parameter
  space) and NSGA-II over layer arrangement (data-flow space) of *existing pretrained models*; no gradient
  training. Lesson: evolve a low-dimensional recipe over strong components, not raw weights.
- Darwin Godel Machine (arXiv 2505.22954): archive of agents, parent sampled from the whole archive
  (not only elites), empirically validated edits, SWE-bench 20%->50%. Uses a foundation model as mutation
  operator, so it is not from-scratch evolution.
- Infini-gram (arXiv 2401.17377): unbounded-order n-gram LM from suffix arrays over 5T tokens; interpolating
  with neural LMs cut perplexity up to 73%. Count-based, no gradients; limited here by our tiny corpus.
- cmix / PAQ family: many context models + a mixer, ~1.17 bpb on enwik8 at huge cost. Caveat: PAQ/cmix mixers
  and its LSTM are trained online by gradient descent, so the exact recipe is not allowed; our Bayesian
  mixer above is a gradient-free substitute.
Honest takeaway: none of these shows a from-scratch, gradient-free path to general language ability. The
solid, reusable lessons are recipe-level evolution over strong components, archive-based parent selection,
and count/mixing models that adapt inside a document.

## Is there hope for AGI? Literature pass 2 (web, 2026-09-29; snippet/abstract level)
Evidence that capable general systems exist and that evolutionary/harness search helps:
- ARC Prize community leaderboard (arcprize.org/leaderboard/community): top entries 99-100% on ARC-AGI-3
  *Public Demo* (Tycho $2,986, Retodict $654, baseline1 $400, NOOA $332), all SELF-REPORTED and unverifiable
  by ARC Prize except semi-private ARC-AGI-1/2. Approaches: LLM agent + executable world model built in code,
  hypotheses that must retrodict logged frames before spending live actions, persistent memory.
- Secondary aggregator benchlm.ai (not primary; treat as unverified): ARC-AGI-2 top ~95%, ARC-AGI-3 top ~62.7%
  as of 2026-09-24. Frontier models are far ahead of our grid components (3/200 dev tasks).
- AlphaEvolve (arXiv 2506.13131): LLM-guided evolutionary code search matched or beat best known solutions on
  ~75% of 50+ math problems (new SOTA ~20%), 48-multiplication 4x4 complex matmul, ~23% kernel speedup. The LLM is
  the mutation operator; evolution supplies selection and archive.
- Darwin Godel Machine (arXiv 2505.22954): SWE-bench 20%->50% by archive-based self-modification of agent code.
- Sakana evolutionary model merge (Nature MI): gradient-free recipe search over pretrained models.
- Dual-process (System 1/System 2) LLM systems are an active line (routers/meta-cognition, e.g. Pangu Embedded,
  Cognitive Decision Routing, latent System1-2 communication); results are efficiency/accuracy tradeoffs, not AGI.
Evidence for limits:
- "On the Fundamental Limits of LLMs at Scale" (arXiv 2511.12869): hallucination, context compression,
  reasoning degradation, retrieval fragility, multimodal misalignment argued as persistent under scaling.
- Continual/online learning without forgetting repeatedly named as a missing capability. A quoted expert survey
  (76% say scaling alone unlikely to yield AGI) comes from a secondary blog; not verified.
Assessment: no source shows a from-scratch, gradient-free path to general intelligence. Every strong result uses a
gradient-pretrained model as the substrate; the gains from evolution/agents come at the harness level (code, prompts,
world models, tools, archives). AGI itself is undefined here and unproven by any of it. The credible research bet for this
project is the harness-level evolution track over pretrained models (System 1 NanoJev / System 2 Gemma 4),
measured against Gemma 4 alone, with the from-scratch line kept as a labeled side experiment.
Caveat: the reported harness gains were obtained with frontier-scale models; how much a harness adds to small
open models (Gemma 4 E2B/E4B, NanoJev 0.6B) on CPU is untested.

## ARC-focused literature pass 3 (web, 2026-09-29; fetched abstracts/HTML, some numbers from search snippets)
Most relevant to "small authored cells joined by evolution/breeding" on low power:
- Combining LLMs and Genetic Search for ARC-AGI-2 (arXiv 2609.27242, fetched): programs are 20 fixed-width genes
  (op + 5 args, 41 ops), so mutation/crossover always yield valid programs; crossover copies each gene slot from
  one of two parents; 2 islands x 4,000 programs, tournament size 2, 2% elitism, ~8,000 evals/generation, fitness =
  pixel match on train pairs. First 60 ARC-AGI-2 public tasks: LLM alone 2/60, LLM-seeded GA 6/60 (10%), UNSEEDED GA 0
  solutions (control incomplete, stochastic). Lesson: breeding needs good seeds; random initialization found nothing.
  Our authored cells are the seeds; a validity-preserving fixed-width genome is a design to copy.
- Neural Cellular Automata for ARC-AGI (Xu & Miikkulainen, arXiv 2506.15746; ALife 2025): a tiny shared per-cell network
  iterated over the grid; snippet reports ~10k params solving 23/172 tasks and <2.5k-param models similar on many;
  GRADIENT-trained per task (PDF text extraction did not confirm the exact numbers). Closest published match to
  "cells". Gap I found no paper on: evolving (gradient-free) tiny NCAs per task or breeding NCA cells.
- Modular NEAT / CoDeepNEAT (Reisinger 2004; Miikkulainen 2017): co-evolve a population of small modules and a
  population of blueprints that say how modules combine. Direct precedent for cells + wiring co-evolution.
- ARC-AGI-2 Technical Report (arXiv 2603.06590): 27.08% semi-private via test-time LoRA training of a 200M model on
  4xL4 (gradient, GPU). Report says a 2026 analysis found 49% of the private eval set solved by at least one team,
  dominated by variations of brute-force program search (from search snippet, unverified).
- Vector Symbolic Algebras for ARC (arXiv 2511.08747): cheap heuristics, no brute force or big nets (snippet only).
- Public CPU-only DSL+heuristic ensemble repo exists (SebAustin/arc-prize-2026-arc-agi-2), quality unverified.
- Duck Harness (ARC-AGI-3 Milestone 1 winner): small open models on a single GPU plus a minimal coding harness.
Takeaway: the honest evidence says (a) validity-preserving fixed-width genomes + good seeds make breeding work at
~10% on a tiny slice of ARC-AGI-2 with a few thousand programs on 4 GB GPUs; (b) tiny per-cell NCAs are the published
"small cells" route but are gradient-trained; (c) no paper shows unseeded evolution solving ARC. Proposed experiment:
gradient-free (mu,lambda)-ES/GA over ~1-2.5k-param NCA cells per task, seeded by authored cells, on CPU.

## Biology / neuroscience transfer candidates (web pass 4, 2026-09-29; abstract-level)
Analogy rule: borrow the *mechanism*, then test it against a control; a metaphor is not a result.
1. Hebbian / in-lifetime plasticity evolved without gradients (Najarro & Risi, NeurIPS 2020, arXiv 2007.02686):
   evolution searches synapse-specific local learning rules (>450K plasticity params in their setup), weights
   self-organize during the agent's life; rapid adaptation to unseen damage in <100 steps without reward. Transfer:
   ARC needs few-shot adaptation from demonstrations. Evolve a small *plasticity rule* per cell type that updates
   cell state/weights while the graph reads the demo pairs, then freezes for the test input. This is
   gradient-free test-time learning and matches our demonstration-conditioned routing/state.
2. Developmental / indirect encodings (HyperNCA arXiv 2204.11674; gene-regulatory-network encodings): a small genome
   grows a larger structure by local rules; compressed genomes reuse parts and yield modularity. Transfer: genome =
   compact cell templates + growth rules that instantiate wiring, so breeding acts on the rule not on every weight.
3. Clonal selection / affinity maturation (immune systems): clone the best-matching candidates and hypermutate
   inversely to affinity (better match -> smaller mutation). Transfer: per-task evolution at inference with
   mutation size scaled by current train-pair error; cheap, no gradients. (No ARC/program-synthesis application
   found in search; untested.)
4. Sparse distributed representations + lateral inhibition (cortical columns, Thousand Brains): sparse codes,
   competition between columns, many small models voting. Transfer: winner-take-all routing among cells and
   voting across cell ensembles for the two ranked ARC attempts. (No ARC result found.)
5. Predictive coding: local prediction-error signals instead of global backprop. Transfer: cell-local error
   used as fitness or as a plasticity signal. (No ARC result found.)
6. Modular NEAT / CoDeepNEAT: co-evolve module and blueprint populations (already logged above).
Honest status: only (1) and (2) have published gradient-free evolved examples, all on RL/control tasks, none on ARC.

## LLM-in-the-loop weight/cell iteration (owner idea, 2026-09-29)
The maker agent reads per-task failure diagnostics from an experiment, proposes targeted edits to cell weights or
wiring, the edit is validated, and successes enter an archive (AlphaEvolve / DGM pattern). This keeps the
no-gradient, directly-authored rule and fits the original maker/harness design. Risks: (a) the LLM sees dev results,
so dev scores become exposure and every LLM edit must count as an evaluation; (b) hold out a sealed split and score it
once; (c) log the diagnostics given to the LLM and the edit it made for provenance; (d) compare against a
no-LLM mutation control at equal evaluation count so any gain is attributable to the LLM.

## Beyond archive sampling: what real Darwinian dynamics add (owner critique, 2026-09-29)
Owner point: DGM-style "sample a parent from an archive by performance/novelty" is closer to a temperature/epoch
schedule over a tree of variants than to evolution. Agreed. Evolution proper = a co-existing population with
heredity, variation, differential reproduction under finite resources, turnover (death), recombination between
contemporaries, changing environments and interacting lineages. Mechanisms to add, each testable against a control:
1. Finite resources + niche competition: each ARC task is a food source whose reward is shared among the cells/graphs
   that solve it (implicit fitness sharing; classic coevolution technique, not re-verified this session). Forces
   specialization instead of one generalist elite. Cap population size; mortality/aging so elites do not live forever.
2. Homologous sexual recombination between contemporaries: crossover speeds up building-block assembly and repairs
   disruptive mutations (arXiv 1403.6600, Evolutionary Computation 2017; classic BBH). Needs fixed typed slots so
   swapped parts are homologous.
3. Horizontal gene transfer (our import_def) plus recombination: modelling work reports HGT + recombination raises
   selection efficiency and lets genomes grow without irreversible mutation load (PNAS 2022, snippet level). HGT alone
   accumulates load; pair it with recombination.
4. Symbiogenesis / major transitions: merge two viable graphs into one and select at the new level, then prune
   redundancy. NOTE: our MNIST-1D A1+A2 crossover child (0.5295 vs parents 0.4655/0.4360) already behaved like a
   symbiogenic merge; make it a named operator with an interface-evolution phase.
5. Coevolution of environments and solvers (POET, arXiv 1901.01753; MAP-Elites niches): evolve a population of task
   generators (ARC-like transformations) kept "not too easy, not too hard" for current solvers, so difficulty rises
   with competence; keep elites per niche.
6. Baldwin effect (Hinton-Nowlan 1987; Meta-Learning by the Baldwin Effect, arXiv 1806.07917): lifetime learning
   smooths the fitness landscape. Measure fitness AFTER per-task adaptation (plasticity rule or in-graph state from
   the demos) so evolution selects genomes that learn quickly; optionally a Lamarckian variant that writes adaptations
   back (this is where LLM edits fit).
7. Evolvability selection: score a lineage by the fitness of its offspring, not only its own fitness.
Constraint reality: real evolution gets its power from large populations x many generations. On a low-power CPU we can
only afford small populations, so genomes must be tiny and evaluations cheap; expect these mechanisms to be tested at
small scale and none is guaranteed to beat a strong mutation-only control.

## Distillation-as-competition to cut evolution cost (owner idea, 2026-09-29)
Evolution from a blank genome is sample-hungry (my NCA runs: 0 solves at 1.5k params, tiny wins only at ~400 params).
Ways to distill knowledge into the population so selection starts far ahead, all gradient-free:
1. Dense synthetic curriculum with competing generators: for each transformation family, generators emit many
   (input, output) pairs (cf. Hodel's re-arc generators for ARC-1). Fitness on hundreds of samples per family removes
   the 3-pair overfitting; generators coevolve to stay not-too-easy for the current solvers (POET-style competition).
2. Distill an LLM/maker's knowledge into the gene pool: authored cells written by a strong model for many families,
   validated on generator samples, are the seeds (the unseeded-GA-found-0 result says seeds are what matter).
3. Amortize the search: log which cells/compositions solved which tasks and evolve/count a fast selector cell
   (System-1 router: demo features -> which cells to try first). Cuts inference time under the 12 h CPU cap.
Caveats: distillation needs a teacher that already solves the tasks, so the teacher is the public training set
(1000 tasks) and authored cells; risk of overfitting to public training versus the harder hidden ARC-AGI-2 eval
distribution; any synthetic data must be released open and derived only from permitted sources; check ARC Prize
rules on externally generated data before relying on LLM-generated training data.

## Evolved-NCA scratch results (Claude, local, numpy; scripts kept in scratchpad, not the repo)
ARC-AGI-2 public training, same-shape tasks under 14x14 (313 exist), seeded at identity, gradient-free (mu+lambda)
Gaussian mutation vs random perturbation of the seed, equal evaluations, exact-match scoring.
- Run 1, 1,488 params (16 ch, 8 hidden, 10 steps), 3000 evals/task, first 40 tasks: ES 0/40 train, 0/40 test;
  random 0/40; identity 0/40.
- Sanity, synthetic 6x6 color swap: 1.5k params, 6000 evals: loss 0.137 -> 0.047 (ES) vs 0.137 (random), unsolved.
  414 params, 3 steps: solved on train in ~8-14k evals (7-11 s/core), test generalized in 1 of 3 configs.
- Run 2, 414 params (10 ch, 4 hidden, 3 steps), 8000 evals/task, 10 workers: first 30 of 313 tasks done before I
  stopped it: ES 0 train, 0 test; random 0 train (loss lower for ES on inspected tasks, e.g. 0.014 vs 0.029).
Reading: selection lowers loss but raw-weight NCA evolution solves no real ARC task at these budgets. Consistent with
unseeded-GA = 0 in arXiv 2609.27242. Partial run (30/313), one seed, not a benchmark.
CORRECTION/UPDATE: my `pkill` failed (command missing), so the single-core Run 2 was never stopped and finished all 40
tasks: 414 params, 8000 evals/task: ES 0/40 train, 0/40 test; random 0/40; identity 0/40 (1,382 s). ES reached lower
loss than random on several tasks (e.g. 0.021 vs 0.028) but solved none. The parallel 313-task run was stopped at 30/313
(also 0 solves) using Stop-Process. Both runs shared CPU for a while, which only affected wall time.

## Hand-authored cell library baseline (research/claude_arc/, 2026-09-30) — no gradients, no LLM at inference
57 authored grid cells (geometry, tiling/mirroring, crop, object pick, fill holes, denoise, gravity, symmetry completion,
boolean ops on halves, separator-block pick, upscale/downscale, ...) composed to depth <=2 by exhaustive enumeration, verified
on the demo pairs, optional task-level color map fitted by exact consistency (counting). Two ranked attempts, exact match on
all test inputs. CPU only, ~0.5 s/task/worker, 3 workers.
- ARC-AGI-2 public TRAINING (1,000 tasks): 53 solved (5.3%); 65 tasks had a demo-consistent chain, so 12 were false
  positives (fit demos, wrong on test). Most common solving chains: identity+color map (4), mirror_4 (3), symfill (3).
- ARC-AGI-2 public EVALUATION (120 tasks), scored once as a baseline: 0/120 (0.0%); no task even had a demo-consistent chain.
  The evaluation set is now exposed for this library; do not tune cells on its tasks. Develop on training only.
Reading: authored primitive composition solves easy training tasks but nothing on the evaluation-style distribution that
resembles the hidden test. My earlier 'low single digits to ~10%' guess for a CPU authored/evolved solver is NOT supported by
this measurement. Reported tiny gradient-trained models reach ~4-8% on ARC-AGI-2 (TRM, CompressARC; literature figures).
Files: research/claude_arc/cells.py, solve.py, results_training_d2.json, results_evaluation_d2.json.

### Update 2026-09-30: fitted cells (research/claude_arc/fitted.py, fitted2.py), development on TRAINING only
Added task-fitted cells whose tables/choices are set by exact consistency on the demo pairs (counting; no gradients):
local-rule tables over per-cell features for same-shape tasks (color, 4/8-neighbor counts, neighbor patches, component size/
rank/border, row/column rays, parity, row/col color sets; accepted only if conflict-free, non-identity, and every test key seen),
plus shape-changing cells (fractal kron, 8-orientation block tiling, crop-by-criterion with ambiguity rejection).
ARC-AGI-2 public TRAINING (1,000 tasks), depth<=2 enumeration + fitted cells, two attempts, exact match, CPU, 3 workers:
  v1 authored cells only: 53 (5.3%)  ->  v2 + local-rule fitting: 80 (8.0%)  ->  v3 + shape-changing fitted cells: 95 (9.5%).
  Demo-consistent chains: 107; false positives: 12 (unchanged). Fitted cells account for 40 of the 95 solves
  (tile_orient 9, n4count 5, ray 4, n8count 4, comp_rank 4, ray_pair 3, others 1-2 each).
ARC-AGI-2 public EVALUATION (120 tasks), scored again with v3, NOT used for tuning: 0/120, and again NO task had any
demo-consistent chain (fitted or enumerated). Training gains do not transfer to the evaluation-style distribution.
Reading: a hand-written, fitted-by-counting cell library reaches ~9.5% on public training in ~10 min CPU, but the public
evaluation set (closest proxy for the hidden test) requires structure this library does not represent. Do not present
training-set numbers as competition-relevant performance.
