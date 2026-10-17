# Full Research Inventory — CellForge prior work (Oct 1–16, 2026)

Purpose (Sano): stop repeating dead ends; surface anything with a pulse dropped too early.
Sampling: directory census + summary/manifest/README reads across all five locations.
Marked [sampled] where I read summaries, [census] where I listed only. Deeper digs flagged.

## /srv/cellforge-inbox (5.8G) — Oct 1–11 programs [sampled]

| Experiment | What was tried | Outcome / status |
|---|---|---|
| two-loop-20261001 | Shared typed cell registry: 3 families (decimal-subtraction 9/9, spatial-cardinal 40/40, signed-temporal-deduction 8/8), 12 live cells, inheritance receipts, MoE/planning/hard-reasoning runtimes | WORKED as authoring; STALLED as evolution — competition froze at iteration 1, `architecture_complete=false`, replay-only. See notes/cell_family_digest.md |
| weights-cell-population-g1 / -sequential-g2 | Cell populations, literal-weight cells, charged development selection | g1 done; g2 sequential done. Some arms all-false (llm_count) |
| weights-action-population-g2/g3/v2-g3/corrected-g3 | Action populations across generations + corrected variant | Multiple generations run; corrected-g3 is latest |
| weights-action-growth-g2 | Action growth | g2 archived |
| weights-logic-circuit-gated-g1 / -population-g1 | Logic-circuit cells; live founders: cmp2, mux | g1 done; **no g2 found — dropped thread?** |
| weights-real-learning-v2/v3 | "Real learning" arms; v2 parent arm solved 1/4 tasks | v3 exists; status unclear — needs check |
| continuous-spatial-20261011 | Continuous spatial research; NEXT_STEP.md proposes transition-conditioned action-effect memory | **HONESTLY PAUSED — hypothesis proposed, never dispatched. PULSE.** Decision gate defined (beat baseline + matched random + memory-off controls). Public games dev-only |
| continuous-control-v2 | Continuous control | Present; not sampled |
| transition-memory-reviewed-20261011 | Migration audit of transition-memory bundle | Audit PASS (188 manifests verified); science plan frozen for nine-job campaign — **campaign launch status unknown, likely dropped. PULSE** |
| research-20261008 | Founder dispatch, rate limits, input-only author prospective (3 versions) | Attempt logs; prospective author designed |
| readiness-20261008-0455 | Readiness probe (events, rate limits) | Done |
| native-six-DEV-8ea5c110804f5baa | Native six dev bundle with handoff protocol | Archived |
| arc-best-a5-20261001 (+source tar) | ARC best-a5 audit | Archived |
| prospective-action-cell / prospective-arc-moe (tars) | Prospective cell + MoE continuations | Archived, not unpacked |
| windows-migration-20261002 | Windows migration | Done |
| _archive_20260930, head-20261001 | Snapshots | Archived |

## /srv/cellforge-public (367M) — grid experiments [census]

Dozens of small evidence bundles: grid-bank-capacity, grid-containment-census,
grid-context (backoff/fine/mask-pilot), grid-curriculum-baseline, grid-demo-cache,
grid-expanded-capacity/population + numbered replications (772211/772221/772231…),
adjacency-evidence-v1, conditional-state (adapt/control), direction-router (adapt/control),
evolution-language (gen0-v2.json, minipilot-v1). Pattern: single-hypothesis evidence
bundles with receipts. **Value: a library of tested micro-hypotheses; risk: nobody has
mined them for what transferred.** [Deeper dig recommended: which grid-* results imply
reusable mechanisms?]

## /srv/cellforge-private (769M) [census + sampled]

| Item | Notes |
|---|---|
| arcagi2/ | ARC-AGI-2 data (changelog, data, license) — reference only |
| bench/, bfcl-source/ | Benchmark + BFCL source material |
| final/, final_lm/, final_clf/ (2026-09-29) | Timestamped training checkpoints (LM + classifier). **Unmined: do any beat current router?** |
| grammar/tinygrammar.py | Tiny grammar module — ported/planned for parse robustness |
| realtext/ (final_pool.json, valid.txt) | Real-text corpus |
| evolution-language-pilot-v1 (empty) / v2 (partition-policy, selection) | Language evolution pilots; v1 empty = abandoned early? |
| wb6-evolution-v1/ (a.json, b.json, fit-selection.json, report.json) | WB6 evolution run with report — not sampled |
| document-* logs | Crossover/audit/diagnostic logs — **all empty shells**; the audit trail was never written |

## /opt/cellforge/code (500K) — cellforge package [census]

`runtime/ construction/ harness/ ir/ reporting/ memory/ data/ reference/ baselines/
providers/ governance/`. The construction + harness + memory modules are the reusable
core (joint_weight_population, red_queen_history used by Oct-1 work). [Deeper dig:
which harness pieces serve the Kaggle offline entry?]

## /srv/arc3-claude (1.1G) — current program [sampled]

- environment_files/: 25 genuine ARC-AGI-3 games with human baselines
- v6run/: explorer/blitz lineage, model/ (champion, LM, rhae_scorer.py, transfer_set.json, METHODS.md, notes/), dev_v6a/b/c.jsonl, **held_v6.jsonl (SEALED)**
- paper-build/: cellforge_clean.tex/.pdf — paper draft exists
- logs/: explorer result logs (all 25 games swept by superseded systems)

## Dropped threads WITH A PULSE (do not lose)

1. **Transition-conditioned action-effect memory** (continuous-spatial NEXT_STEP.md): hypothesis + decision gate fully specified, never dispatched. Cheapest high-value experiment on the list.
2. **Transition-memory nine-job campaign**: audit PASS, plan frozen — check if it ever launched.
3. **Cell-family competition iteration 2+**: registry built, never ran. The CellForge loop is its successor — but the 3 families are port-ready cells.
4. **Logic-circuit g2**: founders live (cmp2, mux), no second generation found.
5. **Sept-29 final_lm/final_clf checkpoints**: never compared against router_v3.
6. **Grid-* micro-hypotheses**: dozens of tested mechanisms, never mined for reuse.

## Confirmed DEAD ENDS (do not repeat)

- VQ-encoder with reconstruction objective (killed by Sano: wastes capacity on decision-irrelevant detail)
- Replay-only registries (Oct-1 lesson: competition must actually run)
- Document-* audit logs as evidence (empty; the discipline is receipts-in-code, not log files)
- lp85 lucky-solve as training signal (excluded from teaching deliberately)
- Static English STATE descriptions (deprecated by decoder-only decision)

## Thesis papers — author-then-evolve (added 2026-10-16)

Sano's thesis: the LLM ITSELF writes the weights (hand-authored, no backprop);
evolution selects and breeds. "Using gradient descent is kind of cheating" — the
paper states the method as author-then-evolve, with backprop explicitly scoped to
optional local refinement (the Baldwin effect).

1. **"Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights"** (arXiv:2506.16406, Jun 2025):
   prompt-conditioned parameter generator maps task prompts DIRECTLY to LoRA weight
   updates — zero per-task training — 12,000x cheaper than fine-tuning, up to 30%
   BETTER than the strongest gradient-trained LoRAs on UNSEEN reasoning/math/coding
   benchmarks. Paper quote: "prompt-conditioned parameter generation is a viable
   alternative to gradient-based adaptation." → Relevance: the crux-experiment
   blueprint — LLM authors LoRA deltas for the tiny router, no backprop.
2. **"Infinite-Parameter LLMs"** (arXiv:2609.18842, Sep 2026, Boltzbit/Cambridge):
   compact hypernetwork turns live interaction data into low-rank weight modulations;
   "representing information in weights rather than prompts amortises compute and
   frees the context window." → Relevance: directly addresses our context-window
   problem; weights-as-memory instead of tokens-as-memory.
3. **HyperNetworks** (Ha et al. 2016): genotype (hypernetwork) → phenotype (main net);
   "reminiscent of HyperNEAT in evolution." → Relevance: the lineage — hypernetworks,
   HyperNEAT, Sakana evolutionary merges are ONE family tree; our paper sits in it.
4. **Sakana Evolutionary Model Merge**: evolved merge recipes, SOTA results, "much
   smaller memory footprint than gradient descent methods." → Relevance: evolution
   over weight-space beats gradient descent on efficiency — the arena's breeding
   step is this, at the researcher level.
