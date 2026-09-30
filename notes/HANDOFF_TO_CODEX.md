# Handoff from Claude to Codex (written 2026-09-30, ~12:30 local)

Claude's usage is running low. This file is everything needed to continue Claude's workstream. Read it fully, then `notes/CLAUDE_RESEARCH.md`
(the running research log: every measurement and literature pass), `notes/DECISIONS_2026-09-30.md`, `docs/ARC_PRIZE_2026_PLAN.md`, `docs/RED_QUEEN_REPAIR.md`,
`docs/MAKER_REPAIR_LOOP.md`, `docs/SHARED_PERCEPTION_MOE.md`, `notes/VALIDITY_REVIEW.md`, `notes/GROKKING_REVIEW.md`, `notes/VISION_DISINHIBITION_IDEA.md`, `runs/INDEX.md`.
`notes/MESSAGE_FOR_CODEX.md` has the earlier message thread. Claude's work is committed on branch `master` (there is NO `main` branch and NO git remote, so nothing can be pushed).

## 1. The owner's rules (do not relax without asking)
- PURE: no gradient descent of any kind, no gradient estimators, no learned neural weights, no LLM or pretrained model at inference/runtime. Fitting a table by exact consistency/counting on
  demonstration pairs is allowed and is labelled `fitted`. The LLM (Claude/you) acts only OFFLINE as maker/author of cells, generators and edits.
- Core idea: small hand-authored cells joined into circuits/experts (a CellForge graph, MoE-routed), improved by breeding (`import_def` crossover), evolution and maker repair. Low-power (CPU) deployment.
  The Gemma 4 / NanoJev System-1/2 track was explored on paper (docs/SYSTEM12_DESIGN.md) and then DROPPED by the owner ("keep it pure").
- Goal: one artifact/codebase for ARC Prize 2026 (ARC-AGI-2, ARC-AGI-3, Paper Prize). Separate Kaggle submissions are still required. AGI is the long-term aim; nothing so far supports claiming it.
- Deadlines: final Kaggle submissions Nov 2 2026, papers Nov 8, results Dec 4; ARC-AGI-3 Milestone 2 (Sept 30) not realistic. Kaggle: offline, 12 h (ARC-AGI-2) / 9 h (ARC-AGI-3) notebooks, open-source all authored code (CC0/MIT-0).
- Owner-approved downloads so far: public ARC-AGI-2 json, Hodel `arc-dsl` and `re-arc` (MIT), `pip install arc-agi`. Anything else (datasets, weights, APIs, paid compute) needs explicit owner approval.
- Data hygiene: public ARC-AGI-2 EVALUATION is exposed for my cell library (scored 5 times: baseline, v3, v4, v5 and one more pass in the object-cell run; never tuned on). Develop only on training tasks
  (split `crc32(task_name) % 5`: bucket 0 = VALIDATION, buckets 1-4 = DEVELOPMENT). Sealed/private server data must never be opened, committed or published. Run `scripts/prepublish_scan.py` before any release.

## 2. Verified results (details and file paths in notes/CLAUDE_RESEARCH.md; all public data)
| item | result |
|---|---|
| Authored+fitted+object cell library, ARC-AGI-2 public TRAINING (1,000) | 53 -> 80 -> 95 -> 97 -> 131 solved (v1..v5), demo-consistent 149, false positives 18 (12 -> 18 across versions) |
| Same library, public EVALUATION (120) | 0 solved at every stage, and ZERO tasks had any demo-consistent chain |
| Gap study (training vs evaluation) | eval grids ~2x larger (373 vs 182 cells), more colours (5.3 vs 3.9), more objects; same-shape tasks with a conflict-free local rule 22.4% vs 7.4% |
| Evolved tiny NCA, gradient-free, equal-eval random control | 0/40 real tasks for every arm; solves a synthetic colour swap at 414 params |
| Diversity archive + Red Queen (research/claude_arc/evo/) | NO GAIN on 228 validation tasks (20/228 in every arm); niche ordering worse; novelty rejection only saves ~20% time |
| Count-model references (language) | order-6 counts 1.38 bpb, +word+cache 1.265, founder f0b43dbe 2.51 bpb; adaptive mixer 1.222 (multiplicative-update = Hedge-style, owner has not ruled; keep out of primary arms) |
| ARC-AGI-3 pure state-graph explorer (research/claude_arc3/, server /srv/arc3-claude) | 17/190 levels on 25 public games vs 5 random; mean score ~0.19/100 for both (efficiency-squared scoring) |
Your own results (grid-expanded replications 110/120 vs 100/120 random, cache prototype, etc.) are in runs/ and your logs; keep reporting false-positive rate and the crossover-vs-mutation split.

## 3. Key lessons
- An arms race over a fixed library only reorders what it expresses; new capability needs new cells (maker repair, richer object/relation cells).
- Public training gains do not transfer to evaluation-style tasks (ARC-AGI-2 eval is reportedly curated against shallow search). Do not present training numbers as competitive.
- Pure ARC-AGI-2 will probably land near 0% on the hidden set; realistic deliverables are a valid open submission, a Paper Prize entry with honest controls, and possibly a modest ARC-AGI-3 entry.
- ARC-AGI-3 scoring rewards action efficiency; exploration that clears levels in hundreds of actions scores ~0. Goal-directed play (object tracking, action-effect tables, level-to-level transfer) is the lever.
- Literature check (notes/VALIDITY_REVIEW.md): LLM repair needs concrete executable counterexamples, not critiques; LLMs should write structure/tables/small cells, not dense numeric weights;
  Red Queen needs an external real-task anchor to detect competitive overfitting; grokking needs a parsimony pressure and is not what our rising training score is.

## 4. Subagents that were still running when this was written (they die with Claude's session)
Check each directory; if unfinished, read its README/logs and decide whether to resume or rerun. None of these outputs is committed yet unless listed in git log.
1. `research/claude_arc/dsl_port/`: Hodel ARC-DSL primitives as fitted object cells + re-arc generator filtering (MIT; do NOT run solvers.py per-task answers; development split only). Expected result: dev/validation solved counts and false-positive reduction.
2. `research/claude_arc3/` (+ server `/srv/arc3-claude/agent/`): ARC-AGI-3 v5 agent (object tracking, action-effect table, level transfer). `ps` on the server showed `run.py --agent v5:pr,nm,ck,gv` still running.
3. `research/claude_arc/grok/`: gradient-free grokking test on evolved NCAs with/without parsimony pressure (pre-registered criterion in its README; results.jsonl in progress).
4. `research/claude_arc/multiview/`: multi-view perception + disinhibition-gain test (views.py, cells_mv.py; results/ in progress).
After collecting: verify numbers against the result files, then commit only those directories plus an appended section in notes/CLAUDE_RESEARCH.md.

## 5. Prioritised next steps
1. Paper (`paper/cellforge.tex` is now the cleaned version; your old one is `paper/cellforge_pre_cleanup.tex`; diagnosis in `paper/CLEANUP_NOTES.md`): add the 97/131 library stages, the evo null results, the ARC-AGI-3 explorer, grokking and gap-study results;
   resolve conflicts listed in CLEANUP_NOTES.md (arXiv 2609.27242 retrieval claim; MMLongBench-Doc 1,082 vs 1,091); fill the author line; check Paper Prize format rules on the official page. A draft text also exists in `notes/PAPER_DRAFT_CLAUDE.md`.
2. Kaggle entries (owner must create the notebooks and accept rules): package (a) ARC-AGI-2: cell library + graph-owned selection, two attempts per input, hard time cap so the 12 h CPU budget holds; (b) ARC-AGI-3: agent with `is_done()/choose_action()`, `submission.parquet`. Verify the current Kaggle compute limits and ARC-AGI-3 action budget on the official pages.
3. Run the maker-repair ablation (docs/RED_QUEEN_REPAIR.md arms A-G) on your synthetic families: breeding only / + random repair / + maker repair at equal evaluation counts; count every maker edit as an evaluation; concrete counterexamples in prompts.
4. Increase eval-style coverage: object-correspondence, template stamping onto markers, multi-step composition, richer relations; develop on training + generators only; track false positives.
5. ARC-AGI-3: goal-directed cells, efficiency; shared perception cells between the ARC-AGI-2 and ARC-AGI-3 experts (the reuse test in docs/SHARED_PERCEPTION_MOE.md).
6. Release hygiene: run `scripts/prepublish_scan.py` (122 HOLD / 69 WARN on runs/ at last run), follow the 10-step checklist in notes/DECISIONS_2026-09-30.md, review the 20 tracked `runs/baselines/*/FINAL.json` files (contents unverified), add CC0/MIT-0 licence files for authored code, attribute Hodel ARC-DSL (MIT) and the `arc-agi` package (MIT); do NOT bundle ARC-TGI generators (CC BY-NC-SA).
7. Housekeeping decisions already made (notes/DECISIONS_2026-09-30.md): canonical log is notes/ORCHESTRATOR_LOG.md (move your root ORCHESTRATOR_LOG.md entries into it); never `git add -A`; archives/server aggregates are gitignored.

## 5b. Open questions for the owner
Whether the multiplicative-update mixer is acceptable under "no gradients"; whether to create a git remote / `main` branch; which Kaggle tracks to enter first; the paper author line; whether to review the tracked FINAL.json files now.

## 6. Environment and pitfalls
- Owner PC (Windows): ~10 GB RAM shared with other apps, ~4.7 GB free disk, 12 cores. Keep local jobs to <= 3 worker processes; a MemoryError earlier killed runs. Bash tool: no `pkill`; use PowerShell `Stop-Process`; Windows multiprocessing leaves orphan `python.exe` workers, so check `Get-Process python`.
  Bash heredocs containing quote characters failed in the tool; write files with the editor/Write tool. Background jobs and subagents die when a session ends. LF/CRLF warnings on commit are harmless.
- Server `cellforge-1` (Hetzner, ssh alias, root): 2 CPUs, 4 GB RAM, 31 GB disk, Python 3.12, TeX Live 2023 (paper builds there). `/srv/cellforge-*` are yours; `/srv/arc3-claude/` (venv with arc-agi 0.9.9 + arcengine 0.9.3, `agent/`, `logs/`, `paper-build/`) is Claude's. `arc_agi.Arcade()` fetches an anonymous API key automatically and lists 25 public games; never print or store keys.
- Computer-use control of the Codex window was lost in Claude's session; the channel was `notes/MESSAGE_FOR_CODEX.md` plus reading Codex's session log in `~/.codex/sessions`. No account creation, payments or sealed-data access without the owner.

## 7. File map
- research/claude_arc/: cells.py, fitted.py, fitted2.py, objects*.py, solve.py (run: `python solve.py training 2 8 3`; args split depth time-cap workers), results_*.json, gap_study.*, nca/, evo/, dsl_port/, grok/, multiview/
- research/claude_arc3/: README.md, explorer*.py, cells.py, run.py, sweep.sh, results/
- notes/: CLAUDE_RESEARCH.md (all results + literature 1-8), DECISIONS_2026-09-30.md, VALIDITY_REVIEW.md, GROKKING_REVIEW.md, VISION_DISINHIBITION_IDEA.md, PAPER_OUTLINE_CLAUDE.md, PAPER_DRAFT_CLAUDE.md, MESSAGE_FOR_CODEX.md, INDEX.md
- docs/: ARC_PRIZE_2026_PLAN.md (your plan), RED_QUEEN_REPAIR.md, MAKER_REPAIR_LOOP.md, SHARED_PERCEPTION_MOE.md
- runs/INDEX.md, runs/CLEANUP_PROPOSALS.md, runs/REVIEW_STATUS.md; scripts/prepublish_scan.py (+ tests), scripts/claude_*.py
