> **2026-10-02: start at `/LAB.md` (repo root). It supersedes the reading order below; everything listed here is evidence or history, not current instructions.**

# notes/ and docs/ index

Generated 2026-09-30 11:30 from file times, read-only `git ls-files` / `git status --short`, and first lines / keyword counts of each file. Owner-agent attributions are inferred from file names, mentions and notes/MESSAGE_FOR_CODEX.md and notes/RESEARCH_LOOP.md; "Likely"/"Unknown" means not stated in the file. Nothing was moved or edited. Codex is editing these directories concurrently, so dates drift.

Reading order for a new agent: docs/ARC_PRIZE_2026_PLAN.md -> notes/RESEARCH_LOOP.md -> notes/MESSAGE_FOR_CODEX.md -> notes/CLAUDE_RESEARCH.md (last sections) -> runs/INDEX.md.


## notes/

| File | Purpose | Owner agent | Last modified (2026) | Lines | Git | Status |
|---|---|---|---|---|---|---|
| `notes/CLAUDE_RESEARCH.md` | Claude side-research log: count-model baseline, literature passes 1-8 (ARC-AGI-2/3, Sakana, DSL, pure CPU approaches), evolved-NCA scratch results, hand-authored cell library baseline (57 cells; 53 -> 80 -> 95 of 1,000 training, 0/120 public evaluation per notes/MESSAGE_FOR_CODEX.md), distillation and real-evolution ideas | Claude | 09-30 11:18 | 341 | tracked, modified | current (append-only log) |
| `notes/MESSAGE_FOR_CODEX.md` | Claude-to-Codex handoff: owner decision (no gradients, LLM-written generators + hand-written cells), measured baseline numbers to put into the plan, requests (report false positives, more cells/generators, breed-then-repair in maker_operator.py) | Claude (addressed to Codex) | 09-30 11:23 | 59 | tracked | current; Codex response not verified |
| `notes/ORCHESTRATOR_LOG.md` | Chronological orchestrator log, 52 "###" entries: starts with the OCR v3 run p1 (orchestrator Claude Opus 5.5) and continues with the Codex-era language/grid experiment entries (latest entry: grid-extended-train-census-v1) | Mixed: Claude (early), Codex team paper_notes agent (later; RESEARCH_LOOP.md assigns it this file) | 09-30 04:13 | 858 | tracked, modified | current but see root ORCHESTRATOR_LOG.md (newer, untracked) |
| `notes/PAPER_DRAFT_CLAUDE.md` | Draft Paper Prize text (negative results with controls on ARC-AGI-2, CPU/no-gradient); status DRAFT for owner review; submission rules not verified | Claude | 09-30 10:50 | 193 | tracked | current draft |
| `notes/PAPER_OUTLINE_CLAUDE.md` | Short outline for the same Paper Prize entry | Claude | 09-30 10:45 | 28 | tracked | likely superseded by PAPER_DRAFT_CLAUDE.md (same day; not stated) |
| `notes/RESEARCH_LOOP.md` | Active coordination file: owner instructions (continue loop, one cohesive language model, no gradients), agent roles (tool_reasoning, language_reasoning, paper_notes, root), server job rules, running record of receipts | Codex (root agent; file states "Root owns ... this coordination file") | 09-30 11:26 | 328 | untracked | current (actively edited) |
| `notes/VALIDITY_REVIEW.md` | Literature validity check of the breed-then-repair maker loop and Red Queen idea (abstract/search level) | Likely Claude (format matches CLAUDE_RESEARCH literature passes; author not stated) | 09-30 11:19 | 48 | tracked | current |

## docs/

| File | Purpose | Owner agent | Last modified (2026) | Lines | Git | Status |
|---|---|---|---|---|---|---|
| `docs/ARC_PRIZE_2026_PLAN.md` | Bounded offline ARC Prize 2026 entry plan: readiness, feasibility, evidence table, Claude baseline status (read-only) | Codex (Claude supplied numbers via MESSAGE_FOR_CODEX.md) | 09-30 11:11 | 540 | untracked | current (owner-designated plan) |
| `docs/BREEDING.md` | LLM-directed breeding protocol breeding/1; contains an "owner override" section authorizing automatic mutation/crossover (2026-09-30) | Unknown (original build, amended later) | 09-29 22:16 | 98 | tracked, modified | current with amendment |
| `docs/DELEGATION.md` | Hierarchical delegation protocol delegation/1 (contracts/unit tests down the ladder) | Unknown (original build, 09-29) | 09-29 17:13 | 72 | tracked | historical/current for maker runs |
| `docs/DOCUMENT_CROSSOVER_PROTOCOL.md` | Document understanding crossover protocol v2 (MMLongBench-Doc, freeze official repo commit) | Likely Codex (untracked; no Claude mentions) | 09-29 22:06 | 179 | untracked | current protocol (documents results: runs/document-crossover) |
| `docs/EXTENSION_DESIGN.md` | Extension design: vocabularies, modalities, component reuse; new protocol versions only | Unknown (original build) | 09-29 11:29 | 55 | tracked | historical |
| `docs/FRONTIER_BENCHMARKS.md` | Frontier benchmark protocol separating three task interfaces; scores only when exported artifact itself answers | Likely Codex (untracked) | 09-29 20:17 | 111 | untracked | current |
| `docs/FRONTIER_COMPARATOR_PLAN.md` | Document comparator predeclaration, plan only (document-frontier-comparator-plan/1) | Likely Codex (untracked) | 09-29 22:06 | 113 | untracked | current plan; not executed per its title |
| `docs/LANGUAGE_CORE_OPTIONS.md` | Language core decision proposal, not measured scaling; top note corrects the primary arm: Hedge-style multiplicative mixer prohibited (no gradients) | Likely Codex (untracked) | 09-30 01:19 | 397 | untracked | current with corrections; parts on adaptive mixer superseded by owner decision |
| `docs/MAKER_GUIDE.md` | Maker agent prompt contract maker-guide/1 (OCR graph authoring rules) | Unknown (original build) | 09-29 11:11 | 176 | tracked | current for maker runs |
| `docs/MAKER_GUIDE_CLF.md` | Addendum: mnist1d/1 (maker-guide-clf/1) | Unknown (original build) | 09-29 17:48 | 62 | tracked | current for mnist1d |
| `docs/MAKER_GUIDE_LM.md` | Addendum: tinylm/1 and cell-cluster-expert ladder (maker-guide-lm/1) | Unknown (original build) | 09-29 11:45 | 98 | tracked | current for tinylm |
| `docs/MAKER_GUIDE_REAL.md` | Addendum: tinyreal/1 (maker-guide-real/1) | Unknown (original build) | 09-29 16:34 | 26 | tracked | current for tinyreal |
| `docs/MAKER_OPERATOR.md` | Manual maker operator protocol: proposal accounting in src/cellforge/construction/maker_operator.py, not an LLM client | Likely Codex (untracked) | 09-29 23:31 | 53 | untracked | current |
| `docs/MAKER_REPAIR_LOOP.md` | Breed-then-repair: harness as weight-writing loop for the maker LLM (owner idea 2026-09-30) | Likely Claude (MESSAGE_FOR_CODEX.md asks Codex to read it) | 09-30 11:17 | 73 | tracked | current proposal; implementation status unknown |
| `docs/RED_QUEEN_REPAIR.md` | Combined design: Red Queen arms race plus maker repair; explicitly a design to be tested, not a result | Likely Claude (builds on VALIDITY_REVIEW and CLAUDE_RESEARCH; author not stated) | 09-30 11:20 | 100 | tracked | current design note |
| `docs/ROADMAP.md` | Candid roadmap: can direct LLM-authored construction scale beyond OCR | Unknown (original build) | 09-29 11:29 | 100 | tracked | historical |
| `docs/SHARED_PERCEPTION_MOE.md` | Design note: shared perception, routed experts for the ARC Prize 2026 tracks; no code run, numbers quoted from named repo files | Likely Claude (19 Claude mentions; pointer in CLAUDE_RESEARCH.md) | 09-30 11:18 | 171 | tracked | current design note |
| `docs/SYSTEM12_DESIGN.md` | System 1 / System 2 composition, design only | Likely Codex (untracked) | 09-30 01:19 | 118 | untracked | current design note |

## Root-level and paper/ files related to notes

| File | Purpose | Owner agent | Last modified (2026) | Lines | Git | Status |
|---|---|---|---|---|---|---|
| `ORCHESTRATOR_LOG.md` | Root-level log, 12 "###" entries (grid scaling genes, selector, capacity etc.); untracked; newer than notes/ORCHESTRATOR_LOG.md (mtime later) and apparently continues it | Probably Codex paper_notes agent (unverified) | 09-30 11:26 | 39 | untracked | current; duplicate filename with notes/ORCHESTRATOR_LOG.md - owner should decide which is canonical |
| `AGENT_HANDOFF.md` | Original spec: CellForge OCR v3 recursive architecture handoff (Rev 3.0, 09-29) | Unknown (project origin) | 09-29 10:41 | 512 | tracked | historical; describes itself as superseding v1/v2 |
| `MEMORY.md` | Original spec: evidence-backed visual memory for OCR v3; read with AGENT_HANDOFF.md | Unknown (project origin) | 09-29 10:41 | 161 | tracked | historical |
| `paper/CLEANUP_NOTES.md` | Diagnosis for cleaning cellforge.tex; build on server cellforge-1; check_numbers.py | Likely Claude | 09-30 11:22 | 70 | tracked | current |
| `paper/cellforge.tex` | Main paper source (tracked, modified) | Codex paper_notes agent (per RESEARCH_LOOP.md); Claude cleanup copies alongside | 09-30 11:23 | 755 | tracked | current |
| `paper/cellforge_clean.tex` | Cleaned copy of the paper; original left untouched (paper/CLEANUP_NOTES.md) | Claude (cleanup notes dated 09-30) | 09-30 11:21 | 755 | tracked | proposal/alternative |
| `paper/cellforge_pre_cleanup.tex` | Snapshot of paper before cleanup | Unknown | 09-30 11:23 | 1313 | tracked | reference copy |
| `paper/check_numbers.py` | Number-consistency check for the paper | Unknown | 09-30 11:21 | 44 | tracked | tool |

## Where to find what

| Need | Location | Notes |
|---|---|---|
| Current plan | `docs/ARC_PRIZE_2026_PLAN.md` | Bounded offline ARC Prize 2026 entry plan; plan and readiness, not a submission or score prediction (its own opening). |
| Research log (Claude) | `notes/CLAUDE_RESEARCH.md` | Literature passes 1-8, baseline numbers, distillation and evolution ideas. Sections are dated; last section is the shared-perception pointer. |
| Orchestrator log (Codex-era experiments) | `notes/ORCHESTRATOR_LOG.md` and newer untracked root `ORCHESTRATOR_LOG.md` | Two files with the same name; see flag above. |
| Coordination | `notes/RESEARCH_LOOP.md` (roles, server rules, owner instructions), `notes/MESSAGE_FOR_CODEX.md` (Claude -> Codex requests), `notes/VALIDITY_REVIEW.md` | RESEARCH_LOOP is the active one. |
| Paper | `paper/cellforge.tex` (+ `.pdf`), `paper/cellforge_clean.tex`, `paper/CLEANUP_NOTES.md`, `paper/check_numbers.py`; Claude drafts at `notes/PAPER_DRAFT_CLAUDE.md`, `notes/PAPER_OUTLINE_CLAUDE.md` | `paper/build.log` is stale (per CLEANUP_NOTES.md); build on server with latexmk. |
| Experiment receipts | `runs/INDEX.md` (this index family), `runs/CLEANUP_PROPOSALS.md` | Every runs/ entry mapped to its receipts, status and references. |
| Claude ARC-AGI-2 code (CPU, no gradients, no LLM at inference) | `research/claude_arc/` (`cells.py`, `fitted.py`, `fitted2.py`, `solve.py`, `results_*_d2*.json`, `objects*.py`, stats scripts) | Read-only for Codex per MESSAGE_FOR_CODEX.md. |
| Evolution experiments (Claude) | `research/claude_arc/evo/` (`archive.py`, `redqueen.py`, `common.py`, `test_evo.py`, archive/result JSON) | Untracked. |
| Hodel ARC-DSL port | `research/claude_arc/dsl_port/` (README.md describes it; `third_party/`, `cache/`) | Untracked. |
| ARC-AGI-3 code | `research/claude_arc3/` (`explorer.py` training-free state-graph explorer, `cells.py`, `run.py`, `sweep.sh`) | Runs against server dir `/srv/arc3-claude` (hard-coded in `run.py` default and `sweep.sh`). Untracked. |
| Evolved-NCA scratch | `research/claude_arc/nca/` (README.md, nca_*.py, logs; tracked) | CLAUDE_RESEARCH.md says these scripts were originally kept outside the repo. |
| Codex grid/language code | `src/cellforge/` (runtime, reference operators, construction/maker_operator.py, cli.py), `scripts/`, `tests/` | Modified files listed in git status. |
| Server: ARC-AGI-3 (Claude) | `/srv/arc3-claude` on `cellforge-1` | Not listed or opened in this pass. |
| Server: Codex | `/srv/cellforge-inbox` (uploaded source bundles, bench-*/clf-* workspaces), `/srv/cellforge-public` (job outputs, *-resource.txt, final_results*.jsonl), `/srv/cellforge-private` (sealed: arcagi2, final*, bfcl-source, realtext, grammar, document-crossover-v1, evolution pilots) | Private must never be read or published; listing viewed only. |

