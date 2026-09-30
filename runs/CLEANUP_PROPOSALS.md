# runs/ cleanup proposals

**NOTHING HAS BEEN MOVED, RENAMED, DELETED, COPIED OR EDITED IN runs/.** This file only proposes actions for the owner to approve. Every proposal is unexecuted. Codex is writing into runs/ and notes/ at the same time, so all sizes below are a snapshot (2026-09-30, about 11:30 local) and will change. Companion file: `runs/INDEX.md` (entry-by-entry table, including "Referenced by" and sensitivity flags).

Method: sizes from `du`/file stats; duplicate detection by SHA-256 of every file of 20 KB or more under runs/ (files smaller than 20 KB not hashed); reference analysis by text search of tests/, scripts/, src/, docs/, notes/, paper/, research/. A text hit is not proof of a runtime dependency, and absence of a hit is not proof of safety (receipts embed hashes and relative paths, and Codex scripts may build paths dynamically).

## Snapshot facts

- About 268 top-level entries, about 515 MB, about 4,250 files (initial count was 265 entries / 496 MB; Codex added `grid-expanded-replication-772211*` while this was written).
- 841 files under runs/ are tracked by git (about 35.8 MB); everything else is untracked (293 untracked paths in `git status --short runs`) and 11 tracked files are modified. The bulk of runs/ from 09-30 (all `grid-*`, `arc-*`, `*-evidence-*`, `parallel-*`, `document-crossover`, `action-*`, `unified-byte*`, `wb6-*`, `language-*`) is not in git at all.
- Largest entries: `lm1-open` about 119 MB (graphs/ 62 MB, tasks/ 29 MB, `_accept/` 24 MB), `real1-open` about 71 MB (graphs/ 51 MB), `arc-grid-genes` about 40 MB, `action-deduction` about 30 MB, `lm1` about 30 MB, `grid-curriculum-baseline-v1` about 23 MB, `real1` about 20 MB, `grid-expanded-population-v1` about 18 MB, `baselines` about 18.6 MB.
- Largest single files: `lm1-open/graphs/31a40974.../proposal.json` 59.6 MB; `real1-open/graphs/f3fe3035.../proposal.json` 52 MB; `grid-curriculum-baseline-v1/original-synthetic-data.json` 22.9 MB; `lm1-open/_accept/O2.json` 22.8 MB and `lm1-open/tasks/r1-O2/proposal.json` 22.8 MB (nearly equal size but not byte-identical); `real1-open/tasks/s1-solo/proposal.json` 19.7 MB; `baselines/grad_n400/params.npz` and `grad_n20k/params.npz` 8.2 MB each; `lm1/graphs/2e63d1f3.../proposal.json` 7.8 MB.
- Tarballs: 77 top-level `.tar.gz` (19.1 MB) of which 48 are source bundles (10.5 MB) and 29 are server `-results`/`-receipts` tarballs; plus 33 `*.tar.manifest.json`, 40 `*-resource.txt`, 21 `*-launch.json`. 99 tarballs and 1 zip exist in total including nested ones (23.3 MB). No two tarballs are byte-identical (hash check) and none share the same size.
- Exact content duplicates (files of 20 KB or more): 98 groups, about 57.6 MB of redundant bytes. Breakdown of the largest: `action-deduction` snapshots vs composed graphs 10.1 MB; `grid-object-age8-7720x1` vs `grid-object-replication-7720x1` vs `arc-grid-genes` founders about 9.3 MB plus 3.0 MB plus 2.8 MB for paired seed directories; `arc-grid-genes` internal (expanded-breeding-v1 vs v2, scaling-two-slot-v1 vs v2, object-association-v1 vs t16-v1) 7.2 MB; `document-crossover/*-staging` + `unified-byte/artifact` + `evolution-language-minipilot-v1` founder 4.9 MB; `arc-grid-genes` vs `grid-scaling-*` 3.0 MB; `action-deduction` vs `action-language` 2.4 MB. Winner directories duplicate their `*-attempt-N` directories.

## Proposed cleanups (none performed)

| # | Proposal | Size affected | What references it (dependency risk) |
|---|---|---|---|
| 1 | Replace byte-identical duplicate files by NTFS hardlinks (same path, same bytes, one copy on disk). This changes no path and no content, so it is the only proposal with near-zero reference risk. Start with `grid-object-age8-7720x1` vs `grid-object-replication-7720x1` (same graph.json for shared attempts), `*-winner` vs `*-attempt-N`, and `action-deduction/composed-v*` vs `harness-snapshot-v*`. Verify with the SHA-256 list before and after. | up to about 57 MB | LOW for read-only archives; MEDIUM if any tool later rewrites a file in place (a hardlinked write would change both names). Do not hardlink anything Codex is still writing (`grid-expanded-*`, `grid-expanded-replication-*`, `arc-grid-genes`). |
| 2 | Archive (copy first, verify, only then consider removal) the two large open-arm construction runs `lm1-open` and `real1-open` (graphs/ and tasks/ proposal.json files, 190 MB together). Keep `ledger.jsonl`, `run.json`, `branches.json` in place. | about 190 MB | `lm1-open` and `real1-open` have no code references in tests/scripts/src; text search finds them only in `notes/ORCHESTRATOR_LOG.md` (see INDEX). Their tracked files include modified `ledger.jsonl`/`branches.json`; ledger chains reference graph hashes, so removing graphs breaks ledger verification. Do not delete; archive only. |
| 3 | Archive `lm1-open/_accept/` and `lm1/_accept/` (31 MB, gitignored, evaluation-side, not opened). Needs an owner decision because they may hold held-out acceptance results. | about 31 MB | gitignored already; references unknown (contents not opened). Verify against run ledgers (`anchor_eval` events) before moving. |
| 4 | Consolidate `document-crossover/*-staging/` (compact-v4, compiled-v5, phase-v3: each about 3 MB, each a copy of src/scripts/runs used to build the matching `*-source.tar.gz`) by relying on the tarballs. | about 9 MB | `docs/DOCUMENT_CROSSOVER_PROTOCOL.md` and `scripts/check_document_readiness.py` mention `runs/document-crossover`; staging dirs may be read by server launch receipts (`server-*-launch-*.json`). Unknown whether any launch command reads a staging path. Keep until launch receipts are checked. |
| 5 | Stale scratch directories: `_scratch/` (3 plugin test scripts, 7 KB, tracked: 3 files), `m1d1-h2/` (empty `dev/`, 0 KB), `m1d-h2/` (untracked, 317 KB, 3-line ledger), `language-core-controls/cancelled-gradient-draft/` (cancelled under the no-gradient rule), `lm1/tasks/r1-L3/scratch/` (2.5 MB, byte-identical to the sibling proposal.json). | under 4 MB | `_scratch` has no textual reference; `cellforge smoke` defaults to `runs/_smoke` (src/cellforge/cli.py) which does not currently exist. `m1d-h2` is referenced by notes/ORCHESTRATOR_LOG.md. Low risk but low benefit; keep unless disk pressure. |
| 6 | Superseded version siblings (v1 when v2 exists): `grid-context-mask-pilot-v1.tar.gz`(+manifest), `grid-structure-census-v1`, `grid-scaling-selector-evidence-v1`, `grid-object-association-control-v2` (vs v3), `grid-object-t16-evidence-v1` (vs v2), `grid-maker-slot-v1.tar.gz` (vs v2), `grid-population-pilot-v1.tar.gz` (vs v2), `language-evolution-calibration-v1.tar.gz` (vs v2) and `evolution-language-gen0-v1-failure.json`. Supersession is inferred from names only, not stated in receipts. Keep failures as evidence. | about 3 MB | Receipts of v2 often quote v1 hashes (example: gen0-v1 failure amended by v2). Treat v1 as evidence, not clutter. HIGH if deleted; fine to move into an archive with a pointer after owner review. |
| 7 | Server `-results`/`-receipts` tarballs (29, about 8.6 MB): these are copies of `/srv/cellforge-public` outputs. Candidates for archival once the matching result directory is verified complete locally. | about 8.6 MB | The matching `REPORT.md` usually cites them (for example `grid-expanded-population-v1/REPORT.md` and the launch JSON). Flagged REVIEW in INDEX (per-task outcomes); do not publish. |
| 8 | Source-bundle tarballs (48, about 10.5 MB) plus manifests (33): each is a frozen snapshot with sha256 manifest; they are the reproducibility record, and receipts cite their SHA-256. Recommend KEEPING; at most move to an archive folder together with their manifests. | about 10.5 MB + manifests | Receipts embed bundle hashes; moving breaks path-based lookups by docs/ARC_PRIZE_2026_PLAN.md and notes/RESEARCH_LOOP.md (they name paths like `runs/adjacency-evidence-v1.tar.gz`). |
| 9 | Duplicated fitted artifacts inside `arc-grid-genes` (`expanded-breeding-v1` vs `v2`, `scaling-two-slot-v1` vs `v2`, `object-association-v1` vs `object-association-t16-v1` AUTHORED_FAMILY_300.json 5.1 MB): use hardlinks (proposal 1) rather than removal. | about 7 MB | `tests/test_expanded_grid_genetics.py`, `tests/test_grid_expanded_genome.py`, `tests/test_grid_population_matched.py`, `tests/test_grid_serial_gene.py` and many scripts read `arc-grid-genes`. HIGH if moved or deleted. |
| 10 | `grid-curriculum-baseline-v1/original-synthetic-data.json` (22.9 MB, regenerable from the recorded seeds per its REPORT.md). Candidate to compress or archive after verifying the generator reproduces the recorded fingerprint hashes. | 22.9 MB | `docs/ARC_PRIZE_2026_PLAN.md`, `notes/ORCHESTRATOR_LOG.md`. MEDIUM: the file is the evidence for "2,400 distinct fingerprints". |
| 11 | `baselines/grad_n400`, `grad_n20k` params.npz (16.5 MB, gradient-trained OCR baselines, tracked in git). Candidate to move to Git LFS or an archive if the repository is to be published. | 16.5 MB | `docs/FRONTIER_COMPARATOR_PLAN.md`, `docs/LANGUAGE_CORE_OPTIONS.md`, `paper/*.tex`, `scripts/measure_grid_curriculum.py`. MEDIUM (paper cites them). |
| 12 | Logs and error files: `bench_results/*.err` (empty), `*.raw`, `lm_baselines/mlp.log`, `real_baselines/mlp.log`, `arc-grid-genes` and `grid-*-resource.txt` (40 files, 4 KB each). Keep resource logs (receipts cite them); the empty `.err` files can be ignored by git. | negligible | `scripts/run_bench_batch.sh` writes into `bench_results/`. |
| 13 | Sensitive-material review before any commit or publication (see INDEX flags): `bench_results/server-aggregate-20260930.jsonl`, `baselines/author_visible_strings.txt`, `*/anchor`, `*/dev`, `*/_accept`, `document-crossover/`, all `*-results.tar.gz`/`*-receipts.tar.gz`, `parallel-*` server receipts. Not opened by this pass. | n/a | Owner decision. Do not `git add -A` in runs/ until reviewed: almost all of it is currently untracked. |
| 14 | Decide tracking policy: most runs/ content from 09-30 is untracked (no history). Options: commit curated receipts only (REPORT.md, DECISION.json, FREEZE.json, ledger.jsonl, manifests), keep graphs/tarballs out of git. | n/a | `git status` shows 293 untracked paths in runs/ and 11 modified tracked files that Codex is editing; committing mid-run risks capturing half-written receipts. |

## Recommended .gitignore additions (proposal only; `.gitignore` was NOT edited)

Existing rules already cover `runs/*/scratch/`, `runs/*/dev/`, `runs/*/anchor/`, `runs/*/_accept/`, `private/`, `__pycache__/`, `*.pyc`, `*.egg-info/`.

```
# proposed
runs/_smoke/
runs/**/__pycache__/
runs/**/*.err
runs/**/fit_cache.pkl
research/claude_arc/evo/fit_cache.pkl
# heavy or sealed-adjacent artifacts (owner decision; receipts cite their hashes, keep copies elsewhere)
runs/*-results.tar.gz
runs/*-receipts.tar.gz
runs/document-crossover/*-staging/
runs/lm1-open/graphs/
runs/real1-open/graphs/
runs/bench_results/server-aggregate-*.jsonl
runs/baselines/author_visible_strings.txt
```

Caveats: ignoring files that are already tracked has no effect until they are untracked; do not untrack without owner approval (reference paths in receipts). The `runs/lm1-open/graphs/` and `real1-open/graphs/` lines are in the "owner decision" group because `lm1-open`/`real1-open` graphs are partly tracked already (86 and 38 tracked files).

## Suggested archive layout (for when the owner approves; copy first, verify hashes, keep stubs)

```
archive/                      (outside the repo or in a separate, ignored folder)
  2026-09-30/
    source-bundles/           <entry>.tar.gz + <entry>.tar.manifest.json (kept together)
    server-results/           <entry>-results.tar.gz, <entry>-receipts.tar.gz
    superseded/               v1 siblings listed in proposal 6
    construction-runs/        lm1-open/, real1-open/ graphs and tasks (keep ledger.jsonl in runs/)
    staging/                  document-crossover/*-staging
    ARCHIVE_MANIFEST.json     sha256 + original path for every file moved
runs/
  <entry>/ARCHIVED.md         one-line pointer + hash, created only when something is actually moved
```

Order of operations suggested: (1) wait until Codex finishes the active jobs (`grid-expanded-*`, `grid-expanded-replication-*`); (2) owner reviews REVIEW-flagged rows; (3) hardlink dedupe (proposal 1) with before/after hash check; (4) archive by copy, verify with `ARCHIVE_MANIFEST.json`; (5) only then decide on removal; update docs/notes references in the same change; (6) run the tests that read runs/ (`tests/test_compiled_runtime.py`, `test_action_language.py`, `test_grid_*`, `test_vision_specialist.py`, `test_specialists.py`, `test_long_context_memory.py`, `test_expanded_grid_genetics.py`) on a copy, not while jobs are running.

## Do-not-touch list (code or tests read these paths)

`runs/unified-byte`, `runs/unified-byte-memory`, `runs/grid-associative`, `runs/grid20`, `runs/arc-grid-genes`, `runs/p1`, `runs/parallel-vision`, `runs/parallel-deduction`, `runs/action-language`, `runs/document-crossover`, `runs/real1` (found by text search in tests/ scripts/ src/). Also every directory named in `docs/ARC_PRIZE_2026_PLAN.md`, `notes/RESEARCH_LOOP.md` and `notes/ORCHESTRATOR_LOG.md`.
