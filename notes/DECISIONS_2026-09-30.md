# Decisions delegated by the owner (2026-09-30): "decide with AGI as a goal in mind"

Principle used: AGI is the long-term aim, but no artifact here is AGI and no result supports a claim of it. What serves a credible path toward it
is (a) results that are reproducible and honestly attributed, (b) evaluation data that stays sealed so any future claim means something, and
(c) low coordination friction between the agents. So: prefer reversible actions, keep provenance, never publish anything unreviewed.

1. **Two ORCHESTRATOR_LOG.md files.** Canonical log = `notes/ORCHESTRATOR_LOG.md` (tracked, 858 lines, referenced by notes/RESEARCH_LOOP.md). The
   root `ORCHESTRATOR_LOG.md` (39 lines, modified by Codex today) is a live working log and is left untouched while Codex is active. Codex is asked
   (notes/MESSAGE_FOR_CODEX.md) to write to `notes/ORCHESTRATOR_LOG.md` from now on and, when idle, to append the root entries there and leave a one-line
   pointer at the root. Rationale: one provenance log, no deletion of another agent's live file.
2. **Untracked runs/ from 09-30.** Do NOT `git add -A`. Default policy: commit only small receipts (DECISION, FREEZE, REPORT, PLAN, ledger files) after the
   flagged-row check; archives and server aggregates are gitignored (`.gitignore` updated: `runs/**/*.tar.gz`, `runs/*.tar.gz`, server-aggregate jsonl,
   crash dumps). Already-tracked files stay tracked. Rationale: protects sealed/held-out material that would invalidate later independent evaluations,
   while keeping lineage.
3. **Flagged rows (54 in runs/INDEX.md).** Gate, not a task: nothing flagged is committed, published or shared until reviewed. The review is a
   release checklist item before any public release of code/results (the ARC Prize requires open-sourcing authored code, so this review must happen before
   the submission, not after). Not reviewed yet.
4. **`lm1-open` and `real1-open` (190 MB).** Leave in place. They are tracked, referenced by the paper/notes, and archiving risks breaking paths; the space is
   not limiting (server 31 GB free, PC ~4.7 GB free is the tighter constraint, handled by keeping new data on the server). Revisit after the Nov 2 submission.
5. **Paper.** `paper/cellforge.tex` is the cleaned version; the pre-cleanup copy is preserved. The author placeholder remains for the owner.

## Release checklist (added 2026-09-30, before publishing code for the ARC Prize; deadline Nov 2 2026)

Per-row verdicts for the 54 flagged runs/ rows are in runs/REVIEW_STATUS.md (42 safe-to-commit, 9 safe-after-redaction, 2 owner decides, 1 hold). Nothing is published until every box is ticked.

1. [ ] Wait for Codex to finish active jobs (arc-grid-genes, grid-expanded-*, grid-bank-capacity-v1, grid-scaling-independent16-v1); never commit half-written receipts.
2. [ ] Run `python scripts/prepublish_scan.py <release-dir>` on the exact tree to be published. Exit code must be 0 (no HOLD: anchor, _accept, dev, final*, sealed*, heldout*, private*, server-aggregate, *.tar.gz, *.pkl, .env, id_rsa*). Review every WARN (FINAL.json names, large files, token-like strings). Run `python -m unittest scripts/test_prepublish_scan.py` first.
3. [ ] Resolve the UNKNOWN-OWNER-DECIDES rows (baselines/author_visible_strings.txt, grid-expanded-replication-772211) and the HOLD row (bench_results server aggregate and graph tarballs); confirm the 20 tracked FINAL.json files are not final-set data.
4. [ ] Apply the SAFE-AFTER-REDACTION edits on a copy: drop anchor/_accept/dev from construction runs, remove /srv/cellforge-private paths from document-crossover receipts, drop document-crossover/sources and *-staging unless licensed, strip __pycache__ from parallel-vision tarballs.
5. [ ] Remove secrets: no API keys, tokens, ssh config, server hostnames or local absolute paths (C:\Users\...) in published files; check git history of files already tracked, not only the working tree.
6. [ ] Confirm no evaluation task answers are embedded: no ARC public-evaluation or audit (SHA256 mod 5 == 0) task content, no MMLongBench-Doc heldout items, no /srv/cellforge-private content (arcagi2, bench, final*, realtext, document-crossover-v1) in code, tests, fixtures, receipts or paper.
7. [ ] License the authored code (CC0 or MIT-0 as the prize requires; add LICENSE file and headers where needed).
8. [ ] Third-party attributions: Hodel ARC-DSL (research/claude_arc/dsl_port) with its license and attribution, the arc-agi package, MMLongBench-Doc sources if any are kept; add a THIRD_PARTY notice and confirm license compatibility with CC0/MIT-0.
9. [ ] Paper author line: replace the placeholder in paper/cellforge.tex with the owner's chosen author name.
10. [ ] Re-run the tests that read runs/ on a clean clone of the release tree, then tag the commit; owner does the actual push.
