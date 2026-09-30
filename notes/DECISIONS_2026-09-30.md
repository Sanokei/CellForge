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
