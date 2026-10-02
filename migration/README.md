# CellForge Windows migration snapshot — 2026-10-02

This branch preserves the current desktop source and report changes. It does not restart experiments, change STOP files, merge into master, or alter scientific claims.

The associated private GitHub release contains the complete screened project snapshot, including uncommitted run evidence, worker folders, linked worktrees, and a verified complete Git history bundle. `inventory.json` records each original path, size, and SHA-256. `scope-summary.json` records counts by folder.

The ZIP is content-addressed: each unique file is stored as `objects/<sha256>`; restore each manifest entry to its recorded path by copying the matching object. `receipts/history.bundle` preserves 27 refs, including all local branches. Clone the bundle to recover commit history, then overlay the desired snapshot root; the manifest, rather than a stale checkout, defines the current files.

Excluded: Git administrative/authentication metadata and rebuildable cache/environment directories (`.git`, `__pycache__`, `.pytest_cache`, `.uv-cache`, `.venv-tests`, `.venv`, `node_modules`). Linked worktrees under `.claude/worktrees` are preserved independently. No desktop authentication credentials were copied. ARC server private data was neither accessed nor copied. Synthetic local project datasets and generated image fixtures are included. All 296 nested project archives were screened before inclusion.

Hetzner backup: `cellforge-1:/srv/cellforge-inbox/windows-migration-20261002/`. The snapshot is separate from all live server working directories.

Local cleanup requires verified server/GitHub receipts and resolved ownership. This branch does not claim cleanup is complete.

Supplement: 126 authored CellForge Temp scratchpad files (854,588 bytes) are preserved in cellforge-temp-scratch.zip. Raw ARC benchmark/dependency downloads and eight private session-output folders/two image folders remain excluded and untouched; see temp-inventory.json. These exclusions and linked-worktree ownership must be resolved before complete local cleanup.

Artifact coverage verified: all 28 v2 parts were uploaded, their GitHub SHA-256 digests matched, and every part was downloaded and hashed in order. Reassembled bytes (228,744,131) match the fully restored Hetzner snapshot. Current transport manifest: parts-v2.json. The original four-part parts.json is retained as the superseded transport plan after an HTTP 408; use only v2 parts for restoration.

Local cleanup has not occurred. Mixed private session outputs (219 files, 56 empty) and the two project/reference screenshots remain local. Direct inspection of login-related excerpts and copying the private-session images were rejected by automatic approval review. Seven linked Claude worktrees still need a current ownership handoff; old timestamps and lack of matching processes alone are insufficient.
