# CellForge Windows migration snapshot — 2026-10-02

This branch preserves the current desktop source and report changes. It does not restart experiments, change STOP files, merge into master, or alter scientific claims.

The associated private GitHub release contains the complete screened project snapshot, including uncommitted run evidence, worker folders, linked worktrees, and a verified complete Git history bundle. `inventory.json` records each original path, size, and SHA-256. `scope-summary.json` records counts by folder.

The ZIP is content-addressed: each unique file is stored as `objects/<sha256>`; restore each manifest entry to its recorded path by copying the matching object. `receipts/history.bundle` preserves 27 refs, including all local branches. Clone the bundle to recover commit history, then overlay the desired snapshot root; the manifest, rather than a stale checkout, defines the current files.

Excluded: Git administrative/authentication metadata and rebuildable cache/environment directories (`.git`, `__pycache__`, `.pytest_cache`, `.uv-cache`, `.venv-tests`, `.venv`, `node_modules`). Linked worktrees under `.claude/worktrees` are preserved independently. No desktop authentication credentials were copied. ARC server private data was neither accessed nor copied. Synthetic local project datasets and generated image fixtures are included. All 296 nested project archives were screened before inclusion.

Hetzner backup: `cellforge-1:/srv/cellforge-inbox/windows-migration-20261002/`. The snapshot is separate from all live server working directories.

Local cleanup requires verified server/GitHub receipts and resolved ownership. This branch does not claim cleanup is complete.

Supplement: 126 authored CellForge Temp scratchpad files (854,588 bytes) are preserved in cellforge-temp-scratch.zip. Raw ARC benchmark/dependency downloads and eight private session-output folders/two image folders remain excluded and untouched; see temp-inventory.json. These exclusions and linked-worktree ownership must be resolved before complete local cleanup.
