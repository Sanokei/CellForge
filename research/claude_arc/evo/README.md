# evo/: diversity archive and Red Queen arms race over the hand-written ARC cell library
No gradients, no learned weights, no LLM, CPU only, max 3 worker processes. Splits: crc32(task_name)%5; bucket 0 = VALIDATION
(228 public training tasks, never used for selection), buckets 1-4 = DEVELOPMENT (772). Evaluation-set tasks are never read.
Only cells.py / fitted.py / fitted2.py are imported (objects routine excluded because that file is under concurrent edit);
the search loop is re-implemented in common.py (same semantics: depth<=2 enumeration, demo-consistent incl. fitted colour map,
2 attempts, fitted routines appended after chains, up to 4 consistent chains).

Run from this directory (cd research/claude_arc/evo):
    python test_evo.py                       # unit tests (signature, novelty archive, niche archive, generators, solves, smoke)
    python archive.py build                  # ~15 min with contention: probe signatures, DEV solve table, archive.json
    python archive.py eval 0.15              # also 0.3 and 5 -> mech1_val_cap<cap>.json (VALIDATION bucket)
    python redqueen.py evolve 80 <seed>      # seeds 0 1 2 -> rq_seed*.json, ctrl_seed*.json (~3 min each, run in parallel)
    python redqueen.py val 0.15 0 1 2        # -> mech2_val_cap0.15.json (VALIDATION + DEV secondary)
Files: common.py, archive.py, redqueen.py, test_evo.py, fit_cache.pkl (fitted-routine results, reused by all arms).
