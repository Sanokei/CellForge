Gradient-free evolved tiny NCA scratch experiments (Claude, 2026-09-30). Run from a directory containing arc2/data (the public
ARC-AGI-2 json, cloned from github.com/arcprize/ARC-AGI-2): `python nca_evo.py <budget> <n_tasks>` (run 1), `python nca_run2.py 8000 40`
(run 2, 414 params), `python nca_par.py 8000 10` (parallel, stopped at 30/313), nca_diag*.py (synthetic color-swap sanity checks).
Results and caveats: notes/CLAUDE_RESEARCH.md ('Evolved-NCA scratch results').
