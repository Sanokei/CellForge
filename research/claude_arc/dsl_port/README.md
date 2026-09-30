# dsl_port: Hodel ARC-DSL object-centric fitted cells

Adds ARGA-style fitted cell families on top of the hand-written library in `research/claude_arc/`
(`cells.py`, `fitted.py`, `fitted2.py`, `solve.py` are imported/copied, never edited). No gradients, no learned weights,
no LLM calls, CPU only, <=2 worker processes.

## Files
- `third_party/arc_dsl/` : `dsl.py`, `arc_types.py`, `constants.py`, `LICENSE` copied unmodified from
  https://github.com/michaelhodel/arc-dsl (MIT); see `ATTRIBUTION.md`. `solvers.py` and re-arc are NOT copied.
- `adapter.py`  : numpy <-> tuple-of-tuples grids; `wrap_cell` (try/except, 30x30 cap) for any DSL Grid->Grid function.
- `objfam.py`   : the families. SEL (abstraction x selector x action), CROP (selector x crop x 8 orientations x up/downscale),
                  REC (recolor-by-property tables), LINE (connect/shoot/enclosed fill), MACRO (depth<=2 DSL macros + colour map).
                  Entry point `run_all(train, test_in, tcap)` -> candidates with `prog(np_grid)` closures and test predictions.
- `mine_bigrams.py` / `mined_bigrams_dev.json` : static aggregate primitive n-gram counts from solvers.py, DEVELOPMENT ids only.
- `gen_examples.py` : offline re-arc wrapper (own process; dev families only). `coverage.py` : per-family coverage table.
- `evaluate.py` (stages `base`, `dsl`), `analyze.py` : scoring. `test_dsl_port.py` : unit tests on synthetic grids.

## Split
`crc32(task_name) % 5 == 0` -> VALIDATION (scored only at the end); buckets 1-4 -> DEVELOPMENT. Training tasks only
(`ARC2/data/training`); evaluation tasks are never opened.

## Commands (run from this directory; set `ARC2` / `REARC` to override the default scratchpad paths)
```
python -m unittest test_dsl_port -v
python mine_bigrams.py                                   # static stats, dev ids only
python gen_examples.py 40                                # re-arc examples for dev families -> cache/gen/*.json
python evaluate.py dsl  cache/res_dsl.jsonl  --workers 2 --tcap 12 --gen-k 20
python evaluate.py base cache/res_base.jsonl --workers 2
python analyze.py cache/res_base.jsonl cache/res_dsl.jsonl
python coverage.py cache/coverage.jsonl --episodes 4 --tcap 4 --workers 2
```
Before any multi-process run check that other jobs are not saturating the machine (the PC has ~10 GB RAM).

## Contamination handling
`solvers.py` is never imported or executed; it is parsed with `ast` and only aggregate counts over development ids are kept.
Generators are offline curriculum/selection tools: they exist only for known ARC-1 ids and are not part of any test-time path.
Generated examples exist only for DEVELOPMENT families (hard guard in `gen_examples.py`), so validation numbers use no
generator filtering.
