# Local operational readiness, 2026-10-06

The desktop can import and exercise the cell runtime without installation. A
full evolution phase cannot yet run legitimately: its saved legacy seed and ARC
training data are absent, and no absolute credit reading was obtained. No maker
turn, campaign, remote job, dependency installation, commit or deployment was
started during this readiness pass. This is operational evidence, not a new
weights result or evidence of chatbot capability.

## Scope and repository

The parent relayed owner authorization for desktop execution using existing
Codex access, bounded necessary checks, and a total budget of **62,000 Codex
credits** (superseding 65,000). Separate paid APIs/cloud infrastructure, Hetzner
access, top-ups, commits and deployments remain excluded. No model-worker batch
is authorized by a percentage reading alone; an observable budget method or an
explicit initial call cap must be settled before dispatch. ARC evaluation access
and closed lane restrictions were not expanded.

Repository: `D:\Github\CellForge`, HEAD
`64a961d3c84bc557c6541206243c2e99c36fe181`. Existing dirty work was preserved,
including the user's `.gitignore` change. This pass changes only this report,
its evidence log, a handoff note, and status documentation. No protected test or
floor was edited. No credentials, auth/session files or `.env` content were read.

## Desktop and runtime

- Intel Core i5-7400 @ 3.00 GHz; four logical CPUs.
- Physical RAM: 17,070,628,864 bytes; available at observation:
  9,065,775,104 bytes. D: free space: 740,284,461,056 bytes.
- GPU remains unverified: WMI hardware queries were denied and `nvidia-smi`
  was absent from PATH. The checks here required no GPU.
- Default Python 3.12.4 lacks NumPy/Pillow. Existing bundled Python
  `C:\Users\wkeif\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
  is 3.12.14 and successfully imports NumPy 2.3.5 and Pillow 12.3.0.
  It was used without changing its environment or installing anything.
- No Python job was observed in the process-name snapshot. Existing Codex
  desktop/plugin processes were left alone. This is not a comprehensive job
  inventory, and no server status was queried in this pass.

## Checks actually executed

Each selected check process had a 30-second subprocess timeout, with
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`PYTHONDONTWRITEBYTECODE=1`. No full campaign or broad test suite was run.

1. The default `scripts/evolution_epochs.py` preview succeeded with disabled
   execution, disabled workers, zero allocated calls and two planned phases.
   Plan hash:
   `0427aa8d9ec8e7f74b5269c0b566db85928af13ea0aca20eb2092846580a2ce3`.
   `runs/weights-readiness-preview-20261006` did not exist afterward.
2. `head_driver` and `skillcore` imports succeeded; 53 templates were registered
   without loading the missing legacy core.
3. Three existing `tests.test_skill_niches` checks passed in 0.023 seconds
   (0.563 seconds including interpreter startup): demonstration isolation,
   tensor/chain round-trip and a valid zero-tensor negative control. Names and
   exact output are in `local-readiness-20261006.log`.
4. An in-memory typed checkpoint diagnostic using the existing synthetic glyph
   truth-cell fixture passed: export/prediction, query-output stripping, valid
   shape-preserving zero override that changes the prediction, preservation of
   the original artifact, scalar-override rejection and tampered-hash rejection.
   Process duration was 0.344 seconds. This fixture is an oracle test input, not
   an LLM founder, seed run, independent benchmark or measured evolution gain.

The standard and ARC-goal suites were not rerun in this bounded readiness pass.
No claim is made that they are green. The protected zero-control helper defect
already documented in `docs/EVOLUTION_INTEGRATION.md` remains unchanged and must
be resolved under the appropriate authorization before trusting that gate for a
new scientific qualification.

## Seed and task dependency

`head_driver` imports hardcoded defaults from `head_arc_cells`:

- Seed source:
  `/srv/cellforge-inbox/two-loop-20261001/runs/weights-arc-moe-smallskill-v2`.
- Task source: `/srv/cellforge-private/arcagi2/data/training`.
- Required seed files include `pop1_core.py`, `parent.json` and `TASKS.json`.
  Task IDs from `TASKS.json` then require their corresponding training JSONs.

Neither source exists on this desktop. Git's tracked-file search and a
filename-only scan of repository `runs`, `research` and `data` found no
`pop1_core.py`, `parent.json`, `TASKS.json`, `FINAL_WEIGHTS.json`, `SKILL_BANK.json`
or `BUDGET.jsonl`. The run index and historical notes reference server artifacts;
they do not supply those artifacts. The documented deployment command uses the
same server-only seed and virtual environment. An arbitrary replacement core or
truth-cell fixture would not preserve historical provenance.

Do not try `--execute-phase` merely to discover these missing inputs: the epoch
runner deliberately reserves the phase allowance before constructing the driver,
so a missing seed would leave an incomplete reservation requiring reconciliation.

Minimal options for the parent/owner:

1. Supply the path to an existing authentic local export of the seed and its
   original provenance, task manifest and authorized public-training task files.
   Add explicit local path configuration and verify content/partition hashes
   before attempting one disabled-worker phase. No such export was found here;
   this option does not authorize a server transfer or download.
2. If no export exists, explicitly scope a new founder experiment with a
   separate bootstrap that does not pretend to resume the historical run.
   Existing generated niches can supply search examples and untouched synthetic
   holdouts, but oracle fixture weights must never be relabeled LLM founders.
   This requires a concrete small experiment and maker limit before model calls.

## Maker route and accounting

Installed desktop CLI: `codex-cli 0.160.0`.
`codex login status` returned `Logged in using ChatGPT`. This establishes the
configured login route; it does not establish remaining credits or a hard cap.
The existing maker removes `OPENAI_API_KEY` and `AZURE_OPENAI_API_KEY` from its
child environment and invokes `codex exec`. Its default executable/ledger paths
are server paths unless `CODEX_BIN`, `MAKER_LEDGER` and `MAKER_CAP` are explicitly
configured. The legacy direct maker also retains a server executable default.

The current project meter records `used_percent`/`resets_at` from Codex rollout
events. It does not count absolute Codex credits. Concurrent activity can consume
the same account budget; percentages and tokens cannot be converted into credits
using an assumed exchange rate.

The installed CLI's generated protocol schema exposes read-only
`account/rateLimits/read` with optional credit data. Two bounded attempts were
made using the supported `app-server proxy` and `app-server --stdio` routes:
both processes exited 1 before the initialization response (about 1.9 seconds
each). No account balance, quota baseline or usage estimate was returned, and
neither attempt created a model turn. No raw authenticated endpoint, credentials
or security-setting change was used as a workaround. The exact cause was not
established from the captured response stream.

Therefore **no current absolute credit baseline is available** and no hard
62,000-credit enforcement claim is justified. The ongoing Codex conversation
itself can consume the shared service budget; zero additional maker calls does
not imply zero total credits consumed. Before any authorized model trial, settle
the budget observation method and a conservative initial call cap with headroom.
Do not launch worker batches while that remains unresolved.

## Interpretation and next bounded milestone

The current bank combines literal numeric tensors with hand-implemented grid
templates and demonstration-error routing. Typed composition and evolution
receipts improve reproducibility; they do not demonstrate a general learned
language model or a capable chatbot. The immediate operational blockers are
authentic input artifacts and accepted accounting, not installing dependencies.

After those inputs are supplied, the next local milestone should be one explicit
zero-worker phase under a short wall-clock limit and a small fixed development
panel. Verify checkpoint round-trip, valid zero/random controls, seed separation,
absence of query-output leakage and no retained-bank regression. Only then
propose a separately bounded founder/mutation comparison with matched random
proposal counts and untouched evaluation data. Stop after the agreed milestone;
an open-ended AGI criterion is not an execution plan.

Source SHA256 observed for reproducibility:

| File | SHA256 |
| --- | --- |
| `scripts/head_driver.py` | `36059225227652d03f0f509dd4a8f8bba5c6231ee766f04f9420ab80151724df` |
| `scripts/skillcore.py` | `3f6b926ecb7650f041c2e0d8ee42dc0271588c76701ce9e30414b3007bcc5b8c` |
| `scripts/evolution_epochs.py` | `cf008d2953239c0cfe3d71d289cd3120413539119bf3e4ab72416b10a23f8e31` |
| `scripts/weight_workers.py` | `54890bfe12e565c667f80b68a4df7bd5e286a0b5553c170bf9c96d3e2cf4aa2e` |
