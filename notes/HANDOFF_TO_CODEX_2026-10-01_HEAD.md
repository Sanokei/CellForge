# Handoff to Codex from head-center (Claude), 2026-10-01 — read this first, then work the list at the bottom

You have been idle since ~18:45 UTC. Meanwhile a working loop was built. Everything below was measured on the server; nothing here is a claim of AGI or a frontier result.

## 1. What exists (all in the shared repo; server work dir `/srv/cellforge-inbox/head-20261001/`, ssh alias `cellforge-1`)
A **skill bank**: narrow skill cells whose behaviour lives in literal tensors written by `gpt-6.1-sol low` (the owner-configured server route: `codex exec`, ChatGPT subscription, read-only sandbox, no API key), competing against matched random-tensor controls, merged by a demonstration-error router, bred into chains, and scored on real ARC.

| piece | file |
|---|---|
| cell templates (structure = code, behaviour = tensors), chains, leave-one-out router, cached scorer | `scripts/skillcore.py` |
| 39 niche generators with hidden ground truth (`apply`) and verification | `scripts/niches.py` |
| continuous driver: niche stages -> breeding -> maker mutation, ARC scoring, resumable | `scripts/head_driver.py` |
| maker call helper (the `codex exec` invocation) | `scripts/head_arc_cells.py` (`ask_maker`) |
| guard: lane snapshot + 54 tests; status file you can read without ssh | `scripts/lane_snapshot.py`, `tests/test_hard_lane_focus.py`, `reports/lane-snapshot/HEAD_STATUS.md`, `HARD_LANES.json` |
| niche tests (truth solves own tasks, random controls, speed paths) | `tests/test_skill_niches.py` (14 tests) |
| memory notes | `C:\Users\wkeif\.claude\projects\...\memory\skillbank-system-state.md` |

Run: `ssh cellforge-1`, then in `/srv/cellforge-inbox/head-20261001`:
`PYTHONPATH=/srv/cellforge-inbox/two-loop-20261001/runs/weights-arc-moe-smallskill-v2/src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 setsid nohup /srv/arc3-claude/venv/bin/python head_driver.py runs/weights-head-skillbank-vN --rounds 12 --seed-from runs/weights-head-skillbank-v(N-1) > log 2>&1 < /dev/null &`
Add `--fake` for an ORACLE run (true tensors instead of the maker): it tests plumbing and measures the ceiling of a niche set; it is never a maker result. Stop a run cooperatively with a `STOP` file in its run dir. Error cache is shared: `/srv/cellforge-inbox/head-20261001/ERRCACHE.pkl`.

## 2. Measured (public ARC training; see section 3 for the split)
- Held-out ARC tasks solved: **1 -> 53 of 579** (run v10, 22:50 UTC). Report split (never used for any selection): **0 -> 28 of 302**. DEV 0 -> 13 of 100.
- All 39 defined niches earned by `v10` (5 added after v8: lineconn +3, cropcolor +3, blockpick, seqcrop, countout +0). All written by the maker in ONE generation except `agree` (second generation fixed one failing pattern from counterexamples = the one successful maker mutation).
- Biggest lifts: context tables (+7), tiling (+6), neighbour rules (+6), objects (+5), scaling (+4), association (+3). Language and OCR niches are earned (association, sequence, recall, induction, seqops, selcopy, grammar, successor, agree, grammar2 (agreement), analogy, tooluse, glyph, ocrdigit, ocrseq, legend) but add ~nothing on ARC; they exist for language/OCR competence, not ARC score.
- Breeding (chains, admitted only on strict ARC dev+validation gain): ~1 child per 2 rounds; the newest child (halfbool -> neighbour) gained +2 on the clean report split.
- Maker weight-mutation of ARC near-misses: ~35 attempts, 0 admitted. The maker answers "no tensors of this template can express the rule". Each answer revealed a missing STRUCTURE; adding the structure by hand then gained tasks (objmove2, flood_through, snap_to_line, context_table2).

## 3. Rules that make the numbers honest. Do NOT break them.
1. Held-out = 1000 public-training tasks minus the 100 declared DEV ids minus the reserved SHA256-mod-5 audit population (excluded if EITHER the bare id or `<id>.json` hashes into it). Split by `sha256("val:"+id) % 2` into **validation (selection)** and **report (never used to select anything)**. Quote report-split numbers as the clean ones. Never read or score the ARC evaluation split.
2. The maker sees only generated niche tasks (demonstrations, plus its own previous attempt as predicted-vs-expected). It never sees ARC held-out or report tasks.
3. Every submitted tensor is charged; random arm gets the same number of literal positions. Report random-control rates (they are not zero for in-context tables; that is expected).
4. Cell ids are content hashes and the error cache is keyed by them: **never change the behaviour of an existing template**. Add a `*2` template instead (see `object_move` -> `object_move2`).
5. Do not edit `HARD_LANES.json`, `tests/hard_lane_*`, `scripts/lane_snapshot.py`, the split/audit code in `head_driver.py`, or `ARC_FLOOR.json`. Ask in `notes/CODEX_TO_HEAD_<date>.md`.
6. Before claiming anything: `python -m unittest tests.test_skill_niches tests.test_hard_lane_focus -v` and paste the result. Known red: `lanes_exist`, `hard_lanes_earn_credit` (tool use and multistep reasoning lanes do not exist yet). That is the honest state.
7. No API calls, downloads or gradients beyond the maker route above without the owner.

## 4. How to add a niche (the unit of work)
1. Template in `scripts/skillcore.py`: a pure function `f(params, grid, demos=None)`, a `TEMPLATES[...]` entry with `shapes`, `fn`, `aware` (True if it reads demonstrations), `doc` (this text IS the maker's spec: be exact). Splice before `def load_legacy`.
2. Niche class in `scripts/niches.py` with `patterns(limit, seed)` returning dicts with hidden `truth` tensors and `apply`; set `verify=True` and a `discriminates()` filter when simpler cells could solve the task; register in `NICHES` and `LIMITS`; append to `QUEUE` in `head_driver.py`.
3. Local test: extend the name lists in `tests/test_skill_niches.py`; run with `--fake` on the server to get the oracle ceiling on real ARC before spending maker calls.
4. Real run as above; read `HISTORY.jsonl` / `LEADERBOARD.json` in the run dir.
`general=True` niches expect ONE cell to solve every pattern (in-context learners); instance niches expect one cell per pattern.

## 5. Your task list, in priority order (nobody else is doing these)
A. **Tool use, real design.** The `tooluse` niche is weak (random allowed-tool sets pass 25-75%; its only tensor is a multi-hot). Build a real one in the same framework from `/srv/cellforge-private/bfcl-source`: pick a function from several by a description token, extract typed arguments, multi-argument calls. Needs a template whose tensors really carry the skill. Register as lane `tool-use` (tell me the run dir; the guard already requires `weights-hard-tool-use`).
B. **Multistep reasoning.** Chains are two-step programs today. Add depth-3 breeding (`breed()` currently chains depth-1 parents only) and a niche family of 3-step composite tasks; or revive `resource-planning` (stuck at 7/8 for 34 generations: the greedy policy cell cannot express the 11-step optimum; give it lookahead structure instead of more generations). Register as `reasoning-multistep` (`weights-hard-multistep-reasoning`).
C. **Structures the maker said it cannot write** (from declined mutations, selection tasks only): follow a wall path from a marker without taking interior spurs (`cc9053aa`, `f1bcbc2c`); rotate objects around an anchor pixel (`1a244afd`); swap interior colours with the frame (`952a094c`); size/colour+holes mapping conflicts (`009d5c81`). Each = one new template + niche + oracle ceiling.
D. **Real-vocabulary text.** All language niches use 10-token or 24-word toy vocabularies. Data: `/srv/cellforge-private/{final_lm,grammar,realtext}`, `data/tinylm`, `reference/language_curriculum.py`. Design a byte-level interface (tensors stay small: selectors, small embeddings) and port: in-context word association, next-byte n-gram, induction on bytes, agreement on real sentences. Needs a text task type beside the grid one.
E. **ARC-3.** `arc_agi` is installed on the server (`/srv/arc3-claude`). Perception cells (components, bounding boxes, motion) and action-selection cells in this framework; frames are grids, actions are tool calls (links to A).
F. **Independent verification** (cheap, valuable): re-run `scripts/head_heldout_eval.py`-style scoring on a FRESH seed split; read `scripts/skillcore.py` for bugs (router tie bias, leave-one-out contexts, `passes`); try to break the audit/split exclusion; report anything that inflates the numbers.
G. **Speed.** v9 added a persistent error cache, a pixel-key cache and a contender prefilter (a niche stage now takes ~50 s end to end on a warm cache; cold start used to cost 5+ min). Next: vectorise `nkey`/`nkey2` with numpy windows, and parallelise cold error computation over the server's 2 cores.

## 6. Open question for the owner (not yours to decide)
Structure changes are made by hand today (by me, guided by the maker's declined mutations). Letting the maker propose new structures (code) would be faster but goes beyond the weights-only standard; structural repairs would need charged, provenance-tracked rules. Do not start that without the owner.

## 7. Coordination
Write `notes/CODEX_TO_HEAD_<date>.md` when you start a lane or find a bug. I watch `reports/lane-snapshot/HEAD_STATUS.md` (refresh: `python scripts/lane_snapshot.py`). Do not write inside `runs/weights-head-skillbank-v*` of a run that is live; start your own run dir. The server has 2 CPUs: one driver at a time plus your experiments is the limit.
