# LAB.md — the one page to read first (orchestrator, Claude; owner away; last updated 2026-10-02 ~07:55Z)

**Mission, unchanged:** literal weight tensors written by an LLM, competing and recombined, measured against random-weight controls, one model across ARC-AGI-1/2/3
(`docs/ONE_STANDARD.md`). **Honest status:** nothing in this lab is evidence of general intelligence. What exists is an LLM maker (gpt-6.1-sol, server `codex exec`) that
writes sandboxed programs plus small tensors, and a verifier that recomputes everything from the saved weights. That is useful machinery and a real, measured capability of the
*maker*; it is not AGI and the numbers below must not be quoted as if it were.

## 1. Numbers (each recomputed from saved weights unless marked; ARC data never leaves the server)
| what | result | controls | caveat |
|---|---|---|---|
| ARC-1/2 public-training pool, skill bank (narrow templates, tensors written for GENERATED niches, never saw ARC tasks), report split 302 | 29 / 302 (9.6%) | zero 0, random 0, 0, 1 | the clean transfer number: weights written for other tasks solve unseen ARC tasks |
| same pool, **structure bank** (test-time authoring: houses write code + <=256 tensors from each task's demonstrations; protocol frozen, run once), report split 302 | **287 / 302 (95.0%)** recomputed in a clean process, test outputs stripped; `tests.test_arc_goal` 14/14 green on the server | zero 0, random 0, 0, 0 | **only 38 of the 287 survive removing the cells authored for that same task (26 skill-niche cells, 12 structures from other tasks); 249 (87%) exist only because a structure was authored from that task's demonstrations** (`reports/arc12/TRANSFER_AUDIT.json`). Public ARC training tasks are widely published, the maker may recall them; colour-permuted+transposed probe 18/20 |
| ARC-3 (25 public games): scripted state-graph explorer, no LLM | 17 levels at 3000 actions | random agent 5 levels | baseline, from `/srv/arc3-claude/logs` |
| ARC-3 managed LLM weight writing (`runs/weights-arc3-managed1`, 59 manager/A-B cycles, 175 writer attempts) | **33 credited levels in 19 of 25 games** (lp85 8/8 won in 332 actions, tu93 4/9, sc25 3/6, vc33 3/7, 15 games 1 each), recomputed in a fresh process, 0 mismatches (`reports/arc3/VERIFY.json`) | zero tensors, 3 random draws, BLIND run: 0 in every credited game; random agent 4 levels, scripted explorer 17 (at 3000 actions) | public games, each cell tuned to its own game by feedback; cross-game transfer 2 of 168 plays; randomised A/B: donor cells did not help (3/17 vs 3/15). Not a generalisation claim |

Read the structure bank honestly: it shows the maker route + sandbox + acceptance pipeline works end to end and that the maker can write a correct program for almost any
ARC-1/2 training task it is shown. It does not show that stored weights generalise (`TRANSFER_AUDIT.json` splits that out) and it says nothing about novel tasks. The only
novelty test the lab owns is the ARC-AGI-2 public EVALUATION split (120 tasks), which the owner excluded; revisiting that is the owner's decision (section 6).

## 2. Regime (what is running, what is closed)
- **Nothing is running right now** (checked 07:50Z: no driver, no scoring job; `python scripts/lab.py status`). Track A (ARC-1/2) is closed: final score packaged in `runs/weights-head-structbank-final/`,
  `tests.test_arc_goal` 14/14 green on the server, transfer split in `reports/arc12/TRANSFER_AUDIT.json`. Track B (ARC-3 policy cells under an LLM manager, `scripts/lab_manager.py`) finished
  three campaigns and is the only open research line; resume it with `python scripts/lab_manager.py run runs/weights-arc3-managed1 --cycles N --slots 3` on the server (resumable, stops at the budget cap or a STOP file).
- **Closed (stand-down, nothing deleted):** all Codex lanes (arithmetic, reasoning, spatial, signed-deduction, resource-planning, tool-use v2, multistep v2, wall/anchor/frame/
  reference/corridor lanes), the arena, the tournament, the skill-bank driver, every `weights-*` pilot. Their receipts and tensors stay as evidence. Reason: in the closed lanes
  random tensors matched or beat the maker (multistep: random 44/48 and 46/48 vs maker 41/48; tool-use v2: random champion 36/48), or the maker solved them in generation 1.
- **Managers:** `lab_manager.py` runs one manager call per cycle that orders writer slots (which game, which house, what to change). The code guarantees that every cycle
  submits weights: a failed or empty manager call is replaced by a deterministic fallback, a writer that "declines" must still submit an attempt, crashes are logged and the
  other slots continue. Managers write no weights.
- **Budget:** the ChatGPT weekly limit is ONE pool for this laptop's Codex app and the server maker route. 57% used at 2026-10-02 03:10Z, 61% at 07:26Z after this whole session (230 maker calls, 1% ~ 58 short calls), resets ~2026-10-06 21:33Z. Every server
  maker call goes through `scripts/maker.py`, which logs the meter to `BUDGET.jsonl` and refuses at the cap (default 85%, `BUDGET_CAP.json`). How the 50% used before this session splits between desktop turns and server
  lane calls is NOT measured (a single 90-minute desktop turn used 12M input tokens, so that is a plausible large share; that is inference).

## 3. Where things are
- Laptop repo: source `scripts/`, tests `tests/`, notes `notes/`, docs `docs/`, evidence `runs/` (486 entries, 861 MB; do not `git add runs/` unreviewed, see `runs/CLEANUP_PROPOSALS.md`).
  `git status`: ~845 untracked paths and no remote; nothing since 2026-09-30 13:08 is committed.
- Server `cellforge-1` (ssh alias; 2 CPUs, no GPU, 29 GB free): `/srv/cellforge-inbox/head-20261001/` (orchestrator work dir: scripts, `runs/weights-head-*`, `BUDGET.jsonl`),
  `/srv/cellforge-inbox/two-loop-20261001/` (Codex lane dirs, untouched), `/srv/cellforge-inbox/_archive_20260930/` (219 older entries moved there on 2026-10-02 with
  `MANIFEST.txt`; nothing deleted), `/srv/arc3-claude/` (ARC-3 games and venv), `/srv/cellforge-private/arcagi2/data` (ARC data, never copied out).
- One command: `python scripts/lab.py status` (budget, processes, ARC-3 status, final score); `sync`, `stop`, `budget N`.

## 4. Rules that keep the numbers honest (unchanged, condensed from the handoff)
Held-out = 1000 public-training tasks minus the 100 DEV ids minus the SHA256-mod-5 audit population; the report split (302) is never used to select, tune or retry anything;
never read the ARC evaluation split; the maker sees only generated tasks or a task's own demonstrations and test INPUTS; every maker call is charged and logged; controls get the
same positions; nothing self-reported counts, only recomputation; never edit `ARC_FLOOR.json`, `HARD_LANES.json`, the standard/arc-goal tests.

## 5. Known red (left red, not weakened)
The two historical pilot failures were archived with owner approval; both standard
suites passed before the new probe. The new single-seed shared-founder probes
are intentionally below the full qualification standard and now cause two
standard failures and three semantic errors. Broader architecture replay has two
additional failures; details and logs are below. No tests or floors were changed.
`tests.test_arc_goal` requires server ARC_ROOT and independently passed 14/14.

## Owner decisions and Codex resumption (2026-10-02)
The owner returned and authorized committing/pushing/merging the source snapshot,
archiving both legacy pilots, and resuming useful research. This supersedes the
owner-away stand-down for the bounded experiment below; other closed lanes stay closed.
The ARC-3 direction was explicitly delegated to a subagent.

Archive state verified locally: `weights-arc-best-a5-v1` is preserved under
`runs/_archive_legacy/`; `weights-cell-population-g1` retains its location for
population recomputation, with `STANDARD.archived.json` and `ARCHIVED.md`.
No tests were changed. On 2026-10-02 the two standard suites ran 25 tests in
38.842 seconds, OK. The previous five failures are historical, not current.

Research decision: stop spending on routine game-specific ARC-3 campaigns.
Test a single shared policy/tensor set that learns from its action/frame history,
without game-ID routing. First step: one metered founder proposal on a frozen
three-game development panel, with zero, three random-tensor draws and blind
controls, then stop and assess. See `docs/ARC3_NEXT_EXPERIMENT.md` for the full
bounded protocol. Existing public games are development evidence, never newly
held-out games. Neither a passing pilot nor the existing ARC floor establishes AGI.
The ARC-AGI-2 evaluation split remains excluded unless the owner explicitly lifts
that restriction; a clarification is pending.

Broader architecture verification was also run: 78 tests in 47.756 seconds,
FAILED (2). `test_action_verification` rejects the corrected-g3 saved population
against changed installed `cell_population` source and differing replay budgets /
membership; `test_circuit_population` expects admission using claimed outcomes
without the execution evidence now required by `CircuitArchive`. These failures
predate the shared-policy runner changes. Do not weaken the fail-closed admission
or rehash historical receipts to hide them. Full log:
`reports/owner-resumption-gates-20261002.log`.

Fresh server ARC goal recomputation (existing `goalcheck3` package): 14 tests in
304.723 seconds, OK. This validates the saved public-training result and the
owner's existing floor; it is not new-game evidence.

First shared-founder probe `runs/weights-arc3-shared-probe-20261002`: zero levels,
zero actions on all three games because the source called unavailable builtin
`str`. Controls crashed too. Independent replay matches. This is a runtime
invalid proposal, not evidence against shared adaptation. One separate charged
runtime-repair probe was started with the exact error and allowed builtin list,
no game observations or tuning feedback; it stops after one proposal as well.
The maker meter moved from 61% to 62% across the first probe (shared desktop
activity included; this delta cannot be attributed to that call alone).

The new probe is below ONE_STANDARD: one seed, no charged random-edit population,
no crossover/router qualification, no standard semantic evaluator. It truthfully
ships STANDARD.json and remains red: 25 standard tests in 4.666 seconds, two
failures and three errors. See `reports/shared-probe-standard-qualification-20261002.log`.
The legacy archival was not used to conceal these new qualification failures.

Runtime-repaired shared policy: all ar25/lp85/sc25 policy and control runs were
valid, 1500 actions each, zero levels in every arm. No-op counts: 1363, 1500, 353;
distinct frames: 107, 1, 689. Two charged writer calls total, no managers; meter
62% after repair. Do not expand the 16-call campaign from this result. The next
hypothesis must address observed action-effect / no-op recovery failures before
a larger population study. Evidence summary: `reports/ARC3_SHARED_PROBE_20261002.md`.

## 6. Decisions only the owner can make
1. ARC-AGI-2 public EVALUATION split: run the frozen structure-bank protocol once on its 120 tasks (about 120-250 maker calls, roughly 2-5% of a weekly limit). This is the only
   honest novelty test; without it "95% on training" means "the maker knows ARC training tasks".
2. Commit policy: ~845 untracked paths, no remote. Suggested first step: commit `scripts/ tests/ docs/ notes/ src/ reports/ LAB.md AGENTS.md` (not `runs/`).
3. Whether to archive the two legacy runs that keep the standard tests red.
4. Whether Codex should resume at all, and on what, given the shared weekly limit.
5. How far to push ARC-3: 19 of 25 public games have a credited level and growth had flattened (31 -> 33 credited levels over the last ~14 cycles). Another campaign costs about 1% of the weekly limit per 58
   calls. The open design question is generalisation (cells are game-specific: cross-game transfer 2 of 168 plays); a meaningful next step is a held-out set of ARC-3 games, which the lab does not have.

## October 5 implementation-only continuation

The latest owner instruction authorizes real code integration with static review
only, explicitly excluding tests, prototypes, experiments, model calls,
deployment and new spending for this pass. No closed campaign was reopened and
no evaluation access was expanded. The earlier test results do not validate the
new integration.

Opt-in evolution policy, complementary-success parent selection, versioned
cell/circuit and router-roster contracts, persisted exploration/challenge/recovery
scheduling, a bounded gpt-6-luna tensor-proposal pool, and immutable run manifests
are integrated with `scripts/head_driver.py`. Defaults retain legacy selection;
new modes have an explicit niche list (empty by default), disabled workers and
zero call allowance. The pool cannot edit the evaluator or runtime routing.
Group-relative selection is an evolutionary mechanism, not an implementation
of PPO/GRPO and not an AGI result. Details: `docs/EVOLUTION_INTEGRATION.md`.

The corrected ARC runner now treats missing engine observations as errors and
compares exploration/terminal evidence during replay. Its ten isolated regression
passes occurred earlier, before the owner's no-test instruction. No runtime
verification of the subsequent evolution integration has been performed.

Current server jobs remain unobserved: `cellforge-1` did not resolve in the
desktop environment and no supported authenticated Hetzner connector was
exposed. The existing key was not transmitted; credentials, trust, firewall and
provisioned resources were untouched. The local checkout lacks the historical
`runs/weights-*` artifacts and NumPy. Budget/access decisions remain with the
parent/owner; source changes have not been committed or deployed.

The next implementation milestone adds opt-in literal router replacement over
a frozen ordered expert roster, complete matched random-router attempts,
zero/incumbent controls, development-only champion retention, and journal-bound
resume checks. Every explicit-policy checkpoint now executes through typed
`predict_bank` while honoring all evaluator-supplied weights, including
zero/random ablations. Router evolution remains disabled by default; its fixed
roster phase replaces cell breeding/mutation rather than claiming joint evolution.
This milestone received source/AST/whitespace inspection only, with no tests,
model calls, runs, deployment or empirical performance claim. Details and
remaining limits are in `docs/EVOLUTION_INTEGRATION.md`.

A subsequent implementation-only milestone connects bounded alternating cell
and router phases through `scripts/evolution_epochs.py` and
`scripts/checkpoint_transitions.py`. Each explicit invocation addresses one
phase in a new manifest-bound child run; no next phase auto-launches. Full
archives, tried pairs, RNG/curriculum state and the separate retained bank carry
forward, while phase call allowances remain independently reserved under a
fixed campaign cap. Cell phases can expand the search archive; a later router
phase can retain a full-bank development gain only with all incumbent anchors
preserved. Such retention is `bank_gain`, not evidence of learned routing or
independent generalisation. Default plans are disabled. No phase has been run;
the same no-test/no-experiment/no-spend/no-deploy restrictions remain in force.

## October 6 bounded desktop readiness

The parent relayed fresh owner authorization for bounded desktop operational
checks and necessary runtime preparation, superseding the preceding static-only
restriction for that scope. The updated limit is 62,000 Codex credits; no separate
paid API/cloud service, Hetzner work, top-up, deployment or commit is authorized.
Closed campaigns and excluded evaluation access remain unchanged.

The existing bundled Python 3.12.14 supplies NumPy 2.3.5 and Pillow 12.3.0; no
installation was needed. A disabled zero-call epoch preview and driver imports
succeeded. Three selected synthetic component tests and in-memory typed-bank
checks passed with one numerical thread and 30-second child limits. This does
not validate a full phase or constitute a scientific weights result.

The historical seed (`pop1_core.py`, `parent.json`, `TASKS.json`) and required ARC
training files are absent locally; configured defaults still point to the server.
No founder artifacts were invented and no new run directory was created. Codex
CLI login reports ChatGPT, but supported account RPC attempts exited before a
response. The project percentage meter is not an absolute credit meter, so paid
worker batches remain blocked pending agreed accounting/initial call limits.
Full evidence and minimal options: `reports/LOCAL_READINESS_20261006.md` and
`notes/CODEX_TO_HEAD_2026-10-06.md`.

The parent subsequently authorized an explicit NEW local baseline and official
public training inputs. `fresh_inputs.py` and opt-in driver/epoch configuration
now initialize declared random literals without historical `pop1_core.py` or
parent artifacts. Eight training JSONs and the Apache-2.0 license were downloaded
from pinned official ARC-AGI-2 commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49`; two report IDs remain reserved with no
files downloaded. Existing historical splits and excluded evaluation access
were preserved. No model calls were made.

Four targeted input tests passed. Bounded execution exposed and fixed canonical
partition ordering in the phase scope hash. Independent replay then rejected a
partial-domain tiling baseline because saved/control tensors could throw on
rectangular inputs; those attempts remain intact. A separately documented
rectangular-grid baseline using random assoc_induct/glyph_recolor cells completed
phase 0 in 1.093 seconds, with valid saved/zero/three-random replay in 1.078 seconds.
Every arm scored 0/4 search and 0/4 development-selection tasks. This is an
operational null result, not a qualified weights study or useful learned model.
Artifacts: `runs/weights-local-fresh-rectangular-20261006/`; full report:
`reports/FRESH_LOCAL_BASELINE_20261006.md`. Paid batches remain unstarted.

The owner subsequently authorized model calls and continuation within the existing
62,000-credit ceiling; the parent bounded the initial tranche to two calls at
concurrency one. A fresh batch at `runs/weights-local-maker-initial-20261006/`
halted before dispatch under the unchanged worker `budget_unavailable` guard:
zero reservations, zero model calls, zero released proposals. Official read-only
account RPC diagnosis removed a sandbox startup denial but still obtained no
quota/credit receipt; no meter was fabricated. This batch is terminal/blocked,
not a weights result. Details: `reports/LOCAL_MAKER_BLOCKER_20261006.md`.

## 2026-10-06 native-agent proposal result (bounded local experiment)

One native gpt-6-luna proposal, with low reasoning, was independently evaluated
and replayed: 0/4 search, 0/4 development; random/zero/reference each also 0/4
and 0/4; all arms had zero runtime errors. Full 102-position charges per native
and random arm are retained. Credit usage is unknown, not free. No report access,
second author call, admission, server work or deployment. See
`reports/NATIVE_AGENT_RESULT_20261006.md` for immutable receipt hashes, validation
and pending search-only failure-journal integration. This result provides no
advantage over controls and no chatbot/general-capability evidence.

### Native failure-memory update

The completed native result now has independently reverified search-only
failure memory. A second packet is prepared under
`runs/weights-native-agent-two-20261006/`; no second author has been dispatched.
It contains only original search demonstrations, previous tensors and one
leave-one-out training failure (34 differing pixels). No selection/report
feedback was exported. See `reports/NATIVE_SEARCH_FEEDBACK_20261006.md`.

### Native V2 terminal result and template boundary

The second native proposal scored 0/4 search and 0/4 development, matching its
parent/zero/random controls, with no runtime errors; saved-runtime replay
passed. Cumulative charges are 204 candidate and 204 random positions; credits
remain unknown. No third model call. Search-only review proves `assoc_induct`
cannot exactly express any of the three shown transformations: equal input
colors require differing output colors. Stop this template/task loop and
review a spatial cell or composition first. Evidence and qualification limits:
`reports/NATIVE_AGENT_V2_RESULT_20261006.md`.

### Spatial candidate plan, not a new author run

A search-only review froze task f5b8619d with existing ray_cast -> tile_orient
(167 literal leaf positions). A hand-coded feasibility probe matches 3/3
shown demonstrations; zero chain 0/3. snap_to_line solves original1a07d186's
3 demos with zero tensors, so it cannot establish an authored-weight gain.
No third model call or query/dev/report prediction occurred. Frozen packet,
selection rationale, accounting caveat and remaining importer work:
`reports/SPATIAL_NEXT_PACKET_PLAN_20261006.md`.

### Spatial importer ready, still no third native author

Narrow support for the frozen ray_cast -> tile_orient packet is prepared and
source-bound. Twelve isolated tests and independent packet/source preflight
passed. Charges are all 167 newly authored leaf positions per arm; hand-coded
feasibility is a separate baseline. Current native ARC charges remain 204 per
arm because no spatial attempt occurred. Exact author paths and freeze:
`reports/SPATIAL_IMPORT_READY_20261006.md`. This remains host-defined primitive
selection/composition selected on search evidence, not invented algorithms.

### Native spatial text route: two accounted responses, no gain

A separate two-attempt text envelope preserves the parent-transcribed native
responses. First wrapper invalid and charged; second valid but0/4 search and
0/4 development, no native runtime errors. Hand-coded reference1/4 search,
0/4 development; random controls each have3 errors across2 task outcomes,
kept null. Batch334 positions per arm; native ARC cumulative538 per arm.
Credits and actual compacted prompt hash remain unknown; task-name provenance
is explicit. Independent replay matched. No new author or report access.
Details: `reports/SPATIAL_TEXT_RESULT_20261006.md`.

### Four-author native spatial batch terminal

The parent-authorized four-slot trial at
`runs/weights-native-spatial-storm01-20261006/` is fully consumed and replayed.
Two responses were invalid (duplicate JSON key; prohibited orientation value),
one valid response scored 0/4 search and 0/4 development, and author04 scored
1/4 search and 0/4 development. The hand-coded baseline also scored 1/4 and
0/4; every matched random draw scored zero with no runtime errors. No admission
or development gain is claimed. New native/random restrictions were frozen
identically before authoring: 103 free plus64 fixed values, all167 charged.

Batch charges668 positions per arm; cumulative native ARC charges1206 per arm.
Credit use remains unknown. The parent resumed the same four startup-blocked
threads as tools-free text authors; common author01 context is parent-attested,
not a host capture of actual platform prompt bytes. A host author02 copying
error was corrected from the supplied text before slot consumption; the bad
capture and explicit correction remain preserved. No author repair or retry.

Seven focused tests passed; independent saved-runtime replay matched all four
results. The required combined standard suites timed out at30 seconds after
failures; a bounded diagnostic run confirmed7 failures in15 contract/milestone
tests, and a semantic fixture encountered local temporary-directory access
denial. Tests/floors and sandbox settings were unchanged. No server work,
additional author call, report access, commit or deployment. Full evidence,
limitations and bounded next decision: `reports/NATIVE_STORM01_RESULT_20261006.md`.

### Synthetic transfer curriculum prepared; founders not dispatched

The parent authorized a finite synthetic curriculum on the unchanged spatial
primitives: two founders plus at most one search-failure replacement each.
`runs/weights-native-transfer-curriculum01-20261006/` freezes six demonstrations,
twelve search cases, twenty-four development inputs, independent oracle/source,
criteria and matched random seeds. Development labels remain uncomputed; both
native and random winners must be selected using search alone before opening.
No authors consumed; charges0 per arm, maximum668 per arm; credits unknown.

Ten focused integrity tests passed on separate fixtures. A structural generator
exhaustion was fixed before freezing without viewing development labels. The
external preparation wrapper's recursive-call counting assertion is preserved;
independent saved-runtime audit/preflight confirms the freeze, separate-fixture
checks and sealed-data boundary. This is a synthetic host-algorithm transfer
probe, not ARC qualification. Exact 4,663-byte common founder text, hashes,
commands, limits and remaining broad-gate failures:
`reports/NATIVE_TRANSFER_CURRICULUM_READY_20261006.md`.

### Synthetic transfer terminal: criterion passed, one functional behavior

Both parent-delivered founders validated and scored12/12 search. No replacement
was eligible. Native and random winners were selected on search alone before
the one-time development opening. Both natives and the hand-coded reference
scored24/24; both random controls and zero tensors scored0/24; all runtime errors0.
All four permutation checks passed. Independent saved-runtime final replay
matched. The frozen synthetic transfer criterion passed, with no ARC or general
capability qualification and no automatic population admission.

The two numeric genomes are functionally identical:16 orientation maxima differ
but every forward decision matches. Synthetic batch charges334 positions per
arm; prior native ARC1206 per arm stays separate; credits unknown. Both founders
are search-perfect, so authoring is closed after2 completions. This development
panel is now spent. Full results, hashes, provenance, qualification limits and
unchanged broad-gate failures: `reports/NATIVE_TRANSFER_CURRICULUM_RESULT_20261006.md`.

### Two-specialist composition protocol frozen; horizontal founder pending

Parent authorized a finite typed-bank experiment retaining accepted column
specialist A unchanged and authoring a horizontal specialist plus literal2x2
router. At most4 completions and346 new positions per native/random arm.
`runs/weights-native-composition01-20261006/` freezes new data, controls, source,
criteria and exact5,287-byte horizontal founder packet. No dispatch or consumed
response. New mixed development labels remain unmaterialized.

Seven focused checks passed, including exceptions swallowed by the legacy
scorer being surfaced explicitly and a perfect random router blocking comparative
acceptance. Native/random winners use search alone. Fixed identity is reported
separately; matching it is not learned-router superiority. specialist_runtime's
ray_cast/resource guard is unchanged; no service admission. Exact packet/hash,
guarded preflight, staging commands and limitations:
`reports/NATIVE_COMPOSITION_READY_20261006.md`.

### Horizontal founder passes search; router packet frozen

The parent-delivered gpt-6-luna/xhigh horizontal founder is valid and12/12 on
search; matched random0/12, errors0. Independent replay matched. No specialist
replacement is eligible. Charges167 per arm; credits unknown. The ordered
column/horizontal roster is frozen, and the exact2,713-byte tools-free router
packet is ready. New final development remains unopened. Raw provenance,
immutable hashes and next command: `reports/NATIVE_COMPOSITION_ROUTER_READY_20261006.md`.

### Typed-bank composition terminal:24/24, identity-router equivalence

Both horizontal and router founders were valid/search-perfect, so no replacements
were eligible. Two completions consumed173 positions per arm; credits unknown.
The search-selected native bank scored24/24 final tasks; both single experts,
matched random router, zero router and zero/random-horizontal controls scored
12/24. Fixed identity and reference banks also24/24. Errors0, permutation checks
4/4; fresh-process final replay matched. Frozen comparative acceptance passed.

The authored router is literally W=identity,b=zero: no learned-router superiority.
It routes12 tasks to each specialist. SELECTED_SKILL_BANK.json exports the exact
340-position bank, including167 inherited positions. specialist_runtime still
rejects ray_cast and was not changed; no service admission. Final panel spent;
authoring closed. Results, hashes, provenance limits and unchanged broad failures:
`reports/NATIVE_COMPOSITION_RESULT_20261006.md`.

### Actual ARC iteration preparation, 2026-10-06

The parent redirected work from synthetic composition to actual ARC. A new,
unlaunched packet at `runs/weights-native-arc-iteration01-20261006/` uses existing
public-training SEARCH task bd283c4a: separate Codex program and literal-weight
structure arms, each capped at one founder plus one conditional training-failure
revision. Zero calls and zero candidate executions. Closed campaigns stay closed;
report/evaluation remain untouched. Corrected struct_pilot feedback to show an
actual failing demonstration, with bounded grid differences; nine focused tests
pass. Required standard suites remain red and hit the 30-second child limit.
Execution needs an authorized bounded Linux runtime; the current no-Hetzner
scope, missing local ARC3 engine/games and unknown credit usage remain explicit.
ARC3 next diagnostic is action-effect/no-op recovery at 150 actions per game,
not continuation of the old campaign. Full packets, resource limits, evidence
and blockers: `reports/ARC_ITERATION_READINESS_20261006.md`.

### Direct ARC prediction baseline terminal, 2026-10-06

One tools-free native Codex direct prediction solved bd283c4a's single public-
training SEARCH query exactly; zero mismatched pixels. Independent replay matched.
The frozen host-inferred program also solved it; identity missed 72 pixels. This
is direct reasoning, not weight learning or held-out transfer. One call, xhigh,
exact backend and credit usage unknown; raw response frozen before scoring, no
label feedback or retry. The proposed literal frequency-packing extension remains
static design only. Required standard suites remain red and timed out at30.031s.
Evidence: `reports/NATIVE_DIRECT_GRID_RESULT_20261006.md`. No remote work,
arbitrary authored-code execution, sandbox weakening, installation or deployment.

### Direct ARC search batch02 terminal, 2026-10-06

The three independently authored direct-grid predictions for 1a07d186, 3194b014,
and f5b8619d each matched exactly, with zero pixel differences; all three fresh-
process replays matched. Identity scored 0/3. Combined with the prior bd283c4a
baseline, direct inference is 4/4 on the EXISTING, previously exposed SEARCH
panel, using four separate task-specific model calls. This is not a shared
learned-weight model or held-out transfer. Exact backend/credit usage unknown;
xhigh reasoning, three calls consumed this batch, no label feedback or retries.
No new template or unrelated standard rerun. Existing full-standard status stays
red/incomplete. Evidence: reports/NATIVE_DIRECT_BATCH02_RESULT_20261006.md and
runs/weights-native-arc-direct-batch02-20261006/TERMINAL.json. ARC3 work remains
with the separate runtime owner; no overlap or remote work occurred here.

### Next public-training batch blocked before selection, 2026-10-06

Batch03 froze selection seed 202610060303 before new task contents. Local exposure
inventory scanned 1,613 text artifacts (171,444,454 bytes), collecting 9,874 candidate
8-hex tokens for later intersection with official training IDs; all eight known
training tasks and both reserved report IDs are covered. Nontext and remote
history remain unverified, so future tasks must be called newly selected, not
never seen. No new task has been selected, downloaded, authored or scored.
Automatic approval review rejected GitHub training metadata retrieval because
relayed delegation was not accepted as owner approval under the no-API/download
rule. No workaround was attempted. Explicit owner approval is needed for the
bounded official metadata/four-training-file/license acquisition. Evidence:
runs/weights-native-arc-direct-batch03-20261006/BLOCKED.json. Existing standard
status remains red/incomplete; no unrelated suite was rerun.

### Batch03 acquisition approved and packets prepared, 2026-10-06

The one metadata retry passed automatic review with the supplied owner transcript
evidence; the bounded four-task/two-license acquisition also passed. Earlier
blocked receipts remain preserved. Seed 202610060303 selected ARC1 tasks 46442a0e
and c8cbb738 plus ARC2 tasks 984d8a3e and 0c786b71 after known-record exclusions
and removal of 391 shared IDs from the ARC2 pool. Pins and Apache-2.0 licenses
are recorded. Newly selected means absent from scanned local text records;
remote/nontext and model exposure remain unverified. Four label-free direct-grid
packets are prepared; zero authors or query scores. Protocol and evidence:
reports/NATIVE_DIRECT_BATCH03_PREPARED_20261006.md. No evaluation/private access,
remote work, credential changes, new framework or unrelated standard rerun.

### Newly selected direct ARC batch03 terminal, 2026-10-06

All four direct predictions matched exactly: ARC1 2/2 (46442a0e,c8cbb738), ARC2
2/2 (984d8a3e,0c786b71), zero direct pixel differences and four matching independent
replays. Identity scored 0/4. Selection was seed-frozen and excluded known local
IDs/cross-version duplicates; remote/nontext/model exposure remains unverified.
This is newly selected PUBLIC TRAINING, not proven unseen or held-out evidence.
Four isolated per-task author calls, xhigh, default backend/credits unverified;
raw responses frozen before scoring, no feedback/retry. No genome result.
Earlier exposed-search panel was 4/4; combined descriptive total is 8/8 unique
training tasks with eight independent calls, not a shared learned model.
Evidence: reports/NATIVE_DIRECT_BATCH03_RESULT_20261006.md and batch03/TERMINAL.json
under runs/weights-native-arc-direct-batch03-20261006. Existing red standard status
carried forward; no unrelated rerun, template implementation or ARC3 overlap.

### Bounded direct ARC batch04 prepared, 2026-10-06

Seed 202610060404 was frozen before contents. Reused official pins/manifests and
prior exposure inventory, excluded every previous selection, and selected ARC1
ed36ccf7/ef135b50 plus ARC2-exclusive e7639916/712bf12e. Four official task files
matched pinned Git blobs; existing Apache-2.0 licenses were verified/reused.
Packets contain demonstrations and query inputs only. Zero author calls or query
scores. Four calls maximum, no query-feedback retries; then stop for a deliberate
capability/weights checkpoint. No batch05/larger campaign automatically. Newly
selected does not prove absence of remote/nontext/model exposure. Evidence:
reports/NATIVE_DIRECT_BATCH04_PREPARED_20261006.md. Existing red qualification
status carries forward; no framework/full-suite rerun or ARC3 source overlap.

### Batch04 terminal and capability/weights checkpoint, 2026-10-06

Batch04 scored 3/3 submitted exactly (ARC1 2/2, ARC2 1/1), with three matching
replays. Of four planned slots, e7639916 was user-cancelled during creation;
parent list_threads verification says no author was created. It remains unscored,
not a synthetic zero, and is never retried. Identity scored 0/3. Direct trials now
have 11/11 scored unique public-training tasks out of 12 planned, with distinct
exposure histories and no held-out/shared-weight claim. Credits/backend unknown.
No additional authors, replacement slot, batch05 or larger campaign are prepared.
The explicit stop and checkpoint are saved in batch04/TERMINAL.json and
CHECKPOINT_STOP.json under runs/weights-native-arc-direct-batch04-20261006.
Results: reports/NATIVE_DIRECT_BATCH04_RESULT_20261006.md. Research decision:
reports/CAPABILITY_WEIGHTS_CHECKPOINT_20261006.md distinguishes direct solver
ability from actual tensor learning, synthetic composition and unresolved ARC
development gain. Existing red qualification status carries forward. No template
implementation, full-suite rerun, sandbox weakening or ARC3 source overlap.
