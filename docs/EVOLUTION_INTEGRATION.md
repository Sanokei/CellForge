# Explicit evolution policy and bounded weight workers

Implemented October 5, 2026. **Static review only: no tests, model calls,
benchmarks, deployment, or experiments were performed for this integration.**
The earlier ten passing ARC runner regression checks predate these changes and
do not validate them. All closed campaigns remain closed under `LAB.md`.

## What is wired into the real driver

`scripts/head_driver.py` accepts an optional `--evolution-policy` JSON file.
Omitting it with no stored policy retains the legacy selection/curriculum path.
An explicit policy remains active even when every value equals the defaults;
disabled workers and an empty niche list cannot fall through to legacy work.
Explicit policy modes
require a new `weights-*` run or an identical frozen resume; they cannot retrofit
an existing unversioned run. Configuration is persisted as `EVOLUTION_POLICY.json`.
No process or model call starts merely by adding these source files.

The opt-in sequence is:

1. Validate frozen source, loaded legacy code/weights, task-content hashes and
   disjoint search/development-selection/report IDs before resumed work.
2. Optionally obtain literal proposals from a bounded gpt-6-luna worker pool.
   An explicit niche list controls founder work; an empty list opens no niches.
3. Score each candidate with host-owned functions, record individual outcomes,
   then select measured task champions. A worker cannot change the evaluator,
   acceptance rules, template, or generated-output schema.
4. Select breeding partners from aligned development successes, check the typed
   boundary, and compose inherited cells. Existing non-regression admission
   remains in force.
5. For worker mutations, score every reserved task attempt and its paired random
   full-template replacement before selecting either arm's champion. Rank on
   the development-selection partition; preserve search and selection anchors.
6. Advance the persisted curriculum only after a completed cell wave. No report
   outcomes enter this loop, and report scoring remains a separate final action.

An additional disabled-by-default router phase is now wired into this driver.
When `router.enabled` is true, configured founders and seed imports finish first,
then the expert roster freezes and router-only waves replace cell breeding and
mutation. The two modes are alternative bounded stages in this milestone;
single-run variable-roster cell/router evolution is not implemented. The bounded
outer phase integration below composes these existing stages in separate runs.

This is group-relative evolutionary selection over explicit proposals. It does
not implement PPO/GRPO gradients, clipped objectives, or a learned language-model
optimizer. "Champion" means measured best under that recorded development scope.

## Selection, curriculum and routing have different jobs

`evolution_policy.complementary_pairs` consumes `cellforge-parent-success/1`:

```json
{
  "schema": "cellforge-parent-success/1",
  "partition": "search_fitness",
  "task_ids": ["development-task-id"],
  "scope_hash": "SHA256 of the declared task scope",
  "evaluator_hash": "frozen manifest source_hashes_hash",
  "outcomes": {"cell-content-hash": {"development-task-id": true}}
}
```

Outcomes are booleans or null. Missing/null observations are unknown and excluded
from the common comparison panel. Fitness rewards successes inversely weighted
by the number of eligible cells that solve each task. The second parent is
weighted by success not supplied by the first. Zero complementarity falls back
to eligible mate fitness, then seeded uniform selection. An empty eligible-pair
set explicitly produces mutation-only status. Different niche names are not
required by this selector. Parent IDs, task/evaluator scope, seed and RNG state
are retained; no report data is used.

This independently implemented rule follows the method in Sakana AI's
[Natural Niches source](https://github.com/SakanaAI/natural_niches/blob/main/natural_niches_fn.py),
whose repository is Apache-2.0 licensed. No JAX dependency or pretrained weights
were imported.

The curriculum chooses development exposure and per-wave operator limits:
exploration prioritizes less-exposed/unknown tasks; challenge prioritizes mixed
success; recovery prioritizes previously solved anchors with current failures.
Recovery never resets the archive. Phase, exposure attempts, outcomes, chosen
task IDs, anchors and RNG state persist across waves. The cycling idea is only a
scheduling analogy to [Sakana Digital Ecosystems](https://pub.sakana.ai/digital-ecosystem/),
not a reproduction of that gradient-trained NCA system or evidence of gains.

Runtime routing remains the separate literal `route.W @ -demo_error + route.b`
calculation. Curriculum rules do not secretly become router weights.

## Typed cells, circuits and expert order

`cell_contracts.py` defines `cellforge-cell-interface/1`, immutable references
binding existing numerical IDs to template/version/layout, and distinct checks
for tensor crossover versus sequential composition. Crossover requires matching
template/version/named shapes/dtypes; composition compares typed output/input
encodings and context versions. Neither condition proves useful cooperation.

`skillcore.make_cell` and `chain_cell` deep-copy inherited material. Existing
numerical IDs stay unchanged. `typed_chain_cell` attaches the explicit contract.
`export_bank` creates a `cellforge-skill-bank/1` artifact with tensors, ordered
expert identities, contracts and content hash. `predict_bank` restores declared
templates directly rather than guessing from tensor shapes, verifies the full
artifact/roster, and then routes on demonstration errors. The legacy numeric
export/evaluator remains available for runs without an explicit policy.

Every explicit-policy checkpoint now writes `SKILL_BANK.json`, regardless of
whether typed chaining or router evolution is enabled. Its generated
`arc_eval.py::predict(weights, task)` calls `predict_bank` with the supplied
numeric tensors. The saved artifact is verified first; supplied controls must
match every tensor name and shape and contain finite nonboolean numbers. The
adapter then uses those values for every expert and router coefficient. It
does not substitute the saved weights. Zeroed experts that become numerically
identical retain their declared topology and ordered slots. This preserves the
meaning of zero/random controls without guessing a template from a tensor shape.
Training-range limits are not silently imposed on evaluator-supplied controls.

Policy-mode scoring no longer loads or writes the shared legacy error cache:
its keys omit source/task versions. New-mode demonstration errors are computed
against the frozen inputs and cached only in the current process.

`remap_router` supports pure expert-order permutations and remaps both matrix
axes plus bias. Adding/removing experts requires explicit replacement router
tensors; this helper does not initialize beneficial new coefficients.

## Literal router proposals and controls

`router_evolution.py` has six pure, stdlib-only helpers: `positions`,
`validate_params`, `make_genome`, `random_params`, `proposal_prompt` and
`select_champion`. A `cellforge-router-genome/1` identity hashes the ordered
expert IDs together with the full literal `route.W[N,N]` and `route.b[N]`.
Lineage is recorded separately from numeric identity. Worker outputs must
contain all N*N+N coefficients as finite nonboolean numbers in [-9, 9].

The prompt contains only a bounded, deterministic search-fitness sample:
negative leave-one-out demonstration errors and which experts succeeded on
the development query. Any row with unknown expert execution is omitted from
teaching feedback. Success labels are authoring targets; runtime features remain
demonstration errors alone. No development-selection or report outcome enters
the proposal prompt. Selection uses the complete frozen search/selection panels.

Each durable reservation, including invalid, missing, refused or uncertain
output, receives one complete seeded random-router counterpart and the same
N*N+N charge in each arm. Journals preserve raw attempt IDs, literal candidate
and random tensors, seeds, full partitioned outcomes, incumbent and zero-router
controls. The zero-router control changes router weights only; checkpoint
evaluation separately supports whole-bank zero/random ablations.

An arm's champion must strictly improve development-selection accuracy while
preserving every previously solved search and selection task. Highest selection
score wins, with content hash as the tie breaker. A candidate is admitted only
when it also strictly exceeds the zero router and every paired random router's
selection score. A rejected wave retains the incumbent. These adaptive
development comparisons do not establish independent generalisation or a
qualified equal-search-budget study across multiple seeds.

`router_competition` and `router_champions` events validate shapes, hashes,
charges, complete panels, ranking and acceptance before state replacement.
Resume verifies the active genome against the latest actual accepted journal
identity, preserving it across later rejected waves. Changed rosters, pending
competitions, in-flight calls and sealed reports fail closed before new work.

An individual router phase rejects simultaneous cell curriculum or complementary
parent selection. Those phases schedule cell pairs/mutations and inherited-cell
anchors; applying them to a frozen router requires a separate explicit design.
The router instead has its own task-exposure bound. Alternating phases carry cell
curriculum state across those router intervals, without changing either phase's
frozen policy. Concurrent cell and router evolution remains outside this design.

## Bounded alternating epochs

`evolution_epochs.py` joins the real cell and router driver paths through
explicit, separate phase directories. An epoch contains one cell phase followed
by one fixed-roster router phase. Its read-only planning path makes no run
directories and does not import the driver. Execution requires an enabled plan
and an explicit phase index, and runs exactly that one phase. It never launches
the next phase automatically.

The plan caps epochs at eight and each phase at twelve requested rounds. The
sum of every allocated phase call allowance must fit the campaign cap; unused
allowances are not recycled. Workers remain disabled with zero allowances in
the default plan. Campaign and phase STOP files are respected. An incomplete
phase cannot be silently replaced or retried; its reservations require manual
reconciliation. A planning/configuration file is not authorization to execute.

Campaign STOP reaches worker reservation and dispatch checks, so a later batch
cannot treat it as merely a between-phase signal. A durable worker safety halt
blocks checkpoint publication and successor phases, even if a final driver
round otherwise reports completion. Ordinary exhaustion of an explicit call
allowance remains distinct from uncertain/unmetered safety halts. Already
dispatched calls still cannot be forcibly cancelled by this cooperative STOP.

Each phase gets its own immutable manifest, explicit predecessor checkpoint,
source/task scope, and call accounting. The driver binds `PHASE_CONTEXT.json`
before freezing inputs and before constructing workers. Unlike the legacy
`--seed-from` path, it imports the entire validated archive and binds its
literal tensors, metadata and lineage before work. Mutable implicit tiling-bank
imports are disabled for this path. Changed evaluator/task scope and changed
recomputed incumbent outcomes fail before proposals.

Transitions distinguish the **search archive** from the **retained full bank**.
Cell phases preserve every inherited expert in order and may append accepted
cells or compositions. They use an explicitly labelled default reference router
for the expanded roster; they do not replace the previously retained full bank.
Router phases freeze that expanded archive and optimize its complete router.
The resulting bank can replace the incumbent only with strict development-
selection improvement and no lost solved tasks across either development panel.
Otherwise the previous bank remains intact while the expanded archive is carried
forward for later search. A reference-router bank gain is not evidence that a
router proposal learned anything; no independent capability claim follows.

`checkpoint_transitions.py` validates content hashes, typed topology references,
numeric identity, ordered archive preservation, task panels and retention rules.
The driver also checks bank exports against the actual registered templates.
Phase checkpoints carry completed niche names, tried pairs, breed/mutation
counters, RNG state and the cell curriculum's wave, exposure, anchors and
outcomes. Router phases preserve that cell state; later cell phases continue it.
Worker budgets, active-work markers and router journals are never copied as
unused allowances into another phase.

An archive exceeding the router's configured expert limit stops before router
proposals; no experts are silently pruned. Checkpoint transitions remain
development-only comparisons, with no report evaluation. This source integration
has not been run or tested and does not constitute an evolutionary result,
a qualified scientific experiment, or a chatbot.

## Worker limits, provenance and interruption

`weight_workers.WeightWorkers` uses the existing `maker.ask` route. It has finite
parallelism, proposal count, lifetime call allowance, per-call timeout, STOP
handling, durable reservations and strictly validated numeric output. It records
raw response references and reserved positions; invalid, missing, refused and
uncertain submissions are not free attempts. `attempts_for(tag)` is the driver's
source for matched control counts. Caller tags with durable attempts are rejected
before dispatch, preventing blind paid replay after a crash.

New mutation competition preserves every proposal's literal tensors, parent,
operator, seed, partitioned outcomes and paired random tensors. Random generalist
controls use one fixed candidate across their entire task panel, with the same
number of proposals as the LLM group. Existing default controls previously drew
a different random cell for each task; that accounting bug is corrected here.

`evolution_manifest.py` binds sources/config/partitions and appends validated,
hash-chained events. Loaded `pop1_core.py`, `parent.json`, run-local evaluator
dependencies, applicable `STRUCTURES.json`, and search/selection task contents
are included. Changed or sealed resumes fail before scoring or maker submission.
The minimal `STANDARD.json` explicitly remains unqualified; operational journals
are not a substitute for the full scientific standard.

An unfinished niche stage or in-flight operator fails closed on restart and
requires receipt reconciliation, without automatically repeating calls. A saved
`mutation_ready` checkpoint resumes after completed breeding; a `complete`
checkpoint advances the phase exactly once. State writes are atomic in new
modes. Journal contention must be retried before dependent work; uncertain I/O
must be reconciled rather than blindly appended again.

Limits: concurrent calls can overshoot a shared meter between observations;
timeouts/`close()` do not guarantee cancellation of running threads or remote
work; the existing maker parses a response before the wrapper's size bound;
Windows cannot supply the same directory-fsync guarantee as POSIX. The pool
halts on stale/unmetered/uncertain budget state and period changes. Keep it
disabled until resources, scope and execution are explicitly authorized.

## Configuration surface

This reviewable example makes no calls and opens no niches. It is not a request
to run the closed campaign:

```json
{
  "schema": "cellforge-evolution-policy/1",
  "parent_selection": "complementary_success",
  "alpha": 1.0,
  "typed_chains": true,
  "niches": [],
  "router": {"enabled": false, "task_limit": 100, "max_experts": 64},
  "curriculum": {"enabled": true, "task_limit": 100},
  "workers": {
    "enabled": false,
    "model": "gpt-6-luna",
    "reasoning": "low",
    "parallelism": 2,
    "proposals_per_prompt": 2,
    "max_calls": 0,
    "timeout_seconds": 300
  }
}
```

Unspecified fields use the strict defaults in `evolution_policy.DEFAULT`.
Unknown fields, invalid numbers and model-route mismatches fail validation.
Enabled workers require a positive explicit allowance; new evolution modes
cannot fall through to the legacy unbounded maker route. A config file does not
authorize a campaign, spending, evaluation access, or deployment.

Router cost bounds, if later authorized: N is capped by `router.max_experts`
(at most 64, hence at most 4,160 coefficients per proposal); search teaching
exposure is at most `router.task_limit` (at most 100). Each wave reserves at
most `workers.proposals_per_prompt` calls (at most 8), bounded again by the
pool's remaining lifetime allowance. For C reservations and T full development
tasks, comparison records require at most (2C+2)*T routed outcomes plus status
measurements and search-example construction. Demonstration features are cached
in process. This is a counting bound, not measured time/memory or a process
resource limit. STOP prevents later waves; scoring has no hard CPU deadline.

Static verification for the router/checkpoint milestone consists of source
inspection, AST parsing without importing changed modules, and whitespace diff
checks only. There is no runtime correctness, accuracy or speed result.

Static inspection also found an existing protected-gate limitation:
`tests/arc_goal_checks.py` passes the tensor dictionary to
`standard_checks._zeros_like`, which currently recurses lists only and therefore
returns scalar `0.0` for a dictionary. Its prediction wrapper counts evaluator
exceptions as incorrect predictions. That can make an invalid zero control look
like a successful ablation. These protected files were not changed. The typed
entry point explicitly rejects malformed scalar controls; any later scientific
evaluation must demonstrate valid shape-preserving controls and resolve this
gate defect under owner authorization before relying on the gate as evidence.

Hetzner remains an access blocker: the SSH alias is unresolved and no supported
authenticated Hetzner connector is exposed in this task. The existing secret was
not transmitted or copied, and security/trust configuration was not changed.

## October 6 desktop readiness evidence

Subsequent owner authorization allowed a bounded local operational check. The
bundled Python runtime imports the driver and provides NumPy/Pillow without an
installation. The disabled epoch preview, three selected existing synthetic
checks and an in-memory typed-bank export/ablation diagnostic passed. No full
phase, founder generation or model-worker call was run. The earlier static-only
qualification statements still apply to end-to-end evolution behavior.

Authentic historical seed files and ARC training data are absent locally, and no
absolute credit reading was obtained. The current percentage-based maker meter
does not enforce the owner's 62,000-credit ceiling. See
`reports/LOCAL_READINESS_20261006.md` for check outputs, configured paths,
account-probe results and the inputs needed before a bounded phase can proceed.

## Explicit fresh local initialization

An optional epoch-plan `fresh_inputs` field now takes exactly an absolute
`manifest` path and its `sha256`. `fresh_inputs.py` verifies a new random baseline,
pinned local public-training bytes and three disjoint task populations before
campaign reservation. The fresh driver emits a typed evaluator without loading
legacy code or inventing historical parent files. Server defaults are unchanged.
The direct driver also accepts `--fresh-inputs` with an explicit evolution policy;
it cannot combine that option with `--seed-from`.

Report IDs are frozen in the manifest but their files remain unavailable to this
fresh development loader. New random baseline tensors have explicit non-LLM
provenance. One local zero-worker phase and independent saved/zero/random replay
have now completed; this supersedes the earlier no-full-phase status only for
that operational slice. All arms scored zero. The rejected partial-domain
baseline and initial checkpoint-ordering failure remain preserved. See
`reports/FRESH_LOCAL_BASELINE_20261006.md` for exact evidence and remaining limits.
