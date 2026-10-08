# Bounded continuation: evaluation integrity and composition proposal

## State and scope

Inspected `D:\Github\CellForge`, branch `master`, HEAD `64a961d` (merged
shared-policy pilot). Origin is the existing Sanokei/CellForge repository.
The pre-existing `.gitignore` addition for `.env` was preserved. No commits,
pushes, deployment, maker/API calls, dependency installation, new experiment
directories, or live game evaluations were made. No protected tests, floors,
lane registry, saved tensors, or historical receipts were edited.

`LAB.md` and its owner-resumption section govern this pass. Some older status
paragraphs are stale: a remote and committed source now exist, but no
`runs/weights-*` directories exist in this checkout. The October 2 reports are
available; their original shared-probe tensors and traces are not. No local
`.agents/skills`, `.codex` repository instructions, or user Codex memory summaries
were found at the inspected locations.

Remote status is **unknown**. The existing command, with batch mode, strict host
checking and an 8-second connection timeout, failed:

```text
ssh: Could not resolve hostname cellforge-1: No such host is known.
```

`C:\Users\wkeif\.ssh\config` is absent; `ssh -G cellforge-1` reports the literal
alias as hostname and default identity locations. The missing information is
the existing host/IP mapping and usable pre-established SSH identity/trust
configuration. Authentication was not reached, so credential validity is unknown.
A Hetzner API variable exists in `.env`; its value was neither printed nor used.
Do not equate the October 2 "nothing running" note with current server status.

## Architecture and actual research bottleneck

`WeightGenome`/`NicheArchive` store literal tensors and lineage; the independent
cell and circuit archives distinguish specialist competition from composition.
`skillcore.chain_cell` executes two inherited cells in sequence.
`head_driver.breed` currently samples base cells from different niches, checks
composite tasks, then requires strict ARC development/selection gain without
regression. Its default router is largely a manually constructed selector based
on demonstration error. This is a useful baseline, not proof of an evolved,
LLM-authored router. `CircuitArchive` now requires actual execution evidence;
its historical test expecting admission from claimed outcomes must stay red
until reconciled by the owner, not by weakening admission.

The project therefore already supports much of the user's small-specialists and
breeding direction. The missing evidence is that inherited cells and routing
produce reliable gains on unseen combinations, beyond the best specialist and a
simple routing/composition baseline. The newer ARC-3 path is restricted Python
plus small tensors, not a demonstrated general neural chatbot.

Latest recorded shared-policy outcome: zero levels on ar25/lp85/sc25 at 1500
actions per game; lp85 had 1500 no-ops. This supports testing action-effect
learning and recovery, not extending routine per-game campaigns. The structure
bank's reported 287/302 is mostly task-specific authoring: only 38 survived
removing same-task cells in the saved audit. Neither result establishes AGI.

## Small implemented correction

- `scripts/arc3_env.py`: missing initial, post-action, or post-reset observations
  now produce explicit errors. An issued action/reset remains charged. Previously
  a missing post-step observation could terminate silently, leaving a control
  apparently valid. Normal budget exhaustion and early wins remain valid.
- `scripts/arc3_shared_pilot.py`: replay comparison includes game/seed/budget,
  terminal state, reset/no-op/distinct-frame counts, action histogram, final frame,
  bounded trace and aggregate totals. Trace lists/tuples compare identically after
  JSON round trips. Trace coverage remains the existing first 400 actions.
- `tests/test_arc3_run_integrity.py`: new isolated fault-injection checks load the
  production functions with mocked engine, numerical and worker dependencies.
  They test the evidence boundary, not game competence or sandbox execution.

This is evaluation maintenance supporting the next weight experiment, not a
capability result. Existing saved artifacts were not rewritten; a future replay
that exposes inconsistent evidence must report a mismatch.

## Validation and limits

Environment: Python 3.12, bytecode writes disabled; numeric checks configured for
one BLAS/OMP thread. No policy workers or external services ran.

```text
python -m unittest tests.test_arc3_run_integrity -v
Ran 10 tests in 0.126s
OK

Same new tests, loading the two production modules from git HEAD:
Ran 10 tests in 2.180s
FAILED (failures=20)  # includes subtest assertions; expected regression evidence

python -m unittest tests.test_one_standard tests.test_standard_semantic -v
Ran 25 tests in 5.155s
FAILED (failures=8)

python -m unittest tests.test_arc3_shared_pilot tests.test_arc3cell tests.test_arc_goal -v
Ran 16 tests in 1.434s
FAILED (failures=3, errors=2)
```

The standard failures are five milestone and three real-run checks, all because
`STANDARD.json` evidence is absent. The integration invocation includes 11 passing
ARC meta-tests, three missing-real-score failures, and two import errors for
missing NumPy. `ARC_ROOT` is unset. Existing recorded server passes have **not**
been reproduced in this pass. Earlier sandbox attempts hit temporary-directory
access errors and stopped at their 90/45/60-second deadlines; the completed
standard/integration checks used the approved local execution escalation with
60-second watchdogs. New regression tests used a 30-second watchdog. All scoped
test processes finished; the task-created workspace temporary directory was
removed. Syntax parsing and `git diff --check` pass.

Full output: `reports/continuation-integrity-20261005.log`,
`reports/continuation-integrity-before-20261005.log`,
`reports/continuation-standard-20261005.log`, and
`reports/continuation-integration-20261005.log`.

## Proposed next experiment — not started or authorized by this note

A separate **small composition probe** is outside the current closed skill-bank
campaign and beyond the one-founder ARC-3 authorization in `LAB.md`. It needs an
explicit bounded owner/orchestrator scope update. Merely running on the allowed
desktop or cellforge-1 does not reopen that campaign. Do not start its driver.

After access/dependencies and original provenance are available, use four existing
compatible, LLM-authored specialist cells, selected only from authorized
development evidence. Freeze their tensors, source and primitive competence
tests. Reuse the existing chain execution mechanism and enumerate their 12 ordered
two-cell chains; do not create a large niche library or change existing templates.
Permit at most **two** metered proposals: an authored literal joint router, then
one mutation from search-fitness failures. Charge both proposals even if invalid;
compare two matched random proposals with identical edit positions and ranges.
This tests reuse and routing; it does not establish founder evolution or the full
ONE_STANDARD milestone.

Pre-register three separate partitions: search fitness (maker feedback),
development selection (choose the final candidate once), and a sealed final
partition containing unseen ordered combinations of the known primitives plus
disjoint inputs/seeds. Final examples, pair identities, labels, outcomes and
derivatives must not enter maker prompts or selection. A candidate bank may contain
all possible chains; no final outcomes may decide which are retained. Hash manifests
and reject cross-partition duplicates before evaluation. Repeatedly consulted
validation data remains development data. Do not use the ARC report or excluded
ARC-AGI-2 evaluation split for this experiment.

Compare best single cell, simple demonstration-error routing over the same full
bank, uniform routing, matched random edits, and zero tensors. Remove each
contributing cell and shuffle its placement to test causal dependence; keep the
same frozen primitive tests to catch regressions. Report separately whether
composition beats singles and whether authored routing beats the simple router.
Freeze one final candidate and all controls, evaluate final data once in a clean
process, and stop. Failure, invalid controls or no gain means no expansion.
Synthetic unseen combinations are mechanism evidence, not novel-game or chatbot
capability evidence. Final-set provenance and leakage checks must be resolved
before generating or inspecting it; otherwise the deliverable remains a
development-only result.

Proposed resource ceiling (requires approval, not a runtime estimate): existing
cellforge-1 only, one scoring job/CPU, no GPU or external API/cloud purchase;
two maker calls at the existing 900-second per-call timeout, at most 10,000
scored case-arm invocations including replays and controls, 15 minutes total
scoring wall time, and no more than 3 GiB total process memory. Thus at most
45 minutes sequential wall time plus startup, with incomplete work reported at
the cap. Retain the 85% shared-pool refusal cap and stop at one percentage point
of additional measured usage; record teacher tokens/time and all numeric edits.
The ledger is stale, so no current headroom or per-call price is asserted.
No automatic retries or continuation. A passing pilot could justify a separately
approved multiseed study, not a general-intelligence claim.

## Interfaces for the parent's cloud-side reuse assessment

Latest steering: the parent is assessing Sakana AI reuse with cloud research
workers. Do not implement the above proposal before reconciling that assessment;
no further desktop experiments are scheduled.

- `src/cellforge/construction/niche_archive.py::WeightGenome` has `params`,
  `template`, `parents`, `origin`, `meta`. Identity hashes canonical JSON of
  template plus tensors; metadata/lineage are separate. JSON includes hash and
  parameter count. This reference implementation validates the specific
  37-position `adjacency_recolor/1` shapes (source/marker 1x10 plus scalar biases,
  target length 10, neighbor gate 1x4 plus scalar bias), finite non-boolean scalars
  of magnitude at most 100, default cap 64. It is not a general checkpoint loader.
  `from_json` checks the hash; call `validate` explicitly for the shape contract.
- `scripts/skillcore.py::make_cell(template, params, **meta)` produces
  `{kind: cell, template, params, id}`. Template functions accept
  `(params, grid, demos)`. `chain_cell(a,b)` stores both parts and parent IDs;
  `run(cell, grid, demos)` executes A then B, transforming demonstration inputs
  for B and excluding each demonstration from its own transformation context.
- `to_tensors(cells, router)` exports `index:tensor_name` keys; chains nest `A.`
  and `B.` prefixes. Router tensors are `route.W` (N by N) and `route.b` (N).
  `predict(tensors, task)` uses demonstrations to compute leave-one-out errors,
  selects `argmax(W @ -errors + b)`, and returns one predicted grid per test
  input in the evaluator's list-of-attempts format. `default_router` uses an
  identity matrix plus a small older-cell tie preference.
- `scripts/arc3cell.py::Policy` instead accepts restricted `act(params, mem,
  frame, avail, info)` source and tensors. Frames are 64x64; return is action ID,
  x, y. Memory persists within a game only. This path limits cells to 256 tensor
  positions and uses a separate worker with per-step timeout and Linux limits.
  These interfaces differ from pretrained LM/LoRA checkpoint interfaces; a
  compatibility adapter and a separately labeled scientific comparison would
  need explicit design, rather than silently substituting a checkpoint.
- No root project LICENSE/LICENCE or pyproject license declaration was found.
  The tracked license inventory contains only
  `research/claude_arc/dsl_port/third_party/arc_dsl/LICENSE`. That third-party
  license does not establish a license for all CellForge source. Verify each
  external Sakana artifact's code, weights and dependency licenses separately
  before redistribution; this pass did not publish or import external artifacts.

## Subsequent owner-directed implementation, static review only

The owner subsequently requested substantive integration without tests,
prototypes or experiments. The earlier validation output above applies only
to the earlier ARC evidence fix, not to the newer evolution code.

Implemented in the real driver: complementary-success parent selection with
unknown/zero-score handling; typed/versioned cell and circuit references;
ordered router roster validation and permutation; separate persisted curriculum;
bounded gpt-6-luna proposal workers; all-attempt matched controls and host-owned
group selection; immutable source/config/partition manifests and journals.
Two parallel implementation workers owned `weight_workers.py` and
`evolution_manifest.py`; the parent arranged an independent static review.

New modes activate by **policy presence**, including an explicit default/disabled
policy. They cannot fall through to the legacy full queue or maker route. Niche
lists default empty, workers default disabled, and call allowance defaults zero.
In-flight stages and previously reserved worker tags fail closed rather than
automatically repeat paid work. Frozen/sealed resume checks precede legacy module
loading and worker construction. Source binding includes the actually loaded
legacy code and parent tensors. No closed campaign was resumed.

Detailed interfaces, configuration, provenance and limitations are documented in
`docs/EVOLUTION_INTEGRATION.md`. Static AST parsing of the six implementation
files and `git diff --check` completed; no changed modules were imported or run.
No tests, model calls, new experiment directories, installation, deployment or
publication occurred during this subsequent integration. Runtime behavior and
capability remain unverified. The Hetzner access blocker is unchanged; no key was
transmitted and no credentials or trust settings were altered.

## Literal router and typed checkpoint milestone

The parent subsequently requested actual integration of evolved router tensors
and canonical typed checkpoint inference, again static-only. `head_driver.py`
now exposes a disabled router policy which freezes the expert roster after
explicit founders/imports and compares full W[N,N]/b[N] worker proposals with
all-reservation matched random proposals plus zero/incumbent controls. Strict
selection gain, preserved development anchors and superiority to every random
attempt are required for retention. Router journals bind hashes, exact expert
order, shapes, budgets, panels and selected identities; resume rejects unsettled
work and verifies the latest accepted router rather than the latest attempted one.

The parent assigned `router_evolution.py` to a separate helper owner and the
router event schemas/status API to the existing manifest owner. Main integration
owns `head_driver.py`, `skillcore.py`, `evolution_policy.py`, `cell_contracts.py`
and these documentation updates. No overlapping edits were made to worker-owned
files by the integration task.

Every explicit-policy `arc_eval.py` now uses `predict_bank` with evaluator-supplied
tensors. Saved contracts establish topology; supplied numeric values drive the
actual prediction, even for all-zero or duplicate ablated experts. New modes
avoid the legacy shared score cache, whose keys omit source/task versions.

This is staged fixed-roster integration, not joint variable-roster evolution or
a chatbot. Router/cell curriculum combination, larger circuits, hard scoring
resource limits, automatic receipt recovery and language-task interfaces remain.
No runtime source was imported, no tests or experiments executed, and no server
jobs or paid work started for this milestone. AST parsing and whitespace checks
are static evidence only. Full configuration/cost bounds:
`docs/EVOLUTION_INTEGRATION.md`.

## Bounded alternating phase integration

The next parent-directed source milestone connects existing cell and router
drivers through separate immutable phase scopes. Main integration adds
`Driver(..., phase_context=...)` and `run_phase()` in `head_driver.py`; the parent
assigned disjoint new files `evolution_epochs.py` and `checkpoint_transitions.py`
to separate workers. Each explicit invocation requests one phase index. No
automatic launch, copied allowance, changed frozen manifest or hidden seed import
is permitted.

The entire accepted search archive and its cell-evolution state carry forward:
completed niches, tried pairs, breed/mutation counters, RNG state and curriculum
exposure/anchors. The retained full bank stays separate while cell phases expand
the archive under a labelled reference router. Router phases freeze that expanded
roster. Strict development-selection gain and preservation of every incumbent
search/selection success govern `bank_gain` retention, without attributing that
gain to router learning unless separate router journal evidence supports it.

Phase seed digests enter the child manifest before worker construction. Stable
implementation/task scope and recomputed incumbent outcomes must match before
new proposals. Campaign/phase STOP remains effective; incomplete reservations
block advancement pending manual reconciliation. A roster above the configured
router cap stops before proposals rather than pruning inherited experts.

Typed checkpoint inference and protected evaluation files are unchanged by this
phase milestone. Runtime correctness remains unverified: source inspection and
AST/whitespace checks only, no tests, imports of implementation modules, research
runs, model calls, installation, deployment or commits.

Static follow-up found a protected-gate defect to resolve before interpreting
future ablations: `arc_goal_checks.check_weights_matter` sends a tensor dict to
`standard_checks._zeros_like`, whose list-only recursion returns scalar zero for
dict input. `solved` counts evaluator exceptions as wrong answers. Protected
tests remain untouched; a malformed scalar is rejected by typed inference and
must not be counted as evidence that weights matter. This is a future evaluation
blocker, not a reason to edit owner-controlled checks in this implementation pass.
