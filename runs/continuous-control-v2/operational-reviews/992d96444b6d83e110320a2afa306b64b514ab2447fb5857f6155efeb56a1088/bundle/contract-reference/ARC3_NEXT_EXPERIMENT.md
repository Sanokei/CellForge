# ARC-3: next experiment, not another game-specific campaign

Decision proposal, 2026-10-02. Based on `LAB.md`, `scripts/lab_manager.py`,
`docs/ONE_STANDARD.md`, and the saved ARC-3 verification and transfer records.
This is a protocol proposal; no maker calls or game evaluations were made for it.

## Recommendation

Stop extending `weights-arc3-managed1` merely to increase its credited-level
total. Preserve its 33 recomputed levels as evidence of game-specific competence.
The relevant weakness is transfer: 2 successes in 168 cross-game plays, and the
randomised donor comparison did not show a benefit (3/17 versus 3/15). These small
samples do not establish that transfer is impossible, but they do not support
spending another campaign on the same recipe.

Spend the next research tranche on a single policy with literal inherited weights
that learns action effects during play. Every game must receive exactly the same
source, tensors, router and initial memory. Object discovery, action-effect
tracking, revisiting-state avoidance and short planning should be inferred from
observations, rather than selected by a game name. This targets a missing ability
needed by the project: reusable adaptation rather than a collection of solutions.
It cannot guarantee AGI, and an ARC-3 success would not establish that endgame.

## What would actually run

First take one measurement with `scripts/arc3_shared_pilot.py run
runs/weights-arc3-shared-probe`: one generic shared founder, no game facts or
probes in its prompt, played unchanged on the fixed public development panel
`ar25`, `lp85`, `sc25`, at 1500 actions and seed 0. It stops after one maker call
regardless of outcome. Zero/random/blind controls run at the same positions and
budget. A crashed control cannot earn weight-dependence credit. `verify` replays
the saved founder without a maker call. The truthful undersized `STANDARD.json`
deliberately leaves standard contract/scientific requirements unmet; report the
resulting red tests. Save maker prompt, raw response and route receipts, tensors,
source, action traces, game results and control errors. The following 16-call
study is a later proposal, not an automatic continuation of that first probe.

Use the existing sandbox and `act(params, mem, frame, avail, info)` interface.
Do not build a new framework first. Any minimal harness change must directly
support the weight experiment and be charged/documented as infrastructure.
Do not edit the project's tests or floors. A new experiment must ship a truthful
`STANDARD.json`; do not claim the full standard is met by this pilot.

1. Freeze the development suite as the current 25 public games. All have already
   been exposed to the maker/research process. They are development data, even if
   some are excluded from a particular writer's prompt. No retroactive split can
   make them an honest novelty test.
2. Fix a 1500-action budget per game, counting resets, and reset all policy memory
   at each game boundary. Record actions to first level and complete trajectories,
   plus levels completed. Hide game IDs, tags, baseline solutions and game-specific
   probe cards from the policy. Give writers development failures and observations,
   labelled explicitly as development material.
3. Author four founders, each with literal tensors and a generic observation-driven
   mechanism. Reuse the existing 256-entry limit for individual cells. Where cells
   cooperate, the router must also consist of authored literal weights, receive
   observable features only, and be evaluated with its cells as one joint genome.
   Track individual cell competition separately from joint-genome competition.
4. Make eight charged mutations from failures, with each feedback batch covering
   several games and preserving previous opponents. Then make four charged
   crossover proposals using live selected parents. Record which literal entries
   are inherited and which are rewritten. Invalid, neutral and rejected proposals
   use the same budget as successful ones. This is a 16-writer-call pilot, with no
   manager calls. Stop there and audit before expanding.
5. Select on development only. Evaluate every eligible proposal on the same game
   positions and action limits as its paired controls. Preserve all results rather
   than reporting only each game's best different policy. The final score belongs
   to one frozen joint genome used across the entire suite.

The pilot tests shared ARC-3 adaptation. It does not yet satisfy the required
single-model ARC-1/2/3 architecture. Any follow-on must preserve a joint lineage and
joint evaluation across those benchmarks, using authorised development data only.

## Controls and decision gate

Keep zero tensors, three random tensor draws and blind observations, but do not
let these weak ablations be the only controls. Include paired random mutations
with the same edit vocabulary, proposal count and evaluation budget. If a zero or
random tensor creates an invalid policy, report that explicitly; compare against
valid random policies as well. A control that merely crashes proves little about
useful numeric structure. Evaluate the existing scripted explorer at the same
1500-action budget: its reported 17 levels at 3000 actions is not a matched
comparison to this pilot. Include a no-crossover counterpart if making a
crossover claim, and charge that arm equally.

Predeclare a pilot expansion gate: one frozen genome must beat the matched
scripted explorer on at least three games and improve total completed levels
over both its best founder and the matched random-edit champion without reducing
the number of games with a solved level. Independently recompute the results from
saved tensors and source. Report every other game and every failed arm. This is
an engineering gate, not statistical evidence of general intelligence. Passing
it justifies a larger, three-seed, equally budgeted study; failing it stops this
approach until there is a concrete new hypothesis.

Representation perturbations (e.g. consistent colour permutations with action
semantics unchanged) can diagnose dependence on fixed colours, but require
validated environment wrappers. They are not new games or a novelty benchmark.
Do not silently alter goals, actions or scoring when applying such probes.

## Budget and true novelty

Use `scripts/maker.py` for every writer call and retain its shared-pool ledger and
85% refusal cap. Also stop this pilot at either 16 writer calls or a one-percentage-
point increase in measured weekly usage, whichever comes first. The call-to-limit
conversion in `LAB.md` is an estimate, not a guarantee. Record current usage before
starting; other desktop/server activity can consume the same pool. No open-ended
manager fallback or automatic continuation beyond the tranche.

Before claiming new-game generalisation, acquire genuinely withheld games with
documented provenance and prior-access history, freeze the full policy and
adaptation protocol, and evaluate once without returning outcomes to development.
The maker may adapt only through the frozen in-game mechanism within the common
action budget; no fresh target-specific writer call. Public games risk maker
pretraining familiarity even if unseen in this repository. Independently authored
withheld games with undisclosed mechanics would provide stronger evidence.
Acquisition/downloads and any external service must follow owner authorisation.

Do not touch the excluded ARC-AGI-2 evaluation split. That is a separate owner
decision. Without genuinely withheld ARC-3 games, the honest deliverable is a
measured improvement in shared-policy development transfer, with novelty still
unresolved.
