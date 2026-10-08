# Actual ARC iteration readiness, 2026-10-06

The next packet is prepared at `runs/weights-native-arc-iteration01-20261006/`.
No author was dispatched and no candidate program was executed. This is a new
bounded protocol, not a reopening of an archived lane or an ARC result.

## First task and budget

Task `bd283c4a` is one of the four already declared SEARCH tasks from official
ARC-AGI-2 public training, pinned at commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49`. A host-written feasibility calculation
matches both demonstrations: rank colors by global frequency, then fill columns
left to right and bottom to top. Equal-frequency behavior is not established by
these two examples. No query label was inspected or scored for this diagnosis.
This is a program hypothesis, not an LLM-authored weight result. The current
uniform-color cell cannot express this spatial rearrangement; the available
sequence sorter operates on one row, not this grid transformation.

- `AUTHOR_PROGRAM.json`: direct Codex program baseline using the existing
  `run(params, grid, demos)` interface with an empty parameter map. Constants
  may live in code. This arm never earns literal-weight credit.
- `AUTHOR_WEIGHTS.json`: separately authored reusable structure and at most 256
  meaningful literal tensor entries, with zero and matched random controls.
- `PROTOCOL.json`: at most one founder plus one conditional training-failure
  revision per arm, four author calls total. Stop an arm as soon as its demos
  pass. Parent dispatches authorized Codex authors; no maker/API fallback.
- One numerical thread, one trial at a time, 2-second grid-call limit, 30-second
  trial limit, 1536 MiB address-space limit. At most 512 authored tensor positions
  and 512 matched random positions if both weight attempts use the full cap;
  actual positions must be charged, including invalid attempts. Extra diagnostic
  random draws and their evaluation costs are recorded separately.
- Absolute Codex usage is unknown. The existing 62,000-credit ceiling is a
  ceiling, not evidence that the proposed tranche is free or metered.

Select only using leave-one-out training execution. Freeze the candidate before
any query score. Preserve every prompt, raw response, errors, actual predicted
grid, difference map, revision parent and hash-chained receipt. Replay the frozen
candidate independently. Existing search queries and the four-task development
panel have prior exposure: a new score on them is development evidence, not a
fresh held-out result. Reserved report IDs `d282b262` and `11dc524f` remain absent;
ARC-AGI-2 evaluation access remains excluded.

Protocol hash:
`1f416bd8ae9e141fbcc46079eea9b78e177ee0caa3fdbae58e89d012ae85b4e8`.

## Existing seam and the small correction

`struct_pilot.attempt` already implements candidate -> sandbox execution ->
training feedback -> one revision, and saves the attempt source/tensors/results.
The native import modules already supply raw-response provenance and immutable
receipts. These are the integration seams; a replacement service is unnecessary.
The server pilot's direct maker call and historical split runner must not be
invoked blindly from this local batch.

Fixed `struct_pilot.feedback_of`: previously it always returned demonstration 0,
even if that example passed and another failed. It now finds the first actual
leave-one-out failure, returns its index, prediction, runtime error or grid
differences, and caps difference details at 64 pixels. Shape errors and
non-reproducible failures remain explicit. It never reads query data.

Nine focused tests pass (six new feedback tests plus the three existing static
whitelist tests). They cover a later-example failure, leave-one-out context,
query-access prohibition, runtime/shape errors, bounded differences, replay
inconsistency, and control/rejection paths. No model source was executed by
these tests. Log: `native-arc-iteration-feedback-tests-20261006.log`.

One older acceptance weakness remains important for the upcoming adapter:
`structcode.evaluate` discards errors from zero/random `fit` calls. The new
protocol explicitly requires retaining those errors and disallows weight credit
from crashing controls. Do not use its boolean `controls_ok` alone. Historical
acceptance results were not rewritten or reinterpreted as new evidence.

## Runtime and data availability

This is `D:\Github\CellForge`, a dirty shared checkout. Existing work was
preserved. Eight official ARC-AGI-2 training JSONs are available locally; a
separate local ARC-AGI-1 dataset has not been verified. `ARC_ROOT` is unset.
No evaluation/report data was acquired.

The bundled Python has NumPy and Pillow but no `arc_agi` or `arcengine`.
`ARC3_DIR` is unset, and local ARC-3 game files were not found. WSL's read-only
`--list --quiet` check returned usage with exit 1, so it did not establish a
usable Linux runtime; Docker is not on PATH. No software was installed.

The existing Windows structure worker lacks an address-space limit and inherits
the environment. The prepared batch does not execute authored programs through
that fallback. A verified bounded Linux sandbox is required; source validation
alone is not a substitute. The current LAB authorization explicitly excludes
Hetzner work, so no SSH connection, deployment, remote maker, job change or
credential/security change occurred. Current remote jobs are unknown. LAB's
2026-10-02 "nothing running" statement is historical, not a current observation.

## ARC-3 action-effect path and next bounded batch

`arc3_env.play` creates an OFFLINE local environment, calls
`arc3cell.Policy.step`, and feeds `act(params, mem, frame, avail, info)` legal
actions. `_do` maps action IDs to `arcengine.GameAction`; action 6 carries click
coordinates. Reset after GAME_OVER is charged. Policy memory starts fresh per
game; zero/random/blind controls share the action budget. Blind hides both frame
and frame hash. This path is already present and is the place to test learning
from action effects.

The old shared founder/repair ended with zero levels. Its no-op counts on
ar25/lp85/sc25 were 1363/1500/353 out of 1500 actions. LAB expressly prohibits
expanding that old campaign; the next hypothesis must address action-effect and
no-op recovery. A new separately named hypothesis can use the same public
development games, without game-ID routing: observe frame changes after each
action, distinguish ineffective actions, switch exploration after repeated
no-ops, and test a shared policy with fresh memory per game.

Proposed ARC-3 tranche, not launched: one shared founder and at most one
development-failure revision; first diagnose 150 charged actions per game on the three games,
with matched zero/three-random/blind and scripted baselines. Candidate plus five
controls across three games costs at most 2,700 actions per attempt; two attempts
plus one three-game scripted baseline cost at most 5,850 actions. Set `wall=30`:
39 plays have a maximum sequential allowance of 19.5 minutes plus startup. This
is a diagnostic budget, not a full-game benchmark or an automatic launch.
Save complete trajectories with `keep_trace=True`; its current 400-action cap
accommodates the 150-action diagnostic. A later 1500-action study would need a
narrow trace extension and a separately agreed budget. No new framework needed.

ARC-3 cannot currently run locally: engine, game files and the bounded sandbox
are missing/unverified. Remote access would require scope adjustment from the
current explicit no-Hetzner boundary. Public-game levels and win counts are
development diagnostics, not RHAE or a private benchmark score.

## Qualification and remaining blockers

This work makes the next experiment concrete; it does not improve ARC accuracy
or establish reusable neural learning yet. The prepared STANDARD explicitly
declares no scientific acceptance. Full standard qualification remains red.
No protected test, floor, lane registry or historical run was changed.

Required broad check was invoked in a one-thread child with a strict 30-second
limit; it remains incomplete and red (standard failures and semantic errors
preceded timeout). Exact result:

```text
python -m unittest tests.test_one_standard tests.test_standard_semantic -v
STATUS: TIMEOUT
ELAPSED: 30.016s
```

Full partial output: `native-arc-iteration-standard-gates-20261006.log`.
The prepared two-attempt tensor protocol does not satisfy the three-seed,
population, mutation, crossover and improvement milestones. No qualification
was claimed from the passing focused tests. `git diff --check` also passed.

Immediate blockers: an authorized bounded execution runtime; a native adapter
that records control errors and raw trajectory instead of invoking the server
maker; parent reservation of the four-call maximum under unknown current credit
usage. ARC-3 additionally needs its existing engine/game installation and
complete action-effect traces. The research/paper worker remains parent-owned.

## Direct-prediction fallback, authorized by parent

`AUTHOR_DIRECT_GRID.json` now contains both training demonstrations and the one
query INPUT, encoded as digit rows; no query output is exported. Its separate
`DIRECT_GRID_PROTOCOL.json` permits one tools-free Codex response, frozen before
host-only exact scoring. Author calls and query scores are still zero. This path
needs no authored-code execution and can proceed locally. It is direct reasoning
evidence, with identity and a training-derived host program as separate baselines;
it earns no genome/weight positions. No verified fixed-template composition for
full-grid frequency packing was found. The sequence frequency sorter accepts one
row; adding a new host algorithm now would be infrastructure, not learned weights.

Direct protocol hash:
`d9deb8813bf22b4e48811da993560ce6a3cef152992d4f76a6cb636da5f788df`.
