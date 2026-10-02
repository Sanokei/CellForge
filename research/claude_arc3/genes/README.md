# Observation-only ARC-AGI-3 genes

`GeneAgent` implements the generator's push contract:

```python
from research.claude_arc3.genes import GeneAgent

agent = GeneAgent(budget=600, seed=0)
while not agent.is_done(frames, latest_frame):
    action = agent.choose_action(frames, latest_frame)
    latest_frame = env.step(action)
    frames.append(latest_frame)
agent.close()
```

The caller supplies `arcengine` (the real SDK or the generator's existing shim).
Actions are `GameAction` values; click data use pixel `x,y`. Only `frame`, `state`,
`available_actions`, and `levels_completed` are read. The agent never receives an
environment, genome, palette labels, transition function, or oracle. Generator
internals are used solely by the benchmark to create instances, verify their
solvability, and report optimal-length baselines.

## Gene inventory

| Gene | Evidence or behavior | Location |
|---|---|---|
| Tile lattice | Infer tile size from rendered pixel boundaries; map clicks to centers | `perception.TileGrid` |
| Object inventory | Find singleton colors with positions; no semantic color constants | `perception.singletons` |
| Avatar detection | Vote for translated objects, scoring action-consistent direction diversity | `perception.AvatarDetection` |
| Action-effect table | Learn permuted action-to-displacement counts; no-ops do not define directions | `learning.ActionEffects` |
| Blocked-edge learning | Distinguish failed traversal from successful movement | `learning.TerrainLearning` |
| Hazard learning | Remember lethal locations and singleton colors; never mark all floor lethal | `learning.TerrainLearning` |
| Goal hypothesis tests | Prioritize untested objects, retain unsuccessful visits | `learning.GoalHypotheses`, `agent` |
| Key/door dependency | Detect collection plus remote disappearance; release blocked edges | `learning.KeyDoorLearning` |
| Push detection | Correlate a second object's displacement with avatar contact | `learning.PushDetection` |
| Hazard motion | Predict witnessed motion with two-position context or linear extrapolation | `learning.HazardMotion` |
| Navigation | Shortest paths honoring observed directed blockage | `planning.navigation` |
| Push planning | Joint avatar/block BFS; reposition avatar; reject non-goal static corners | `planning.push_plan` |
| Click effect learning | Learn toggle sets through individual interventions | `planning.ClickEffects` |
| Binary goal testing | Solve learned XOR effects for both possible uniform-color goals | `planning.ClickEffects` |
| Sequence progress/replay | Detect disappearance after leaving an object; remember progress prefix and tests by depth | `agent` |
| Frontier exploration | Favor unvisited reachable cells; leave walls when calibration stalls | `agent` |
| Recovery/lifecycle | Reset on death or push deadlock; isolate learned state between levels | `agent` |

These are Python behavioral genes, not CellForge IR chromosomes or trained weights.
The classes can be reused individually; the director behaviors compose them in
`GeneAgent`. Optional `disabled` names support ablations (`avatar`, `effects`,
`terrain`, `goals`, `keydoor`, `push`, `hazard_motion`).

## Synthetic-only validation

Run against a checkout of branch `claude/arc3-gan-generator`:

```powershell
python -m pytest tests/test_arc3_genes.py -q
python -m research.claude_arc3.genes.benchmark --generator-root PATH_TO_GENERATOR_CHECKOUT --seeds 10 --budget 600 --output research/claude_arc3/genes/results/synthetic.json
python -m research.claude_arc3.genes.benchmark --generator-root PATH_TO_GENERATOR_CHECKOUT --seeds 3 --budget 300 --ablations --output research/claude_arc3/genes/results/ablations.json
```

The main sweep covers all six families, two levels per episode, ten layouts per
family, permuted actions, randomized palettes, obstacles, distractors, and optional
no-op actions. Each variant gets the same instance and action budget. JSON receipts
include generator commit, source hashes, genomes, per-episode outcomes, action
counts, deaths, optimal lengths, efficiency, and aggregate results. Aggregate
`mean_score` is the generator's score averaged over episodes, not an ARC Prize score.
The random baseline samples full-frame click coordinates like the generator's
random agent; its click comparison includes the benefit of target perception.

## Scope and limitations

The perception assumes aligned square tiles, a wall-colored top-left corner, and
single-tile objects with individually distinguishable colors. The binary click
solver is bounded at 16 targets and assumes deterministic XOR effects. Push search
is bounded at 30,000 states and models one movable block. Moving-hazard prediction
is a hypothesis, not a certified temporal planner. Sequence inference can confuse
collection and other remote disappearances. Mixed keyboard/click tasks currently
favor keyboard actions. These assumptions match this synthetic curriculum; no
public games or public-game implementation code were used for evaluation. Results
do not establish generalization to private ARC-AGI-3 games.

See `REPORT.md` for the measured results and ablation interpretation.
