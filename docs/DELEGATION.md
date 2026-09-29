# Hierarchical delegation protocol (`delegation/1`)

Purpose: scale direct authoring by delegating down the ladder. Each level
specifies the level below with **contracts** (unit tests) and integrates what
comes back. Workers never spawn workers directly; they request subtasks and
the orchestrator commissions them (budget and permissions stay external).

## Roles

| role | authors | writes contracts for | integrates |
|---|---|---|---|
| orchestrator | nothing numeric | expert scopes (behavioural goals) | accepts graphs |
| architect / expert-maker | router, expert wiring, root | circuits (clusters) | circuits → experts → model |
| circuit-maker | micro cells (≤40 slots) + one cluster | (optional) its own cells | cells → cluster |

## Phase A — decomposition (architect)

Deliverables in the architect's task directory:

* `plan.json`:
  ```json
  {"experts": [{"name": "ex_...", "purpose": "...", "circuits": ["cl_a", "cl_b"]}],
   "router": {"purpose": "...", "circuits": ["cl_r"]},
   "circuits": [{"name": "cl_a", "purpose": "...", "inputs": {"x": [32]},
                 "output": {"y": [4]}, "contract": "contracts/cl_a.json",
                 "namespace": "cl_a."}],
   "state": {"dim": 0, "meaning": "..."}}
  ```
* `contracts/<circuit>.json` — per `construction/contracts.py`. A contract
  must specify behaviour precisely enough that a circuit-maker who never sees
  the text corpus could satisfy it (e.g. "output[0] ≥ 1 iff prev is a vowel",
  "output = +1 if the most recent subject noun was singular, -1 if plural").

## Phase B — circuits (circuit-makers, in parallel)

Each circuit-maker receives one circuit spec + contract. It writes
`proposal.json` containing only `set_param` / `put_def` ops for its
namespace (parameter and definition names start with the given prefix; the
cluster itself carries the circuit's name and `"role": "cluster"`), and
checks it with:

```bash
cellforge component-test --run <run> --task <task> --proposal proposal.json --contract <contract>
```

Component tests are free (not dev-evals). A circuit-maker may not change the
contract; if it believes the contract is wrong or unsatisfiable under the
40-slot cell cap, it says so in `notes.md` with a counterexample.

## Phase C — integration (architect, resumed)

The architect merges accepted circuit fragments, adds experts/router/root,
and uses dev-evals to test the whole model. It may request circuit revisions
(new contract version → new circuit task) through the orchestrator.

## Accounting

Every contract version, component-test result, dev-eval and accepted fragment
is recorded (task dirs + ledger). Construction efficiency is compared across
arms as final-eval quality per maker token / dev-eval / wall-clock.
