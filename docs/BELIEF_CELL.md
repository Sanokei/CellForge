# Structured-event belief cell

`scripts/belief_cell.py` provides a small numerical cell family for tracking an
actor's represented location belief about an object. This is a standalone,
stdlib-only implementation scaffold. It is **not learned social intelligence,
language understanding, emotion recognition, or a qualified ONE_STANDARD run**.
It does not parse natural language. There are no emotion labels, demographics,
personality assumptions, host tool actions, or connections to a live agent.

The caller supplies structured events and explicit observer sets. Actor/object
IDs are positional slots. The host loops over slots; the cell does not infer
identities, attention, visibility, deception, intentions or higher-order beliefs.
Those capabilities must not be inferred from this interface.

## Literal parameters and forward contract

Exactly **293 finite numeric literals** implement a `12 -> 16 -> 5` MLP:

| Tensor | Shape | Flat offset | Count |
| --- | --- | --- | --- |
| W1 | 16 x 12 | 0 | 192 |
| b1 | 16 | 192 | 16 |
| W2 | 5 x 16 | 208 | 80 |
| b2 | 5 | 288 | 5 |

Matrices flatten in row-major order, with outputs as rows. Inputs are old
five-way location distribution, observed-new five-way location distribution,
object-match bit, observation bit, in that order. The forward method computes
`h = tanh(W1*x+b1)`, then `softmax(W2*h+b2)` using a maximum-shifted softmax.
There are no embeddings, hidden recurrent tensors, copied outputs, learned
routers, residual identity paths or reference-rule fallbacks. Every actor/object
slot, including unrelated and unobserved slots, passes through this same MLP.

Distribution indices 0..3 represent four abstract locations. Index 4 is
`UNKNOWN`: an explicit unknown/outside/unspecified category, not a uniform
distribution over the other locations. The neural output may express a mixture.
The 293 count excludes host state, code, input encodings, generator and oracle.
It measures stored neural coefficients only, not total system complexity.

`parameter_schema()` returns the versioned layout and hash encoding.
`BeliefCell(tuple_of_293_numbers).forward(KernelInput(...))` returns a typed
`Distribution`. `parameter_hash` is SHA-256 of ASCII
`cellforge-belief-mlp/1`, one NUL byte, and all 293 coefficients encoded as
big-endian binary64. Integers convert to binary64; negative zero normalizes to
positive zero. This hashes the interpreted coefficients and schema, not an
author's original textual representation.

`to_dict()` returns exactly `schema`, `parameters` (a list), `parameter_hash`,
and `status="unverified-literal-weights"`. `from_dict()` requires that exact
contract and verifies the hash. No code is evaluated and no provenance claim
is authenticated by the loader. A later writer must preserve original prompts,
responses, numeric conversion, counterexamples, source, lineage and budget
receipts separately. A matching hash is content identity, not proof of learning
or LLM authorship. A future change to forward semantics needs a new schema.

## Typed host boundary and reference semantics

All public state values use frozen dataclasses and immutable tuples:

| API | Contract |
| --- | --- |
| `Distribution(values)` / `Distribution.at(location)` | Exactly five finite probabilities in [0,1], sum within 1e-9 of one; no silent normalization |
| `KernelInput(old, observed_new, object_match, observation)` | Typed distributions and actual bool flags; unseen observed-new must be `UNKNOWN` |
| `BeliefState(actor_count, object_count, beliefs)` | 1..4 actors and 1..4 objects; actor-major matrix of typed distributions |
| `BeliefState.unknown(actor_count, object_count)` | Initialize every belief explicitly to unknown |
| `WorldState(object_locations)` | Separate tuple of 1..4 actual location codes 0..4 |
| `MoveEvent(object_id, destination, observed_by)` | Object 0..3, destination 0..4, unique tuple of observers 0..3 |
| `Scenario(initial_world, initial_beliefs, events)` | Matching object dimensions, existing observer/object slots, 1..64 events |
| `apply_event(cell, beliefs, event)` | Call the kernel once per slot, returning new beliefs; no world-state argument |
| `advance_world(world, event)` | Set only the moved object's actual location, independent of observers |
| `reference_update(input)` / `reference_step(beliefs, event)` | Authored target rule for synthetic examples, never a model result |

The event constructor enforces global bounds; applying it to a state also
checks that its slots exist in that particular state. Standalone `WorldState`
does not contain actor identities, so `advance_world` only checks its object
bound. `Scenario` checks both dimensions together. No object/actor creation,
deletion, arbitrary identifiers, nested belief graph or free-text input exists.

The host encodes a move separately for each actor and object. Object-match
means the current slot is the moved object; observation means the actor is
listed in `observed_by`. For non-observers, the destination distribution is
always replaced by `UNKNOWN` before reaching the cell. The occurrence of an
event and object-match bit remain visible; this is destination masking, not a
claim that the actor sees no trace of an unseen event. World state never enters
the MLP. Observer selection itself is supplied by the host, not learned.

The reference target replaces the old distribution **only if both bits are
true**. Otherwise it preserves the old distribution, including mixtures.
Observed moves to unknown set the belief to unknown. Unseen moves keep a stale
belief even as the actual world changes. Moves of another object preserve its
unrelated slots. A same-location event is a valid observation/set-location event.
The reference does not track what an actor believes another actor believes.

These are reference invariants, not promises made by untrained weights. In
particular, the zero cell outputs a uniform distribution in every slot and
fails preservation. The host never repairs those failures using the oracle.
Finite-logit softmax generally cannot produce exact one-hot reference targets;
future scoring must declare a probability loss/tolerance and any argmax tie
policy rather than silently claim exact distribution equality.

`BeliefError` exposes a `.code`, including `invalid_integer`, `invalid_number`,
`invalid_distribution`, `invalid_input`, `hidden_observation`,
`invalid_parameters`, `invalid_artifact`, `hash_mismatch`, `invalid_state`,
`invalid_world`, `invalid_event`, `slot_out_of_bounds`, `invalid_scenario`,
`invalid_example` and `numeric_overflow`. Bool is rejected as a numeric literal
or slot ID; coercion from strings, NaN/infinity, wrong dimensions and absent
slots are errors. Extremely large finite coefficients can be stored, but a
nonfinite affine computation raises `numeric_overflow`; it is never reported
as a successful prediction. Operations allocate bounded replacement state and
leave inputs unchanged even if an error interrupts a computation.

## Original synthetic events and future integration

`generate_scenario(seed, actor_count=4, object_count=4, event_count=12)` uses
only a local stdlib random generator. Seed is an integer in 0..2**32-1. The
generator accepts 2..4 actors, 2..4 objects and 6..64 events, ensuring room for
observers/non-observers and unrelated objects. Its six-event prefix includes
an all-observed placement, a partially observed move, an observed move to
unknown, an unrelated-object move, an all-observed placement and a move unseen
by everyone. Remaining events vary destinations, objects and observer sets.
It starts with unknown world locations and unknown beliefs. This authored
generator uses no downloaded benchmark, corpus or ARC task. A fresh seed alone
does not establish independent held-out generalization.

`reference_examples(scenario)` returns up to 1024 typed `KernelExample` values,
with event index, actor/object slots, kernel input and expected distribution.
Each old belief comes from the reference rollout: these are **teacher-forced**
local targets. They do not measure recursive model performance. A later
evaluation must also feed each model prediction into its own next step and
account for accumulating errors, alongside matched controls.

```python
from scripts.belief_cell import (
    apply_event, advance_world, generate_scenario, random_cell, reference_examples,
)

scenario = generate_scenario(7)
cell = random_cell(11)  # Untrained initialization; not an LLM founder.
examples = reference_examples(scenario)  # Authored targets for a later writer.
beliefs, world = scenario.initial_beliefs, scenario.initial_world
for event in scenario.events:
    beliefs = apply_event(cell, beliefs, event)  # Carries model errors forward.
    world = advance_world(world, event)         # Separate host truth.
artifact = cell.to_dict()  # In memory only; no automatic file/run creation.
```

`zero_cell()` and `random_cell(seed)` are explicitly untrained controls. This
change supplies no trained or LLM-written founder, fitting, mutation,
competition, crossover, selection, registry admission, live failure journal,
agent binding or maker calls. A later authorized integration may supply the
same 293 literal positions via `BeliefCell`/`from_dict`, but must add its own
provenance, equal-budget controls, frozen task splits and scientific gates.
No default enabled route has been added to grid, main or agent contracts.

## Operational verification

Run only the bounded self-check explicitly:

```text
python -B scripts/belief_cell.py --check
```

The self-check covers count/layout and a known forward path, artifact/hash
round-trip, deterministic initialization/generation, typed finite output,
observed/unseen/unknown/unrelated reference updates, world separation,
destination masking, an explicit untrained-kernel failure to match the oracle,
and rejection of selected malformed/nonfinite/out-of-bounds inputs. Its probe
weights are an authored arithmetic fixture, not a submitted founder. It does
not search weights, train a model, create runs or access any external data.

Executed locally on 2026-10-06 with the existing Python interpreter: exit code
0, **34 deterministic operational assertions passed**, 293 parameters, zero
model calls and zero training steps. The canonical zero-cell parameter hash was:

```text
5e0666d2a8f200d0580e4348f42c22abcfa1cb6b40ec62aec990552632f57b53
```

This result validates only the stated operational checks. The full standard,
semantic, ARC and historical lane suites are outside this narrow check
authorization and are not claimed green. No scientific milestone,
generalization gain, language competence or full social awareness is claimed.
