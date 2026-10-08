"""Bounded structured-event belief tracking, not natural-language understanding.

The only neural computation is a literal 12 -> 16 -> 5 tanh/softmax MLP.
Host state handling and the synthetic reference rule are explicit scaffolding.
Importing this stdlib-only module performs no I/O, training or model calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import struct


MAX_ACTORS = 4
MAX_OBJECTS = 4
LOCATION_COUNT = 5
UNKNOWN = 4
MAX_EVENTS = 64
INPUT_SIZE = 12
HIDDEN_SIZE = 16
OUTPUT_SIZE = 5
PARAMETER_COUNT = 293
SCHEMA = "cellforge-belief-mlp/1"
_LAYOUT = (("W1", (16, 12), 0, 192), ("b1", (16,), 192, 16),
           ("W2", (5, 16), 208, 80), ("b2", (5,), 288, 5))


class BeliefError(ValueError):
    """An explicit contract/numerical error with a stable machine-readable code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(code + ": " + detail)


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise BeliefError(code, detail)


def _integer(value: int, low: int, high: int, label: str) -> None:
    _require(type(value) is int and low <= value <= high, "invalid_integer",
             f"{label} must be an integer in [{low}, {high}] (not bool)")


def _number(value: float, label: str) -> float:
    _require(type(value) in (int, float), "invalid_number", label + " must be a literal number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise BeliefError("invalid_number", label + " is not finite binary64") from error
    _require(math.isfinite(result), "invalid_number", label + " must be finite")
    return 0.0 if result == 0 else result  # Canonicalize negative zero for hashing.


@dataclass(frozen=True)
class Distribution:
    """P(location=0,1,2,3,unknown); unknown is a category, not uniform doubt."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(type(self.values) is tuple and len(self.values) == LOCATION_COUNT,
                 "invalid_distribution", "exactly five probabilities in a tuple required")
        values = tuple(_number(value, "probability") for value in self.values)
        _require(all(0 <= value <= 1 for value in values)
                 and abs(math.fsum(values) - 1.0) <= 1e-9,
                 "invalid_distribution", "probabilities must be in [0,1] and sum to one")
        object.__setattr__(self, "values", values)

    @classmethod
    def at(cls, location: int) -> Distribution:
        _integer(location, 0, UNKNOWN, "location")
        return cls(tuple(float(index == location) for index in range(LOCATION_COUNT)))


@dataclass(frozen=True)
class KernelInput:
    old: Distribution
    observed_new: Distribution
    object_match: bool
    observation: bool

    def __post_init__(self) -> None:
        _require(type(self.old) is Distribution and type(self.observed_new) is Distribution,
                 "invalid_input", "typed old and observed-new distributions required")
        _require(type(self.object_match) is bool and type(self.observation) is bool,
                 "invalid_input", "object_match and observation must be bool")
        _require(self.observation or self.observed_new == Distribution.at(UNKNOWN),
                 "hidden_observation", "unseen destinations must be masked to unknown")

    def vector(self) -> tuple[float, ...]:
        return (self.old.values + self.observed_new.values
                + (float(self.object_match), float(self.observation)))


def parameter_schema() -> dict:
    """Return a fresh layout description; rows are outputs, columns are inputs."""
    return {"schema": SCHEMA, "parameter_count": PARAMETER_COUNT,
            "input_order": ["old[0:5]", "observed_new[0:5]", "object_match", "observation"],
            "hidden_activation": "tanh", "output_activation": "softmax",
            "dtype": "finite-binary64", "flattening": "row-major",
            "tensors": [{"name": name, "shape": list(shape), "offset": offset, "count": count}
                        for name, shape, offset, count in _LAYOUT],
            "hash_encoding": "SHA256(schema ASCII + NUL + 293 big-endian binary64 values); -0 => +0"}


def _affine(weights: tuple[float, ...], inputs: tuple[float, ...], bias: float) -> float:
    try:
        result = math.fsum((bias,) + tuple(weight * value for weight, value in zip(weights, inputs)))
    except (OverflowError, ValueError) as error:
        raise BeliefError("numeric_overflow", "nonfinite affine accumulation") from error
    _require(math.isfinite(result), "numeric_overflow", "nonfinite affine result")
    return result


@dataclass(frozen=True)
class BeliefCell:
    """293 supplied literals, with no reference-rule bypass and no learned claim."""

    parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(type(self.parameters) is tuple and len(self.parameters) == PARAMETER_COUNT,
                 "invalid_parameters", "exactly 293 literal parameters in a tuple required")
        object.__setattr__(self, "parameters", tuple(
            _number(value, "parameter") for value in self.parameters))

    @property
    def parameter_hash(self) -> str:
        payload = SCHEMA.encode("ascii") + b"\0" + struct.pack(">293d", *self.parameters)
        return hashlib.sha256(payload).hexdigest()

    def forward(self, inputs: KernelInput) -> Distribution:
        _require(type(inputs) is KernelInput, "invalid_input", "KernelInput required")
        values, parameters = inputs.vector(), self.parameters
        hidden = tuple(math.tanh(_affine(parameters[row * 12:(row + 1) * 12], values,
                                        parameters[192 + row])) for row in range(16))
        logits = tuple(_affine(parameters[208 + row * 16:208 + (row + 1) * 16], hidden,
                               parameters[288 + row]) for row in range(5))
        maximum = max(logits)
        # Subtraction can underflow to -inf for extreme finite logits; exp(-inf)
        # is the correct zero limit. Affine overflow itself is always an error.
        exponentials = tuple(math.exp(value - maximum) for value in logits)
        denominator = math.fsum(exponentials)
        return Distribution(tuple(value / denominator for value in exponentials))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "parameters": list(self.parameters),
                "parameter_hash": self.parameter_hash, "status": "unverified-literal-weights"}

    @classmethod
    def from_dict(cls, payload: dict) -> BeliefCell:
        _require(type(payload) is dict and set(payload) == {
            "schema", "parameters", "parameter_hash", "status"},
            "invalid_artifact", "exact belief-cell artifact fields required")
        _require(payload["schema"] == SCHEMA and payload["status"] == "unverified-literal-weights",
                 "invalid_artifact", "unsupported schema/status; this loader verifies no provenance")
        _require(type(payload["parameters"]) is list and len(payload["parameters"]) == PARAMETER_COUNT,
                 "invalid_parameters", "artifact parameters must be a 293-element list")
        cell = cls(tuple(payload["parameters"]))
        _require(type(payload["parameter_hash"]) is str and payload["parameter_hash"] == cell.parameter_hash,
                 "hash_mismatch", "parameter hash differs from canonical literals")
        return cell


def zero_cell() -> BeliefCell:
    """Untrained zero control. Its output is uniform, including unknown."""
    return BeliefCell((0.0,) * PARAMETER_COUNT)


def random_cell(seed: int) -> BeliefCell:
    """Untrained initialization/control; no search, fitting or selection."""
    _integer(seed, 0, 2**32 - 1, "seed")
    rng = random.Random(seed)
    return BeliefCell(tuple(rng.uniform(-0.25, 0.25) for _ in range(PARAMETER_COUNT)))


@dataclass(frozen=True)
class BeliefState:
    """Immutable actor-major matrix. IDs are positional slots, not person traits."""

    actor_count: int
    object_count: int
    beliefs: tuple[tuple[Distribution, ...], ...]

    def __post_init__(self) -> None:
        _integer(self.actor_count, 1, MAX_ACTORS, "actor_count")
        _integer(self.object_count, 1, MAX_OBJECTS, "object_count")
        _require(type(self.beliefs) is tuple and len(self.beliefs) == self.actor_count,
                 "invalid_state", "one immutable row per actor required")
        for row in self.beliefs:
            _require(type(row) is tuple and len(row) == self.object_count
                     and all(type(value) is Distribution for value in row),
                     "invalid_state", "one typed distribution per object required")

    @classmethod
    def unknown(cls, actor_count: int, object_count: int) -> BeliefState:
        _integer(actor_count, 1, MAX_ACTORS, "actor_count")
        _integer(object_count, 1, MAX_OBJECTS, "object_count")
        return cls(actor_count, object_count, tuple(
            tuple(Distribution.at(UNKNOWN) for _ in range(object_count)) for _ in range(actor_count)))


@dataclass(frozen=True)
class WorldState:
    """Separate host truth. UNKNOWN means outside/unspecified in this tiny world."""

    object_locations: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(type(self.object_locations) is tuple and 1 <= len(self.object_locations) <= MAX_OBJECTS,
                 "invalid_world", "one to four object locations in a tuple required")
        for location in self.object_locations:
            _integer(location, 0, UNKNOWN, "world location")


@dataclass(frozen=True)
class MoveEvent:
    """Host-supplied move/set-location with an explicit observer set; no parser."""

    object_id: int
    destination: int
    observed_by: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.object_id, 0, MAX_OBJECTS - 1, "object_id")
        _integer(self.destination, 0, UNKNOWN, "destination")
        _require(type(self.observed_by) is tuple and len(self.observed_by) <= MAX_ACTORS,
                 "invalid_event", "bounded observer tuple required")
        for actor in self.observed_by:
            _integer(actor, 0, MAX_ACTORS - 1, "observer")
        _require(len(set(self.observed_by)) == len(self.observed_by), "invalid_event", "duplicate observer")


def _validate_event(state: BeliefState, event: MoveEvent) -> None:
    _require(type(state) is BeliefState and type(event) is MoveEvent,
             "invalid_input", "BeliefState and MoveEvent required")
    _require(event.object_id < state.object_count
             and all(actor < state.actor_count for actor in event.observed_by),
             "slot_out_of_bounds", "event references an absent actor/object slot")


def _slot_input(state: BeliefState, event: MoveEvent, actor: int, obj: int) -> KernelInput:
    observed = actor in event.observed_by
    return KernelInput(state.beliefs[actor][obj],
                       Distribution.at(event.destination if observed else UNKNOWN),
                       obj == event.object_id, observed)


def apply_event(cell: BeliefCell, state: BeliefState, event: MoveEvent) -> BeliefState:
    """Apply the same kernel to EVERY slot; do not repair its wrong predictions.

    No world-state argument exists. Even unrelated/unobserved slots are computed
    by the MLP, so initialized cells generally fail preservation invariants.
    """
    _require(type(cell) is BeliefCell, "invalid_input", "BeliefCell required")
    _validate_event(state, event)
    rows = tuple(tuple(cell.forward(_slot_input(state, event, actor, obj))
                       for obj in range(state.object_count)) for actor in range(state.actor_count))
    return BeliefState(state.actor_count, state.object_count, rows)


def advance_world(world: WorldState, event: MoveEvent) -> WorldState:
    """Host bookkeeping only; observers do not determine actual object motion."""
    _require(type(world) is WorldState and type(event) is MoveEvent,
             "invalid_input", "WorldState and MoveEvent required")
    _require(event.object_id < len(world.object_locations), "slot_out_of_bounds", "absent world object")
    return WorldState(tuple(event.destination if obj == event.object_id else old
                            for obj, old in enumerate(world.object_locations)))


def reference_update(inputs: KernelInput) -> Distribution:
    """Authored oracle target, NOT weights and never called by BeliefCell.forward."""
    _require(type(inputs) is KernelInput, "invalid_input", "KernelInput required")
    return inputs.observed_new if inputs.object_match and inputs.observation else inputs.old


def reference_step(state: BeliefState, event: MoveEvent) -> BeliefState:
    """Synthetic ground truth only; not a substitute for running the cell."""
    _validate_event(state, event)
    return BeliefState(state.actor_count, state.object_count, tuple(
        tuple(reference_update(_slot_input(state, event, actor, obj)) for obj in range(state.object_count))
        for actor in range(state.actor_count)))


@dataclass(frozen=True)
class Scenario:
    initial_world: WorldState
    initial_beliefs: BeliefState
    events: tuple[MoveEvent, ...]

    def __post_init__(self) -> None:
        _require(type(self.initial_world) is WorldState and type(self.initial_beliefs) is BeliefState,
                 "invalid_scenario", "typed initial world/beliefs required")
        _require(len(self.initial_world.object_locations) == self.initial_beliefs.object_count,
                 "invalid_scenario", "world and belief object counts differ")
        _require(type(self.events) is tuple and 1 <= len(self.events) <= MAX_EVENTS,
                 "invalid_scenario", "one to 64 structured events required")
        for event in self.events:
            _validate_event(self.initial_beliefs, event)


def generate_scenario(seed: int, actor_count: int = 4, object_count: int = 4,
                      event_count: int = 12) -> Scenario:
    """Original local synthetic events; no corpus, downloads or benchmark tasks.

    Six prefix events ensure observed/unseen changes, observed unknown and an
    unrelated object. Slot identities/destinations and the suffix vary by seed.
    Seeds are reproducibility controls, not evidence of independent holdouts.
    """
    _integer(seed, 0, 2**32 - 1, "seed")
    _integer(actor_count, 2, MAX_ACTORS, "generator actor_count")
    _integer(object_count, 2, MAX_OBJECTS, "generator object_count")
    _integer(event_count, 6, MAX_EVENTS, "generator event_count")
    rng = random.Random(seed)
    actors, objects = list(range(actor_count)), list(range(object_count))
    rng.shuffle(actors)
    rng.shuffle(objects)
    first = rng.randrange(4)
    second = (first + rng.randrange(1, 4)) % 4
    everyone = tuple(range(actor_count))
    events = [MoveEvent(objects[0], first, everyone),
              MoveEvent(objects[0], second, (actors[0],)),
              MoveEvent(objects[0], UNKNOWN, (actors[1],)),
              MoveEvent(objects[1], rng.randrange(4), everyone),
              MoveEvent(objects[0], first, everyone),
              MoveEvent(objects[0], second, ())]
    for _ in range(event_count - len(events)):
        events.append(MoveEvent(rng.randrange(object_count), rng.randrange(LOCATION_COUNT),
                                tuple(actor for actor in everyone if rng.randrange(2))))
    return Scenario(WorldState((UNKNOWN,) * object_count),
                    BeliefState.unknown(actor_count, object_count), tuple(events))


@dataclass(frozen=True)
class KernelExample:
    event_index: int
    actor_id: int
    object_id: int
    inputs: KernelInput
    expected: Distribution

    def __post_init__(self) -> None:
        _integer(self.event_index, 0, MAX_EVENTS - 1, "event_index")
        _integer(self.actor_id, 0, MAX_ACTORS - 1, "actor_id")
        _integer(self.object_id, 0, MAX_OBJECTS - 1, "object_id")
        _require(type(self.inputs) is KernelInput and type(self.expected) is Distribution,
                 "invalid_example", "typed input/expected distribution required")


def reference_examples(scenario: Scenario) -> tuple[KernelExample, ...]:
    """At most 1024 teacher-forced oracle examples, never an autonomous score."""
    _require(type(scenario) is Scenario, "invalid_scenario", "Scenario required")
    state, result = scenario.initial_beliefs, []
    for index, event in enumerate(scenario.events):
        for actor in range(state.actor_count):
            for obj in range(state.object_count):
                inputs = _slot_input(state, event, actor, obj)
                result.append(KernelExample(index, actor, obj, inputs, reference_update(inputs)))
        state = reference_step(state, event)
    return tuple(result)


def operational_checks() -> dict:
    """Small deterministic contract checks only; no fitting or capability score."""
    checked = []

    def check(name, condition):
        _require(condition, "check_failed", name)
        checked.append(name)

    def rejects(name, code, call):
        try:
            call()
        except BeliefError as error:
            check(name, error.code == code)
        else:
            raise BeliefError("check_failed", name + " was accepted")

    cell, scenario = zero_cell(), generate_scenario(7, 2, 2, 6)
    check("parameter_layout_293", sum(item[3] for item in _LAYOUT) == len(cell.parameters) == 293)
    check("literal_artifact_round_trip", BeliefCell.from_dict(cell.to_dict()) == cell)
    random = random_cell(11)
    check("seeded_initialization", random == random_cell(11) and random.parameter_hash != cell.parameter_hash)
    check("seeded_generator", scenario == generate_scenario(7, 2, 2, 6))
    examples = reference_examples(scenario)
    check("example_count_and_input_shape", len(examples) == 24
          and all(len(example.inputs.vector()) == 12 for example in examples))
    check("zero_output_uniform", cell.forward(examples[0].inputs) == Distribution((0.2,) * 5))
    check("random_output_finite_normalized", all(math.isfinite(value)
          for value in random.forward(examples[0].inputs).values))
    # Single-path literals check W1/b1/W2/b2 offsets and tanh+softmax direction.
    probe = [0.0] * PARAMETER_COUNT
    probe[0], probe[192], probe[208], probe[288] = 1.0, 0.5, 2.0, 0.25
    probe_input = KernelInput(Distribution.at(0), Distribution.at(UNKNOWN), True, False)
    logit = 2.0 * math.tanh(1.5) + 0.25
    check("forward_known_path", abs(BeliefCell(tuple(probe)).forward(probe_input).values[0]
                                   - 1.0 / (1.0 + 4.0 * math.exp(-logit))) < 1e-14)
    old, world = scenario.initial_beliefs, scenario.initial_world
    for index, event in enumerate(scenario.events):
        new, new_world = reference_step(old, event), advance_world(world, event)
        check(f"reference_event_{index}", all(
            new.beliefs[actor][obj] == (Distribution.at(event.destination)
                if actor in event.observed_by and obj == event.object_id else old.beliefs[actor][obj])
            for actor in range(old.actor_count) for obj in range(old.object_count)))
        check(f"world_event_{index}", all(new_world.object_locations[obj] == (
            event.destination if obj == event.object_id else world.object_locations[obj])
            for obj in range(old.object_count)))
        old, world = new, new_world
    obj = scenario.events[-1].object_id
    check("unseen_move_separates_world_and_belief", old.beliefs[0][obj] != Distribution.at(world.object_locations[obj]))
    check("untrained_kernel_not_oracle", apply_event(cell, scenario.initial_beliefs, scenario.events[0])
          != reference_step(scenario.initial_beliefs, scenario.events[0]))
    a, b = MoveEvent(0, 0, ()), MoveEvent(0, 3, ())
    check("unseen_destination_masked", all(
        _slot_input(scenario.initial_beliefs, a, actor, obj) == _slot_input(scenario.initial_beliefs, b, actor, obj)
        for actor in range(2) for obj in range(2)))
    mixture = Distribution((0.2, 0.3, 0.1, 0.4, 0.0))
    check("reference_preserves_mixture", reference_update(KernelInput(
        mixture, Distribution.at(1), False, True)) == mixture)
    rejects("reject_parameter_count", "invalid_parameters", lambda: BeliefCell((0.0,) * 292))
    rejects("reject_nonfinite", "invalid_number", lambda: BeliefCell((float("nan"),) + (0.0,) * 292))
    rejects("reject_bool_parameter", "invalid_number", lambda: BeliefCell((True,) + (0.0,) * 292))
    rejects("reject_distribution", "invalid_distribution", lambda: Distribution((0.0,) * 5))
    rejects("reject_bool_slot", "invalid_integer", lambda: MoveEvent(True, 1, ()))
    rejects("reject_duplicate_observer", "invalid_event", lambda: MoveEvent(0, 1, (0, 0)))
    rejects("reject_absent_slot", "slot_out_of_bounds", lambda: apply_event(
        cell, scenario.initial_beliefs, MoveEvent(3, 1, ())))
    rejects("reject_hidden_location", "hidden_observation", lambda: KernelInput(
        Distribution.at(0), Distribution.at(1), True, False))
    tampered = cell.to_dict()
    tampered["parameters"][0] = 1.0
    rejects("reject_hash_mismatch", "hash_mismatch", lambda: BeliefCell.from_dict(tampered))
    rejects("reject_numeric_overflow", "numeric_overflow", lambda: BeliefCell(
        (1e308,) * PARAMETER_COUNT).forward(examples[0].inputs))
    return {"scope": "deterministic operational checks only", "passed": len(checked),
            "checks": checked, "parameter_count": PARAMETER_COUNT, "zero_parameter_hash": cell.parameter_hash,
            "learned_or_generalization_claim": False, "model_calls": 0, "training_steps": 0}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run bounded deterministic operational checks")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(operational_checks(), indent=2, allow_nan=False))
    else:
        parser.print_help()
