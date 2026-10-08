"""Explicit local grid/belief adapters, not a learned coordinator or tool runner.

Use with scripts on PYTHONPATH, as for skillcore. Host code constructs Binding
from already-read artifact bytes, an expected SHA-256, source pins and provenance;
LocalDispatcher(root, budget, bindings, clock_ms=...) owns the task tree. Call
delegate(parent_id, task, budget), dispatch(task_id), cancel(task_id, reason),
or receipt(task_id). Routing is only the Task.capability enum. No CLI, artifact
discovery, external tools, model calls, file writes or automatic retries exist.

All calls are serialized by the trusted host. Deadlines/cancellation are checked
before and after synchronous computation, not hard process termination. Source
pins assume a fresh, trusted Python process without monkeypatching. Receipts are
in-memory evidence, not durable journals or scientific qualification. A host
must retain bindings, inputs and receipts itself. Reconstructing a dispatcher
does not resume its budget. Work units are accounting units, not Codex credits.

Grid admission supports only tile_orient, assoc_induct, glyph_recolor and ray_cast.
Before predict_bank, a shape-only walk covers every expert's leave-one-out
demonstrations and every possible routed query, including recursively staged
demonstration inputs AND outputs. Tile dimensions conservatively expand by four
on each axis regardless of saved/override tensors; the others preserve shape.
Every intermediate must fit 30x30, with at most 4096 leaf calls and 1,000,000
aggregate input/output/context grid-cell visits. Each h*w ray_cast call also
charges 8*h*w ray launches and 8*h*w*(max(h,w)-1) in-bounds ray path steps,
assuming every pixel launches every direction without early stopping. Their
sum with grid visits is capped at 1,000,000 and reserved as work units, not FLOPs
or a hard CPU limit. Unsupported shape effects fail closed.
This intentionally rejects some safe particular weights to bound all controls
without executing numerical templates during admission.

The first belief dispatch accepts only declared location IDs or null (the
UNKNOWN category), never user-supplied mixed probability vectors/maps. Task
validation rejects those values before dispatch; no argmax/coercion is applied.
Exact neural distributions are canonical recurrent storage: pass
belief_state_from=prior_task_id to the next dispatch. Its entity lists, world and
presentation input must match that prior result; the prior distributions, not
one-hot projections, feed the MLP. belief_state(task_id) exposes the immutable
state. Without belief_state_from a dispatch is a new categorical initialization.
Completed Result locations are presentation only: maximum probability >=0.95
selects a category, otherwise null. This is a probability threshold, not calibrated
confidence. Receipts distinguish unknown-category from below-threshold projection.
Unknown event/visibility metadata is rejected, never mapped to the UNKNOWN
location category. Real state advances separately by host bookkeeping. Event
actor is metadata; visibility comes only from the explicit observation mask.

Preserving those raw values between calls does NOT mean the cell preserves a
distribution when it computes its next output. The analytical reference is
near-categorical memory, not general probability memory. Its frozen-weight
post-selection stress report records 3072/3072 categorical slot predictions,
but 1168/1536 distributed-input probability errors >0.05 (maximum error 0.79).
See runs/weights-belief-analytical-reference-20261006/post-selection-stress/
STRESS_REPORT.md; this adapter did not rerun that stress or qualify generalization.
Recurrent outputs are retained verbatim even if they drift; they are never
repaired with one-hot projections or the reference updater. New categorical
initialization must not be used to claim a continuous-distribution rollout.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import agent_contracts as contracts
import belief_cell


_OPERATIONS = {contracts.Capability.GRID: 'grid.predict',
               contracts.Capability.STATE_UPDATE: 'belief.update'}
_COMMON_SOURCES = {'agent_contracts.py', 'specialist_runtime.py'}
_SOURCES = {'grid.predict': _COMMON_SOURCES | {'skillcore.py', 'cell_contracts.py'},
            'belief.update': _COMMON_SOURCES | {'belief_cell.py'}}


def _require(condition, code, detail):
    if not condition:
        raise contracts.ContractError(code, detail)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'invalid_artifact', 'duplicate JSON field')
        result[key] = value
    return result


def _json(raw):
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique)
        _require(type(value) is dict, 'invalid_artifact', 'artifact must be an object')
        _encoded(value)  # Reject NaN/Infinity, including overflowed exponents.
        return value
    except (ValueError, UnicodeError, RecursionError, OverflowError) as error:
        raise contracts.ContractError('invalid_artifact', 'invalid finite artifact JSON') from error


def _monotonic_ms():
    return time.monotonic_ns() // 1000000


@dataclass(frozen=True)
class Binding:
    """Host-only, content-bound artifact; source names are an exact fixed allowlist."""
    operation: str
    artifact_bytes: bytes
    artifact_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    provenance: contracts.Provenance

    def __post_init__(self):
        _require(type(self.operation) is str and self.operation in _SOURCES,
                 'capability_unavailable', 'only grid.predict and belief.update are implemented')
        _require(type(self.artifact_bytes) is bytes and 0 < len(self.artifact_bytes) <= 8 * 1024 * 1024,
                 'invalid_artifact', 'bounded immutable artifact bytes required')
        _require(self.artifact_sha256 == _sha(self.artifact_bytes),
                 'hash_mismatch', 'artifact bytes differ from host pin')
        _require(type(self.source_hashes) is tuple and all(type(pair) is tuple and len(pair) == 2
                 and type(pair[0]) is str and type(pair[1]) is str for pair in self.source_hashes),
                 'invalid_artifact', 'immutable source/hash pairs required')
        _require(len(self.source_hashes) == len(_SOURCES[self.operation])
                 and {name for name, _ in self.source_hashes} == _SOURCES[self.operation],
                 'invalid_artifact', 'exact adapter dependency source pins required')
        _require(type(self.provenance) is contracts.Provenance and bool(self.provenance.evidence_refs),
                 'invalid_artifact', 'explicit artifact origin and evidence references required')
        _json(self.artifact_bytes)


def _verify_sources(binding):
    root = Path(__file__).resolve().parent
    for name, expected in binding.source_hashes:
        _require(_sha((root / name).read_bytes()) == expected,
                 'source_changed', 'source pin mismatch: ' + name)


def _grid_nodes(artifact):
    _require(artifact.get('schema') == 'cellforge-skill-bank/1'
             and type(artifact.get('experts')) is list and 1 <= len(artifact['experts']) <= 8,
             'unsupported_input', 'one to eight typed grid experts required')
    def count(ref, depth):
        _require(type(ref) is dict and depth <= 2, 'unsupported_input', 'grid chain depth exceeds two')
        if ref.get('kind') == 'cell':
            _require(ref.get('template') in ('tile_orient', 'assoc_induct', 'glyph_recolor', 'ray_cast'),
                     'unsupported_input', 'grid template has no admitted shape bound')
            return 1
        _require(ref.get('kind') == 'chain' and type(ref.get('parts')) is list
                 and len(ref['parts']) == 2, 'invalid_artifact', 'invalid grid expert topology')
        return 1 + sum(count(part, depth + 1) for part in ref['parts'])
    return sum(count(ref, 0) for ref in artifact['experts'])


def _grid_preflight(artifact, train_shapes, test_shapes):
    """Pure shape propagation, including skillcore.run's recursive demo staging.

    No grids or tensors are allocated/executed here. Worst-case template bounds
    cover every legal saved/override value, so controls cannot enlarge admission.
    Query work is overcounted for all experts since the router choice is unknown.
    """
    nodes = _grid_nodes(artifact)
    calls = visits = ray_launches = ray_steps = 0

    def bounded(shape):
        height, width = shape
        _require(1 <= height <= 30 and 1 <= width <= 30, 'resource_limit',
                 'possible grid intermediate %sx%s exceeds 30x30' % shape)
        return height * width

    def run_shape(ref, shape, demos):
        nonlocal calls, visits, ray_launches, ray_steps
        if ref['kind'] == 'chain':
            first, second = ref['parts']
            middle = run_shape(first, shape, demos)
            staged = []
            for i, (before, after) in enumerate(demos):
                other = demos[:i] + demos[i + 1:]
                staged.append((run_shape(first, before, other), run_shape(first, after, other)))
            return run_shape(second, middle, staged)
        output = (shape[0] * 4, shape[1] * 4) if ref['template'] == 'tile_orient' else shape
        cells = bounded(shape) + bounded(output)
        cells += sum(bounded(before) + bounded(after) for before, after in demos)
        calls += 1
        visits += cells
        if ref['template'] == 'ray_cast':
            # Eight fixed, nonzero unit directions; every while iteration moves
            # closer to an edge. Include launches even for a 1x1 grid (zero
            # in-bounds steps). Ignore source masks/stops for every tensor arm.
            launches = 8 * shape[0] * shape[1]
            ray_launches += launches
            ray_steps += launches * (max(shape) - 1)
        _require(calls <= 4096 and visits <= 1000000
                 and visits + ray_launches + ray_steps <= 1000000, 'resource_limit',
                 'possible grid execution exceeds leaf-call, grid-cell or total work cap')
        return output

    for expert in artifact['experts']:
        for i, (before, after) in enumerate(train_shapes):
            bounded(after)  # Scorer compares this target even when output shapes differ.
            visits += bounded(after)
            run_shape(expert, before, train_shapes[:i] + train_shapes[i + 1:])
        for query in test_shapes:
            run_shape(expert, query, train_shapes)
    return {'policy': 'worst-case-template-shapes/3', 'nodes': nodes,
            'leaf_calls_bound': calls, 'grid_cell_visits_bound': visits,
            'ray_launches_bound': ray_launches, 'ray_path_steps_bound': ray_steps,
            'work_units_bound': visits + ray_launches + ray_steps,
            'max_intermediate_shape': [30, 30]}


class LocalDispatcher:
    """Host-only fixed routing. ToolRequest records are never accepted/executed here."""

    def __init__(self, root: contracts.Task, budget: contracts.Budget,
                 bindings: tuple[Binding, ...] = (), *,
                 clock_ms: Callable[[], int] = _monotonic_ms):
        _require(type(bindings) is tuple and len(bindings) <= 2
                 and all(type(binding) is Binding for binding in bindings),
                 'invalid_artifact', 'explicit local bindings required')
        _require(len({binding.operation for binding in bindings}) == len(bindings),
                 'invalid_artifact', 'duplicate adapter binding')
        self._clock = clock_ms
        self._bindings = {binding.operation: binding for binding in bindings}
        self._artifacts, self._receipts, self._belief_states, self._outputs = {}, {}, {}, {}
        self._grid = None
        for binding in bindings:
            _verify_sources(binding)
            artifact = _json(binding.artifact_bytes)
            modules = [contracts]
            if binding.operation == 'belief.update':
                belief_cell.BeliefCell.from_dict(artifact)
                modules.append(belief_cell)
            else:
                _grid_nodes(artifact)
                # Explicit trusted local modules only; artifact text cannot name imports.
                try:
                    import skillcore
                    import cell_contracts
                except ImportError as error:
                    raise contracts.CapabilityUnavailable('grid.predict dependencies unavailable') from error
                self._grid = skillcore
                modules.extend((skillcore, cell_contracts))
            for module in modules:
                _require(Path(module.__file__).resolve() == Path(__file__).resolve().with_name(module.__name__ + '.py'),
                         'source_changed', 'adapter dependency resolved outside this scripts directory')
            _verify_sources(binding)
            self._artifacts[binding.operation] = artifact
        registry = contracts.CapabilityRegistry(tuple(contracts.Specialist(
            _OPERATIONS.get(capability, capability.value), capability,
            _OPERATIONS[capability] if _OPERATIONS.get(capability) in self._bindings else None)
            for capability in contracts.Capability))
        self._boundary = contracts.HostBoundary(root, budget, registry, now_ms=self._clock())
        self._tasks = {root.task_id: root}

    def delegate(self, parent_task_id: str, task: contracts.Task, budget: contracts.Budget):
        message = self._boundary.delegate(parent_task_id, task, budget, now_ms=self._clock())
        self._tasks[task.task_id] = task
        return message

    def cancel(self, task_id: str, reason: str):
        self._boundary.cancel(task_id, reason)

    def receipt(self, task_id: str) -> dict:
        _require(task_id in self._receipts, 'invalid_contract', 'no dispatch receipt for task')
        return deepcopy(self._receipts[task_id])

    def belief_state(self, task_id: str) -> belief_cell.BeliefState:
        """Exact stored neural output, not a claim of learned probability preservation."""
        _require(task_id in self._belief_states, 'invalid_contract', 'no completed recurrent belief state')
        return self._belief_states[task_id]

    def _result(self, task, evidence, status, output=None, error=None):
        evidence.update(status=status.value, error_code=error.code if error else None,
                        detail=str(error)[:1000] if error else None)
        evidence['receipt_hash'] = _sha(_encoded(evidence))
        self._receipts[task.task_id] = deepcopy(evidence)
        provenance = contracts.Provenance('specialist_runtime', contracts.Origin.ADAPTER,
            ('input-sha256:' + task.input_hash, 'receipt-sha256:' + evidence['receipt_hash']))
        return contracts.Result(task.task_id, task.trace_id, status, provenance,
            _encoded(output).decode('utf-8') if output is not None else None,
            error.code if error else None, str(error)[:1000] if error else None)

    def dispatch(self, task_id: str, *, tensor_override: dict | None = None,
                 belief_state_from: str | None = None) -> contracts.Result:
        """Execute once; failed/late outputs are errors. Override is a host evaluator input."""
        _require(task_id in self._tasks, 'invalid_contract', 'unknown task ID')
        _require(task_id not in self._receipts, 'task_closed', 'dispatch already attempted; no implicit retry')
        task = self._tasks[task_id]
        operation = _OPERATIONS.get(task.capability)
        evidence = {'schema': 'cellforge-local-dispatch/1', 'task_id': task_id,
                    'trace_id': task.trace_id, 'input_hash': task.input_hash,
                    'operation': operation, 'adapter_called': False, 'charged_work_units': 0}
        try:
            self._boundary.charge(task_id, 1, now_ms=self._clock())
            evidence['charged_work_units'] = 1
            if operation not in self._bindings:
                raise contracts.CapabilityUnavailable(task.capability.value)
            binding, artifact = self._bindings[operation], self._artifacts[operation]
            _verify_sources(binding)
            evidence.update(artifact_sha256=binding.artifact_sha256,
                            source_hashes=dict(binding.source_hashes),
                            artifact_provenance={'producer': binding.provenance.producer,
                                'origin': binding.provenance.origin.value,
                                'evidence_refs': list(binding.provenance.evidence_refs)})
            value = json.loads(task.input_json)
            prior_state = None
            if operation == 'grid.predict':
                _require(belief_state_from is None, 'unsupported_input', 'grid adapter cannot consume belief state')
                shape = lambda grid: (len(grid), len(grid[0]))
                evidence['grid_preflight'] = _grid_preflight(artifact,
                    [(shape(row['input']), shape(row['output'])) for row in value['train']],
                    [shape(row['input']) for row in value['test']])
                units = evidence['grid_preflight']['work_units_bound']
            else:
                _require(tensor_override is None, 'unsupported_input', 'belief controls require a separate explicit binding')
                units = self._belief_size(value)
                if belief_state_from is not None:
                    prior_state = self.belief_state(belief_state_from)
                    prior_task = self._tasks[belief_state_from]
                    prior_input = json.loads(prior_task.input_json)
                    prior_output = self._outputs[belief_state_from]
                    _require(prior_task.trace_id == task.trace_id
                             and all(value[key] == prior_input[key] for key in ('actors', 'objects', 'locations'))
                             and all(value[key] == prior_output[key] for key in ('real_state', 'actor_beliefs')),
                             'unsupported_input', 'recurrent state trace/slots/world/presentation must match prior result')
                    evidence['prior_receipt_sha256'] = self._receipts[belief_state_from]['receipt_hash']
                evidence['state_initialization'] = 'prior-neural-distributions' if prior_state is not None else 'explicit-categorical-input'
                evidence['belief_state_from'] = belief_state_from
            self._boundary.charge(task_id, units, now_ms=self._clock())
            evidence['charged_work_units'] += units
            if operation == 'grid.predict':
                override = deepcopy(tensor_override)
                evidence['tensor_override_sha256'] = _sha(_encoded(override)) if override is not None else None
                evidence['artifact_hash'] = artifact.get('artifact_hash')
                evidence['numpy_version'] = self._grid.np.__version__
                evidence['adapter_called'] = True
                predictions = self._grid.predict_bank(deepcopy(artifact), contracts.grid_bank_input(task), tensors=override)
                _require(type(predictions) is list and len(predictions) == len(value['test'])
                         and all(type(row) is list and len(row) == 1 for row in predictions),
                         'invalid_output', 'grid bank returned an unexpected candidate shape')
                output = {'schema': 'cellforge-grid-result/1', 'predictions': [row[0] for row in predictions]}
            else:
                evidence['adapter_called'] = True
                output, updated_state = self._belief_update(value, artifact, evidence, prior_state)
            evidence['output_sha256'] = _sha(_encoded(output))
            result = self._result(task, evidence, contracts.Status.COMPLETED, output=output)
            self._boundary.finish(result, now_ms=self._clock())
            self._outputs[task_id] = deepcopy(output)
            if operation == 'belief.update':
                self._belief_states[task_id] = updated_state
            return result
        except (contracts.ContractError, belief_cell.BeliefError, ValueError, TypeError,
                KeyError, IndexError, OverflowError, OSError, RecursionError) as error:
            if not isinstance(error, contracts.ContractError):
                error = contracts.ContractError(getattr(error, 'code', 'adapter_failed'), str(error)[:1000])
            status = (contracts.Status.UNAVAILABLE if error.code == 'capability_unavailable' else
                      contracts.Status.CANCELLED if error.code == 'cancelled' else contracts.Status.FAILED)
            evidence.pop('receipt_hash', None)
            result = self._result(task, evidence, status, error=error)
            try:
                self._boundary.finish(result, now_ms=self._clock())
            except contracts.ContractError:
                pass  # A cancelled/expired ancestor rejects late results; receipt retains failure.
            return result

    @staticmethod
    def _belief_size(value):
        _require(all(len(value[key]) <= 4 for key in ('actors', 'objects', 'locations')),
                 'unsupported_input', 'belief cell supports at most four actors/objects/known locations')
        _require(all(item is not None for item in value['event'].values())
                 and all(type(item) is bool for item in value['observation_mask'].values()),
                 'unsupported_input', 'uncertain event or observation requires an explicit encoder; no guessing')
        return len(value['actors']) * len(value['objects'])

    @staticmethod
    def _belief_update(value, artifact, evidence, prior_state):
        actors, objects, locations = value['actors'], value['objects'], value['locations']
        def encode(location):
            return belief_cell.UNKNOWN if location is None else locations.index(location)
        state = prior_state if prior_state is not None else belief_cell.BeliefState(len(actors), len(objects), tuple(tuple(
            belief_cell.Distribution.at(encode(value['actor_beliefs'][actor][obj]))
            for obj in objects) for actor in actors))
        world = belief_cell.WorldState(tuple(encode(value['real_state'][obj]) for obj in objects))
        event = belief_cell.MoveEvent(objects.index(value['event']['object']),
            encode(value['event']['location']), tuple(i for i, actor in enumerate(actors)
                if value['observation_mask'][actor]))
        cell = belief_cell.BeliefCell.from_dict(artifact)
        updated = belief_cell.apply_event(cell, state, event)
        real = belief_cell.advance_world(world, event)
        evidence.update(parameter_hash=cell.parameter_hash, parameter_count=belief_cell.PARAMETER_COUNT,
                        decoding='presentation-only max-probability>=0.95; not calibrated confidence; no neural-state repair',
                        actor_role='event actor is metadata; explicit observation mask controls visibility',
                        input_distributions={actor: {obj: list(state.beliefs[i][j].values)
                            for j, obj in enumerate(objects)} for i, actor in enumerate(actors)},
                        distributions={actor: {obj: list(updated.beliefs[i][j].values)
                            for j, obj in enumerate(objects)} for i, actor in enumerate(actors)})
        projection_reasons = {}
        def decode(distribution, actor, obj):
            maximum = max(distribution.values)
            if maximum < 0.95:
                projection_reasons[actor][obj] = 'below-probability-threshold'
                return None
            index = distribution.values.index(maximum)
            _require(index == belief_cell.UNKNOWN or index < len(locations),
                     'invalid_output', 'cell predicted an undeclared location slot above threshold')
            projection_reasons[actor][obj] = 'unknown-category' if index == belief_cell.UNKNOWN else 'location-category'
            return None if index == belief_cell.UNKNOWN else locations[index]
        for actor in actors:
            projection_reasons[actor] = {}
        output = {'schema': 'cellforge-language-state-result/1',
                'real_state': {obj: None if real.object_locations[j] == belief_cell.UNKNOWN
                               else locations[real.object_locations[j]] for j, obj in enumerate(objects)},
                'actor_beliefs': {actor: {obj: decode(updated.beliefs[i][j], actor, obj)
                    for j, obj in enumerate(objects)} for i, actor in enumerate(actors)}}
        evidence['projection_reasons'] = projection_reasons
        return output, updated
