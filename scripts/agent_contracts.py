"""Data contracts and host-owned admission checks; no agent or tool execution.

Only trusted host code may hold HostBoundary. Models/weights supply message
data, never Python objects, host method access, registry bindings or permissions.
This in-memory boundary is not a sandbox, scheduler or durable accounting log.
"""
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import ClassVar


class ContractError(ValueError):
    """A stable machine-readable code plus a human-readable explanation."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(code + ': ' + detail)


class CapabilityUnavailable(ContractError):
    def __init__(self, capability: str):
        super().__init__('capability_unavailable', capability)


class Capability(str, Enum):
    GRID = 'grid.pattern'
    LANGUAGE = 'language.text'
    STATE_UPDATE = 'language.state_update'
    MEMORY = 'memory.lookup'
    PLANNING = 'planning.decompose'


class Origin(str, Enum):
    HOST = 'host'
    MODEL = 'model'
    WEIGHTS = 'weights'
    ADAPTER = 'adapter'


class Status(str, Enum):
    COMPLETED = 'completed'
    UNAVAILABLE = 'unavailable'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


def _require(condition, code, detail):
    if not condition:
        raise ContractError(code, detail)


def _text(value, label, limit=256):
    _require(type(value) is str and 0 < len(value.encode('utf-8')) <= limit,
             'invalid_contract', label + ' must be nonempty bounded UTF-8 text')


def _integer(value, low, high, label):
    _require(type(value) is int and low <= value <= high, 'invalid_contract',
             '%s must be an integer in [%s, %s]' % (label, low, high))


def _names(values, label):
    _require(type(values) is tuple and len(values) <= 32, 'invalid_contract', label)
    for value in values:
        _text(value, label)
    _require(len(set(values)) == len(values), 'invalid_contract', label + ' repeats')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'invalid_contract', 'duplicate JSON key')
        result[key] = value
    return result


def _no_constant(value):
    raise ContractError('invalid_contract', 'nonfinite JSON: ' + value)


def _object(encoded):
    _text(encoded, 'JSON object', 65536)
    try:
        value = json.loads(encoded, object_pairs_hook=_unique, parse_constant=_no_constant)
        _require(type(value) is dict, 'invalid_contract', 'JSON object required')
        canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, RecursionError, OverflowError) as error:
        raise ContractError('invalid_contract', 'invalid finite JSON object') from error
    _require(len(canonical.encode('utf-8')) <= 65536, 'invalid_contract', 'JSON too large')
    return value, canonical


def _canonical(instance, attribute):
    value, encoded = _object(getattr(instance, attribute))
    object.__setattr__(instance, attribute, encoded)
    return value


@dataclass(frozen=True)
class Provenance:
    producer: str
    origin: Origin
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.producer, 'producer')
        _require(type(self.origin) is Origin, 'invalid_contract', 'explicit origin required')
        _names(self.evidence_refs, 'evidence references')


@dataclass(frozen=True)
class Task:
    schema: ClassVar[str] = 'cellforge-agent-task/1'
    task_id: str
    trace_id: str
    capability: Capability
    input_json: str
    provenance: Provenance

    def __post_init__(self):
        _text(self.task_id, 'task ID')
        _text(self.trace_id, 'trace ID')
        _require(type(self.capability) is Capability, 'invalid_contract', 'known capability required')
        _require(type(self.provenance) is Provenance, 'invalid_contract', 'provenance required')
        payload = _canonical(self, 'input_json')
        _require(payload.get('schema') == _schemas(self.capability)[0],
                 'invalid_contract', 'capability input schema mismatch')
        if self.capability is Capability.GRID:
            grid_bank_input(self)
        elif self.capability is Capability.LANGUAGE:
            _language_input(payload)
        elif self.capability is Capability.STATE_UPDATE:
            _state_input(payload)

    @property
    def input_hash(self):
        return hashlib.sha256(self.input_json.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Budget:
    """Integer consumables plus structural bounds and a host-clock deadline.

    Work units are host-defined accounting units, not currency or Codex credits.
    Every delegation/tool admission costs one unit, in addition to reservations.
    """
    work_units: int
    tool_requests: int
    delegations: int
    hops: int
    fanout: int
    deadline_ms: int

    def __post_init__(self):
        for key, maximum in (('work_units', 1000000), ('tool_requests', 10000),
                             ('delegations', 1024), ('hops', 32), ('fanout', 32),
                             ('deadline_ms', 2**63 - 1)):
            _integer(getattr(self, key), 0, maximum, key)


@dataclass(frozen=True)
class Delegation:
    schema: ClassVar[str] = 'cellforge-agent-delegation/1'
    parent_task_id: str
    task: Task
    specialist_id: str
    budget: Budget
    allowed_tools: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self):
        _text(self.parent_task_id, 'parent task ID')
        _text(self.specialist_id, 'specialist ID')
        _require(type(self.task) is Task and type(self.budget) is Budget
                 and type(self.provenance) is Provenance, 'invalid_contract', 'typed delegation required')
        _names(self.allowed_tools, 'allowed tools')


@dataclass(frozen=True)
class Result:
    schema: ClassVar[str] = 'cellforge-agent-result/1'
    task_id: str
    trace_id: str
    status: Status
    provenance: Provenance
    output_json: str | None = None
    error_code: str | None = None
    detail: str | None = None

    def __post_init__(self):
        _text(self.task_id, 'task ID')
        _text(self.trace_id, 'trace ID')
        _require(type(self.status) is Status and type(self.provenance) is Provenance,
                 'invalid_contract', 'typed status and provenance required')
        if self.status is Status.COMPLETED:
            _require(self.error_code is None and self.detail is None
                     and bool(self.provenance.evidence_refs), 'invalid_contract',
                     'completed result requires evidence references and no error')
            _canonical(self, 'output_json')
        else:
            _require(self.output_json is None, 'invalid_contract', 'errors cannot carry successful output')
            _text(self.error_code, 'error code')
            _text(self.detail, 'error detail', 4096)
            if self.status is Status.UNAVAILABLE:
                _require(self.error_code == 'capability_unavailable', 'invalid_contract',
                         'unavailable result requires capability_unavailable')


def _schemas(capability):
    stem = {Capability.GRID: 'grid', Capability.LANGUAGE: 'language',
            Capability.STATE_UPDATE: 'language-state',
            Capability.MEMORY: 'memory', Capability.PLANNING: 'planning'}[capability]
    return 'cellforge-' + stem + '-task/1', 'cellforge-' + stem + '-result/1'


@dataclass(frozen=True)
class Specialist:
    specialist_id: str
    capability: Capability
    adapter_ref: str | None = None

    def __post_init__(self):
        _text(self.specialist_id, 'specialist ID')
        _require(type(self.capability) is Capability, 'invalid_contract', 'known capability required')
        if self.adapter_ref is not None:
            _text(self.adapter_ref, 'host adapter reference')


@dataclass(frozen=True)
class CapabilityRegistry:
    """Explicit host bindings only. References are labels, never import paths."""
    specialists: tuple[Specialist, ...] = field(default_factory=lambda: tuple(
        Specialist(capability.value, capability) for capability in Capability))

    def __post_init__(self):
        _require(type(self.specialists) is tuple and len(self.specialists) == len(Capability)
                 and all(type(item) is Specialist for item in self.specialists),
                 'invalid_contract', 'one explicit entry per capability required')
        _require({item.capability for item in self.specialists} == set(Capability)
                 and len({item.specialist_id for item in self.specialists}) == len(Capability),
                 'invalid_contract', 'duplicate or missing specialist')

    def require(self, capability):
        _require(type(capability) is Capability, 'invalid_contract', 'known capability required')
        item = next(item for item in self.specialists if item.capability is capability)
        if item.adapter_ref is None:
            raise CapabilityUnavailable(capability.value)
        return item


@dataclass(frozen=True)
class ToolSchema:
    """Small exact schema: required flat arguments of type string/integer/boolean."""
    name: str
    version: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        _text(self.name, 'tool name')
        _text(self.version, 'tool schema version')
        _require(type(self.parameters) is tuple and len(self.parameters) <= 32,
                 'invalid_contract', 'bounded immutable parameters required')
        for pair in self.parameters:
            _require(type(pair) is tuple and len(pair) == 2, 'invalid_contract', 'parameter pair required')
            _text(pair[0], 'parameter name')
            _require(pair[1] in ('string', 'integer', 'boolean'), 'invalid_contract', 'unsupported argument type')
        _names(tuple(pair[0] for pair in self.parameters), 'parameter names')

    def validate(self, request):
        _require(request.tool == self.name and request.version == self.version,
                 'permission_denied', 'tool schema mismatch')
        values, _ = _object(request.arguments_json)
        _require(set(values) == {key for key, _ in self.parameters},
                 'invalid_contract', 'exact tool arguments required')
        for key, kind in self.parameters:
            value = values[key]
            if kind == 'string':
                _text(value, key, 4096)
            elif kind == 'integer':
                _integer(value, -(2**53 - 1), 2**53 - 1, key)
            else:
                _require(type(value) is bool, 'invalid_contract', key + ' must be boolean')


@dataclass(frozen=True)
class ToolRequest:
    schema: ClassVar[str] = 'cellforge-agent-tool-request/1'
    request_id: str
    task_id: str
    tool: str
    version: str
    arguments_json: str
    provenance: Provenance

    def __post_init__(self):
        for key in ('request_id', 'task_id', 'tool', 'version'):
            _text(getattr(self, key), key)
        _require(type(self.provenance) is Provenance, 'invalid_contract', 'provenance required')
        _canonical(self, 'arguments_json')


@dataclass(frozen=True)
class ToolAuthorization:
    """Host-issued one-use approval. Constructing an equal record grants nothing."""
    request: ToolRequest
    approval_ref: str
    expires_ms: int


@dataclass
class _Scope:
    task: Task
    budget: Budget
    allowed_tools: tuple[str, ...]
    parent: str | None = None
    remaining: list[int] = field(init=False)
    children: int = 0
    cancelled: str | None = None
    result: Result | None = None

    def __post_init__(self):
        self.remaining = [self.budget.work_units, self.budget.tool_requests, self.budget.delegations]


class HostBoundary:
    """Single-root, serialized host admission state. Never give this to workers.

    Host supplies monotonic milliseconds and authenticated task identity. Parent
    cancellation/termination blocks descendants. Reservations are never refunded.
    No method executes a specialist, imports an adapter, or performs a tool call.
    """

    def __init__(self, root: Task, budget: Budget, registry: CapabilityRegistry,
                 tools: tuple[ToolSchema, ...] = (), allowed_tools: tuple[str, ...] = (),
                 *, now_ms: int):
        _require(type(root) is Task and type(budget) is Budget
                 and type(registry) is CapabilityRegistry, 'invalid_contract', 'typed host configuration required')
        _integer(now_ms, 0, 2**63 - 1, 'host clock')
        _require(now_ms < budget.deadline_ms <= now_ms + 86400000,
                 'invalid_contract', 'root deadline must be within 24 hours')
        _require(type(tools) is tuple and len(tools) <= 32
                 and all(type(tool) is ToolSchema for tool in tools),
                 'invalid_contract', 'explicit immutable tool schemas required')
        _require(len({tool.name for tool in tools}) == len(tools), 'invalid_contract', 'duplicate tools')
        self._tools = {tool.name: tool for tool in tools}
        _names(allowed_tools, 'allowed tools')
        _require(set(allowed_tools) <= set(self._tools), 'permission_denied', 'tool outside host allowlist')
        self._registry = registry
        self._scopes = {root.task_id: _Scope(root, budget, allowed_tools)}
        self._approvals = {}
        self._used_requests = set()
        self._last_ms = now_ms

    def _active(self, task_id, now_ms):
        _integer(now_ms, self._last_ms, 2**63 - 1, 'monotonic host clock')
        self._last_ms = now_ms
        _require(task_id in self._scopes, 'invalid_contract', 'unknown task ID')
        scope = self._scopes[task_id]
        current = scope
        while current is not None:
            _require(current.cancelled is None, 'cancelled', current.cancelled or 'cancelled')
            _require(current.result is None, 'task_closed', 'task or ancestor already ended')
            _require(now_ms < current.budget.deadline_ms, 'deadline_exceeded', 'task or ancestor expired')
            current = self._scopes[current.parent] if current.parent is not None else None
        return scope

    @staticmethod
    def _reserve(scope, costs):
        _require(all(cost <= left for cost, left in zip(costs, scope.remaining)),
                 'budget_exhausted', 'aggregate reservations exceed remaining parent allowance')
        scope.remaining = [left - cost for cost, left in zip(costs, scope.remaining)]

    def charge(self, task_id: str, units: int, *, now_ms: int):
        """Host reserves measured/bounded local work BEFORE doing it."""
        _integer(units, 1, 1000000, 'work units')
        self._reserve(self._active(task_id, now_ms), (units, 0, 0))

    def delegate(self, parent_task_id: str, task: Task, budget: Budget,
                 allowed_tools: tuple[str, ...] = (), *, now_ms: int) -> Delegation:
        parent = self._active(parent_task_id, now_ms)
        _require(type(task) is Task and type(budget) is Budget, 'invalid_contract', 'typed child required')
        _require(task.task_id not in self._scopes and task.trace_id == parent.task.trace_id,
                 'invalid_contract', 'fresh child ID and same trace required')
        _names(allowed_tools, 'child allowed tools')
        _require(set(allowed_tools) <= set(parent.allowed_tools), 'permission_denied', 'child expands tool permissions')
        _require(parent.budget.hops > 0 and budget.hops < parent.budget.hops
                 and budget.fanout <= parent.budget.fanout
                 and parent.children < parent.budget.fanout,
                 'delegation_limit', 'hop/fanout limit exceeded')
        _require(now_ms < budget.deadline_ms <= parent.budget.deadline_ms,
                 'deadline_exceeded', 'child extends or has exhausted parent deadline')
        specialist = self._registry.require(task.capability)
        message = Delegation(parent_task_id, task, specialist.specialist_id, budget,
                             allowed_tools, Provenance('host-boundary', Origin.HOST, (parent_task_id,)))
        self._reserve(parent, (1 + budget.work_units, budget.tool_requests, 1 + budget.delegations))
        parent.children += 1
        self._scopes[task.task_id] = _Scope(task, budget, allowed_tools, parent_task_id)
        return message

    def cancel(self, task_id: str, reason: str):
        """Descendants observe this at every admission; siblings are unaffected."""
        _text(reason, 'cancellation reason', 4096)
        _require(task_id in self._scopes, 'invalid_contract', 'unknown task ID')
        self._scopes[task_id].cancelled = reason

    def _tool_scope(self, request, now_ms):
        _require(type(request) is ToolRequest, 'invalid_contract', 'typed tool request required')
        scope = self._active(request.task_id, now_ms)
        _require(request.tool in scope.allowed_tools, 'permission_denied', 'tool not allowed for task')
        self._tools[request.tool].validate(request)
        _require((request.task_id, request.request_id) not in self._used_requests,
                 'permission_denied', 'request already consumed')
        return scope

    def authorize_tool(self, request: ToolRequest, approval_ref: str,
                       expires_ms: int, *, now_ms: int) -> ToolAuthorization:
        """HOST ONLY: issue approval after resolving the user's actual permission.

        approval_ref is an audit reference, never proof extracted from model text.
        Issuance is itself the trusted decision; no message field can invoke it.
        """
        scope = self._tool_scope(request, now_ms)
        _text(approval_ref, 'host approval reference')
        _integer(expires_ms, now_ms + 1, scope.budget.deadline_ms, 'authorization expiry')
        key = (request.task_id, request.request_id)
        _require(key not in self._approvals, 'permission_denied', 'request already has an approval')
        pending = sum(task_id == request.task_id for task_id, _ in self._approvals)
        _require(pending < min(scope.remaining[:2]), 'budget_exhausted', 'no room for another pending approval')
        approval = ToolAuthorization(request, approval_ref, expires_ms)
        self._approvals[key] = approval
        return approval

    def reserve_tool(self, request: ToolRequest, approval: ToolAuthorization,
                     *, now_ms: int) -> ToolRequest:
        """Consume exact approval and quota; return DATA only, not an execution receipt."""
        scope = self._tool_scope(request, now_ms)
        key = (request.task_id, request.request_id)
        _require(type(approval) is ToolAuthorization and self._approvals.get(key) is approval
                 and approval.request == request and now_ms < approval.expires_ms,
                 'permission_denied', 'missing, forged, changed or expired authorization')
        self._reserve(scope, (1, 1, 0))
        self._used_requests.add(key)
        del self._approvals[key]
        return request

    def finish(self, result: Result, *, now_ms: int):
        """Close a task with host-recorded outcome; this does not verify competence."""
        _require(type(result) is Result, 'invalid_contract', 'typed result required')
        scope = self._active(result.task_id, now_ms)
        _require(result.trace_id == scope.task.trace_id, 'invalid_contract', 'result trace mismatch')
        if result.status is Status.COMPLETED:
            self._registry.require(scope.task.capability)
            value, _ = _object(result.output_json)
            _require(value.get('schema') == _schemas(scope.task.capability)[1],
                     'invalid_contract', 'capability output schema mismatch')
            _validate_output(scope.task, value)
        scope.result = result


def _grid(value):
    _require(type(value) is list and 1 <= len(value) <= 30,
             'invalid_contract', 'grid requires 1..30 rows')
    width = len(value[0]) if type(value[0]) is list else 0
    _require(1 <= width <= 30 and all(type(row) is list and len(row) == width
             and all(type(pixel) is int and 0 <= pixel <= 9 for pixel in row) for row in value),
             'invalid_contract', 'rectangular 1..30 column ten-color integer grid required')


def grid_bank_input(task: Task) -> dict:
    """Pure seam to skillcore.predict_bank: fresh train/test data, no test outputs.

    The host must separately supply a verified cellforge-skill-bank/1 artifact
    and any evaluator tensor override. Nothing is loaded or executed here.
    """
    _require(type(task) is Task and task.capability is Capability.GRID,
             'invalid_contract', 'grid task required')
    value, _ = _object(task.input_json)
    _require(set(value) == {'schema', 'train', 'test'}
             and value['schema'] == _schemas(Capability.GRID)[0],
             'invalid_contract', 'exact grid task fields required')
    for key, fields in (('train', {'input', 'output'}), ('test', {'input'})):
        examples = value[key]
        _require(type(examples) is list and 1 <= len(examples) <= 16,
                 'invalid_contract', 'one to sixteen ' + key + ' examples required')
        for example in examples:
            _require(type(example) is dict and set(example) == fields,
                     'invalid_contract', 'unexpected grid example fields (test outputs forbidden)')
            for grid in example.values():
                _grid(grid)
    return {'train': value['train'], 'test': value['test']}


def _language_input(value):
    _require(set(value) == {'schema', 'operation', 'text', 'context', 'max_output_bytes'},
             'invalid_contract', 'exact language task fields required')
    _require(value['operation'] in ('complete', 'classify', 'extract', 'transform', 'respond'),
             'invalid_contract', 'unknown language operation')
    _text(value['text'], 'language text', 32768)
    _require(type(value['context']) is list and len(value['context']) <= 16,
             'invalid_contract', 'bounded language context required')
    for example in value['context']:
        _require(type(example) is dict and set(example) == {'input', 'output'},
                 'invalid_contract', 'language context pairs required')
        _text(example['input'], 'context input', 4096)
        _text(example['output'], 'context output', 4096)
    _integer(value['max_output_bytes'], 1, 32768, 'maximum language output bytes')


def _state_maps(value, actors, objects, locations):
    def positions(mapping):
        _require(type(mapping) is dict and set(mapping) == set(objects)
                 and all(position is None or (type(position) is str and position in locations)
                         for position in mapping.values()),
                 'invalid_contract', 'object locations must be declared IDs or null (unknown)')
    positions(value['real_state'])
    beliefs = value['actor_beliefs']
    _require(type(beliefs) is dict and set(beliefs) == set(actors),
             'invalid_contract', 'one belief map per declared actor required')
    for belief in beliefs.values():
        positions(belief)


def _state_input(value):
    _require(set(value) == {'schema', 'actors', 'objects', 'locations', 'real_state',
                            'actor_beliefs', 'event', 'observation_mask'},
             'invalid_contract', 'exact structured state task fields required')
    for key in ('actors', 'objects', 'locations'):
        _require(type(value[key]) is list and 1 <= len(value[key]) <= 16,
                 'invalid_contract', 'one to sixteen declared ' + key + ' required')
        _names(tuple(value[key]), key)
    _state_maps(value, value['actors'], value['objects'], value['locations'])
    event = value['event']
    _require(type(event) is dict and set(event) == {'actor', 'object', 'location'},
             'invalid_contract', 'explicit move-event fields required')
    for key, domain in (('actor', 'actors'), ('object', 'objects'), ('location', 'locations')):
        _require(event[key] is None or (type(event[key]) is str and event[key] in value[domain]),
                 'invalid_contract', 'event IDs must be declared or null (unknown)')
    mask = value['observation_mask']
    _require(type(mask) is dict and set(mask) == set(value['actors'])
             and all(item is None or type(item) is bool for item in mask.values()),
             'invalid_contract', 'observation mask requires true/false/null for every actor')


def _validate_output(task, value):
    source, _ = _object(task.input_json)
    if task.capability is Capability.GRID:
        _require(set(value) == {'schema', 'predictions'} and type(value['predictions']) is list
                 and len(value['predictions']) == len(source['test']),
                 'invalid_contract', 'one grid prediction per test input required')
        for prediction in value['predictions']:
            _grid(prediction)
    elif task.capability is Capability.LANGUAGE:
        _require(set(value) == {'schema', 'text'}, 'invalid_contract', 'exact language output fields required')
        _text(value['text'], 'language output', source['max_output_bytes'])
    elif task.capability is Capability.STATE_UPDATE:
        _require(set(value) == {'schema', 'real_state', 'actor_beliefs'},
                 'invalid_contract', 'exact structured state output fields required')
        _state_maps(value, source['actors'], source['objects'], source['locations'])
    else:
        # A host binding must add operation semantics before these placeholders
        # can accept successful output. Merely naming an adapter is insufficient.
        raise CapabilityUnavailable(task.capability.value + ' result contract not implemented')
