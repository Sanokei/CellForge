# Coordinator, specialists and tool boundaries

This is a small foundational contract, not coordinator intelligence or a claim
of AGI. The intended architecture is a coordinator that delegates to specialist
agents, which propose tool requests. Grid-pattern cells are one subsystem.
Language would require many task cells, explicit encoders and composition;
neither toy token niches nor this interface establish language understanding.

Only `scripts/agent_contracts.py` and this document are added. Nothing connects
them to the live driver, maker, failure journal or worker pool. No model route,
tool executor, discovery, credentials, background process or campaign is added.

## Implemented surface

The stdlib-only module provides frozen typed `Task`, `Delegation`, `Result`,
`Provenance`, `Budget`, `Specialist`, `ToolSchema`, `ToolRequest` and
`ToolAuthorization` records. Task, delegation, result and request have explicit
version-1 schema names. JSON payloads become canonical immutable strings;
duplicate keys, nonfinite numbers, oversized objects and schema mismatches
are rejected. Integer bounds reject booleans. `Task.input_hash` hashes the
canonical input bytes. Records are in-process types, not a wire decoder;
future decoding must validate exact envelope fields and schema versions.

Provenance distinguishes host, model, weights and adapter producers and retains
explicit evidence references. Delegations carry parent task IDs and share a
trace ID. Completed results require output and evidence references; failures
carry a code and detail without successful output. These references and hashes
are audit data, not signatures, verified execution receipts or permissions.
The module never generates a successful agent outcome. The host must retain
and verify the actual source, input, tensors, execution and outcome evidence.

`CapabilityRegistry()` explicitly lists these specialists, all unavailable:

| Capability | Input contract | Current boundary |
| --- | --- | --- |
| `grid.pattern` | `cellforge-grid-task/1` | Validated input and output; pure bank-input converter |
| `language.text` | `cellforge-language-task/1` | Text contract only; no encoder or language implementation |
| `language.state_update` | `cellforge-language-state-task/1` | Structured event/belief contract only; no neural cell |
| `memory.lookup` | `cellforge-memory-task/1` | Reserved schema name; operation/result semantics unavailable |
| `planning.decompose` | `cellforge-planning-task/1` | Reserved schema name; operation/result semantics unavailable |

A trusted host can construct an immutable registry with an explicit adapter
reference. References are labels: the module never imports, discovers or calls
them. `require()` raises `CapabilityUnavailable` with code
`capability_unavailable` when no binding exists. No grid fallback or fabricated
language answer substitutes for a missing specialist. Memory/planning successful
results remain unavailable even if a host supplies a binding label, until their
operation/result contracts are implemented. The root can represent a coordinator
request without having an implemented specialist; successful completion still
requires a bound capability and supported result contract.

## Host-owned authority and finite work

`HostBoundary(root, budget, registry, tools=(), allowed_tools=(), now_ms=...)`
owns one in-memory task tree. Only trusted host code may hold this object or
invoke its mutation methods. Workers receive serialized data, never this Python
object. A model's text, tensors, provenance fields, approval strings or proposed
delegations cannot install adapters, grant permissions, reset limits or select
another task's identity. The host derives the calling task from its authenticated
worker context rather than trusting a `task_id` from worker output.

Budgets contain integer `work_units`, `tool_requests` and `delegations`, plus
`hops`, lifetime direct-child `fanout` and an absolute `deadline_ms` in the host's
monotonic clock domain. Root lifetime is at most 24 hours. Limits cap work units
at 1,000,000, tool admissions at 10,000, descendants at 1,024, and depth/fanout at
32. These are admission ceilings, not performance claims or spending authority.
Work units are not Codex credits, tokens or money. Future paid/model work needs
separate authorized accounting and an executor; this module supplies neither.

`delegate()` checks availability, same trace, fresh child identity, permission
subset, deadline, strictly smaller hop allowance and nonincreasing fanout. It
atomically reserves the child's entire consumable allocation plus one parent
work unit and one delegation. Sibling reservations therefore cannot each spend
the parent's whole allowance. Unused allocations are never refunded, including
after failure or cancellation; a retry needs a new identity and fresh allowance.
One boundary belongs to one root authorization; rebuilding it is not resume.

`charge()` reserves local work units before the host performs bounded work.
`cancel()` marks a task; every descendant checks ancestors on subsequent
admission. `finish()` closes a task and prevents further descendant work or
late successful results. Deadlines and monotonic time are checked on every
charge, delegation, authorization, tool reservation and result admission.
Cancellation remains recorded in the boundary; a result submitted after
cancellation is rejected. The host logs that cancellation separately.

All methods must be called serially by one host owner. There is no thread-safe
ledger, persistence, automatic retry, scheduler, hard CPU timeout or process
termination. The host supplies its clock, bounds its own parsing and compute,
and must propagate cancellation/deadlines to any future execution process.
This is cooperative admission control, not a sandbox for untrusted Python.

## Tools remain data

The host supplies an explicit tuple of `ToolSchema` objects and a per-root
allowlist. Each schema has a name, version and exact required flat arguments
(`string`, `integer` or `boolean`); unknown arguments and invalid types fail.
Children may only narrow their parent's allowlist. There are no built-in shell,
network, browser, filesystem, model or credential tools.

The host-only `authorize_tool(request, approval_ref, expires_ms, now_ms=...)`
method records a decision about **one exact request** after the host resolves
the actual user authorization. A request includes its task, request ID, schema
version, canonical arguments and provenance. Approval expires no later than
the task. An approval reference is an audit label; reading that label in model
text is never permission. Pending approvals are bounded by remaining quota.

`reserve_tool(request, approval, now_ms=...)` rechecks schema, task ancestry,
deadline, exact request equality and the identity of the host-issued approval.
An equal record constructed by a worker grants nothing. It consumes one work
unit and one tool admission, invalidates the approval and prevents replay of
that task/request ID. It returns the unchanged `ToolRequest` **data**, not tool
output or an execution receipt. There is no dispatcher to execute it.

A future executor must consume admission exactly once, check cancellation and
authority again at dispatch, enforce resource/side-effect scope, and preserve
receipts for failed or uncertain calls. It must not treat a queued data record
as indefinite authorization, retry an uncertain side effect, or expose host
policy methods to a model. This module does not promise to revoke a tool already
executing, authenticate external processes or recover approvals after a crash.

## Exact typed grid-bank adapter seam

A grid task's canonical `input_json` is:

```json
{"schema":"cellforge-grid-task/1","train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]]}]}
```

This contract accepts 1..16 demonstrations and queries and rectangular 1..30 by
1..30 grids of integer colors 0..9. It deliberately forbids test-output fields.
`grid_bank_input(task)` returns a fresh `{"train": [...], "test": [...]}` object.
It reads no files and does no prediction. The future **host adapter** must:

1. Explicitly bind and verify the host-selected `cellforge-skill-bank/1` artifact,
   source versions, expert roster and interfaces from `cell_contracts.py`.
2. Charge its bounded execution and check cancellation/deadline.
3. Call the existing `skillcore.predict_bank(artifact, grid_bank_input(task),
   tensors=tensor_override)`. Preserve `None` for saved tensors or pass the
   evaluator's complete override unchanged, including zero/random controls.
   Never restore saved values into a supplied ablation.
4. Validate the existing return shape: one list containing one grid per query
   (`[[grid0], [grid1], ...]`). Remove exactly that candidate-list layer and
   produce `{"schema":"cellforge-grid-result/1","predictions":[grid0,grid1,...]}`.
5. Record genuine execution/source/artifact/override evidence and only then
   submit a `Result`. Exceptions are failures, not empty successful predictions.

This seam does not import `skillcore` or NumPy. The converter does not establish
competence, evaluator correctness, task success or evidence for a scientific
claim. Existing protected standards, controls and evaluation exclusions remain.

## Language and structured event/belief seams

`language.text` takes exactly `schema`, `operation`, `text`, `context` and
`max_output_bytes`. Operations are `complete`, `classify`, `extract`, `transform`
or `respond`. Context is up to 16 input/output text pairs. Result shape is
`{"schema":"cellforge-language-result/1","text":"actual adapter output"}`,
bounded by the requested UTF-8 byte limit. These are interface verbs, not
implemented abilities. Text is content and never host configuration.

`language.state_update` is a separate narrow symbolic contract:

```json
{
  "schema": "cellforge-language-state-task/1",
  "actors": ["a", "b"], "objects": ["key"], "locations": ["box", "desk"],
  "real_state": {"key": "box"},
  "actor_beliefs": {"a": {"key": "box"}, "b": {"key": null}},
  "event": {"actor": "a", "object": "key", "location": "desk"},
  "observation_mask": {"a": true, "b": false}
}
```

Each entity list contains 1..16 unique IDs. Maps cover every declared entity.
An event represents a proposed move; event fields and locations may be `null`
for unknown. Observation values are `true`, `false` or `null` (unknown), with
no conversion of unknown into false, a default actor or an inferred location.
Result fields are exactly `schema: cellforge-language-state-result/1`,
`real_state` and `actor_beliefs`, using the same IDs and nullable locations.
The contract validates representation, not the correctness of a belief update.
Real-state data and an actor's belief remain separate even when they disagree.

The proposed 12 -> 16 -> 5 MLP with 293 parameters is an unimplemented candidate
for this narrow symbolic subsystem, not a natural-language encoder or a measured
capability. A future natural-text encoder must have its own explicit capability,
versioned structured-output contract, provenance and charged work. The generic
`language.text` string output is insufficient to bind that encoder implicitly.
Extraction uncertainty must remain unknown or produce an explicit error, never
be promoted to certain state or permission. No event or belief value authorizes
a tool. No neural cell, encoder, language execution or successful example is
implemented by this change.

## Validation scope

On October 6, 2026, a bounded `python -B -` in-memory unittest check ran 15 tests
in 0.007 seconds: **OK**. It covered aggregate quotas, unavailable capabilities,
hop/fanout/deadline limits, permission narrowing, cancellation, exact-request
approval, forged/expired/replayed approvals, pending-approval bounds, task/trace
identity, JSON/schema rejection, detached grid inputs and nullable belief state.
Host adapter labels were fixtures only; no adapter or tool was called. The check
was supplied through stdin, with no additional test file or bytecode written.

AST inspection confirmed stdlib imports and no execution/file-opening primitives;
Git whitespace checks for both new files passed. This is not an exhaustive
security review, a full repository test pass or evidence of specialist intelligence.
The protected scientific suites were not run in this restricted contract pass;
no scientific qualification or performance claim is made. There is no
prototype campaign, paid/model call, tool execution, deployment or commit.
Integration and real execution remain with the parent after the live run ends.
