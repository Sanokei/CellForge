# Native-agent proposal route: feasibility and one prepared request

This route is within the current explicit authorization. `docs/ONE_STANDARD.md`
rule 9 already names the offline/human-supervised file outbox/inbox protocol in
`scripts/run_three_arm.py`. The owner's repeated instruction that Codex agents
author weights, and the parent's latest instruction, authorize this tiny separate
native proposal. No additional owner approval is required merely to export it or
evaluate a returned literal artifact within that scope. No protected test, floor,
closed campaign or benchmark-access boundary needs to change.

The blocked CLI worker remains halted. Native-agent work consumes the current
Codex service budget; this is not a free or credit-metered replacement. Record
actual agent/thread/model provenance supplied by the supervising platform and
leave authoring credits unknown when no receipt exists. Never invent a
`maker.ask` invocation, CLI quota receipt, worker reservation or worker signature.

## Existing APIs and their limits

1. `DirectWeightMaker(ledger, budget=..., profile=...).submit(parent, proposal,
   diagnostics=..., fixtures=..., evaluate=..., accept=...)` in
   `src/cellforge/construction/direct_weight_maker.py` is an existing accounted
   manual import API. Proposal fields are `base`, `actor`, `model`, `scope`,
   `rationale`, optional `authoring_cost`, and `ops` containing literal
   `{op:"set_param", name, value}` records. Its underlying `MakerOperator`
   charges invalid/no-change submissions and preserves parent integrity.
   **Its parent is a CellForge IR `Graph`, not a `skillcore` skill-bank cell.**
2. `scripts/run_three_arm.py` and `niche_weight_bridge.run_comparisons` implement
   file outboxes/inboxes for Graph artifacts. They require a domain adapter,
   existing graph genes and three seeds. The default random arm edits one scalar;
   it is not a matched control for a complete 102-position proposal. Running this
   unchanged would neither be a minimal current-bank import nor fair accounting.
3. The current grid implementation already provides public
   `cell_contracts.validate_tensor_override(params, reference_params)` for exact
   names/shapes and finite literal number types, followed by
   `skillcore.valid_params(template, params)` for registered shapes and [-9,9]
   bounds. Then `skillcore.make_cell(template, params, **host_provenance)` creates
   a content-addressed cell. `skillcore.run(cell, query_input, demonstrations)`
   executes it without query outputs. `skillcore.task_outcome` returns null for
   runtime failures, distinct from false for wrong answers. Typed bank export
   and `predict_bank` are available when needed, but a host default router must
   not be relabeled as agent-authored router weights.

**Gap:** there is no complete accounted manual/native import endpoint for this
skill-bank path. `head_driver.parse_cells` alone silently skips invalid entries
and does not account for native authorship/attempts. The CLI worker's private
validator should not be repurposed to manufacture worker receipts. A small,
separate host adapter using the public APIs above is feasible, but has not been
implemented in this feasibility pass. It must consume one immutable submission,
including invalid/refused replies, before evaluation; preserve exact raw response
and packet hashes; and record unknown credit usage honestly.

## Prepared one-task outbox

Directory: `runs/weights-native-agent-one-20261006/`.

Give the native author ONLY `AUTHORING_PACKET.json` and `RESPONSE_SCHEMA.json`,
or paste their contents into a context without prior task/evaluation history.
Do not grant that author the dataset directory, host manifest or evaluation logs.
The packet forbids tools, browsing, task lookup, code and gradients. It contains
the three demonstration pairs for search task `1a07d186`, the existing
`assoc_induct` semantics and the exact required response structure. No query
pairs, selection examples, report examples or prior evaluation scores are included.

Response is exactly one JSON array entry:

```text
[{"task_id":"1a07d186","params":{"sim":<10x10 numeric array>,"temp":<1 numeric value array>,"keep":<1 numeric value array>},"rationale":"..."}]
```

All **102 numbers** must be explicit, finite and within [-9,9]. No extra fields,
booleans, numeric strings, duplicate JSON keys, omitted entries or executable
expressions. The rationale limit is 4096 UTF-8 bytes (the JSON Schema's character
limit is supplemented by this byte check in the host). The task ID/template are
host-bound and cannot be changed by a response.

Packet SHA256:
`480180c9a379148bf4b0d6d603a2011991334aeb4ccaa8336bce766764091aa3`.
Schema SHA256:
`1eaf67ee775689c554fb02d62ceaf112c5180ed288cc332b82a7c40b4b1ef667`.
`REQUEST_MANIFEST.json` binds task/source/template hashes and records provenance
as pending. `STANDARD.json` explicitly marks the request unqualified, with no
founders/proposals claimed yet. No native agent was spawned in this pass and no
response has been received or evaluated.

The parent can now assign one bounded native author to return the literal JSON.
Before importing it, freeze the separate one-submission accounting record,
matched 102-position random-control seed/budget and evaluation protocol. Use
only the existing search/selection panels; never export selection feedback to
the author. Preserve runtime errors as invalid evidence. Do not reopen the
blocked CLI batch or rewrite its frozen artifacts. A passing tiny proposal still
would not satisfy the multi-seed, crossover, router and scientific standard.
