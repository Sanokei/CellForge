# Frontier benchmark protocol

This document separates three task interfaces. A score is reported only when the
exported CellForge artifact itself receives the task input and produces the task
output. Author LLMs may design the artifact, but may not answer evaluation items
at inference time.

## Multiple-choice text: `mc-loglik/1`

MMLU, ARC-Challenge, and HellaSwag are already installed on the separate
evaluation server. `src/cellforge/reference/bench_operator.py` fixes a seeded,
nested item order and reports both total-continuation-bits accuracy and
length-normalized accuracy with Wilson intervals. The real-text artifacts have
only lowercase letters and limited punctuation. The operator drops unsupported
characters, including digits, and reports character coverage; this is a
material limitation, especially for MMLU. A result from this adapter is not a
general answer-generation score. Selection uses text bpc until a candidate
clears chance on an independent benchmark check; sealed benchmark items remain
outside the maker loop.

## ARC-AGI-2: `arc-grid/1` (next artifact interface)

This is distinct from ARC-Challenge. The [official ARC-AGI-2 repository](https://github.com/arcprize/ARC-AGI-2)
provides 1,000 public training tasks and 120 public evaluation tasks. Each task
contains demonstration input/output grids and one or more test inputs. Success
requires the exact output grid, including its dimensions, for every test input.
The official description permits two attempts per test input. The 120 public
evaluation tasks are a one-time final check; the private competition tasks are
not available locally. Source pinned on the evaluation server at commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49`, under
`/srv/cellforge-private/arcagi2`. No evaluation task contents are copied into
the author workspace.

Required model contract: `predict_grid(demonstrations, test_input) -> up to two
rectangular integer grids`, colors 0–9, dimensions 1–30. The operator will
validate shape and values and score exact task success; per-task details must
remain in the private evaluation process. A development curriculum can
use the 1,000 training tasks, partitioned by task ID before construction.

The exact-grid scorer and output validation are implemented in
`src/cellforge/reference/arc_grid_operator.py`. It accepts a predictor callback;
`src/cellforge/runtime/grid.py` now supplies a graph inference adapter with
the `grid-graph/1` interface. Each demonstration updates graph-owned recurrent
state; the final query supplies zero output tensors, and state resets per
prediction. The graph emits one or two dimension/color-logit heads. The
adapter only decodes their argmax values and rejects malformed/nonfinite
heads; it does not solve transformations or repair outputs. A saved/reloaded
authored color-permutation cell passes synthetic exact-grid tests. This is
interface validation, not a learned transformation or an ARC-AGI result.
No competitive authored grid artifact exists yet.

The first demonstration-conditioned artifact (`grid20/1`, builder
`scripts/build_grid_moe.py`) has 20 transformation experts and passes
`ladder-micro/1`. On 200 public training development tasks, both it and a
fixed-identity routing control solved 0 tasks (222 test inputs, no invalid
outputs). Receipts and frozen hashes are in `runs/grid20`. This is negative
development evidence, not an independent evaluation or frontier comparison.
The server CLI in `arc_grid_operator.py` partitions public training by task-ID
SHA-256 modulo five (zero reserved for audit), then selects in hash order.
It publishes aggregates and a selected-data manifest hash. It deliberately
does not access public evaluation tasks. This split protocol was fixed before
the first official-data measurement; later candidates must disclose aggregate
development feedback exposure.

The present `tinyreal/1` character model has an eight-character input window
and no grid output head. Encoding grids as prose or asking the author LLM to
solve them would change the tested system. Until a competitive grid artifact is
authored for this interface and independently evaluated, ARC-AGI-2 status is
**unscored**; current text artifacts remain incompatible.

## Tool use: BFCL V4 agentic (next artifact interface)

The [official Berkeley Function Calling Leaderboard project](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
includes multi-turn and agentic evaluations. Its V4 agentic domain covers
multi-step web search, memory management, and format sensitivity. This tests
more than picking a tool name: an agent must form structured calls, consume
tool observations, recover from errors, and finish the user's task.

The official evaluator source is checked out sparsely on the evaluation server
at `/srv/cellforge-private/bfcl-source`, pinned to commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Its web-search category
requires a SerpAPI key or an explicitly configured alternate search backend;
memory and format categories can be prepared without that service. No BFCL
items have been scored by a CellForge artifact.

Required model contract: a typed message and tool-schema input, plus a
structured `call(name, arguments)` or `respond(text)` output. The harness owns
tool execution and supplies observations to the model; it never silently
repairs arguments or lets a construction-time author answer an item. Report
the official category scores and success rate, invalid-call rate, tool-call
count, and latency. Freeze tool schemas and use the official evaluator version
and dataset manifest. For transfer, hold out entire tool families or domains
from construction, then compare a new domain with and without reused cells,
circuits, and experts at matched authoring cost.

The current `tinyreal/1` vocabulary lacks braces, digits, and other syntax
needed for tool calls, and its graph has no typed action output. BFCL V4 status
is **incompatible, unscored** until an authored action head and tool-observation
state are built. A text adapter that invents JSON outside the model would not
measure CellForge tool use.

## Evolution and negative evidence

Use development performance and independently checked component tests to
choose survivors. Preserve unsuccessful proposals as mutation donors with
their source evaluation, transfer only an explicit part, and re-evaluate the
child against its parent. A specialist can recover a component that fails when
applied globally. Track parent hashes, donor proposal and evaluation, changed
definitions, author tokens, and dev-eval count. Freeze the selected artifact
before opening any final benchmark score. Success on one domain does not prove
domain transfer; use held-out domains and a no-reuse control.
