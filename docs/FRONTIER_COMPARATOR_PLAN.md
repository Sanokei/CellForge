# Document comparator predeclaration — plan only

Protocol: `document-frontier-comparator-plan/1`.
Source verification: September 29, 2026, America/New_York.
Draft freeze timestamp: `2026-09-30T02:04:58Z` (September 29, 10:04:58 p.m. EDT).
This freezes this proposed design before baseline execution, not the dataset,
endpoints or executable evaluation. No baseline was performed. No credentials
were accessed, paid requests made or large weights downloaded.

The full objective remains independent matched-protocol superiority against
explicit frontier comparators across the declared scope. This document designs
the document comparison; a win here alone would not establish the full objective
or AGI. MMLongBench-Doc is demanding, not universally hardest.

## Sources and candidate identity

| Candidate | Verified primary evidence | Pinning requirement |
| --- | --- | --- |
| OpenAI `gpt-6-astra`, Responses endpoint | [Official model page](https://developers.openai.com/api/docs/models/gpt-6-astra) lists demanding-work positioning, image input, text output and supported reasoning efforts. | Page's snapshot section exposes only `gpt-6-astra`, without a dated ID. Do not invent a dated snapshot or claim immutable weights. Operator must record requested/returned model ID, serving fingerprint when available, date and response metadata; obtain documented version stability before claiming a pinned longitudinal comparison. Otherwise label endpoint-date-specific. |
| Anthropic `claude-fable-5-1`, Claude API Messages endpoint | [Official models overview](https://platform.claude.com/docs/en/models/overview) identifies this ID for demanding reasoning and image input; default effort is high. | [Official versioning documentation](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions) says post-4.6 dateless IDs are pinned snapshots, while serving infrastructure may change. Record ID, date, provider/API version and response metadata. |

These are two explicitly versioned candidate IDs, not an exhaustive universal
frontier set or evidence of account availability. Recheck official sources and
pin source snapshots at execution freeze; retirement or unavailable versions
requires a dated amendment before final scoring, never silent substitution.

The [official MMLongBench-Doc repository](https://github.com/mayubo2333/MMLongBench-Doc)
describes a three-stage evaluator using GPT-4o answer extraction and lists older
model interfaces including GPT-4o/GPT-4V and Gemini 1.5. Historical published
numbers and that extraction procedure are context only, not matched baselines
for the custom protocol below. The pinned local source is
`d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384`; its actual annotation inventory has
1,082 questions/135 PDFs, differing from the README's 1,091. Never relabel a
custom short-answer result as the official leaderboard score.

## Input and trial contract

All systems receive the identical complete ordered page raster list rendered
from each PDF at 72 DPI, with identical image-byte hashes, plus the same question
and instruction: “Answer the question using the supplied pages. Return only a
short answer; if the pages do not establish it, return NOT_FOUND. At most 192
UTF-8 bytes.” Question follows the pages/observations in every adapter.
No PDF text layer, ground-truth evidence, external search or oracle crop selection
is supplied. CellForge may use its frozen authored OCR internally; vendor models
receive the same raster pages, not CellForge's lossy OCR instead of the images.
Provider-side rescaling, image-detail settings and encoding are recorded as
unavoidable implementation differences. Any provider all-page limit is checked
before freeze; no question-dependent page pruning or silent truncation.

One candidate answer per question, one attempt, no model repair or answer judge.
No external tools for vendor baselines; CellForge internal computation and
authored specialist calls must be declared with hashes and counts. Same maximum
64 pages, 16 million pixels/page, 8,192 question bytes and 120 seconds of model
execution per item. Render/preprocessing time is separately recorded; end-to-end
latency, failures and cold/warm compile costs are also reported. Overflow or
timeout remains failure in the denominator. Items violating a common contract
must be declared before result access and reported as coverage failures rather
than quietly removed. If full benchmark cannot fit, amend capabilities/protocol
to preserve the full intended scope instead of claiming success on a convenient
subset. Final resource limits require a hashed executable policy before scoring.

## Output budget versus tokenizer differences

A 48-byte-token CellForge diagnostic cap is not 48 OpenAI/Claude tokens. The
planned common *visible-answer* cap is 192 UTF-8 bytes, enforced by scoring
complete output: over-cap answers fail, never truncate into a correct answer.
CellForge byte decoding needs at least 192 byte tokens plus termination;
vendor adapters use provider-native output limits sufficient for 192-byte text,
and record actual native/visible/reasoning token usage. Predeclare native limits
and high reasoning effort on answer-free development transport fixtures before
final freeze. Hidden reasoning cannot be equated to byte decoding or FLOPs;
equal wall time and visible-answer bytes are a resource-constrained system
comparison, not equal internal compute. If billing/output-token caps include
hidden reasoning, record that explicitly rather than starving one endpoint under
a purported equal-token rule. A different cap needs a separate amendment/run.

## Scoring, isolation and superiority gate

Custom primary metric: normalized short-answer exact match against frozen
operator-owned references/acceptable variants, using Unicode NFC, outer-space
trim and whitespace collapse only. No author LLM extracts or repairs predictions.
Freeze accepted-variant rules without candidate outputs; semantically equivalent
answers missed by exact match are a known limitation. Report raw exact match,
failure/invalid/over-cap rates, coverage, latency, usage and cost. Any official
three-stage evaluation is a separately labeled protocol with its own frozen
extractor, budget and fresh comparison, not merged into this result.

Use every question in the operator's finally frozen held-out document-family
manifest. Current 117 heuristic families do **not** prove isolation: duplicate
bytes were resolved, but related editions remain unverified. Do not score final
or tune against it until family overlap is audited and hashes/ownership/exposure
are recorded. Public benchmark pretraining contamination remains possible for
vendor and author models; disclose it and require an independently sourced,
fresh document-family replication before broad superiority claims.

Freeze CellForge artifact lineage, runtime snapshot, baseline IDs/settings,
input/scorer manifests and this plan amendment together. For each comparator,
report paired differences and 95% document-family cluster bootstrap intervals
(10,000 replicates, fixed operator seed), with multiplicity control across both
predeclared comparisons. Superiority requires positive lower bounds against
both baselines, no hidden denominator exclusions, and fresh independent
replication. A development-only improvement, endpoint error or timeout on the
other side does not establish frontier superiority.

## Readiness and execution status

Plan only; benchmark baselines unscored. Funded endpoint access is **not
established** and was deliberately not inspected. External comparator execution
awaits authorized funded endpoint access, immutable/dated identity evidence,
all-page limit verification, final split isolation and executable scorer/policy
freeze. Do not state that funding is definitively absent from this read-only
review. These block this comparison, not all useful goal work. Current runtime
v5 and broker-backend work is diagnostic and separate from baseline execution.
