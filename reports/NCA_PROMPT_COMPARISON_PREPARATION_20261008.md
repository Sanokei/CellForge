# Prospective NCA prompt-contract comparison: preparation only

## Status and limits

A distinct comparison is frozen at
`runs/weights-nca-prompt-comparison-20261008/`. It is **unapproved for execution**:
zero author invocations, reservations, consumed attempts, charged positions,
constructed random controls or predictions. No new model/API call, runtime
evaluation, dispatch integration, publication, commit, push or deployment occurred.

The original tranche remains terminal HALT. Its packets, source, runtime,
receipts, tests and outputs were not changed. The historical founder is not the
prospective A response. Historical approvals and quota attestations do not carry
over. This preparation provides no evidence of capability gain.

The only implementation is `scripts/nca_prompt_comparison_prepare.py`, with
`prepare` and `verify` operations. It does not import the historical intake or
runtime, generate tensors, reserve an attempt, execute a model or score anything.
It reads and checks 72 historical source/artifact bindings, constructs data-only
packets and freezes an exclusive-create directory. Verification requires an
externally retained contract hash, exact regenerated content, intact upstream
bindings, and no additional execution-state files or directories.

## Exact prompt intervention

A is the **exact original founder packet**, including its original ambiguity.
The response schema is copied byte-for-byte for both arms. B changes exactly two
JSON paths, documented in `PROMPT_DIFF.json`:

1. `/architecture/sharing`: replace only the original prohibition substring:

   Original: “No demonstration conditioning, task routing, per-case weights,
   fitting, or executable structure changes.”

   Replacement: “No demonstration conditioning, task routing, per-case weights,
   gradient or runtime fitting, or executable structure changes. Manual numerical
   reasoning to choose literal weights is allowed; do not use tools, gradients,
   or code.”

2. Add `/objective`:

   “Choose the one global tensor set to maximize the number of exact output-grid
   matches across all 17 training demonstrations. Break ties by lower mean
   per-demonstration normalized pixel error: for each demonstration, mismatched
   pixels divided by output pixels, then the unweighted mean across all 17
   demonstrations.”

All other architecture text, all demonstrations, tensor schema and response
contract have identical values and canonical subtree bytes. The patch retains
both full old/new sharing values, exact changed substrings, packet lengths and
hashes, and the common unchanged-projection hash. Reversing these two changes
reconstructs A's original bytes exactly. The two clarifications are bundled;
even a completed comparison cannot isolate their individual effects.

## Frozen prospective design

- Two **new independent founders**, A then B, one invocation each. No mutation,
  replacement, cross-arm feedback, prior weights or reference/random outcomes are
  shown to either author. Scientific inputs are only the respective packet and
  unchanged response schema. Generic platform context may remain and must be
  disclosed; preventive tool isolation is not claimed.
- Same fixed T8 architecture, 96 arrays, 1,488 positions, seven-value vocabulary
  `[-1, -0.5, -0.25, 0, 0.25, 0.5, 1]`, `gpt-6.1-sol`, low reasoning, concurrency
  one, 120-second author wall limit and 2 GiB author memory limit.
- At most two scientific author invocations total: 2,976 positions in the author
  accounting arm and 2,976 in the matched-random accounting arm. Each prompt arm
  gets one 1,488-position opportunity. Invalid, refused, neutral, duplicate,
  timeout and uncertain attempts are charged. No refunds or retries.
- Fresh matched-random seeds: A `2026100803`, B `2026100804`. Each future draw is
  independent uniform choice from the same seven values in sorted tensor-name,
  C/row-major order, only after the corresponding author invocation is actually
  consumed. Reservation alone is insufficient. No random tensor is constructed
  here. Exact RNG implementation remains a pre-dispatch integration gate.
- Reuse original phase1 identity, neighbor-smoothing (`neighbor_majority`) and
  zero tensor files, saved predictions and results unchanged, without reevaluation.
  These files are hash-bound references, never author inputs.
- A valid completed/audited/evaluated A and valid matching random are required
  before B. A perfect score does not change B into a mutation or suppress B.
  Any invalid response, failed audit, tool use, unknown/gapped/truncated event
  stream, timeout, uncertainty, source mismatch, runtime error or invalid control
  terminally stops the comparison. An A timeout prevents B; the comparison is
  inconclusive. Charged but uncomputed controls remain explicitly incomplete.
- Preserve exact response bytes before parsing, exact native-event line/trace
  hashes and all five native usage counters. Only complete observed-zero-tool
  provenance may qualify a response. Backend request counts and credits remain
  unknown; equal literal charges do not establish equal monetary expenditure.

The evaluator-only inventory retains all **11 tasks / 32 demonstrations**,
including the **5 tasks / 15 demonstrations** excluded for output-size mismatch.
Only the original **6 tasks / 17 demonstrations** enter this restricted
comparison. These are previously exposed public-training demonstrations.
Cancelled `e7639916` stays excluded, unopened, unscored and unreplaced.

Primary analysis is B minus A exact demonstrations, denominator 17. Secondary
analyses are exact tasks (6), unweighted mean normalized pixel error, numerical
identity equivalence, panel-local prediction-vector identity equivalence, and
descriptive fixed/random comparisons. One fixed-order pair cannot establish
causality, significance, replication, generalization or architecture incapability.

If a later report uses a positive-feasibility label, the candidate must strictly
beat the best fixed reference and every consumed matched random on exact
demonstrations, while preserving every selected-reference success. All required
controls must be valid and complete. B>A alone is only a descriptive comparison
and cannot become a feasibility or capability-gain claim. None of this qualifies
as held-out evidence, whole-panel feasibility, ARC benchmark performance,
admission or ONE_STANDARD qualification.

## Remaining execution gates

Preparation is not approval or an operational dispatcher. A future decision must
bind both exact packet/schema hashes and the contract, with fresh route,
resource, packet-only isolation, audit and quota attestations. A new reviewed
two-independent-founder integration is required. **Never use the old
mutation-specific reservation/dispatch path for B.**

Before each prospective dispatch, require a fresh authoritative true
`ordinaryUsageAllowed` value and current shared usage below the **85% STOP**.
Bind the quota period/reset timestamp; drift needs fresh attestation and approval.
Stale or unavailable meters, unmetered permission and historical quota approvals
cannot authorize execution. Freeze and test the exact RNG and capture
implementation before dispatch. No extra diagnostic invocation is authorized by
this preparation.

## Frozen bytes and SHA256

All paths below are relative to the new preparation directory.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `CONTRACT.json` | 1649 | `f4f89d4eaeec4adbd50f4f8bd42c5e35ad56cf535396032f36f516a5b93a09f4` |
| `author/A_PACKET.json` | 16752 | `f26d3567f9d02903bbb525072803ed4713714108665a30458df99f2cd1532e37` |
| `author/B_PACKET.json` | 17203 | `2138f4a0a1600a497efff13e8fce97b83d1b70255842d5a45371986956d9d387` |
| `author/RESPONSE_SCHEMA.json` | 15494 | `d7ebc66e50abf32d43db94d4984201725d8dc8c122f2795a2af57aef1d272f72` |
| `PROMPT_DIFF.json` | 2146 | `89afebe33fe03acab0b81e9e23f9c9f2cf351f4673a157e455f9876bde4c81e2` |
| `PROTOCOL.json` | 8277 | `69fa8d809a063c606bfbe8f9cd16208d089e6c1800efb57e8f9e1ba735cd6704` |
| `SOURCE_BINDINGS.json` | 14852 | `fd2265c4175c952ef2bfc06edf11b986477e7f8e7e0534297909f9c576e1f25d` |
| `EVALUATOR_ONLY.json` | 12762 | `f0cbc86ec48eb05ef9fb87503408e1329888f353cac7dca13c4993bd0c3cc576` |
| `INITIAL_STATUS.json` | 456 | `050669d1bea9fe5e7ac320a0880557b8ab09ac9a9fe8c9f184bb3a08d50b80d1` |
| `STANDARD.json` | 345 | `23b21b78947f62510849f9e1906e7a7f75ad31449261f908bd4a084bffba1c53` |
| `source/scripts/nca_prompt_comparison_prepare.py` | 28027 | `928851e2a30594a12b1f9abb28f744ed8f347005fffac4f5faa92c15b9092f92` |
| `source/tests/test_nca_prompt_comparison_prepare.py` | 21484 | `d2c3b2db2ceaf9db9ae82c553c0fac8e0753cdb7c75078d4cb59090d145de21d` |

The contract was frozen at 2026-10-08 07:35:35.020171 UTC (file timestamp)
and verified afterward. Recheck without execution:

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/nca_prompt_comparison_prepare.py verify \
  --repo . --contract-sha256 f4f89d4eaeec4adbd50f4f8bd42c5e35ad56cf535396032f36f516a5b93a09f4
```

## Verification and prior qualification

Fresh preparation-only command, after the final source freeze:

```text
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_nca_prompt_comparison_prepare -v
Ran 30 tests in 0.268s
OK
```

These tests cover exact A and schema bytes, the exact B diff and unchanged
subtrees, training-only input scope, full inventory, no weight/control leakage,
budgets/order/seeds, current quota gates, descriptive/feasibility distinctions,
immutable-source failure, tampered/locally rehashed artifacts, exclusive creation,
missing/extra files or directories, path traversal, symlinks and strict JSON.
They create only isolated temporary preparation fixtures. No author or candidate
evaluation is part of this test module.

The full qualification suites were **not rerun for this preparation**, per the
parent's preparation-only scope. The prior terminal-tranche command was:

```text
python -m unittest tests.test_one_standard tests.test_standard_semantic -v
Ran 25 tests in 1.796s
FAILED (failures=7, errors=3)
```

Source: `reports/nca-tranche-terminal-20261008/qualification-tests.log`, saved on
2026-10-08 at 07:14:02.779671 UTC according to its file modification timestamp.
This is a receipt timestamp, not a claimed exact process-start timestamp. The
11,228-byte log has SHA256
`e092cdc7aa90a72001fdba7fb7cb095cf39a3fe48f03b36d5533c5c41782108a`.
The
prior result remains red; it is not a fresh qualification of this new directory.
No protected test or floor was changed.
