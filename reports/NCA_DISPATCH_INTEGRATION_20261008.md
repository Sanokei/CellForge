# Parent/server dispatch integration checklist

**Preparation only. Do not run reservation, submission, intake or scoring steps
until the parent explicitly authorizes the particular founder invocation.** No
such action was performed to write this checklist. Existing freezes are unchanged.

## 1. Transfer boundary and gates

Transfer to the isolated author input directory only:

| File | Bytes | SHA256 |
|---|---:|---|
| `FOUNDER_PACKET.json` | 16752 | `f26d3567f9d02903bbb525072803ed4713714108665a30458df99f2cd1532e37` |
| `RESPONSE_SCHEMA.json` | 15494 | `d7ebc66e50abf32d43db94d4984201725d8dc8c122f2795a2af57aef1d272f72` |

Programmatically decode/transfer bytes and verify both hashes on the server;
never retype or reserialize them. The host-side capture wrapper, stable config,
route/quota attestations and dispatch identifier remain separate from author
input. The evaluator, phase1 tensors/results, actual task-ID map, contracts and
reference/random scores remain in this cloud workspace and need not be sent to
the author. Generic system/tool instructions remain disclosed context.

Before reservation, the parent verifies final wrapper/config hashes, bounded
file-backed stdin/deadline/RSS capture, exact raw-envelope validation, durable
five-counter NATIVE_USAGE binding, current quota, and packet-only research input.
The parent then supplies a real approval object matching
`nca_author_intake.verify_authorization`. It must bind the exact contract/packet,
parent identity, approval reference and route/quota attestation references.
The route attestation references the audit encoding, native schema and invocation
accounting supplements. Do not fabricate attestations or infer them from fixtures.

One author unit is one bounded CLI invocation, not proof of one HTTP request.
Known diagnostic calls are separate overhead, not founder attempts. No request
or monetary equality claim follows from the two-author/2,976-position cap.

## 2. Cloud-side durable reservation, then submission intent

Run from `/workspace/scratch/b2ab9be79e45/cellforge`. The approval file and host
reference below must be supplied by the parent after the above gates pass.

```sh
export PYTHONPATH="$PWD/scripts:$PWD/src"
export NCA_RUN="$PWD/runs/weights-nca-author-tranche-20261008"
export NCA_CONTRACT=18f8b7c1be20991b3ea021aa43d7bfd4a821aab67bc547989c2a9f222535d7ee
export NCA_ATTEMPT=1
# Set APPROVAL_FILE to the actual parent approval JSON, not a mock fixture.
python - <<'PY'
import json, os
from pathlib import Path
import nca_author_intake as M
r = Path(os.environ['NCA_RUN']); h = os.environ['NCA_CONTRACT']
a = M.read_json(Path(os.environ['APPROVAL_FILE']))
receipt = M.reserve_attempt(r, h, int(os.environ['NCA_ATTEMPT']), a)
print(json.dumps(receipt, sort_keys=True))
print('reservation_binding_sha256', M.digest(receipt))
PY
```

After the server controller is ready with its immutable host dispatch identifier,
but **before** starting the possible external invocation, record submission intent:

```sh
python - <<'PY'
import json, os
from pathlib import Path
import nca_author_intake as M
s = M.mark_submission_intent(Path(os.environ['NCA_RUN']), os.environ['NCA_CONTRACT'],
                            int(os.environ['NCA_ATTEMPT']), os.environ['HOST_REFERENCE'])
print(json.dumps(s, sort_keys=True))
print('submission_binding_sha256', M.digest(s))
PY
```

The binding digests above are canonical-object hashes (`M.digest`), not the
SHA256 of newline-terminated receipt files. The host response audit must use the
correct digests. Submission intent conservatively consumes 1,488 positions in
each arm; unresolved intent stops all later work. Reservation alone allocates
the budget but is not evidence that a model ran. No cancellation/reset helper
exists; do not delete a partial reservation to bypass this boundary.

**Only the parent/server controller starts the separately approved CLI invocation.**
There is no invocation command in this checklist or in the intake module.

## 3. Return and ingest exact response bytes

Return the exact final UTF-8 text bytes, their SHA256/byte count, sanitized host
audit metadata, complete native usage record and wrapper/config/capture receipt
hashes. Raw stdout/stderr traces remain restricted on the server for parent
audit; do not publish private reasoning or expose it to future authors.

The host audit uses the three normalized counters plus credits=`unknown`, while
NATIVE_USAGE preserves all five exact native counters and binds the completed
event/full-trace hashes. Include real packet/raw/submission/parent/route/quota
bindings and verified completion/zero-tool classification. Do not repair invalid
JSON, trim whitespace, add a newline or extract a nicer JSON substring.

With verified `RAW_RESPONSE_FILE` and the genuine `HOST_AUDIT_FILE`:

```sh
python - <<'PY'
import json, os
from pathlib import Path
import nca_author_intake as M
raw = Path(os.environ['RAW_RESPONSE_FILE']).read_bytes()
audit = M.read_json(Path(os.environ['HOST_AUDIT_FILE']))
receipt = M.capture_response(Path(os.environ['NCA_RUN']), os.environ['NCA_CONTRACT'],
                            int(os.environ['NCA_ATTEMPT']), raw, audit)
print(json.dumps(receipt, sort_keys=True))
PY
```

The module durably preserves raw bytes before parsing and charges invalid/refused
responses. Never call capture twice or replace its frozen bytes.

## 4. Failure accounting and evaluation

- No trustworthy final response, timeout, truncation, uncertain submission or
  incomplete capture: retain failed host receipts and call `mark_uncertain`
  once. No synthetic final response, refund, relaunch or mutation.
- A complete final response with forbidden tool/error/unknown-event audit:
  preserve it through `capture_response`; failed audit makes the attempt invalid
  and creates terminal HALT. No candidate claim or further invocation.
- Clean complete audit but invalid data/refusal: preserve the invalid intake;
  the attempt remains charged and blocks further authors. No host repair.
- Pre-dispatch transport failure is not a model attempt. Recover only the same
  original bytes; never relaunch a model to repair transport.

```sh
# Only for an actual consumed invocation with uncertainty; REASON is factual.
python - <<'PY'
import os
from pathlib import Path
import nca_author_intake as M
M.mark_uncertain(Path(os.environ['NCA_RUN']), os.environ['NCA_CONTRACT'],
                 int(os.environ['NCA_ATTEMPT']), os.environ['REASON'])
PY
```

For an intact, authorized completed intake, score once in the cloud using the
unchanged saved runtime. Invalid proposal data gets null outcomes without
candidate execution; HALT prevents evaluation. The first matched random uses
the phase1 saved tensor/result, with no redraw.

```sh
python - <<'PY'
import json, os
from pathlib import Path
import nca_author_intake as M
result = M.evaluate_attempt(Path(os.environ['NCA_RUN']), os.environ['NCA_CONTRACT'],
                            int(os.environ['NCA_ATTEMPT']))
print(json.dumps({'comparison': result['comparison'], 'accounting': result['accounting']}))
PY
```

An independent reviewer then reruns the saved candidate worker into a **fresh**
replay directory and compares deterministic outputs/metrics/injection hashes.
Do not rerun `evaluate_attempt` over its existing output directory or claim
scientific success from receipt hashes alone.

## 5. Conditional mutation is a later parent decision

At most one mutation is eligible only after a valid, clean-audit, independently
replayed founder with valid controls and at least one training failure. A perfect,
invalid/refused, uncertain, source-mismatched or stopped founder permits none.

`M.prepare_mutation_packet(run, contract)` deterministically includes only frozen
founder context, parent literals and ordered training failures. A **new exact
packet-bound parent approval** is required before attempt 2 reservation. Reuse
the above commands with `NCA_ATTEMPT=2` only after that approval. The second
predeclared random tensor is constructed/evaluated only after attempt 2 is
consumed. Stop after this finite tranche; no third attempt or automatic expansion.
