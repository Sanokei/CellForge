# Belief V2: one search-feedback refinement

This is the second reviewed author attempt for the belief cell, with one local
submission allowed in this new directory. No third attempt is authorized
without review. Preparation makes no model calls and designs no replacement
weights. All V1 evidence remains at `../weights-belief-native-packet-20261006/`.

Give a fresh author context **only** `author/AUTHORING_PACKET.json` and
`author/RESPONSE_SCHEMA.json`. Explicitly restrict reads to these files and
output to `author/NATIVE_PROPOSAL.json`; do not expose host files or development
evidence. The platform is known to lack a tool-disabled switch: this is
prompt-restricted access, not technically enforced filesystem isolation.
Directory separation does not prove compliance. Preserve the original author
tool/access history and exact response externally.

The packet retains all 32 V1 search examples, the identical 293-position
12->16->5 parameter schema/forward semantics, and the exact failed V1 literals
as the parent candidate. Its added feedback is restricted to the first original
search scenario, events 0..2. Event 0 supplies an observed-update failure with
exact input, hidden preactivations, tanh values, output contributions/logits,
probabilities and expected output. The following unseen/unrelated events show
actual recursive inputs/outputs alongside independent teacher-forced results.
Every trace value was recomputed and checked against saved V1 search evidence.
The host never injects the correct reference state into the model rollout.

The stated failure mechanism is a numerical observation about these supplied
weights/input: the observed-new units saturate almost equally, so their small
logit contrast loses to the old-location signal. The packet supplies no new
coefficient values, optimizer, gradients or replacement architecture. It makes
no claim that this explains untested contexts or that a repair is known to work.

`host/SEARCH.json`, `host/DEVELOPMENT.json`, `host/runtime/belief_cell.py` and
`host/evaluate.py` are byte-for-byte V1 copies. Thus panel families, seeds,
reference semantics, 32/64 queries, category balance, error handling and scoring
stay unchanged. The author receives no development data, examples, labels,
statistics or outcomes. Reusing the development panel does not create fresh
held-out evidence. A new single random control seed, **2026100632**, is declared
before V2 author output; its distribution remains uniform[-0.25,0.25]. Native
and random each consume all 293 positions per submission, including invalid
native submissions. Credits remain unknown, not free. There is no admission.

`FREEZE.json` binds all static source, packet, data and protocol bytes; retain
its SHA-256 externally before dispatch. `host/PREPARATION_VERIFICATION.json`
records parent hashes, trace equality and unchanged copied source/data hashes.
Frozen files are never edited or rehashed after authoring. New response,
attempt, source and result receipts are separate additive files.

Host commands, from the repo root, use the externally retained V2 hash:

```text
python -B runs/weights-belief-native-refine-20261006-v2/host/evaluate.py --expected-freeze <V2-hash> check
python -B runs/weights-belief-native-refine-20261006-v2/host/evaluate.py --expected-freeze <V2-hash> import --response runs/weights-belief-native-refine-20261006-v2/author/NATIVE_PROPOSAL.json
python -B runs/weights-belief-native-refine-20261006-v2/host/evaluate.py --expected-freeze <V2-hash> replay
python -B runs/weights-belief-native-refine-20261006-v2/host/evaluate.py --expected-freeze <V2-hash> record-source --source-thread-id <id> --source-turn-id <id> --model <actual-model> --reasoning <actual-setting>
```

Do not import until the parent confirms terminal author completion. The frozen
importer permits one local attempt, preserves the exact captured bytes, checks
the strict numeric response and hashes, scores all arms, and supports a fresh
process replay. `attempt_index=1` is local to this directory; protocol field
`family_author_attempt_ordinal=2` records the overall second attempt. Source IDs
are attached later; no credit receipt or source identity is invented.

No natural-language, emotion or general social competence is claimed. The
protected full standard/semantic/ARC/lane suites are not claimed green. No
model call, third author, deployment, new dependency or live source change is
authorized by preparing this packet.
