# One native-agent tensor proposal: measured null result

The one existing native Codex proposal was valid and independently evaluated:
**0/4 search tasks and 0/4 development tasks, with zero runtime errors**.
Random, zero and reference arms each also scored 0/4 on both panels, with zero
runtime errors. The result provides no evidence of an advantage over random
weights and admits no candidate to an existing population. This is a bounded
pattern-recognition experiment, not evidence of chatbot or general capability.

## Evidence and accounting

- Run: `runs/weights-native-agent-one-20261006/`.
- Author thread: `01a10f41-e19a-732a-b546-f94e2edf000b`.
- Author turn: `01a10f41-ec1a-715b-b47f-feb5e193d2a6`.
- Actual model/reasoning: `gpt-6-luna`, `low`. It was not a nonthinking call.
- Author input: frozen `AUTHORING_PACKET.json` and `RESPONSE_SCHEMA.json`,
  containing only search task `1a07d186`'s three demonstrations and template
  semantics. No query answers or selection/report evidence were provided.
- Exact native response captured in `HOST_EVALUATION/RESPONSE_CAPTURE.bin`;
  SHA256 `f0e1d5f99799223aa144bd327639f7512e1a57bdfa5415213c75baff7caef468`.
- One submission consumed before parsing/scoring. Both native and matched
  random arms were charged all **102 literal tensor positions**. The random
  arm uses one uniform[-9,9] draw at every position, seed `2026100602`.
- Authoring credit usage is **unknown/null**, not zero. This is a separately
  accounted native artifact, not `maker.ask` or a CLI-metered worker receipt.
- Result hash:
  `8c432472c2ce1c5531bc38ae51f6030956f09ccbd14c0017d1cf753a114c7c1b`.
- Protocol hash:
  `04663de0bcee1ac6b331f361941bd668efc4a3f0e18b2e59eda7fe83f5bcb6a3`.
- Attempt hash:
  `5b47941ed9906732bc396f2ec8c1ebb9325176f675817f4ae8b0c1dc62bd059c`.

## Validation

`scripts/native_proposal_import.py` strictly validates the existing literal
tensor interface, task binding, numeric range, finite values, exact wrapper and
duplicate JSON keys. Invalid/refused submissions also consume the full attempt.
It freezes host source and data hashes, strips query outputs from predictor
arguments, preserves runtime exceptions as unknown outcomes, and refuses an
automatic repeat or population admission.

Four isolated tests passed in 0.045 seconds:

```text
Ran 4 tests in 0.045s
OK
```

Actual import completed in 0.770 seconds. A separate process using the saved
host runtime recomputed the result identically in 0.670 seconds. See
`reports/native-import-tests-20261006.log`,
`reports/native-import-result-20261006.log`, and
`reports/native-import-replay-20261006.log`. `git diff --check` passed.
BLAS/OpenMP/MKL were each capped at one thread; subprocess timeouts were 30 seconds.

No second author call, report access, paid API call, server action, deployment,
commit or push occurred. Development results remain host-side selection
evidence and must not be supplied to an author. The protected broad milestone
tests were not rerun for this bounded import; no ONE_STANDARD or ARC-goal
qualification is claimed. Earlier frozen run artifacts were preserved.

## Next bounded step and blockers

The parent requested this result promptly before more model work. The completed
`failure_journal.py` has not yet been connected to this result by a trusted host
callback. That integration must bind the immutable proposal/attempt receipt,
recompute search-only evidence, and scope all search rows in their original
order (`1a07d186`, `bd283c4a`, `3194b014`, `f5b8619d`). A founder record must not
misrepresent the diagnostic reference cell as its parent. Only a verified
search demonstration and search outcome may become future author feedback.

No next author prompt has been dispatched. A subsequent attempt requires the
parent's bounded call allowance and must keep credit usage unknown unless a
real meter becomes available. The ordinary CLI route remains blocked by its
quota-read timeout; the native route has not bypassed or repaired that guard.
