# Offline timeout and literal-output burden diagnosis

This is a descriptive audit of four existing invocation receipts and packet bytes.
No new model, tokenizer, candidate, inference or control evaluation was run.

| Invocation | Scientific input packet bytes | Wall seconds | Final response | Peak owned RSS KiB |
|---|---:|---:|---|---:|
| Original founder | 16,752 | 82.256 | Valid, 4,597 bytes | 140,848 |
| Original conditional mutation | 27,507 | 120.056 | None; timeout | 141,524 |
| Fresh comparison A | 16,752 | 81.609 | Valid, 4,597 bytes | 141,392 |
| Fresh comparison B | 17,203 | 120.118 | None; timeout | 139,640 |

The two successful final responses are byte-identical. Each spells out 96 named
arrays and all 1,488 positions despite containing only 24 ones and 1,464 zeros.
This is a substantial fixed serialization task even for the identity response.
The common response JSON schema is 15,494 bytes. Packet byte counts exclude
platform context and cannot be treated as model input-token counts.

Exact native successful-call counters are:

| Invocation | Input | Cached input | Cache-write input | Output | Reasoning output |
|---|---:|---:|---:|---:|---:|
| Original founder | 24,742 | 0 | 0 | 3,892 | 73 |
| Fresh A | 24,742 | 0 | 0 | 3,920 | 101 |

All five counters retain their native meanings; no additive relation is assumed.
The timed-out calls have no completed usage event. Missing usage is not zero
usage. The successful calls leave approximately 38 seconds below the fixed
120-second cap, but two successes cannot estimate a reliable completion-time
quantile. Output bytes divided by total wall time would combine scheduling,
reasoning and serialization latency and is not a measured generation throughput.

B adds only 451 packet bytes (about 2.69%) over A. The mutation adds 10,755 bytes
(about 64.2%) over the founder. Both longer packets timed out, but that observation
cannot establish a size effect: their instructions/content differ, order was fixed,
there are only four observations, and provider-side timing is unavailable.
B's small byte delta especially rules out treating input byte size alone as an
adequate explanation. Neither timeout hit the 2 GiB RSS ceiling; the reported
rejection in each case is wall timeout.

The timeout streams contain only thread.started and turn.started. The capture
protocol exports completed items, so no final item means no observable completed
candidate; it does not tell us whether the backend had generated partial text,
was reasoning, queued work or stalled. The wrapper's previously verified
file-backed stdin and EOF receipts address known host-capture failure modes but
do not diagnose backend time allocation.

## Design implications for a separately approved future study

- Keep these invocations terminal. Increasing the deadline and retrying them
  would change the preregistered experiment and cannot repair the missing scores.
- A prospective resource study would need an explicit latency/timeout outcome and
  fresh finite budget; a different deadline would be a new protocol, not a result
  reinterpretation. No such study or extra call is authorized here.
- Sparse encoding or copying an identity initializer could shorten output, but it
  would alter full-literal authorship/position accounting and require a separately
  reviewed scientific contract. It must not be silently substituted.
- A smaller genuinely weight-sensitive architecture could reduce serialization,
  but must first be justified against the existing training-only locality lower
  bound (the current compatible panel requires at least four local steps).
  Parameter count alone does not establish sufficient representational capacity.

The actionable conclusion is that the present route has two successful dense
identity outputs and two censored responses under the same wall cap. It supplies
no evidence that explicit-objective wording improves weights, no model-capacity
upper bound, and no basis for replacing missing candidates with hand-written ones.
