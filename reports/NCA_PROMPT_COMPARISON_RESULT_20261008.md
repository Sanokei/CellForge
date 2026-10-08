# Prompt-contract comparison: inconclusive after B timeout

The prospectively frozen A-then-B two-founder study is closed. A completed with
an identity duplicate; B timed out without a final response. There is no B score,
B−A effect estimate, positive feasibility result or causal explanation.

## Frozen comparison

A was the exact original 16,752-byte founder packet. B contained identical
architecture, tensor schema and 17 exposed training demonstrations, with only an
explicit exact-demo objective/pixel-error tie-break and clarification permitting
manual numeric reasoning while forbidding tools, gradients and runtime fitting.
Both used the same requested gpt-6.1-sol/low route, 120-second wall limit, 2 GiB,
seven-value vocabulary, fixed T8 graph and 1,488 charged literal positions.
The frozen full inventory remained 11 tasks/32 demos; five tasks/15 demos were
same-size incompatible, leaving six tasks/17 demos for restricted feasibility.
No queries, private evaluation data, gradients or control feedback were supplied.

A was a NEW independently captured invocation, not the historical founder reused
as an observation. Its 4,597-byte final happened to equal the historical response
exactly. The new provenance, native usage and full saved numeric replay remain
separate. A solved 0/17 demos and 0/6 tasks, with mean normalized per-demo pixel
error 0.2877535467628347 and zero runtime errors. Numeric and panel-prediction
identity equivalence were confirmed. Newly drawn matched seed 2026100803 also
solved 0/17 and 0/6, with error 0.8785617985385787 and zero runtime errors.
Independent fresh-process replay matched every output, summary and tensor hash,
and independently confirmed the specified seed03 draw. Fixed identity, smoothing
and zero receipts were reused without modification. A did not meet strict
feasibility superiority over the fixed reference and random control.

## B terminal event and accounting

B started once at 08:15:37 UTC on 2026-10-08 and reached a wall timeout at 120.118 s,
exit -15, peak owned RSS 139,640 KiB. Exact recovered receipts show both EOFs,
empty stderr, only thread.started/turn.started, no final response or native usage,
and no remaining owned invocation. The controller did not relaunch it.

The frozen adapter recorded terminal STOP. Both consumed invocations remain
charged in full: 2,976 positions per proposal arm. Matched seed 2026100804 was never
constructed or evaluated after STOP. Therefore equal charged positions do not
mean a completed paired numerical comparison. No candidate was repaired or
substituted, and no third invocation is authorized. The ledger retains B's last
stage as submitted alongside terminal_stop=true; its generic unresolved_no_retry
label denotes missing candidate completion, not a claim of a running process.

A's accepted trace establishes observed zero tool use for that complete call.
B's partial trace does not establish accepted zero-tool completion. Tool prevention
was never claimed. Generic platform and built-in skill descriptions remained.
Subscription usage was 16% before/after both new calls; consumed monetary cost and
backend request counts remain unknown. Four earlier diagnostic invocations and
the previous closed two-author tranche are separately retained overhead/history.

The two observed timeout cases cannot distinguish planning, generation latency,
provider delay or literal-output burden. Absence of final output is not evidence
of absent internal generation. The one-pair fixed-order design was descriptive
and exposed-training-only even if both arms had completed. Here it is inconclusive.

## Verification and artifacts

State: `runs/weights-nca-prompt-comparison-execution-20261008/`.

- A evaluation: `evaluation/A/candidate.json`, SHA256
  `9c7907c65000f4ee72228bc0b2abc922f25b7d870da33e0919cdeb679a6a69b0`
- Seed03: `evaluation/A/random.json`, SHA256
  `646734afec223bd065df20f71dedcf734bdf5fb73a2c81f9a4c086cd1d40ab3c`
- A independent replay: `independent_replay/A/RESULT.json`, SHA256
  `5dad31a765750d3563dd8486158fae95e4956283e18336db54dee4c5ec71e1e4`
- B host receipt: `host_receipts/B/COMPARISON_B_HOST_RESULT.json`, SHA256
  `5ab1e144e29f8e24742f058ad78089415e74589328d344897d42f0cefd227a1b`
- B wrapper receipt: `host_receipts/B/result.json`, SHA256
  `7db618d2f30323324b4ac8a80184e991c2220e2244f5d38e1d6b570becfd6af2`
- Terminal metadata: `TERMINAL.json`, SHA256
  `df8cadf01e49c38cf7b66740215d79b533c8e96f5844bdecb113d8e202e0915a`
- Terminal ledger head:
  `f1e68c0acd2c712734a5b72f47596fc795e6791fc2280c9e25e0b7be54689692`

Fresh focused command:
`python -m unittest tests.test_nca_prompt_comparison_prepare tests.test_nca_prompt_comparison_intake tests.test_nca_author_intake tests.test_nca_feasibility_preflight tests.test_nca_native_event_envelopes -v`

```
Ran 120 tests in 4.173s
OK
```

Full log: `nca-prompt-comparison-result-20261008/focused-tests.log`.
The latest separately run qualification command remains 25 tests / 7 failures / 3 errors;
it was not rerun for this closure. No scientific qualification is claimed. Frozen
historical runs, source, templates, protected tests and floors remain unchanged.
Restricted raw audit evidence and private server traces must not be published.
