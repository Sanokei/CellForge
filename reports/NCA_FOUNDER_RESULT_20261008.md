# NCA founder 1: valid identity duplicate, no feasibility gain

This is the first consumed author invocation in the bounded two-invocation
tranche. It is not a completed mutation study, benchmark result or qualification.

## Result

The exact 4,597-byte final response passed strict JSON, vocabulary, name and
shape checks: 96 arrays containing all 1,488 required literal positions. Every
literal was injected unchanged into the saved fixed-T8 graph. The founder is
numerically equivalent to the identity reference and produces identical outputs
on every one of the 17 training demonstrations.

| Arm | Exact demonstrations | Exact tasks | Mean per-demo normalized pixel error | Runtime errors |
|---|---:|---:|---:|---:|
| Authored founder | 0/17 | 0/6 | 0.2877535467628347 | 0 |
| Fixed identity | 0/17 | 0/6 | 0.2877535467628347 | 0 |
| Fixed neighbor smoothing | 0/17 | 0/6 | 0.6022913902403066 | 0 |
| Zero | 0/17 | 0/6 | 0.5465838521643475 | 0 |
| First matched random | 0/17 | 0/6 | 0.9526492418829879 | 0 |

The predeclared strict demonstration-exact improvement criterion is not met.
There are no selected-baseline demonstration successes to preserve. Better
pixel error than random is not an authored gain over the identity reference.
Control outputs and the first random tensor were reused unchanged, not redrawn.

The full frozen inventory remains eleven exposed public-training tasks / 32
demonstrations. Five tasks / fifteen demonstrations are structurally
size-incompatible and were not executed. The restricted six-task result must
not be reported as performance on all eleven or as held-out generalization.

## Capture, runtime and accounting

The parent reported one completed founder CLI invocation, requested model
`gpt-6.1-sol` with low reasoning under the frozen route config. Backend-internal
request count and monetary cost remain unknown. The accepted V5 capture reports
exit 0, both EOFs, empty stderr, 82.256 seconds wall time and 140,848 KiB peak
owned RSS. Its complete four-event stream is consistent with observed zero tool
use/errors/reconnects; this is not proof that tools were impossible.

All five native counters are preserved exactly: input 24,742; cached input 0;
cache-write input 0; output 3,892; reasoning output 73. No additive relationship
or monetary conversion is assumed. Subscription usage was reported at 15%
before and after; that coarse percentage is not consumed-credit accounting.
Four known earlier non-scientific diagnostic invocations are separate overhead.

The single durable submission intent consumes 1,488 author positions and 1,488
matched-random positions. One wrapper/control-channel dispatch-admission failure
was definitively reconciled as no launch before the original authorization was
delivered. A later SSH timeout and receipt-transfer CRC failure were recovered
read-only from the completed immutable capture, without model relaunch or response
repair. These transport events are not additional scientific invocations.

The cloud checked exact final/receipt hashes, cross-receipt event/usage bindings,
and the frozen host-audit contract before capture. Full private raw-trace audit
is parent/server-attested; that trace was not imported into the public research
artifacts. Raw final bytes were durably saved before intake parsing.

## Independent replay and evidence

Run root: `runs/weights-nca-author-tranche-20261008/`.

- Raw response SHA256:
  `ea8591a8811eed9a65416cc98edee40f67c9dec7b0c8ade2a4165c42f74e9335`
- Founder literal tensor JSON identity:
  `734f60cfe7cc14d074f7ceccb09b257c9209bd3965f0abd96bdea44f8ce81ec9`
- `attempts/01/EVALUATION.json` SHA256:
  `00d4ee91840354e01f21eebe100fc62807eed325ee620c00019077480e422ef9`
- `independent_replay/founder001/RESULT.json` SHA256:
  `68ee21a52a880537e7aae5ae2e97fac09361159d947fe71be81feddf77ea56af`
- Exact native usage receipt SHA256:
  `b6bb2e362e551b1a0326127424cfd2b9de919b4e979444b9690cd0596ffb37c0`

The independent reviewer used a fresh process with the saved runtime and tensor
injection. Every demonstration prediction, metric, graph and per-tensor hash
matched. Numeric and prediction identity equivalence was independently confirmed.
The complete response, intake, audit, sanitized capture receipts, inference
outputs and injection hashes are retained under `attempts/01/`.

## Interpretation and next gate

This is a null founder result, not evidence that the architecture is incapable.
The founder packet conveys its target through examples but leaves the optimization
objective implicit; its phrase “no ... fitting” is not qualified as gradient
fitting. This is a possible design ambiguity, not a demonstrated cause of the
identity response. The packet is not retroactively changed.

A valid nonperfect founder with valid controls meets the frozen eligibility
condition for one possible mutation. The existing deterministic helper prepared
the unchanged founder context, parent literals and seventeen ordered training
failures only: `MUTATION_PACKET.json`, 27,507 bytes, SHA256
`b65e9d25b3987213f3248c9f56db8b281d9f6001a72c0cc143366a834857ad74`.
No extra feedback/control information or prompt edits were added. At this report's
creation, mutation approval, reservation and dispatch have not occurred. A second
invocation requires a new exact packet-bound parent approval and remains the last
possible invocation in this tranche. Broader ONE_STANDARD qualification stays red.
