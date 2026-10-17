# GAMMA-G2 STAKE DECLARATION (gen-2, generation 1)

Researcher: gamma-g2 | Bankroll before stake: 1311
STAKE: 400 tokens on own transfer outcome

## Reasoning

- The submission is the only gen-2 entry with a MECHANISM aimed directly at
  the diagnosed transfer failure mode (non-recognition -> probe), plus the
  strongest anti-forgetting guarantee in the field (zero gradient steps;
  measured ~0 effect on recognized contexts; dev expected ~dnd1's 0.69%,
  guardrail 0.31%).
- Honest uncertainty: five straight 0.00% transfer scores across all of
  last night's attempts; my novelty proxy is synthetic (d' = 1.88, modest);
  alpha/beta are training with guardrails and may also land nonzero.
- 400/1311 (~30%) is a real commitment without risking bankruptcy:
  worst case (bottom quartile, -100%) leaves 911, well above elimination.
- Payout table per charter: rank 1 -> 3x; >= mean -> 1.1x; < mean -> -50%;
  bottom quartile (rank 3 of 3) -> -100%.

Staking locks at submission. No transfer evals run by researcher; the
judge scores transfer-12 (seeds drawn post-staking-close, held sealed).
