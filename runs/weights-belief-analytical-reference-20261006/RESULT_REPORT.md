# Analytical reference belief-cell result

The separate analytical-reference author baseline passed strict 293-literal validation and scored 32/32 search and 64/64 development queries in both recursive rollout and teacher-forced evaluation. All eight scenarios passed (four search at 8/8 each, four development at 16/16 each). Categorical errors, runtime errors, unresolved queries and probability errors above the frozen 0.05 threshold were all zero. Mean L1 was approximately 3.386e-8.

| Arm | Rollout search | Rollout development | Teacher-forced search | Teacher-forced development |
| --- | --- | --- | --- | --- |
| Analytical reference author | 32/32 | 64/64 | 32/32 | 64/64 |
| Original random | 8/32 | 9/64 | 7/32 | 7/64 |
| Zero weights | 0/32 | 15/64 | 0/32 | 15/64 |
| Host reference oracle | 32/32 | 64/64 | 32/32 | 64/64 |

All arms had zero runtime errors. Original zero/random/oracle outcomes were identical to the saved V1 outcomes; the original random draw was reused as requested, not selected or treated as a new independent draw. Every query category passed for the analytical candidate (8/8 per search category and 16/16 per development category). Reference targets never repaired its model state.

A separate replay process recomputed identical results. Result SHA-256: 1cb5a4b2efc629146b7db2926d4c6ada189b30b9f027df0affe99bd52b824a66. The evaluation receipt binds source, raw response, interpreted weights, attempt and result hashes; RESULT.json retains full per-scenario/query/error metrics.

Source was the parent-relayed native task /root/derive_belief_reference_weights, using an inherited parent model with unknown exact backend name, not gpt-6-luna. Parent reports a text-only response with no tools and author context consisting of task/forward semantics and the saturation failure, without examples or development outcomes. That access history was not independently audited here. Credits remain unknown; 293 literal positions are charged to each analytical/random arm in this separate capture. No extra model call or admission occurred.

This is a declared hand-derived architecture learnability/expressivity probe on the existing synthetic one-hot histories, not a successful V2luna refinement, emergent social reasoning, exact arbitrary-distribution preservation or an independently held-out generalization result. Full scientific gates were not run or claimed green. The V2 Windows startup failure and later interrupted text refinement remain preserved as no-score outcomes.
