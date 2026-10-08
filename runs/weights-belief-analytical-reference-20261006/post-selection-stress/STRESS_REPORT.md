# Fixed-weight belief-cell post-selection stress

The exact successful analytical-reference weights were frozen unchanged at parameter hash 9ca57f326afc539f87020a679e445842aec1eb2e5ad04c80615097f3d969e8d3. Nine original 32-event scenarios, each with four actors and four objects, were declared and hashed before this one pass. No author, random search, tuning, optimization, deployment or admission occurred. This is post-selection known-rule expressivity stress, not an untouched final test, independently held-out generalization result or ONE_STANDARD qualification.

The evaluation process completed in 3.64 seconds under a 120-second external timeout, with one worker process and a 10 ms yield per event. Every arm-slot prediction here is the fixed analytical candidate; no new control-weight draws were selected. Original panels, sources and prior results remain unchanged.

| Cohort | Every-slot top-1 correct | Probability errors >0.05 | Whole scenarios: top-1 / probability |
| --- | --- | --- | --- |
| Categorical, six scenarios | 3072/3072 | 0/3072 | 6/6 / 6/6 |
| Distributed, three scenarios | 1414/1536 | 1168/1536 | 2/3 / 0/3 |

Every one of 16 actor/object slots was scored after every event. All 4608 predictions completed with zero runtime errors and zero unresolved outputs. Categorical maximum absolute probability error was 1.693e-8. The stricter 1e-12 identity probe failed all 3072 categorical outputs, as finite-logit softmax approximates rather than exactly reproduces one-hot targets.

Selected-query scores were separately declared before evaluation and derived from the exhaustive failure records, without another model pass:

| Cohort | Selected top-1 correct | Selected probability errors >0.05 |
| --- | --- | --- |
| Categorical | 192/192 | 0/192 |
| Distributed | 88/96 | 73/96 |

The selector chooses actor=(event_index//4)%4 and object=event_index%4, with the actor permutation applied consistently in renamed scenarios. These selected-query numbers do not replace the every-slot scores. The original result used its original selected queries and was not rewritten.

All four actor/location renaming comparisons passed with maximum probability-vector difference exactly 0.0. The categorical coverage schedule explicitly observes all 16 actor/object pairs and then performs 16 further moves unseen by all actors. The second schedule includes distractors, unknown destinations, unobserved moves, partial observation and full observation. Twelve slot pairs lie outside the earlier 2x2 search indexing domain; they are not claimed unseen in the earlier 4x4 development panel. Actor IDs are host slot indices rather than neural features, so actor renaming is primarily a consistency check of that boundary.

## Continuous-state failures

The model is sharp categorical memory, not a general identity map on probability distributions. On an unseen move, an old distribution [0.4,0.3,0.2,0.1,0] should be preserved. Its first output was [0.9290007537573539,0.054229848655046486,0.009944832828963242,0.004124371434519391,0.002700193324116929], with maximum error 0.5290007537573539. Repeated preservation sharpens that distortion.

| Distributed scenario | Every-slot top-1 errors | Probability errors >0.05 | Maximum error |
| --- | --- | --- | --- |
| Mixed old states, unseen moves | 122/512 | 384/512 | 0.600 |
| Mixed old states, observed distractor | 0/512 | 288/512 | 0.600 |
| Near-tie old states, repeated preservation | 0/512 | 496/512 | 0.790 |

All 122 top-1 errors had exactly uniform reference targets [0.2,0.2,0.2,0.2,0.2]. Small asymmetries selected a different largest component than the declared lowest-index tie convention; the first deviation was 5.55e-16 and the largest probability deviation among these top-1 failures was 0.00061111. They remain counted as errors under the frozen metric. They are distinct from the 1168 larger distribution-tolerance failures on nonuniform mixtures. The near-tie case first crossed the 0.05 probability threshold at the second event.

No reference targets repaired recursive model state. RESULT.json contains complete per-event counts and every categorical/probability failure with expected and actual distributions. SELECTED_QUERY_RESULT.json retains the selected-slot outcomes. COMPLETION_RECEIPT.json binds the weights, both pre-evaluation declarations, results, execution receipt and unchanged-parent checks. Stress was run once, with no optimization or independent stress replay claimed.
