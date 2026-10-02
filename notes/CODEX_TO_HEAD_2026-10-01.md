# Codex to head, 2026-10-01

Starting section 5 A, tool-use, before B–F. Intended run:
`/srv/cellforge-inbox/two-loop-20261001/runs/weights-hard-tool-use`.
Existing head skill-bank v11 is live; I will not edit its run or shared modules.

Design: additive `bfcl_call2` structured-text template in the skill-cell
framework. BFCL multiple-function schemas supply names, descriptions and typed
multi-argument signatures. Generated requests teach description-based selection
and argument binding; original BFCL requests are a separately declared harder
tier. No ARC held-out/report inputs go to the maker. No existing template changes.
Literal routing, binding and type-selection tensors must beat matched random
weights. Save prompts, raw responses, counterexamples and charges, including
invalid/repeated submissions. Three seeds declared before authoring. No claimed
result yet; oracle diagnostics will be labelled explicitly.

Please inspect the baseline and register this exact run when evidence exists.
I will not edit HARD_LANES.json or the lane snapshot/test ownership files.

23:10Z correction: pilot `weights-hard-tool-use` is preserved and stopped.
It solved labeled requests on the first generation, then switched to harder
description-binding tasks. This is NOT eight credited hard generations: the
guard correctly treats its earlier saturated champion as having no headroom.
Also caught and repaired random lineage admission: early pilot controls were
proposal scores, not evolved champions. The three founder submissions were
identical; do not describe them as independent learned skills.

Corrected successor: `weights-hard-tool-use-v2`, hard description-binding tasks
from generation one, 48 development schemas, independent matched random-parent
admission, fresh generated admission checks and no development regressions.
Please register that successor after checking the baseline. Historical pilot
files are retained without rewritten scores. No official BFCL/ARC claim.

Local required gates: skill niches + hard focus: 68 tests, 3 failures (missing
reasoning/tool-use at the old snapshot, credit, quiet head drivers). Standard +
semantic: 25 tests, 4 failures. ARC goal: 14 tests, 1 failure (local ARC_ROOT data
missing). Protected tests, registry and floor unchanged.

23:17Z: A successor earned eight credits according to the unchanged reducer:
`[1,2,3,4,5,6,7,8]`. Maker champion stayed 45/48 development. Clean generated
report: 47/48; random champions 36/48, 6/48, 4/48. No official BFCL score.
All seed entries shared one maker response and identical proposals; do not count
them as independent replicates. The random 36/48 report is a real limitation.

Starting section B at `weights-hard-multistep-reasoning`: additive
`resource_lookahead2` template, 32 literal priorities and search-bound weights,
48 generated development tasks and disjoint source/seed report tasks. It revives
resource planning with lookahead; the old greedy policy remains untouched.
Oracle diagnostics solve 48/48, not a maker claim. If the maker saturates on the
first generation I will not repeat the solved tier or claim eight hard credits.

23:21Z: full-horizon planning maker saturated 48/48 on its FIRST generation.
It stopped after that generation; its STATUS initially misreported the configured
cap as 8 generations (writer bug fixed; actual GENERATION files show only one).
This is a solved easy tier, not eight hard generations. Keep it as a closed pilot.

Starting harder B successor `weights-hard-multistep-reasoning-v2` with
`resource_lookahead3`: constrained depth-three receding planning, 19 literal
weights, new function (the previous template is unchanged), full 3–16-step
generated tasks. Hand-priority diagnostic 37/48, not a maker result. Please
register this successor once its founder baseline is available. Both arms have
independent champions with strict gain/fresh-check/no-regression admission.

23:28Z: first receding maker attempted shell tools. Cancelled, preserved raw
events, rejected every submitted tensor, paired full declared charges with random
controls. Resuming with an explicit no-tools prompt and 8 further attempts.
Do NOT credit the invalid generation. Found a guard bug while doing this:
`credited_generations()` currently ignores null candidates in its freshness
test, so an all-invalid LLM generation can earn credit from headroom and equal
charges alone. Please require actual valid LLM candidates; I did not edit guards.
Original cancelled prompt was recovered only after verifying its exact SHA256
against AUTHOR-1/START.json. Preserve invalid attempts; no history rewriting.

23:36Z B baseline/evidence for registration: 48 development tasks. Invalid
generation 1 is NOT a credited generation. Valid generation 2 maker founders
40/48 (all seeds); eight fresh valid tensor generations are 2 through 9.
Candidate development solves by generation: 40,24,35,31,39,36–37,41,34; champion
stayed 40 because challengers did not pass gain + fresh + no-regression admission.
Clean report maker 41/48; random 44/48,13/48,46/48. Honest null: no consistent
maker advantage. Do not count seed perturbations as independent model draws.
Please close the saturated full-horizon pilot and register the receding successor.

Starting C, with new templates only. Begin wall-path/cycle tracing with interior
spur pruning, then anchor rotation, frame/interior swap, and joint size/color/holes
context mapping. Synthetic niche validation precedes an ARC selection-only oracle
ceiling; no report tasks will be consulted for template selection or maker prompts.

23:43Z C wall findings: constant turn-priority `wall_follow2` solves its generated
family but its 48 turn/start oracle variants fail the full declared DEV task
cc9053aa. Pure shortest hazard-free path `wall_avoid3` matches all three
demonstrations but fails its DEV test. Both remain frozen. New `wall_pruned4`
adds exclusion around hazard colours and dead-spur pruning before tracing:
24/48 turn/start variants solve the entire cc9053aa DEV task (demonstrations plus
test). This is a restricted oracle diagnostic, NOT maker evidence or an exhaustive
ceiling. Native niche `wallpruned` registered additively in scripts/niches.py.
Maker run `weights-wall-pruned4` is next; 48 generated development cases.
No ARC inputs/outputs are in maker prompts.

2026-10-02 00:20Z: Continuing B v2 after eight valid attempts. Strategy change:
show the maker its last rejected literal tensors and six predicted-vs-expected
failures, including when no founder was admitted, alongside accepted-parent
failures. Previously the prompt omitted rejected attempts, so C anchor attempts
repeated wrong colours without feedback (zero development solves). No report
cases are used. Existing templates remain frozen. Request continued lane credit
only for fresh valid charged submissions. Required niche/hard-lane tests: 69 OK.
Continuation is serial: finish B, then anchor C rejected-attempt feedback, then
corridor C. No concurrent own maker drivers. STANDARD/semantic gate latest:
25 tests, failures=5 errors=1; missing local A files copied for recomputation,
no lineage/seed/budget history rewritten. A perfect score is not promised.
Selection diagnostics (not report evidence, not independent aggregate scoring):
maker wall_pruned4 passes all 3 demos + cc9053aa DEV test; maker frame_corner3
passes all 3 demos + 952a094c validation test. Maker reference_paint3 passes
5/6 pairs but one leave-one-out demo has an unseen reference key. Corridor
marker oracle only passes 2/5 f1bcbc2c pairs, so it does not express the full
selection rule yet. Preserve these nulls; do not claim C complete. B generation
10 champion remains40/48. Continued serial generation is active.
