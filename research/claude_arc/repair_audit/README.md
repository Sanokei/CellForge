# Execution-guided repair: next experiment

Owner direction, 2026-09-30: continue the maker/breeding/reusable-cell path after Claude's agents stopped. Keep gradients, gradient estimators, neural learned weights and LLM calls out of runtime. The maker remains this offline authoring session.

Primary-source reading:
- ENPIRE: https://arxiv.org/html/2606.19980v1 . Section 2 freezes verification/reset APIs after human-guided construction, then improves policies using execution evidence. It supports heuristic code policies alongside gradient-based methods; only the code-policy mechanism fits our rules.
- ASPIRE: https://arxiv.org/html/2607.00272v1 . Sections 2.1-2.3 use primitive traces, targeted code repair and reusable skill knowledge; section 3.3 separates debug and evaluation configurations. Some tracks generate code during execution, so the entire architecture is not a drop-in pure runtime.

Neither paper establishes AGI or that reward exploitation causes its gains. Distinguish productive feedback-guided optimization from exploiting a proxy verifier. Randomization is a generalization test/curriculum ingredient, not sufficient evidence of capability creation. This is our interpretation, not a causal result isolated by these papers.

Coordinate the graph-level maker experiment with the active `Continue MoE AGI experiment` chat. This directory owns Claude-side evidence and the protocol; it must not overwrite shared core runtime or that chat's running work.

Predeclare before a new scored maker round:
1. Freeze a deficient parent, existing cell library, task generator, scorer and proposal budget. Choose the failure family from development only; retain 3-5 executable input/expected/predicted examples and actual cell traces.
2. Compare breeding-only, random structural repair and offline maker structural repair at equal candidate evaluations. Count rejected/invalid proposals and every maker edit. Report tokens/cost as unavailable if unmeasured, never as zero.
3. Separate library-slot repair from new-cell authoring. A known bank gene restoring a deliberately removed slot is an engineering control, not new capability or an unbiased maker superiority test.
4. Freeze candidates before reporting on fresh seeds and separately specified structural variants. Keep previous-family regression checks. No public ARC evaluation or sealed/private data access.
5. Report exact whole-output success, demo-consistent query failures, exceptions, cost, accepted operator type and lineage. Pixel similarity is a diagnostic proxy; it cannot admit a child that fails the exact objective.
6. Require improved fresh-task exact success without regression/false-positive increase for admission. If proxy reward rises but exact success does not, label proxy exploitation or overfitting, not intelligence gain. Generalization success remains necessary even for clever solutions.

Success criterion: maker beats random repair on predeclared fresh structural variants across matched seeds with no regression. A single hand-authored cell beating one random edit is only a mechanism sanity check. If improvement is confined to shown examples or known-slot restoration, continue diagnosing transfer rather than claim the general loop works.

Completed Claude evidence: multiview development-selected arm improves validation 29 to 31 of 228, FP unchanged at 5; generator coverage reaches 174 top-1 successes in 1,260 synthetic episodes; full ARC3 v5 seeds clear 15,16,15 of 190, with mean scores approximately .190,.139,.189. No new ARC evaluation was run.
