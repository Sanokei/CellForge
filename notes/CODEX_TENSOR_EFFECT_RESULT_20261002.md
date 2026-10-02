# Shared tensor effect/recovery checkpoint, 2026-10-02

Two reusable cells now use a FIXED generic runtime and LLM-authored numerical tensors, not maker-authored task programs. Each candidate has effect[8], recovery[8], router[4], rates[2], click[4]: 26 literal coefficients. The same coefficients/source operate on all games. Online state estimates action-conditioned frame change/no-op rates; coefficients themselves remain fixed. This is online statistical adaptation, not demonstrated weight-learning across generations.

## Executed provenance
- First one-call batch at16:23:59UTC: GPT-6.1 Sol low, ChatGPT subscription via existing maker.py,21.333s. Both candidates rejected because click offsets exceeded the predeclared +/-20 bound. No clipping, no play. Two proposals/52 positions charged.
- Separate contract-only repair at16:26:10UTC: same route,15.759s, two valid fresh numerical candidates. Two proposals/52 positions charged. No game feedback in repair prompt. Raw response exactly matches saved CANDIDATES.json.
- 33 plays initially completed in19.155s, but ar25 exposed action7 outside the runtime's seven-row storage. Preserved as invalid ar25 evidence.
- Generic dynamic legal-action storage fix, regression-tested in the sandbox. NO new maker call or weight edits. Replay of the identical frozen candidates/selected B completed16:31:38UTC:33 plays,25.759s evaluation,9900 maximum actions. No crashes. Original selection preserved; reporting was already exposed and is NOT newly untouched.
- Latest server check16:32:39UTC: no effect-cell/Codex/sandbox research processes. Meter63%, cap85%. Existing campaigns/STOP files untouched. No API, new server, dataset download, upload or excluded ARC2 evaluation access.

## Actual results (300 actions/game, reporting seed1, public development games)
| arm | ar25 levels/noops/distinct | lp85 levels/noops/distinct | sc25 levels/noops/distinct |
|---|---|---|---|
| authored B |0/4/253|0/285/14|0/74/173|
| zero |0/295/6|0/300/1|0/108/34|
| effect off |0/2/246|0/285/14|0/76/181|
| recovery off |0/4/281|0/285/14|0/74/173|
| swap |0/295/6|0/285/14|0/108/34|
| random11 |0/35/217|0/294/7|0/44/187|
| random12 |0/295/6|0/293/8|0/108/34|
| random13 |0/72/163|1/280/19|0/55/222|
| blind |0/68/121|0/285/14|0/61/108|

No authored level gain; random13 wins lp85. Lower no-op counts and zero/swap sensitivity establish behavioral dependence on numbers, not useful task learning. Recovery-off retains nearly all full behavior; its unique contribution is not established. Effect-off also solves no level. No candidate admitted/promoted, no generalization/AGI claim. This is two contributions inside a shared policy, not independently evolving cell/MoE populations.

## Gates and limits
New tensor/anti-substitution tests pass16/16: numeric finite shape bounds; code/program/task/sample/lookup rejection BEFOREplay; actual no-op/meaningful transition execution; zero/swap behavioral sensitivity; literal-response/prompt/model/runtime/parent preflight; no-change revision rejection; fixed tensors across tasks; irreversible development-only selection seal; forged report acceptance rejected.
The independent worker's seven assertions were preserved. Its positive maker fixture was updated to produce the real maker START receipt; stale prompt/parent negative fixtures remain unchanged. The stale acceptance defect is fixed before play. The first valid run predates this preflight patch; the runtime replay executes it, and earlier evidence is labeled postflight-only.
Required ONE_STANDARD/semantic suites remain red; saved full log reports exact outcomes. Pilot STANDARD records truthfully declare missing population/crossover/equal charged random evolution qualification. No tests/floors were weakened. Random tensors are ablations, not matched charged evolutionary proposals.

Files: scripts/arc3_tensor_effect_cells.py; scripts/run_arc3_tensor_effect_cells.py; tests/test_arc3_tensor_effect_cells.py; tests/test_arc3_tensor_effect_antisubstitution.py; notes/CODEX_EFFECT_CELL_OWNERSHIP_20261002.md. Evidence in runs/weights-arc3-tensor-effect-recovery-{repair-,runtimefix-}20261002 and failed firstbatch directory. No shared campaign or report-script/manuscript edits.

Next scientific gap is useful capabilities, not numeric authorship: this representation learns whether an action changes pixels, but cannot infer goal relevance, spatial action grounding or delayed consequences. It still has generic affine click schedules and only per-action aggregate memory. Any follow-on needs a separately declared development protocol/cap; no automatic 16-call expansion has been launched.

Final test replay:41 tests in the final log;16 new tensor/anti-substitution gates passed. The required suites retain2 failures (charged random-arm budget, insufficient declared seeds) and3 errors (missing semantic evaluator) from existing shared-probe qualification. A transient third failure was a missing per-founder materialized weight reference in the new runtime replay; both files now contain the exact already-saved tensor dictionaries, and that failure is fixed. Final log: reports/arc3-tensor-effect-gates-final-20261002.log. No numerical edits, new maker calls or altered outcomes occurred in this fix.

Revision guard follow-up: validate_revision now requires the actual nonempty prompt text and checks its UTF-8 SHA256 against the receipt, alongside the exact parent digest. A 64-character unrelated digest, wrong prompt text, missing/blank prompt, wrong parent, and unchanged coefficients are rejected. The valid numerical revision fixture passes. No provider call, game evaluation, tensor changes or historical-record rewrite. New suite17/17 passed; combined mandatory replay42 tests retains2 failures/3 errors. Log: reports/arc3-tensor-revision-prompt-gates-20261002.log. Updated isolated guard synchronized to the server.
