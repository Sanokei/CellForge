# Manual maker operator protocol

`src/cellforge/construction/maker_operator.py` implements proposal accounting,
not an LLM client. The existing assistant supplies authored transaction proposals
manually. Record actor, model and method (`manual-llm-maker` or
`non-llm-mutation-control`); optional session fields may supplement provenance.
Never imply an automated model service generated a manual edit.

The caller supplies the exact diagnostics shown, explicitly marked
`partition=development`, `evidence=train-pairs`, with corpus/operator provenance.
Per-task demonstration input/output/error diagnostics are authorized. Audit,
final, sealed, held-out and test fields are rejected recursively. This is a
schema guard, not an information-flow proof: operators must never disguise
reserved labels under allowed field names, and must not provide reserved items
to the assistant. Operator scoring callbacks expose development aggregates only.

Every serializable submitted edit consumes one evaluation-budget unit, including
invalid transactions, runtime failures, rejected candidates and no-change edits.
Input serialization errors are API failures before a proposal exists. The JSONL
receipt contains exact submitted operations, diagnostics, provenance, parent and
candidate hashes, profile/stats, status, error and hash-linked previous receipt.
Only validation-passing changed candidates reach the score callback. A returned
candidate is not automatically promoted: consult its status. Accepted/rejected
are explicit development decisions; `validated-unscored` is not benchmark evidence.

Transactions reuse `apply_ops` on a copy, including checked `import_def` resolution.
Graph validation, an explicit structural profile and mandatory runtime fixtures
check parameter usage and declared output ports/shapes/finite values. These are
fixtures, not hidden-answer checks. Supply representative demo/query inputs and
contracts for grid state, dimensions and both attempt logits. A single fixture
does not prove every possible shape or generalization. The primary protocol
requires bounded resources in the private evaluation process; this API does not
itself enforce a process timeout/RSS sandbox.

Compare maker and non-LLM mutation arms with the same frozen founder, allowed
cells/transactions, diagnostics, proposal-attempt count, scorer and compute caps.
Count assistant proposal attempts even when they fail; include maker inference
cost/time separately. Neither arm uses gradients or fitness-weighted ES estimates.
ARC values must carry authored provenance; language count fitting is a distinct
authorized protocol and must not silently become an ARC seed.

Freeze the exact selected graph before final access. `sealed_once` consumes the
single access token before invoking the evaluator, so a failed call cannot retry
under this ledger. Final aggregates remain outside maker diagnostics and cannot
select more children. The ledger is a local accountability tool, not secure
multi-process locking, tamper-proof storage or enforcement across new ledgers.

Biological ideas remain hypotheses: demo-gated Hebbian state plasticity must
freeze at query time; fixed-slot developmental blueprints may reuse modules;
clonal offspring may target recorded remaining demonstration error; graph-owned
winner-take-all inhibition may produce two ranked outputs. These features are
not implemented by this operator. Existing NCA prototype has 48 cells/1,488
parameters and local tests only; local calls/second are not server ARC scores.

### Breed-then-repair accounting hardened (2026-09-30)
The three-arm interface snapshots serializable repairs before scoring. Missing fields or malformed individual repairs receive invalid attempt receipts while later arms continue. Every receipt distinguishes proposal attempts from actual evaluator calls; unchanged baseline/control edits are scored in this protocol. Evaluator exceptions consume a call and retain CPU/wall timing. Reopening a ledger verifies every previous-receipt link and payload hash before permitting further operations. Eight focused maker/cache tests pass, including malformed-repair continuation, evaluator failure accounting, replay and tamper rejection.
These are interface invariants, not measured maker improvement. Equal attempted edits can have unequal physical evaluations if invalid edits are rejected before execution; reports must show both. Whole-run CPU/RSS and maker authoring cost still require experiment-level measurement. Fresh development/archive/real-anchor acceptance remains a caller obligation. Before the expanded-genome repair experiment, transaction lineage must update the declared alleles together with slot definitions; generic put_def alone can leave expanded_genome metadata stale. Do not silently archive such candidates as validated genotypes. No native/evaluation/audit/final data accessed this turn; no gradients, downloads or paid services. Independent matched frontier superiority remains unproven.

### Atomic typed-slot repair (2026-09-30)
Added set_slot_gene transaction for the frozen seven-slot expanded genome. It changes the declared allele and its definition together, records the immediate parent hash and old/new gene, and removes stale crossover donor/seed metadata when a slot changes. Same-gene edits preserve artifact identity. All transactions now check expanded allele/wiring agreement before and after applying operations, so generic put_def cannot silently produce stale genome records. Operations apply to a copy; invalid slot/bank choices leave the parent unchanged. Nine expanded-genetics/maker tests pass, including canonical compatibility, neutral identity and rollback. This resolves the previously recorded lineage-interface gap; it is not a maker-effectiveness result. The measured three-arm repair experiment remains outstanding. No gradients/native/evaluation/audit/final access, downloads or paid calls. Frontier superiority remains unproven.

### Paired repair admission gate (2026-09-30)
Added construction/repair_acceptance.py as a development-only admission policy helper. It requires strict fresh-niche gain, per-task nonregression on every archive and real-public-training anchor outcome, matched manifests/task populations/source labels, defined nonincreasing false-positive rates, and passed cap/fixture flags. Aggregate ties cannot conceal losing an earlier solved task. Undefined consistency rates reject admission rather than inventing zero. Eleven focused admission/maker/expanded-genetics tests pass, including regression, population substitution and reserved partition rejection. This helper is not yet connected to a scored archive experiment; source labels and caps are caller attestations, not independent enforcement. The previous synthetic pilot remains rejected/unadmitted, no retrospective acceptance. No final/evaluation/audit access or gradients; frontier superiority unproven.

### Admission policy connected to maker lineage (2026-09-30)
MakerOperator.submit and breed_then_repair_round now accept paired admission_parent evidence. The ledger stores the exact parent scores, candidate metrics, gate result and explicit rejection reasons. A permissive caller accept callback cannot override archive/anchor/false-positive regression; it can only further restrict a passing paired admission. Legacy unpaired callbacks remain available for existing diagnostic pilots and do not establish full archive admission. Twelve focused tests pass, including an integration case where an apparent fresh gain with a real-anchor loss is rejected despite accept=True, followed by matched nonregressing evidence and verified ledger replay. This is measured interface behavior using fixtures, not a model-performance or real-anchor experiment. The synthetic pilot remains unchanged and unadmitted. Full scored development admission and repaired-cell capability expansion remain outstanding. No gradients, final/evaluation/audit data or paid calls; frontier superiority unproven.

### Maker diagnostics now contain executed cell evidence (2026-09-30)
Added construction/maker_diagnostics.py: bounded100development tasks/five counterexamples, train-only schema rejects query-answer fields; reference interpreter records actual cell-definition invocation counts for every demonstration/query step. Diagnostic predictor calls and graph/source hashes are logged. Per-call traces reset; focused test verifies prediction parity with compiled output, actual cell calls and rejection of test-bearing task dictionaries. Three original tiling train-pair failures at seed3297000 recorded in runs/maker-cell-diagnostics-v1/diagnostics.json, three charged diagnostic calls, wall0.248s. These replace the missing cells_fired evidence for future edits, not retroactively for the prior pilot. Trace scope is reference execution by cell definition, not compiled physical work or full node paths; route statistics remain available through the adapter. Input provenance is caller-supplied, so schema does not prove disguised data are safe. No gradients/public evaluation/audit/final/native files/download/paid calls. Next full repair runner still requires new-cell proposals, fresh/archive/anchor admission and isolated fixture/cap receipts. Frontier superiority unproven.

### New-cell repair vocabulary and charged transaction (2026-09-30)
Added import_slot_gene: exact donor hash/no-gradient marker, typed grid cluster contract, collision-safe import_def, one bounded seven-to-eight bank expansion, atomic slot activation and parent/donor/imported-definition lineage. A second expansion is rejected; errors leave the original parent unchanged. Twelve genetics/maker/diagnostic tests pass, including actual graph routing of imported tiling, mismatched-hash rejection and rollback.
Recorded one manual maker edit using three previously logged tiling failures with actual cell-call diagnostics. New candidate b96a430e12254af09b2da4ec4b81df050c9301520b55d291c3f82e2d8bc01a2e reconstructs3/3 fresh development demonstration pairs; one charged edit/three predictor calls, typed fixture passes, ledger replay verified. Candidate retained at runs/maker-new-gene-transaction-v1/candidate but explicitly rejected/unadmitted because full archive/real-anchor/cap gates were not supplied. This is a known authored donor absent from the initial bank, not independent maker generation or a comparative repair win. New bank versions differ from their parents, so canonical comparison must use a shared new-bank control, not silently compare incompatible bank hashes. No gradients/final/audit/public-evaluation/native-data/download/paid access. Full equal-budget new-capability repair experiment and independent frontier comparison remain outstanding.

# Frozen new-gene fixture/resource receipt

Candidateb96a430e12254af09b2da4ec4b81df050c9301520b55d291c3f82e2d8bc01a2e. Reference/compiled exact parity, finite outputs and typed state/dimension/color ports passed at1x1and30x30 for demo/query phases. Same graph hash and current source hash verified locally; external resource exit0. Frozen policy limits:1024stored parameters,3GiB address space,4GiB device RSS,60s fixture wall; nice15,threads1. Measured wall0.341s,peak RSS83288064bytes. Source archivebc76277877bddb7974d94174c799f599d6f974f4baaee90938073605a0e93510.

Policy declared before this fixture run, after candidate authoring. Not a predeclared repair-effect experiment, not arbitrary-grid correctness or query throughput. Candidate still unadmitted pending fresh/archive/real-anchor evidence; do not alter prior rejection receipts. These graph-bound fixture/cap receipts can be referenced in a subsequent declared admission protocol. No benchmark labels, gradients, evaluation/audit/final data or paid calls.

### Fixed-rank selection option and owner scope clarification (2026-09-30)
Added --rank-consistent-selection to matched population runner. One fixed output rank must reconstruct every complete demonstration after all task adaptation; all demo calls still execute, same budget. Frozen policy records the flag and exact fitness wording. Legacy default preserved for prior receipt comparability. Constructed two-demo rank-switch predictor passes legacy but fails strict; stable-rank predictor passes. Nine runner tests pass. Four actual repaired/crossover graphs on six original train-only tasks retain scores2,4,6,0 under both methods. No evidence this loophole caused prior gains.
Initial v1 diagnostic incorrectly compared normalized tuple grids to list targets and reported strict0; v2 normalizes both, supersedes those invalid scores. Retain v1 as a harness failure, never use as model evidence.
Owner clarification relayed through the co-researcher: offline maker may directly author numeric weights as well as cells/wiring; existing set_param remains permitted. No code-only restriction is adopted. Maker parameters stay fixed; artifacts and harness/playbook improve through feedback. Reward hacking was intended as clever feedback-guided improvement, not scorer exploitation. Next comparative protocol must test iterative feedback-guided authored weight/cell/wiring proposals, not just known-slot restoration, against equal-budget random edits and breeding alone, with fresh exact outcomes/false positives/lineage. No gradients or gradient estimators/runtime LLM/download/sealed access. Prior coordination note's proposed code-oriented examples do not limit owner-authorized numeric editing. Frontier superiority remains unproven.

# Two-round authored numeric-weight prototype

37stored parameters: source/marker masks and target plus four direction weights and scalar bias. Initial OR gate fails a new horizontal-two-marker target (5/16). First manual numeric edit bias-1.5 reduces one-neighbor false positives (9/16); second writes direction weights[0,0,1,1], giving16/16. Each round logs concrete train failures, actual reference cell calls and literal numeric transaction. Fixed maker model parameters; no gradients or estimators. Artifact weights—not the maker—change.

Two charged proposals per path,32physical selection predictor calls per proposal. Frozen final report on new backgrounds: neutral-founder path5/16; random path8/16; authored path16/16. Three two-entry receipt chains replay-verified. The author knows the generator rule; this is an execution prototype, not a blinded experiment or evidence that feedback improves the maker. Logical masks are the same16patterns across selection/report; background randomization alone is weak generalization. No real ARC/structural-transfer test. Breeding-only path represents identical OR founders and is neutral, not a meaningful diverse-parent crossover baseline. Paths follow every proposal without selection/rollback; random round1was12/16, so final8/16 is not its best-seen score. No candidate archive admission or superiority claim.

v1 failed its first charged fixture attempt because input/output masks were missing. v2 fixed masks but used short-circuit all(), making its declared32selection calls inaccurate. Both retained as harness/protocol failures; v3 evaluates both demonstrations physically and supersedes the call accounting. No hidden-data reuse. Before a causal three-seed study, add diverse contemporaneous parents, per-round feedback/no-feedback controls, truthful authoring cost, selection/rollback, structural report tasks and full admission caps. Numeric writing stays within owner scope.

### Charged numerical recipe transactions, September 30
Added numeric_recipe transaction {from: exact donor graph hash, recipe: explicit numerical edits}. It resolves the donor, checks exact hash, then applies homologous tensor masks or matrix-row scaling atomically. No numerical values are estimated or fitted. MakerOperator.submit and breed_then_repair_round inherit the transaction through apply_ops: missing/stale donors consume attempts with no scoring callback; valid edited graphs receive typed fixtures/profile checks and one charged development evaluation. Existing admission callbacks/gates, immutable freeze and once-only report path remain in force. Candidate construction metadata is unadmitted until archive policy approves; constructing a recipe is not an evaluation or admission.
Sixteen focused numeric/maker tests pass. Integration checks missing resolver and stale donor both charged and rejected, exact resolution evaluated once, conservative callback rejects admission, chain reload verifies, freeze and once-only scoring remain enforced. Initial fixture failures (cell passed directly to bundle interpreter, then missing ladder role) were corrected in test setup; no experiment was run or restarted. This extends actual numerical editing beyond slot-choice operators; it does not establish a maker or evolutionary gain. Prior30/30 known-rule fixture remains historical and is not rescored.
No gradients/estimators/runtime LLM/native/evaluation/audit/final/download/paid access. Full goal active; independent matched frontier superiority remains unproven. Next capability experiment still requires diverse bred parents, feedback-guided numeric proposals, matched random/no-feedback controls, fresh report tasks, rollback and full archive/resource gates.

### Staged iterative numerical repair, September 30
Executed scripts/run_staged_numeric_repair.py in separate prepare/round1/round2/report stages. Two distinct authored parents (up+left versus down+right) are numerically recombined into an OR-gate child. Train-only traced failures are saved before each maker file is manually authored, with exact diagnostic hashes and rationale. Round1 maker bias-1.5 removes one-neighbor false positives; round2 direction weights[0,0,1,1] suppress vertical contributions. Random uses the same edited tensor and discrete value vocabulary each round, one tensor edit per proposal. Baseline is unchanged charged bred child. Three seeds774101/774111/774121, two proposals per arm per seed; strict batch-A pixel-error improvement with batch-B nonincrease, otherwise rollback. Both disjoint selection batches use full train pairs, no query labels.
Frozen fresh synthetic report: maker60/60 (20,20,20), random6/60 (1,1,4), unchanged0/60. Demo-consistent wrong report tasks maker0/random1/unchanged0. Each arm six candidate evaluations and240selection predictor calls; selected edits maker6/random3/baseline0. Initial common baseline measured once per seed outside matched proposal budget; diagnostics and final report work separately accounted. Round1 random outperformed maker on two selection seeds: preserve this negative intermediate evidence. Maker is informed about the generator rule; this is a known-family feedback/rollback execution pilot, not blinded discovery or proof that the maker itself improves. Random receives common maker-path diagnostics in its ledger but does not use them; diagnostic graph hash correctly identifies the maker-path artifact and can differ from random-path parent.
All nine four-entry chains reload and verify;18 candidate graphs reconstructed from transactions, exact parent/candidate/selected/frozen hashes checked, selection metrics recomputed, decisions/diagnostic commitments verified. Report records recounted without rescoring report data. Selection CPU totals baseline.484375/random.46875/maker.453125seconds. Report process wall1.198552seconds and peakRSS35524608bytes, local Windows only; per-seed inference tasks/sec and CPU/task in RESOURCES.json, not server throughput. No whole-stage peak or maker-token cost measured. Frozen graph stores37parameters; no full native-anchor/archive/cap admission asserted.
Receipts runs/staged-numeric-repair-v1/policy.json, maker-1.json, maker-2.json, result.json, VERIFIED.json and RESOURCES.json. No real ARC/native/evaluation/audit/final data, downloads, paid calls, gradients or estimators. Report is authored synthetic development, not a sealed independent final benchmark. Family logic is already exposed from earlier prototypes; new structural samples do not demonstrate independent ARC transfer. Frontier superiority remains unproven and goal active. Next required evidence is feedback-guided numerical/cell repair on failure-derived real training families plus full regression/resource gates and broader independent controls.
