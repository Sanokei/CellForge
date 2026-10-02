# Current Red Queen Repair integration checkpoint, 2026-09-30

No new training, 1D rerun, paid call, model download, server upload or deployment.

Installed `src/cellforge/construction/direct_weight_maker.py` and
`tests/test_direct_weight_maker.py`. The numeric maker wraps the existing
`MakerOperator`, requiring literal reachable-cell weight edits with base hash,
actor/model, rationale, tensor shape and actual changed scalars. Scope supports
cells, router or joint edits. Rejected/unscored proposals do not become parents.
Authored assembly requires explicit readiness evidence; donor import receives
the existing transaction resolver. Optional affine router calibration starts
as identity on existing scores, retains expert-index metadata, and is not a
contextual router or demonstrated gain.

`DirectWeightMaker.compare_repairs` connects supplied proposals to the existing
charged maker and paired fresh/archive/real-anchor admission machinery: three
arms share one child, proposals freeze before evaluation, three budget units
are required, and unchanged baseline is evaluated. Evaluator callbacks must
truthfully enforce data provenance and measured resource caps. The integration
fixture uses synthetic scalar outcomes, not ARC or LLM training. This is an API
integration; an end-to-end diverse-skill competition runner remains incomplete.

Verification: six focused unittest checks passed (0.699 s), and three retained
oracle checks passed (1.548 s). Existing maker/diagnostic/admission tests import
pytest, which is absent from this Python runtime; those suites did not run.
No dependency installation performed. Manuscript LaTeX was not rebuilt.

## Competitor and seeds

Repository notes/docs/config/manifest search found Gemma references in
`docs/SYSTEM12_DESIGN.md`, `docs/ARC_PRIZE_2026_PLAN.md`, and historical notes,
but no Gemma-derived distilled checkpoint manifest. SYSTEM12 explicitly says
design only, no weights downloaded or run; `notes/HANDOFF_TO_CODEX.md` says
that separate track was dropped. Latest owner recollection could refer to a
different task/artifact not represented here; it is not disproved by this search.
No supported external task-record tool was used to locate that artifact.

Live opponent archive: **zero registered opponents; no run started**.
`WEIGHT_SEED_INVENTORY.json` records four loaded/validated numerical candidates:
two historical offline maker edits and two deterministic random controls from
`runs/numeric-weight-repair-prototype-v3`. Each has 37 parameters,
grid_step/recolor_root, state zero, ladder-micro/1. They are a narrow toy
specialist inventory, not a diverse archive or Gemma distillation. Numerical
weights differ; these are not slot genotypes. Historical OCR/language weights
also exist, but their task ports/state/vocabulary are incompatible with the grid
contract; they require separate typed skill adapters and matched suites, not
unconditional merging into one opponent pool. Nothing has been promoted.

The executable `research/claude_arc/evo/redqueen.py` evolves Python library
chains and generators with no numerical CellForge tensors; its README declares
no learned weights. Reusing it unchanged would repeat the easier proxy, so it
was inspected but not run or modified. Current Red Queen intent is in
`docs/RED_QUEEN_REPAIR.md`; maker APIs implement only part of it.

## Exact unresolved inputs and feasible next steps

Preferred: provide the existing Gemma-derived competitor's task/checkpoint
location, revision/hash, runtime contract and teacher-output development
manifest. Verify its actual weights and permitted resources before selecting it.
No new cost is inferred from an uncertain model name.

If no artifact exists, the owner must choose between (a) explicitly approved
offline strong-maker authored seed cells plus compatible fresh suites (existing
CPU-compatible mechanism, no claim of Gemma distillation), or (b) creating the
intended distilled competitor, which needs a pinned capable teacher, permitted
teacher outputs, no-gradient student mechanism, runtime/storage estimates and
any additional download/API approval. Neither choice is silently made here.
The known design's E2B text GGUF is 3.35 GB plus runtime overhead; the current
4 GB server cannot be assumed to fit it alongside other models. It is not
the identified missing distilled student.

Configured maker invocation: existing MakerOperator and new DirectWeightMaker
consume caller-supplied offline proposals; neither calls an LLM service.
`claude.exe` is installed, but that alone establishes no authorized unattended
subscription/API invocation or marginal cost. A real cycle must log author/model,
proposal and parent hashes, numerical changes, readiness/assembly decisions,
development manifests, physical calls/time/memory, and fresh admission receipts.
Three-arm API minimum: three charged submissions, one evaluation per valid arm;
actual corpus CPU/cost remains unmeasured. No fabricated run budget or rate.

Latest approved read-only SSH: no Python/Python3 jobs; server RAM 3819 MiB,
3302 MiB available, no swap. Local Python processes exist with unknown ownership;
none stopped. No new compute begun. Browser use/name remains unverified.

## Manuscript tangent

Edited `paper/cellforge.tex`, Discussion paragraph
"Research-assistant goal drift: substituting an easier proxy."
Documents unchanged weight-evolution objective versus delivered fixed-gene
socket proxy; cites early nine/79 literal set_param proposals and the expanded
breeding run. Labels "got lazy" as humorous interpretation, not motivation.
`paper/cellforge_clean.tex` and PDFs were not changed or rebuilt.
