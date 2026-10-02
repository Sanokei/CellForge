# Document understanding crossover protocol v2

MMLongBench-Doc is the selected demanding document benchmark, without a claim
that it is universally the hardest. Freeze the official repository at commit
`d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384` before running comparisons.
[Official source](https://github.com/mayubo2333/MMLongBench-Doc/tree/d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384)
describes long PDFs with text, tables, charts, images, cross-page questions and
unanswerable questions. The dataset was updated after its initial publication;
historical published scores are not matched contemporary frontier evidence.

## Isolation and frozen inputs

The independent operator must hash the annotation file and every PDF and retain
an inventory before development begins. Assign whole document IDs with
SHA256(`cellforge-mmlongbench-doc-v1:` + exact doc_id), interpret the first eight
bytes as an unsigned big-endian integer, modulo five: zero is heldout and the
remaining residues are development. This is a custom document-group split,
not a claimed official test split. All questions for a document share its split.
Protocol revision v2: Duplicate PDFs identified by identical SHA256 bytes share
the lexicographically smallest exact document ID as canonical ID. Hash that
canonical ID with the same rule and assign ALL associated questions to its split.
The original per-ID inventory was rejected after detecting one duplicate group
crossing splits; retain that rejected inventory and receipt. The v2 grouping is
applied before any development or evaluation and creates no target-informed split.
Related editions must be audited and grouped before development.

The pinned dataset is staged privately on the evaluation server: 135 PDFs and
1082 actual annotation questions, 666,763,962 downloaded bytes. The README's
historical count is not substituted for the pinned annotation inventory. The
original per-document split audit was rejected for one identical-PDF cross-split
conflict. Canonical reassignment and all-page screening completed privately:
133 unique byte groups became 117 conservative family components, using all-page
text fingerprints for 105 groups and all-page raster fingerprints for 28 groups.
All flagged pairs now share a split: 110 development and 25 heldout document IDs.
Similarity screening cannot prove absence of every related edition; the full
split remains **unfrozen**. Authors have not loaded
dataset annotation items, answers or PDFs. Published README schema
examples are source documentation, never inference inputs or scoring targets.
Heldout labels, evidence pages, answer formats, evidence sources and per-item
scores remain outside the author process. Inference receives only document ID,
question and all page images. The operator renders PDFs identically for every
arm and freezes renderer version, resolution, color handling, page ordering,
image hashes, question byte serialization and resource limits before evaluation.
There is no target-guided page selection, OCR repair or host answer construction.

## Crossover and evaluation gates

1. Parents must expose the same versioned document system interface. An integrated
   graph may use `document-question-answer/1` and a `document_answer` entry. A modular
   system may instead use `composed-document-system/1`, freezing every participating
   graph hash, OCR crop/raster policy, token codec, observation serialization and
   resource budget. The actual language graph must consume question tokens and
   ordered page observations and generate its own answer tokens or abstention.
   A single internal document graph is not required. Declaration alone does not
   establish behavioral readiness. No Python question matching or host answer
   construction is allowed in either alternative.
2. Components may cross only when tensor shapes, pixel/token conventions, state
   semantics, output vocabulary and referenced parameter closure match. Every
   imported definition and numeric parameter must have explicit donor lineage.
   Reject dangling references, implicit vocabulary remaps, incompatible state
   dimensions and hidden external inference. Save parent/child hashes and proposal
   receipts before evaluating children.
3. Run candidate, recipient parent, donor parent and no-specialist/zero-observation
   controls on identical development documents. Include shuffled observations to
   detect whether outputs actually depend on relevant page evidence. Keep all
   failures and resource costs. Selection uses development only.
4. Freeze one selected child, all parent controls, system implementation, prompt,
   tool registry and preprocessing. An independent operator evaluates heldout
   documents once and exposes aggregate metrics with document-level uncertainty.
   Repeated selection on heldout results invalidates this confirmation set.
5. A superiority claim requires identified frontier model/version endpoints under
   the same document inputs, questions, answer protocol and budgets. No frontier
   endpoint has been selected or executed here; published historical results do
   not satisfy this gate. Record failures, truncation, latency and cost alongside
   scores. No paid API or credential access is authorized by this protocol.

## Metric

The official repository requires GPT-4o for extracting short answers from model
responses and its evaluator; that official metric is **not executed** here.
No replacement is labeled official F1. The prepared local **custom exact short
answer** scorer is case-sensitive equality after stripping outer whitespace,
with invalid/non-string output counted incorrect. It performs no model-based
normalization, number tolerance, synonym matching or equivalence judgment.
Every arm must emit the same short-answer format; inability to do so is a model
failure once an actual evaluation is launched, not a reason to repair output.
Until the document input interface exists, results are **incompatible/unscored**,
with score `null`, not a fabricated zero.

## Current readiness

The composed adapter is implemented in `src/cellforge/runtime/document.py`.
`cellforge document-answer --artifact LM_DIR --ocr-artifact OCR_DIR --page PAGE.png
--question "QUESTION"` accepts ordered raster pages (repeat `--page`) and a
question. It preserves pixel scale, tiles every page, and inverts dark-ink
grayscale crops for the legacy bright-ink OCR contract. It never extracts PDF
text or chooses evidence pages. Overflow fails explicitly. `--observation-mode
withheld` removes observations; `shuffled --shuffle-seed N` moves recognized
content between fixed page/crop coordinates. The whole composed system and
these preprocessing/control settings must be frozen before evaluation.

Runtime tests include graph-executed digit recognition with a constant-EOS
language fixture that answers with an empty string. That fixture establishes
question/observation transport only, not document QA. Nonoverlapping tiles may
cut glyphs, and the legacy OCR vocabulary cannot recognize alphabetic text.

The optional `--observation-format compact` uses schema
`document-ocr-observation/3`: every crop retains its page, coordinates, source
dimensions, recognized text and failure. Hashes and duplicate byte IDs remain
in operator observations rather than the language prompt. The question appears
last. This changes the input representation and must be frozen as its own arm.
The same-item diagnostic reduced context from 783,438 to 185,055 bytes but still
timed out at 120 seconds. It supplied no benchmark score or winner selection.

`--backend compiled --state-only-prefill` is a separate optional execution arm.
During prompt ingestion it executes the dependency closure of recurrent state,
retaining full execution when any active batch row needs token logits. Reference
execution remains the default, and teacher-forced scoring is unchanged. Tests
compare states, logits and generated tokens; recorded speedups apply to measured
fixtures only. Counter scope differs when discarded prompt logits are omitted;
constant caching reports semantic operations rather than physical recomputation.
Freeze the runtime source and flags separately from observation format. Runtime
speed does not establish reasoning or document-answer quality.

`runs/document-crossover/readiness-v3.json` records initial graph hashes and complete
source receipts. It predates the modular adapter and alphabet-expanded OCR family
and remains historical interface evidence. Subsequent server receipts preserve
staging, family grouping and raster budget audits.
The original OCR parents recognize digit-only 32x128 line rasters and lack
page layout and question conditioning. Alphabet-expanded candidates are now
available but remain unscored on documents. The byte LM
accepts text but its actual synthetic tool-action probe produced three invalid
actions and no calls or responses. Numeric observation projection tests are typed
finite-state component checks, not document understanding.

Development raster feasibility at 72 DPI audited 106 then-provisional canonical
development PDFs: 4710 pages and 547407 crops. Of these, 65 fit default input
budgets; none had at most 256 crops. This audit predates family reassignment.
Batch OCR and compact observation schema `document-ocr-observation/2` were frozen
in development policy v2 with batch size 16 and 120 seconds per arm. One complete
development document/question was evaluated with actual, withheld and shuffled
observations: actual and shuffled timed out; withheld processed 2125 crops in
133 batches, exhausted 48 answer tokens and failed custom exact match. All three
custom exact outcomes were incorrect. This is a one-question negative development
probe, not a full benchmark score; timeout phase was not instrumented in that
frozen snapshot. No heldout inference or paid answer-extraction judge ran.
`runs/document-crossover/server-development-probe-v2.json` preserves full numeric
receipts; `development-summary-v2.json` provides compact aggregate evidence.
The same-model, same-item, same-rendered-pages actual-arm diagnostic subsequently
retained the same 120-second limit and timed out in language processing. All
2125 OCR crops completed in 133 batches in 1.7213 seconds; the language phase
began at observer elapsed 2.1396 seconds with 783438 context bytes. Raster rendering
took 1.6596 seconds. The unfinished language timer is recorded as zero and must
not be interpreted as zero language computation. This is bottleneck evidence,
not a score-improving rerun. See `server-phase-diagnostic-v3.json`.
The separately frozen compact schema/3 diagnostic retained the same models,
development question, rendered pages, batch size and 120-second worker limit.
It reduced language context from 783438 to 185055 bytes (about 76.4%) while
retaining every crop's text, failure and coordinates, but still timed out in the
language phase. OCR completed all 2125 crops in 1.7736 seconds. This input-schema
intervention included question-last ordering and did not use a compiled backend.
No benchmark or custom score was assigned to the diagnostic. See
`server-compact-diagnostic-v4.json`; the runtime bottleneck remains unresolved.
The separate compiled-runtime diagnostic retained compact schema/3, the same
question, models, rendered pages and budgets. Compiled language execution with
state-only prefix prefill also timed out at 120 seconds in the language phase;
all OCR crops completed in 1.7766 seconds and context remained 185055 bytes.
Language counters and generated-token count are unavailable because generation
did not return. Demanded-closure operation counts, when available, must not be
compared directly with full-eager counts. The microbenchmark acceleration did
not establish completion within this full-document budget. Preserve
`server-compiled-diagnostic-v5.json`; no diagnostic score or model selection was
assigned. Prefix/generation progress profiling is still needed.
The next meaningful experiment requires
evaluated page/question/answer behavior with broad recognition and cross-page state.
Digit OCR breeding or assembling specialists behind an interface cannot by itself
establish this capability.
Run `python scripts/check_document_readiness.py` to create a new immutable audit;
the checker loads graphs and source files only, never benchmark annotations.
