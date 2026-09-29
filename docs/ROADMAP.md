# Roadmap: can direct LLM-authored construction scale beyond OCR?

Candid version. Each stage lists what must be **demonstrated** (on independent final
evaluation, with controls) before the next stage is meaningful, and the risks most
likely to stop progress there. Stages are ordered by dependency, not by calendar.

## Where we are

Stage 1 is partly done: LLM makers author glyph recognizers that are perfect in
distribution and degrade under font/size shift (0.38–0.74 exact on an unseen font).
The key observation so far: authors succeed by **transcribing what they see into
templates**. That is a real capability, but it is closer to "an LLM with vision writes
down a lookup structure" than to "an LLM discovers a learning algorithm". Every later
stage depends on whether authors can move from transcription to *generalizing
computation*.

## Stage 1 — Robust glyph recognition (digits, then 36-class alphanumerics)

Demonstrate: ≥0.95 exact on an unseen font and on size/position/blur shifts, with a
held-out font family the authors never see; alphanumeric (36 classes) at similar
robustness; replicated across ≥3 independent construction runs per arm.

Risks: templates don't generalize across fonts; authors need features (strokes,
invariances) that are hard to write by hand; confusable pairs (O/0, I/1, S/5) need
fine discrimination that hand-set margins handle poorly. **Honest expectation:**
achievable for printed monospace text; uncertain for proportional fonts.

## Stage 2 — Real-world text lines (proportional fonts, noise, scanned/photographed text)

Demonstrate: competitive CER on a public printed-text benchmark versus a small
gradient-trained CNN of similar size, at matched parameter budget.

Risks: the number of cases an author must reason about explodes (kerning, ligatures,
blur, perspective). Hand-authored weights scale with author effort, not data.
This is the first stage where I expect gradient training to win decisively unless
authors discover reusable, compositional feature machinery. **This is the critical
go/no-go stage** for the whole approach.

## Stage 3 — Words and sequence structure

Demonstrate: a constructed model exploits sequence context (e.g. a constructed
character-bigram/lexicon prior inside the graph) and improves CER on ambiguous glyphs
under independent evaluation; reuse of Stage 1–2 bundles measured (construction cost
with vs. without reuse).

Risks: a language prior is statistics over large text; writing it by hand means either
very small priors or an author "compressing its pretraining" into weights — useful, but
then the capability is transferred from the author, not discovered, and must be
reported that way.

## Stage 4 — Non-text visual tasks and component reuse

Demonstrate: bundles built for OCR are reused (verified by lineage + ablation) for a
second visual task (shape counting, simple scene classification), lowering construction
cost relative to building from scratch.

Risks: OCR components may be too specialized to transfer; "reuse" can be inflated by
wrapping without functional benefit (bundle count is not evidence).

## Stage 5 — Constructed algorithmic reasoning

Demonstrate: constructed graphs that execute multi-step procedures (multi-digit
arithmetic read from images, sorting, simple planning in grid worlds) with
length generalization beyond the development range.

Risks: authors can write exact algorithms (circuits) — but a hand-written adder is a
unit test, not discovery (per the handoff). The meaningful test is whether authors can
build *general* reasoning machinery rather than one circuit per task.

## Stage 6 — Language

Demonstrate: a constructed model that maps text to useful text on held-out inputs
(e.g., constrained question answering over a small domain), compared against a
gradient-trained model of equal size.

Risks — the largest in the plan:
* Scale: capable language models use 10⁸–10¹² parameters. Even at 1,000 authored
  parameters per maker-hour, direct authoring cannot reach that scale by hand. The
  approach would need authors to design *generators* of weights (compact programs that
  expand to large networks) — which drifts toward writing algorithms, and away from
  "neural" learning, or toward reintroducing fitting.
* Where the knowledge comes from: any linguistic competence in the artifact would be
  distilled from the author LLM's pretraining. That is legitimate engineering but
  must be reported as transfer, not independent discovery.

## Stage 7 — Conversational multimodal system (the long-term vision)

Would require Stages 1–6 plus dialogue state, grounding across vision/speech, and
robustness at a level no current evidence suggests direct authoring can reach.
A practical assistant today should be built from existing trained models
(separate project). CellForge's scientific contribution, if the early stages hold,
is evidence about **what LLMs can construct directly** — which is valuable even if
Stage 7 is never reached.

## Cross-cutting risks

* Evaluation leakage via author perception (authors looking at dev images) — keep
  unseen-family finals and report them separately.
* Author cost grows with task complexity; track tokens per point of final accuracy.
* Single-run anecdotes: require independent construction replicates per arm.
