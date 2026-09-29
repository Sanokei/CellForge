# Orchestrator log — CellForge OCR v3, run `p1` (protocol digits-mono/1)

Orchestrator: Claude Opus 5.5 (this Claude Code session). Makers: Claude Code
subagents (same model family), each commissioned with a task packet and
`docs/MAKER_GUIDE.md`. Author-token usage is not exposed per subagent by the
harness; I record wall-clock, dev-eval counts, and subagent-reported usage when
available. Unknown costs are recorded as unknown, not zero.

## 2026-09-29 — Round 1 (arms established)

Bootstrap `616545f2…` (EOS-constant): anchor CER 1.000, exact 0.000 (n=1000).

Commissioned in parallel from the same bootstrap:
| task | branch | arm | dev-eval budget | memory |
|---|---|---|---|---|
| r1-iter-a | iter-a | iterative, recursive bundles allowed | 25 | yes (empty) |
| r1-iter-b | iter-b | iterative replicate (independent) | 25 | yes (empty) |
| r1-oneshot | oneshot | one-shot author (validate-only, no metrics) | 0 | no |
| r1-flat | flat | flat graph (no sub-bundles/dispatch) | 25 | yes (empty) |

Rationale: the first question is whether direct authoring can produce any
working recognizer at all, and how much iterative feedback matters versus a
single shot. Two iterative replicates give a first (tiny) sense of variance.

### Round 1 results (all four makers completed; none needed restarting)

Maker usage (subagent-reported): ~98–104k tokens, 15–20 tool calls, 5–6 min
wall-clock each. Dev-evals used: iter-a 2, iter-b 2, flat 2, oneshot 0 (validate-only).
Audit: no helper script reads dataset images/labels (grep for images.npz,
labels.json, np.load, lstsq: no hits). All four authored hand-typed 15x12
glyph templates after *viewing* dev PNGs (or, for oneshot, an ASCII print of
3 dev images) — authored weights, permitted by the guide, but this is
the "LLM perception as learning" channel and must be named in the paper.

Reference anchor (dev feedback, n=1000): all four CER 0.000 / exact 1.000.
**The in-distribution task is saturated; it does not discriminate arms.**

Independent final evaluation (server, fresh secret seed per eval, evaluator
account; exact match with Wilson 95% CI, CER):

| artifact | iid (n=2000) | unseen font LiberationMono (1000) | size 18–22 (1000) | length 9 (500) |
|---|---|---|---|---|
| r1-iter-a (recursive, iterative) | 1.000 [.998,1] / 0.000 | 0.738 [.710,.764] / 0.098 | 0.723 [.694,.750] / 0.133 | 1.000 / 0.000 |
| r1-iter-b (recursive, iterative) | 1.000 / 0.000 | 0.382 / 0.240 | 0.379 / 0.471 | 0.000 / 0.111 |
| r1-oneshot (no feedback) | 1.000 / 0.000 | 0.427 / 0.187 | 0.453 / 0.376 | 1.000 / 0.000 |
| r1-flat (flat graph) | 1.000 / 0.000 | 0.407 / 0.197 | 0.570 / 0.276 | 1.000 / 0.000 |
| gradient slot-MLP, 400 train | 0.057 / 0.748 | 0.054 / 0.784 | 0.050 / 0.764 | 0.000 / 0.770 |
| gradient slot-MLP, 20k train | 0.838 / 0.052 | 0.100 / 0.584 | 0.419 / 0.329 | 0.000* / 0.319 |
| random (1+1) edits, 20k evals | 0.000 / 1.074 | 0.000 / 1.058 | 0.000 / 1.072 | 0.000 / 0.848 |
| random (1+1) edits, 25 evals | 0.000 / 1.014 | 0.001 / 1.036 | 0.000 / 1.028 | 0.000 / 0.930 |

\* slot-MLP has 8 character slots by construction, so 9-digit strings are impossible for it.

Interpretation (hypotheses, not established):
* One replicate per arm; differences between iter-b / oneshot / flat are within
  what design idiosyncrasy explains. n=1 construction run per arm is not a
  comparison of arms. iter-a's robustness traces to specific choices
  (normalized agreement score, ±3 px search, soft-edge zeros), not to "recursion".
* iter-b hard-coded 8 cells → total failure on length 9 (a genuine
  authored-design brittleness that iid dev feedback cannot reveal).
* The gradient control is a deliberately simple MLP with no translation
  invariance; a gradient-trained CNN would very likely reach ~1.0 iid. The
  result shows authored templates beat *this* control, not gradient training.
* Exposure: these final results were seen by the orchestrator and will guide
  Round 2. Any Round 2 final claim uses fresh final cases (automatic: new seed).

Decision: Round 2 targets robustness (fonts/sizes/sub-pixel), since iid is
solved. Dev curriculum gains FreeMono + DejaVu Bold and size 18–22; the
unseen-font test (LiberationMono) stays absent from all author-visible data.

## 2026-09-29 — Pivot (owner decision): hierarchical MoE language modeling

Owner direction: OCR is the easy part; focus on the cell -> cluster -> expert
MoE decoder. Owner chose (a) tiny character language modeling as the first
task and (b) an enforced ladder as the primary arm plus an open-recursion arm.
This reverses v3's "no imposed ladder" at the owner's instruction; the
fixed-vs-adaptive hierarchy comparison is kept as ladder/1 vs open/1.

Protocol tinylm/1: English-like text from a private grammar (agreement
across relative clauses), written by the orchestrator and stored only on the
evaluation server. Makers see 2,000 sampled documents (dev_train) and 300
dev_eval documents, never the grammar. Final eval: fresh documents per run
(bpc, top1, bootstrap CI) + grammaticality/novelty/diversity of the model's
own temperature-0.8 generations, checked by the private recognizer
(recognizer accepts 5000/5000 generated sentences; rejects 5/5 probes).
Additional maker rule: no code that computes statistics from text.

Controls (fitted on dev_train, anchor bpc, n=500 docs): uniform 5.000;
unigram 4.224; bigram 2.629; trigram 1.382; gradient MLP (ctx 8, H=256)
training in progress.

Round LM-1 commissioned: ladder arm L1, L2, L3 (runs/lm1); open arm O1, O2
(runs/lm1-open). 25 dev-evals each, independent, same hypothesis text.
