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

### Round LM-1 results (tinylm/1)

Accepted via `cellforge accept`; final evaluation on fresh 1000-document
samples + 200 free-running generations (T=0.8) checked by the private
recognizer. bpc with bootstrap 95% CI (over documents).

| artifact | profile | dev-evals used | final bpc | top1 | grammatical | gram.+novel | experts used | unique params |
|---|---|---|---|---|---|---|---|---|
| L1 | ladder | 6 | 0.6796 [.678,.682] | .810 | .966 | .788 | 6 | 278k |
| L2 | ladder | 11 | **0.6776** [.676,.680] | .809 | **.997** | .884 | 60 | 1.69M |
| L3 | ladder | 5 | 0.6790 [.677,.681] | .809 | .940 | .804 | 4 | 898k |
| O1 | open (no routing) | 3 | 0.6797 [.678,.682] | .809 | .955 | .839 | – | 1.31M |
| O2 | open | interrupted (3 evals, 0.691 dev), resumed, still running | | | | | | |
| gradient MLP ctx8 H=256 | control | – | 0.824 | .780 | .201 | .135 | – | 74k |
| trigram counts | control | – | 1.383 | .617 | 0.0 | 0.0 | – | 65.6k |
| uniform | bootstrap | – | 5.000 | 0 | 0.0 | 0.0 | – | 32 |

Observations (measured) and interpretations (hypotheses):
* Every authored model reaches 0.678–0.680 bpc (CIs overlap): a wide plateau,
  ~0.14 bpc better than the gradient MLP and with 95–99.7% grammatical
  samples vs 20%. Authored models track agreement across nested relative
  clauses via recurrent state (a stack); the MLP with an 8-char window cannot.
* **The task is saturated by grammar reverse-engineering.** All makers read
  ~35-45 documents, wrote down the closed grammar and hand-tallied rates,
  then compiled it into tables/state machines. bpc differences of 0.002 are
  within noise; the plateau is probably near the generator's own entropy
  (unmeasured; a true entropy-rate estimate is a TODO).
  L2's 99.7% grammaticality (vs 94–97% for L1/L3/O1) is the only clear
  separation; single replicate each.
* **No evidence yet for "ladder learns faster than open".** O1 (open, no
  routing) matched the ladder with 3 dev-evals vs 5-11. n=1 vs n=3; the
  ladder was mandated to use dispatch, which the task does not need. What the
  ladder produced is more inspectable (named experts per grammatical
  situation) and reusable, not more accurate.
* Authoring cost (subagent-reported tokens): L1 199k, L3 176k, L2 231k(+),
  O1 186k(+). ~35-41 tool calls each. Open arm costs comparable.
* This is "author compiles a known grammar", i.e. transfer of the author's
  knowledge of English + perception of samples, not discovery of unknown
  structure. It shows the machinery (state, routing, recursion tracking)
  works end-to-end at 0.3–1.7M authored parameters, which is a real engineering
  result, but says little about scaling to data whose generator the author
  cannot read off from 40 samples.

Decision: next task must resist reading-off. Options: (a) a much larger,
noisier grammar with many more lexical/semantic constraints; (b) real text
(e.g. public-domain prose) where no closed generator exists and gradient
controls scale; (c) enforce ladder-micro (<=40-slot cells) with delegated
circuits on the same grammar to test the *delegation* mechanism, since the
plateau makes efficiency (tokens per quality) the only meaningful axis.

**Entropy floor (measured on the evaluation server):** summing the bits of every
random choice the private generator makes gives 0.6734 bits/char over 400 fresh
documents. Because a text can have several derivations, this is an upper bound on
the true entropy rate, so the floor is <= 0.6734. Best authored model (L2):
0.6776, i.e. within 0.004 bpc (<1%). tinylm/1 is exhausted as a discriminating
benchmark: further bpc differences are unmeasurable. Only grammaticality,
inspectability and authoring cost still vary between arms.

## Real-text phase (owner chose real text after tinylm/1 saturated)

tinyreal/1: TinyStories validation file (MIT; 22.5 MB, sha256 6874bae9...
downloaded to the evaluation server only, on the owner's approval of a real-text
download). Stories 150-600 chars, [a-z ,.!?'"] only, split by sha256 bucket:
dev_train 2088 stories (1.12M chars), dev_eval 165, anchor 290, private final
pool 2684 (each final eval draws unused stories, no reuse). Reference levels
(anchor bpc): uniform 5.17, unigram 4.21, bigram 3.19, trigram 2.30.
Arms: (A) delegated ladder-micro/1: architect a1 + circuit-makers under
contracts; (B) solo open/1 maker s1. Same evaluation, budgets recorded.
Expectation stated in advance: authored models will land far above
gradient/MLP controls on real text (no closed grammar to decompile, and the
no-statistics rule bars fitting). The interesting quantities are how far
authored structure gets, and authoring cost per bit gained.

## Benchmarks, circuits and breeding (2026-09-29, later)

* Owner: "tests too easy; use the benchmarks frontier labs use". Candid
  response given: frontier benchmarks read as chance for models of this size
  (a same-size gradient model also scores ~25% on MMLU). Owner chose "frontier
  set only". Downloaded to the evaluation server (private dir only): MMLU test
  (cais/mmlu, 3.5 MB), ARC-Challenge test (0.2 MB), HellaSwag validation (6.3 MB).
  **GPQA not obtained**: gated dataset (HTTP 401); needs the owner to accept
  its terms/login, which the orchestrator cannot do. Protocol `mc-loglik/1`
  (bench_operator.py): normalize to the model's own vocabulary (lowercase; digits/capitals/
  symbols dropped; char coverage 0.89 MMLU, 0.96 ARC, 0.99 HellaSwag), score
  choice text by total bits, report acc and length-normalized acc_norm, Wilson CI.
  Fixed seeded item permutation, nested subsets, sel/sealed split for breeding.
* First bench sweep (n=200 by mistake: an earlier smoke test fixed the subset
  size; fixed to a stored permutation): real-text controls (unigram, trigram,
  MLP-512) sit at chance on all three, as expected.
* Real-text delegated round: architect a1 planned 14 circuits (predicted
  2.75 bpc, <10% chance to beat the 2.30 trigram); 14 circuit-makers ran in
  parallel and ALL passed their contracts (10-19 cases each), ~65-82k tokens each,
  ~1-2 minutes wall-clock each. Phase C integration in progress (a1 resumed).
* Owner idea: LLM breeding/evolution on benchmark fitness. Adopted as
  breeding/1 (docs/BREEDING.md): LLM picks parents and writes crossovers;
  `import_def` op added (deterministic copy of a definition + dependencies
  with prefix isolation; 20 tests pass); benchmark sel/sealed split added.
  Guardrail: selection on chance-level benchmarks selects noise; primary
  fitness is bpc on fresh stories until some candidate is above chance
  on the sealed split.
