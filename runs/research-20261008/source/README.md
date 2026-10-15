# CellForge

LLM agents directly author the weights and wiring of neural networks, with no gradient training,
using recursive cell/bundle graphs, an improvable development harness, and evidence-backed memory.

## Long-term vision (owner's aim — not a current capability)

The long-term purpose of CellForge is to work toward general intelligence: eventually a system you
can talk to like an LLM, with visual input and later speech, whose computation is *constructed*
by LLM authors rather than fitted by gradient descent. OCR is the first vertical slice, not the
end goal.

**This is an aspiration, not a result.** Nothing in this repository has language understanding,
dialogue, speech, or general intelligence. Whether direct authoring can scale from glyph
recognition toward any of that is an open research question; `docs/ROADMAP.md` lays out, stage by
stage, what would have to be demonstrated and where it is most likely to break.

## What is actually built and measured (as of 2026-09-29)

| Component | Status |
|---|---|
| Recursive IR: cells, nested/shared bundles, top-1 lazy dispatch, wrap/unwrap refactors | implemented, 12 tests (G0) |
| Transactions: atomic, base-hash checked, validated, pure-refactor equivalence tests | implemented, tested |
| Reference evaluator (exact match, CER with explicit denominators) | implemented |
| Independent final evaluation on a separate server (fresh secret seed per eval, evaluator account, aggregate-only output) | implemented, used |
| Hash-chained ledger, graph version store, branches | implemented |
| Evidence memory: content-addressed artifacts, append-only events, TF-IDF retrieval, counterevidence links, final-data denylist | implemented; live **visual retriever not yet run** |
| Maker agents (Claude Code subagents) authoring graphs via CLI | live, Round 1 complete |
| Ladder/ladder-micro profiles, top-k lazy dispatch, component contracts, LM controls and final evaluator | implemented, tested (19 tests) |
| Harness evolution (G3), memory-arm comparison (G3M), scaling fixture (G5) | not yet run |

**Round 1 result (protocol `digits-mono/1`: 1–8 digits, DejaVu Sans Mono 20 px, 32×128):**
four independent LLM-authored artifacts, including one authored with *zero* metric feedback,
reach exact match 1.000 on in-distribution final cases (n=2000). Under distribution shift they
differ sharply: best artifact 0.738 exact on an unseen font and 0.723 under size shift; others
0.38–0.57. A gradient-trained slot-MLP control reaches 0.838 iid / 0.100 unseen font; random
direct-edit search reaches 0.000. One construction run per arm; see `notes/ORCHESTRATOR_LOG.md`
for full tables, caveats and exposure disclosures. The control is deliberately simple; this does
**not** show that authoring beats gradient training in general.

**Language-model round 1 (protocol `tinylm/1`, private English-like grammar, owner-directed hierarchical-MoE focus):**
five authored character LMs (three in the enforced cell→cluster→expert ladder, two open) reach 0.678–0.681 bits/char on
fresh text with 94–99.7% grammatical free-running generations (the entropy floor of the generator is ≤0.6734). A gradient
MLP with an 8-character window gets 0.824 bpc and 20% grammatical; trigram counts get 1.38 and 0%. Best model: 60 routed
experts, 1.7M authored parameters, 99.7% grammatical. **Caveat:** the makers read ~40 documents, wrote the closed grammar
down, and compiled it; the task is exhausted (within 1% of the floor) and does not show scaling, nor that the ladder
learns faster than open recursion (an open, unrouted model matched it with fewer feedback runs). Next: real text
(`tinyreal/1`, TinyStories), where no closed grammar exists.

## Scope and non-goals of the current experiment

CellForge tests whether LLM agents can author neural network weights directly, without gradient
training. The current artifact is an offline image-to-text recognizer. It has no language
understanding, dialogue, speech capability, or general intelligence; chat and speech interfaces
are out of scope for this repository. A practical multimodal assistant built from existing models
(e.g. a chat/vision API plus speech-to-text) is a separate project and must not be presented as a
CellForge result.

## Layout

```
src/cellforge/ir/            recursive registry, hashing, validation, stats
src/cellforge/runtime/       parameter-free ops, reference interpreter, bounded decoder
src/cellforge/construction/  bootstrap, transactions, run state / acceptance
src/cellforge/harness/       development diagnostics (agent-improvable)
src/cellforge/memory/        evidence store, retrieval, evidence checks
src/cellforge/governance/    hash-chained ledger
src/cellforge/reference/     protected metrics, protocols, evaluator, operator (final eval)
src/cellforge/baselines/     gradient and random-search control arms (never touch primary)
docs/MAKER_GUIDE.md          the maker agents' contract
docs/ROADMAP.md              staged path beyond OCR, with risks
docs/EXTENSION_DESIGN.md     modality/vocabulary stages and component reuse
notes/ORCHESTRATOR_LOG.md    decisions, results, exposure disclosures
paper/                       LaTeX paper
```

## Quickstart

```bash
pip install -e .[dev]
export CELLFORGE_FONT_DIR=/path/to/licensed/ttf   # fonts are not shipped
python -m pytest -q
cellforge init-run --run runs/demo --protocol digits-mono/1
cellforge status --run runs/demo
```
