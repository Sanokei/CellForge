# Extension design: vocabularies, modalities, component reuse

Constraint: nothing here may change the current `digits-mono/1` experiment. Extensions
enter as **new protocol versions**; old results stay in their own result epoch.

## 1. Vocabulary stages

Protocols carry their own charset and vocab (`reference/protocols.py`):

| protocol | charset | status |
|---|---|---|
| `digits-mono/1` | 0–9 | active |
| `alnum-mono/1` | 0–9, A–Z | defined, not run |
| planned `alnum-prop/1` | + lowercase, proportional fonts | planned |
| planned `words/1` | + space, word-level strings | planned |

Vocab order is always `<pad>, <bos>, <eos>, charset...`, so a token index from an
earlier stage keeps its meaning in later stages (digits keep indices 3–12). A graph
authored for stage k can be carried to stage k+1 by an explicit, logged transaction that
widens output projections (new rows start at an author-chosen value, e.g. a strongly
negative logit) — the widening itself is a computational edit and requires re-evaluation.

## 2. Modality path

The runtime already separates `encode` (modality-specific) from `step` (token decoder).
The decoder interface — `memory tensors + step + prev one-hot + state → logits, state` —
is modality-agnostic. New modalities add new *input contracts*, not new decoders:

| input contract | encode input | notes |
|---|---|---|
| `pixels-32x128/1` | `image [32,128]` | current |
| planned `pixels-HxW/1` | `image [H,W]` | general 2-D images; same ops |
| planned `tokens/1` | `text [L,V]` one-hot | text-to-text tasks (Stage 6) |
| planned `spectrogram/1` | `spec [F,T]` | speech; fixed, documented STFT preprocessing only, no pretrained features |

Rule carried over from the handoff: preprocessing may be fixed numerical transforms
(normalization, patching, STFT) documented in the contract, never recognition logic.

## 3. Component reuse across tasks

* Definitions are content-addressed by graph hash plus definition name; the memory
  records lineage for pure refactors, so a bundle can be traced across versions.
* Planned: a cross-run **component library** (`library/<def-hash>/`) holding exported
  bundle definitions with their parameters, provenance, and the evaluations they were
  accepted under. Importing a library component into a new run is an explicit
  transaction (`import_def`) recorded in the ledger — its origin is visible, and its
  earlier results do not transfer as claims; it must be re-evaluated in the new task.
* Reuse is measured, not assumed: construction cost (maker tokens, dev-evals) to reach a
  target with vs. without library access, on independent finals (Roadmap Stage 4).

## 4. What is deliberately not done now

No chat interface, speech pipeline, or pretrained component is added to the CellForge
runtime. Those would contaminate the experiment's central claim (weights authored, not
imported). A practical assistant built on existing models is a separate project.
