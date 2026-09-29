# LLM-directed breeding (`breeding/1`)

Owner idea (2026-09-29): let the LLM pick the best models and "breed" them, treating
construction as evolution driven by benchmarks. This document says how to do that
without breaking what the project claims.

## What makes it authored, not a search engine

The handoff bans an automatic mutation/selection engine in the primary path
(random/evolutionary search may only appear as separately labelled control arms).
Breeding is therefore run as follows:

* **The LLM chooses.** The orchestrator reads a fitness table (below) and decides
  which candidates become parents and what to recombine. No code ranks and selects.
* **The LLM writes every child.** A crossover child is an ordinary transaction
  proposal. The only new tool is the deterministic materializer `import_def`:
  `{"op":"import_def","from":"<graph ref>","name":"ex_np","prefix":"b_"}` copies a
  definition and everything it references, exactly, from another model version into
  the child. `prefix` isolates the copy (namespaced circuits make "take expert X from A
  and circuit Y from B" a precise edit). It selects and fits nothing.
* **Mutations are maker tasks**, exactly as in construction; a maker may also be asked
  to "reconcile" a crossover (re-tune the router or weights so imported parts
  cooperate), using the normal dev-eval budget.
* Every child records its parents (task packets + ledger), so lineage is auditable.

Population-style comparison arms (labelled, never feeding the primary):
`random-crossover` (import random parts, no LLM choice) and `no-breeding`
(the same number of maker tasks, no recombination), matched on maker tokens.

## Fitness must not become the test

Selecting on a benchmark makes it development feedback. So:

| signal | use | who sees it |
|---|---|---|
| bpc on fresh held-out real-text stories | primary fitness (low noise) | orchestrator: aggregate only |
| benchmark **sel** split (even positions of a fixed seeded sample) | secondary fitness / milestone | orchestrator: aggregate only |
| benchmark **sealed** split (odd positions) | final reporting only, never used to select | nobody until freeze |
| final story pool (private) | final reporting | operator only |

Makers never see benchmark items or per-item outcomes. Because selection over N
candidates on n items inflates the best score, sealed-split numbers are the ones
reported, and the size of the selection effect is reported as
`sel_score - sealed_score` for the chosen candidate.

## Noise and what selection can resolve

A model at chance scores 25% ± 2.7% (95%, n=1000). Selecting the best of 20 chance-level
models on that split selects noise (expected inflation ≈ +3 points). Rules:

* Do not use a benchmark for selection unless some candidate is above chance on the
  sealed split by more than its 95% interval.
* Prefer bpc (per-token, ~10^5 tokens per evaluation) for choosing parents until then.
* Minimum meaningful difference: bootstrap CI over documents from the evaluator; treat
  overlapping intervals as ties and prefer the more inspectable candidate.

## Generation loop

```
population <- accepted models (any run, any profile that matches the target task)
repeat:
    fitness table <- bpc (fresh stories) + sel-split benchmark accuracies, aggregate only
    orchestrator picks parents and crossover/mutation tasks, writes the rationale to the log
    makers author children (import_def + edits) within their budgets
    accept valid children; evaluate; append to population with lineage
freeze best-by-sealed; final evaluation on untouched final pool and sealed split
```

Budget and accounting: maker tokens per generation, dev-evals, and wall-clock are
recorded per child. Report quality per unit of authoring effort against the
`no-breeding` control.
