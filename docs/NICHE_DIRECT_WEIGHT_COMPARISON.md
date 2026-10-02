The archive proposer from `claude/redqueen-archive` is bridged to DirectWeightMaker through `niche_weight_bridge.py`. The original archive module is imported intact from commit 61e6489. Its scalar indexed edits become same-shape `set_param` tensors; actual graph execution and audited transactions replace its toy interpreter for comparisons.

Run from the repository with `PYTHONPATH=src`:

```text
python scripts/run_three_arm.py --gene runs/your-session/your-gene --out runs/your-comparison --adapter cellforge.construction.niche_grid_tasks:adjacency_context
```

Repeat `--gene` for artifacts from any session. Genes are identified and deduplicated by their full graph hashes. The supplied adjacency adapter generates four synthetic grid niches; other domains require a `module:function` task adapter. It does not claim coverage of unrelated skills or a real ARC anchor.

The runner writes `predeclared.json` before loading proposals: seeds 11, 22, 33, four rounds by default, one charged attempt per arm per gene per seed per round. The three arms are unchanged baseline, random literal numeric repair, and supervised LLM repair. Each pair starts from the same frozen gene; this is independent repair attribution, not evolving populations or cross-niche breeding. Random edits use a predeclared scalar vocabulary. Equal charged budget means attempts, not tokens, CPU, or changed scalar count.

The original file layout remains `outbox/<seed>/round<k>.json` and `inbox/<seed>/round<k>.json`. Outboxes contain `targets` and `niches`, with parameter tensors, editable parameter names, exact base hashes and development counterexamples. A human-supervised LLM writes a target-keyed inbox:

```json
{"GENE_HASH": {"actor": "supervising session", "model": "actual author model", "base": "GENE_HASH", "scope": "joint", "rationale": "reason grounded in shown examples", "edits": [{"param": "neighbor_gate.b", "index": [0], "value": -1.5}]}}
```

DirectWeightMaker `ops` can replace `edits`. Supply truthful actor/model provenance. Missing, stale, malformed or unchanged proposals consume their attempt and cannot be adopted. The runner exits with pending inbox paths; rerun the same command after writing responses. Submitted proposals are frozen before scoring. Completed rounds are skipped on resume. Partial comparison receipts cause a fail-closed stop and must be preserved; restart in a fresh output directory.

Custom adapters receive `(graph, shown_seed, fresh_seed)` and return `diagnostics`, typed `fixtures`, `evaluate(graph)` and `accept(parent_metrics, candidate_metrics)`. Generate deterministic independent shown/fresh samples. Expose only development train pairs to the proposer and only aggregate fresh scores in receipts. All three arms use the same fresh seed/tasks. The adjacency adapter accepts only strict total improvements without a per-niche regression or a rise in false positives. Read per-gene results and hash-chained ledgers under `results` and `ledgers`; no LLM API is invoked and no performance gain is assumed.
