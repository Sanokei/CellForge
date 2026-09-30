# Paper Prize outline (Claude draft, 2026-09-30) — negative results with controls

Working title: *What small authored, fitted and evolved cell libraries can and cannot do on ARC-AGI-2 under CPU and
no-gradient constraints.*

Positioning: a transparent, reproducible account of a constrained approach (CPU only, no gradient training, open code) with
honest controls. It claims no competitive score and no AGI. The Paper Prize requires a corresponding working Kaggle entry;
verify current paper rules (format, length, deadline Nov 8) on the official pages before writing the final text.

1. Motivation and constraints: low-power deployment, no gradients, directly authored cells joined by evolution/breeding,
   Kaggle CPU budget (12 h, no internet), open-source release (CC0/MIT-0).
2. Methods
   2.1 Authored cells (57 hand-written grid cells, depth-2 composition, demo-verified, two attempts).
   2.2 Task-fitted cells (tables fitted by counting on demos; strict coverage; colour-invariant variants).
   2.3 CellForge graphs: cells + wiring as genome; crossover via import_def; evolution and random-search controls.
   2.4 Evolved tiny cellular automata (gradient-free) as a negative control.
3. Results (all on public data; development on training, evaluation scored without tuning)
   - Training: 53 -> 80 -> 95 -> 97 of 1,000 (5.3% -> 9.7%), false positives 12 -> 14.
   - Evaluation: 0/120 at every stage; zero tasks with any demo-consistent chain.
   - CellForge breeding: best single genes and breeding pairs tie at 3/200 dev tasks (Codex receipts).
   - Evolved NCA (414-1,488 params, equal-eval random control): 0/40; selection lowers loss but solves nothing.
   - Evolution vs random search on the byte-LM pilot: tied (2.0423 vs 2.0423 bpb).
4. Distribution-gap analysis: evaluation grids ~2x larger, more colours/objects, local-rule expressibility 22.4% -> 7.4%.
5. What did not work and why: local/primitive rules do not express evaluation-style structure; unseeded evolution finds
   nothing; breeding of near-identical genes adds nothing; selection on 2-3 demo pairs overfits (false-positive rate).
6. What might transfer: object/relation cells, generator-based dense fitness, amortised selector, Baldwin-style fitness.
7. Threats to validity: public evaluation exposed; single runs; author-written cells; no private-set score.
8. Reproducibility: code, hashes, run ledgers, exact commands, Kaggle notebook link (to be created by the owner).
