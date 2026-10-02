# Synthetic gene evaluation

Tested on `claude/arc3-gan-generator`, commit
`f6a0ba841c2f1e56b6ec862aa1aafdf49465f628`, on September 30, 2026.
Receipts contain source SHA-256 hashes for the actual generator and agent files.
No public-game internals were used.

## Main sweep

Ten seeds per family, two levels per game, 600 actions per episode. Layout seeds
43000–43009; solver seeds 0–9. Generated sizes 5–9, densities 0–0.15, extras 0–4,
randomized palettes and permuted movement actions. This is a small development
curriculum, not a held-out evaluation or the generator's full difficulty range.

| Family | Genes wins | Random wins | Genes actions | Random actions | Genes mean efficiency |
|---|---:|---:|---:|---:|---:|
| Touch | 10/10 | 7/10 | 315 | 3467 | 0.191 |
| Key/door | 10/10 | 5/10 | 462 | 5265 | 0.353 |
| Push | 9/10 | 0/10 | 1231 | 6000 | 0.171 |
| Sequence | 8/10 | 5/10 | 1533 | 4483 | 0.274 |
| Click | 10/10 | 1/10 | 175 | 5770 | 0.269 |
| Moving hazard | 8/10 | 6/10 | 1405 | 4380 | 0.223 |
| **Total** | **55/60** | **24/60** | **5121** | **29365** | |

Genes completed 111/120 levels, compared with 62/120 for random. Moving-hazard
deaths were 4 versus 60 for random. All action totals include capped failures.
Efficiency is the synthetic generator's per-level squared optimal/action ratio,
averaged over both levels, including unsolved levels as zero.

## Ablations

Three seeds per family, 18 episodes per variant, 300-action budget; these overlap
the development sweep. Disable one mechanism while retaining the rest.

| Variant | Wins /18 | Actions | Deaths | Mean efficiency |
|---|---:|---:|---:|---:|
| Full genes | 17 | 818 | 2 | 0.299 |
| Random | 4 | 4686 | 19 | 0.021 |
| No action effects | 9 | 3610 | 23 | 0.074 |
| No terrain evidence | 13 | 1837 | 88 | 0.267 |
| No goal targeting | 18 | 1350 | 0 | 0.137 |
| No key/door dependency | 14 | 1586 | 2 | 0.271 |
| No push mechanism | 15 | 1353 | 2 | 0.264 |
| No hazard prediction | 17 | 823 | 3 | 0.294 |

Action-effect learning provides the largest measured gain in this subset. Terrain
evidence prevents repeated lethal experiments. Key/door inference and push
planning improve wins and efficiency. Goal targeting trades one failure for
substantially better efficiency, so it is not uniformly beneficial. Hazard
prediction's incremental evidence is modest; avatar disambiguation is shared by
both variants. These ablations disable groups of related behavior, not every
listed gene individually.

## Tests and artifacts

`python -m pytest tests/test_arc3_genes.py -q`: **9 passed**.
Tests cover tile inference/click coordinates, avatar displacement, hidden action
permutations, no-op evidence, lethal-floor ambiguity, key/door disappearance,
directed wall edges, push repositioning/corners, learned click effects, hazard
direction context, and contract budget/death/level lifecycle.

- `results/synthetic.json`: 120 paired episode records plus provenance.
- `results/ablations.json`: 144 episode records plus provenance.
- `README.md`: contract usage, 17 mechanisms, and explicit model assumptions.

The agent's remaining failures are one push game, two sequence games, and two
moving-hazard games. Larger boards, multi-block puzzles, multi-pixel objects,
mixed action interfaces, stochastic effects, hidden inventories beyond collection,
and private ARC-AGI-3 games remain unmeasured. These are Python observation/planning
genes usable through the requested contract; no CellForge IR encoding is claimed.
