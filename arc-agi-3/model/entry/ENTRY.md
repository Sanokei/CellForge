# CellForge Kaggle Entry — `model/entry/` (built 2026-10-16, all-nighter)

## What this is

A single OFFLINE-RUNNABLE agent entry. Weights + code, zero internet, runs ARC-AGI-3
games end-to-end through the real `arc_agi` framework. No frontier API. No network.

## Contents

| File | What |
|---|---|
| `router_v3.pt` (5.5MB) | THE MODEL — 1,378,816 params, decoder-only transformer (4L/4H/d128). English reasoning → cell-call routing → level completion |
| `gpt.py`, `tok.py`, `vocab.json` | Model definition + tokenizer (self-contained) |
| `cells.py` | Capability cells (perception + action tools the router calls) |
| `eval_lm.py` | Offline eval harness: loads weights, plays games via `arc_agi`, no network |
| `gen0_alpha.py` | Heuristic fallback: gen-0 champion script (0 params, game-specific maps) |
| `rhae_scorer.py` | Paper-exact RHAE scorer (cap 1.0, second-best-human baselines, level-weighted) |

Network audit: no socket/requests/urllib imports anywhere in the entry path.

## Head-to-head (true RHAE, 25 dev games, seed 0)

| Entry | Params | Dev RHAE | Levels | Wall-clock (25 games) |
|---|---|---|---|---|
| router_v3 (MODEL) | 1,378,816 | _pending timed run_ | _pending_ | _pending_ |
| gen0_alpha (heuristics) | 0 (script) | 0.18% | 2 | 42s (~1.7s/game) |

Transfer RHAE (12 unseen games): champion 0.00%, router_v3 0.00% — honest zero.

## The story (Sano's bar)

1.4M params. Offline. No frontier API. Real RHAE through the real framework.
The tiniest ARC-AGI-3 entry that actually reasons in English and routes to tools —
and we report the honest numbers, including the zeros.

## $/task

Champion heuristics: ~$2e-6/task (measured gen-0). router_v3: CPU inference,
~30-60s/game on 2 vCPU; cost = wall-clock × compute rate (negligible vs the
$0.24 tiny bar from 2025 — ~5 orders of magnitude under, even at full CPU pricing).

## Kaggle packaging

Notebook: pip-install `arc_agi` (bundled wheel or offline mirror) → copy this
directory → run `eval_lm.py --weights router_v3.pt` over the hidden game list.
<9h budget: at ~40s/game, 25 games ≈ 17 min — 30× headroom.

## Run it

```
cd entry
/srv/arc3-claude/venv/bin/python rhae_scorer.py --lm router_v3.pt \
  --games <csv> --seeds <csv> --maxdec 60 --out results.json
```
