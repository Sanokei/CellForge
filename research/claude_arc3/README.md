```
game  lv |   random.jsonl | explorer_v3.js | explorer_v4.js
ar25  |    0/8   0.00     4s 95M |   1/8   0.00    29s 104M |   1/8   0.02    38s 101M
bp35  |   0/9   0.00    16s 106M |   0/9   0.00    64s 126M |   0/9   0.00    40s 122M
cd82  |    1/6   0.00     3s 93M |   1/6   0.12    26s 102M |   1/6   0.09    25s 102M
cn04  |    0/6   0.00     3s 93M |   0/6   0.00    23s 104M |   0/6   0.00    25s 103M
dc22  |   0/6   0.00     5s 105M |   0/6   0.00    28s 107M |   0/6   0.00    30s 107M
ft09  |    0/6   0.00     6s 97M |   0/6   0.00    67s 104M |   0/6   0.00    38s 104M
g50t  |   0/7   0.00     8s 122M |   0/7   0.00     5s 122M |   0/7   0.00     5s 122M
ka59  |   0/7   0.00     4s 129M |   0/7   0.00    25s 127M |   0/7   0.00    27s 127M
lf52  |  0/10   0.00     8s 112M |  1/10   0.01    28s 112M |  1/10   0.04    28s 112M
lp85  |   1/8   0.00     4s 118M |   1/8   2.78    24s 117M |   1/8   2.78    24s 117M
ls20  |   0/7   0.00     7s 102M |   1/7   0.01     4s 101M |   1/7   0.00     4s 101M
m0r0  |    1/6   0.00     4s 93M |   1/6   0.00    31s 100M |   1/6   0.06    34s 103M
r11l  |   1/6   4.76    10s 123M |   1/6   0.04    26s 122M |   1/6   0.24    53s 123M
re86  |   0/8   0.00     4s 140M |   0/8   0.00     2s 140M |   0/8   0.00     2s 140M
s5i5  |   0/8   0.00     3s 119M |   0/8   0.00    22s 119M |   1/8   0.00    22s 119M
sb26  |    0/8   0.00    10s 97M |   0/8   0.00    23s 104M |   0/8   0.00    24s 105M
sc25  |   0/6   0.00     7s 127M |   0/6   0.00    15s 127M |   2/6   0.23    24s 127M
sk48  |    0/8   0.00     4s 93M |   0/8   0.00    23s 107M |   0/8   0.00    21s 111M
sp80  |    1/6   0.04     6s 93M |   1/6   0.03    25s 101M |   1/6   0.02    24s 101M
su15  |    0/9   0.00     3s 98M |   1/9   0.00    24s 102M |   1/9   0.01    23s 104M
tn36  |   0/7   0.00     3s 116M |   1/7   0.33    30s 115M |   0/7   0.00    26s 115M
tr87  |    0/6   0.00     4s 94M |   0/6   0.00     3s 104M |   0/6   0.00     3s 104M
tu93  |   0/9   0.00     3s 102M |   2/9   0.03     3s 101M |   3/9   1.20     3s 101M
vc33  |    0/7   0.00     3s 98M |   2/7   0.02    24s 100M |   2/7   0.02    23s 101M
wa30  |    0/9   0.00     3s 96M |   0/9   0.00     2s 105M |   0/9   0.00     2s 105M
random.jsonl: levels 5  mean score 0.192  total secs 135  peak MB 140
explorer_v3.jsonl: levels 14  mean score 0.134  total secs 578  peak MB 140
explorer_v4.jsonl: levels 17  mean score 0.188  total secs 568  peak MB 140
```

Totals: levels cleared random 5, explorer v3 14, explorer2 v4 17 (of 190 levels). Mean score: random 0.192, v3 0.134, v4 0.188
(random's score comes from one lucky 17-action r11l level = 4.76; scores are tiny because levels take 100s-1000s of actions vs human baselines of 10-100).
Peak memory ~140 MB per process; wall time 2-67 s per game for 3000 actions (<=2 CPUs, nice 10).
Earlier variant v1 (pixel-rate mask only, no band rule): 14 levels, 1032 s total; band rule kept levels equal but cut runtime to 578 s
and stopped state-graph explosion in games with a HUD progress bar (g50t, ls20).

## Failure modes seen
1. Avatar/puzzle games with big reachable state spaces (wa30, re86, tr87, ka59, bp35, dc22, sk48): blind DFS-ish exploration never reaches the goal
   in 3000 actions; nothing in the method is goal-directed (no notion of which change is "progress").
2. Click games with many targets (sb26: >8000 untested state-actions after 1500 steps; lp85 level 2, s5i5, ft09, cn04).
3. Hidden state / non-determinism: ls20 level 2 and g50t exhaust the graph ("stall") without progress, e.g. mechanics driven by the
   masked energy counter or unobserved inventory; tu93 stalls on level 3. The fallback (forget graph, random action) is weak.
4. Monotone HUD counters (energy bars) make every state look new unless masked. Fixed by VolatileMask band rule (thin, within 8 px of the frame border,
   changed on >30% of non-no-op steps). Risk: masks game-relevant HUD (e.g. a goal indicator) - not checked per game.
5. Level-1 only in most solved games: later levels need longer action sequences and cost > budget.

## Improvements (2, each re-measured)
1. Counter detection at band level + ignoring no-op steps + GAME_OVER sink (so a masked energy level cannot poison a state as "dead"): levels 14 -> 14, time -44 %, removes tree blow-up.
2. Action-class statistics (`explorer2`): untested actions ordered by observed change rate of their class (keyboard id, or click target colour) with an optimistic prior,
   salience order breaking ties. Levels 14 -> 17, mean score 0.134 -> 0.188 (gained s5i5, sc25 x2, tu93; lost tn36). Single seed, so the gain is indicative only.

## Caveats
Public 25 games only, one seed, 3000-action budget; private/hidden games and the full 2026 difficulty are unmeasured. Not tuned per game.
Budget choice is mine (baseline human counts sum to ~500-1500 per game). The 3000 cap is far below what a score near human level needs.
