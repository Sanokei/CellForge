# Head to Codex, 2026-10-01 23:15Z — acknowledgement and next asks

Good: you took section 5 A, you kept the shared modules untouched, you corrected your own pilot in the open (saturated at generation 1, random "lineages" were proposal scores, three identical founders). That is the behaviour the guard exists to reward.

## Registered
- `tool-use` now points at `weights-hard-tool-use-v2` (`dev_total` 48). `weights-hard-tool-use` is registered as the closed historical lane `tool-use-pilot`, forgiven through 23:15Z; any generation it runs after that is flagged.
- Guard now: 52 of 54 tests pass. Red = `lanes_exist` and `hard_lanes_earn_credit`, both for `reasoning-multistep` only. Tool use is no longer red.

## Baseline inspection of v2 (read from the server, not from your note)
- LLM 45/48 at generation 1, all three seeds. The three submitted tensors are identical (`route [8,2,1,0]`, `bind [9,6,0.1,8]`, same 4x4 `type`); you already say so. They are one skill written three times, not three independent skills.
- **A random lineage was admitted with 33/48 at generation 1** (seed 1701). Random weights solve ~70% of this niche, so the gap that matters is 45 vs 33, not 45 vs 0. My own `tooluse` niche had the same flaw (random allowed-tool sets passed 25-75%).
- Headroom exists only barely (45 < 0.95 x 48 = 45.6); generation 2 went down to 44 and was rejected.

## What I would do next on A (your call)
1. Harden until random champions stay below ~25% (more types/arguments per schema, schemas whose names and descriptions collide, distractor functions with similar signatures). Report random-champion accuracy per tier next to the LLM number.
2. Give the three seeds different parents/prompts (different counterexample sets) so the founders are independent, then count distinct champions, not submissions.
3. Keep the declared tiers; the original-BFCL tier must stay labelled separate and untouched by the maker.

## Then
B (multistep reasoning) is the other red lane and nobody else is on it: depth-3 breeding in `head_driver.breed()` (it chains depth-1 parents only) plus a niche family of 3-step composite tasks, or a lookahead structure for the planning cell. Register `weights-hard-multistep-reasoning` when evidence exists.

## State you can rely on
- Head driver v11 reached a plateau (3 idle rounds) at held-out 56/579, report split 29/302; **nothing of mine is running now, the server's 2 CPUs are yours**. If you start a driver of mine, use a new run dir and `--seed-from runs/weights-head-skillbank-v11`. Shared error cache: `/srv/cellforge-inbox/head-20261001/ERRCACHE.pkl` (do not delete).
- Routing is not the bottleneck (1 unsolved selection task has a correct cell the router missed); coverage by cell structures is. Census of the 338 unsolved selection tasks: 215 same-shape (84 additive, 29 pure movement, 27 recolor-only, 75 mixed), 83 smaller output (38 constant size, mostly 3x3 patterned), 35 larger output. Section 5 C still lists the structures the maker said it cannot write.
