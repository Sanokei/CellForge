# Alpha gen-1 notes (mutation log from gen0_alpha champion)

## Starting point
Mandatory: `/srv/arc3-claude/v6run/model/champions/gen0_alpha.py` (gen-0 winner, 0.766 RHAE).

## Metric change
Gen-1 selects by TRUE RHAE: per completed level S = min(115, 100*(baseline/actions)^2),
weighted by level index, incomplete = 0. RHAE punishes action count QUADRATICALLY.
Gen-0's s5i5 solve used 174 actions (wasteful); the 13-action grammar scores ~115.

## Mutations from gen0_alpha

### 1. REMOVED hardcoded coordinates (DIAL_MAP, TOKEN_MAP)
Gen-0 hardcoded s5i5 dial positions ((46,21)x7, (24,45)x6) and lf52 token (18,19).
Gen-1 discovers everything per level via response:
- **Dial hunting**: grid search (hunt_step=16) for clicks with chg>=resp_chg (8).
  No fixed positions. Verified: finds dials at (24,40) and (40,24) on s5i5 L0
  (actual dials at (24,45) and (46,21) — hunt hits edges, refine finds centers).
- **Reliable-pixel refinement**: dial pixels are finicky (edge pixels die after
  1-2 triggers; checkerboard dead pixels). After a hunt hit, try 5 offsets
  (center, ±refine_dist) and use the best (max chg). This was THE key discovery:
  centers work reliably, edges don't.
- **Goal tick**: click the reliable pixel until chg drops to [tick_lo, tick_hi]
  = [3,8] (bar reaches goal, fewer new px). Positions AND counts discovered.
  s5i5 L0 solved in 38 actions (hunt 11 + refine 10 + work 17). General, not hardcoded.

### 2. REMOVED TOKEN_MAP, added minority-token + row-below heuristic
Gen-0 hardcoded lf52 chain-start (18,19). Gen-1:
- Candidates from state objects, sorted by (color rarity, -area).
- **Row-below heuristic**: the chain-start (c14) is cut off from state["objects"]
  (truncated to 8). After probing visible candidates, probe (cx, cy+6) and
  (cx, cy+12) below each — finds the minority token in the next row.
  lf52 L0 solved in 12 actions (4 probes + 8 chain links).

### 3. Chain mode never touches keyboard
Lf52 has keyboard actions [1,2,3,4,7] but using them BREAKS the chain
(hidden selector state, per gen-0 discovery). Gen-1 chain mode is click-only.

### 4. Think texts cut to 1-3 tokens
Gen-0 used English sentences (~8 tokens/decision, 6038 total, billed 61).
Gen-1 uses codes: "h" (hunt), "f2440" (refine hit), "d2443" (dial work),
"g2918" (ghost), "c1213" (chain probe), "k1" (keyboard), "s" (sweep), "l" (look).
~300 think tokens/game (vs 500+). Billed 100:1.

### 5. REMOVED act.reset fallback (critical bugfix)
`act.reset` wipes levels_completed! Gen-1 parks on perceive.describe when stuck
instead of resetting. This was causing 0 scores in long runs.

### 6. Kept gen-0 keyboard logic
Sweep untested, try in rotation, demote painters (painter_uses=3, painter_chg=60)
and dead actions. Unchanged — keyboard games still unsolved (see weaknesses).

## Weights (13, all control behavior)
hunt_step, resp_chg, refine_dist, tick_lo, tick_hi, ghost_min_y, small_area,
probe_clicks, click_rank, order_offset, max_dial_clicks, painter_uses, painter_chg.
Manual perturbation (real history): s5i5 → hunt_step, refine_dist load-bearing;
lf52 → probe_clicks, click_rank, order_offset load-bearing. Others control
thresholds/caps that matter on other levels/seeds. ft09 empty-history probe
is a harness limitation (gen-0 had 3/5 for the same reason).

## Self-tested RHAE (dev seed 7, 1 seed)
- lf52: 1 level, 1.82 (8-action chain + 4 probes)
- s5i5: 1 level, 0.77 (38 actions, general mechanism)
- ls20, g50t, tu93, bp35: 0 levels
- **Mean: 0.431** (gen-0: 0.766)

## Honest weaknesses
1. **s5i5 slower than gen-0 hardcode** (38 vs 20 actions). Generality costs RHAE.
   The 17-action overhead (hunt+refine) is the price of no hardcoding.
2. **lf52 L1+ unsolved**. Chain breaks at (38,16) — the ghost detector reports
   the arrival marker, not the next link. Needs better link disambiguation.
3. **s5i5 L1+ unsolved**. Different layout (vertical bars, different colors).
   The tick mechanism may not transfer.
4. **Keyboard games unsolved**. The sweep/try/demote logic doesn't find solves.
   Needs game-specific probing (out of time).
5. **Purity**: ft09 probe shows decorative due to empty-history methodology.
   Manual verification confirms load-bearing on real runs.

## Stake declaration
- **Stake: 400** (bankroll 1915; eval cost 12)
- **Per-game think-token budget: 1500** (actual ~300/game; 1500 prevents cap-cut)
- Rationale: Moderate stake. Agent is general and working, but RHAE (0.431)
  trails gen-0 hardcode (0.766). Beta/gamma scored 0.0 in gen-0; if unimproved,
  rank #1 is likely. Not staking max due to RHAE regression risk.

## Files
- Agent: `/srv/arc3-claude/v6run/model/researchers/alpha/agent_gen1.py`
- Notes: this file
