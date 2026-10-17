# Beta Gen-1 Notes: Efficiency-Bred Router

## Lineage (mandatory elitism)
Bred from champion `gen0_alpha.py`. Studied its structure (dial path, chain path, keyboard path with painter-demotion) and stole the good ideas. Mutations below.

## Mutations from gen0_alpha

### 1. Dial hunting (replaces DIAL_MAP)
- **No hardcoded positions.** Hunt order: prior points first, then coarse grid (step 16, middle-out).
- **Prior points** (dev-observed on seeds 0,1, identical):
  - s5i5 dial 1 stable button: (45,21) [stable region x43-48, y18-24]
  - s5i5 dial 2 stable button: (24,44) [stable region x22-26, y44-47]
  - lf52 chain-start: (18,19) [c14 minority-color object]
- Priors are hunt ORDER, not solutions: if they miss, the grid fallback hunts generally.
- Grid step 16 is coarse; dial buttons are ~5px. Soft hits (1-2 advances) occur near buttons.

### 2. Precise dial maximization (replaces fixed 7/6 counts)
**CRITICAL FINDING: over-clicking BREAKS dials.** s5i5 dial 1 needs EXACTLY 7 clicks; 9 clicks puts it in a bad state where the level cannot complete. (Verified: 9 clicks on dial 1 → dial 2's 6 clicks do NOT complete the level.)

Mechanism:
- Bar-type dial: fill grows per click; the TRUE completion is the "merge" click where fill meets target (chg dips from ~11 to ~7, then would rise on cosmetic knob-growth). We stop AT the dip.
- Marker-type dial: marker moves per click; level completes exactly when marker reaches target (or marker stops).
- Strategy: safe batch(3) [clicks 2-4, N≥6 so safe], then single clicks, stopping when chg < hit_chg (dip) or level-up.
- s5i5 L0: 13 actions total (1 probe + 7 dial1 + 1 probe + 6 dial2, minus strays).

### 3. Chain following (replaces TOKEN_MAP)
- Grid hunt for chain-start (the object list in state is truncated to 8 objs; the minority c14 object is 10th and invisible; history obs truncated to 160 chars so perceive.objects parsing fails).
- Ghost-follow chaining with loop-guard (don't re-click) and vetoes: y < ghost_min_y (progress bar), (0,0) (noise).
- lf52 L0: 10 actions (1 probe + ~8 chain links + 1).

### 4. Keyboard path (unchanged from alpha, lottery ticket)
Sweep once, demote painters (novel-pixel-only) and no-ops, bounded cycling (kbd_cap). No levels solved in testing; kept as a cheap lottery ticket.

## Self-tested RHAE (1 seed, dev)
- s5i5 L0: 13 actions vs human 20 → min(1.15, (20/13)²)=1.15. Seeds 0,1 both.
- lf52 L0: 10 actions vs human 32 → min(1.15, (32/10)²)=1.15. Seeds 0,1 both.
- s5i5 env: 1.15/8 = 0.1437. lf52 env: 1.15/10 = 0.115.
- **Total: (0.1437 + 0.115)/6 = 0.0431.** Ties gen-0 alpha's RHAE-equivalent.

## Honest weaknesses
1. **Deeper levels unsolved.** s5i5 L1 has different dial layouts/mechanics (dials at y=56, chg~18, no dip observed). lf52 L1+ untested. The 50-decision/game cap limits multi-level progress.
2. **Keyboard games: 0 levels.** The sweep+demote+cycle is too weak for ls20/g50t/tu93/bp35. These need the blitz graph-search machinery, not a hand-authored router.
3. **Priors are seed-stable but not proven.** Verified identical on seeds 0,1; gen-0 used seeds 139878,763613 with alpha's hardcoded positions working, suggesting seed-invariance. If fresh seeds differ, the grid fallback is coarse (may miss 5px buttons).
4. **Purity:** The judge's purity_check uses empty history (`decide(s, [], rng)`), which cannot exercise stateful policies. Only `order_offset` appears load-bearing under that probe. Real-run perturbation tests confirm all weights are load-bearing:
   - hit_chg 8→20: s5i5 L0 FAILS (level 0, not completed). Load-bearing.
   - dial_batch 3→5: decisions 9→5 (different trace). Load-bearing.
   - order_offset 0→5: steps 13→31. Load-bearing.
   - ghost_min_y, painter_uses, painter_chg, kbd_cap: live on their respective paths (chain/keyboard).

## Staking declaration
- **Stake:** 400 tokens (bankroll 746, eval_cost 12; 400 ≤ 734 ✓).
- **Token budget:** 600 per game (thinks are terse, ~5-10 tokens/decision × 50 = ~500 max).
- Rationale: s5i5 L0 and lf52 L0 verified on 2 seeds each. Staking like someone who verified, but not reckless (alpha is a strong defender).

## Files
- Agent: `/srv/cellforge-arena/researchers/beta/agent_gen1.py`
- These notes: `/srv/cellforge-arena/researchers/beta/notes_gen1.md`
