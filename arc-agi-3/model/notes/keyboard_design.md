# Keyboard Action Support — Design Doc
**Date:** 2026-10-16 | **Directive:** Sano, all-in for the win (top prize, organizers' terms)
**Status:** Phase 1 complete (investigation); Phase 2 gated on bred-children dev-6 eval

## 1. Framework keyboard API (mapped)

`GameAction` (arcengine/enums.py): `RESET(0)`, `ACTION1(1)`…`ACTION5(5)`, `ACTION6(6, ComplexAction x,y)`, `ACTION7(7)`.
There are **no literal key-press events** — "keyboard games" are games driven by the
*simple* button actions 1–5, 7 (game-defined abstract buttons); ACTION6 is the click/tap.

Per-game `available_actions` survey (all 25 games, seed 0):
- **Pure keyboard (no 6):** g50t [1-5], re86 [1-5] (both NON-transfer — free training ground),
  tr87 [1-4], wa30 [1-5] (both transfer — measure-only)
- **Click-only ([6]):** ft09, lp85, r11l, s5i5, tn36, vc33
- **Hybrid:** ar25, bp35, cd82, cn04, dc22, ka59, lf52, ls20, m0r0, sb26, sc25, sk48, sp80, su15, tu93

Human `baseline_actions` are per-level ACTION COUNTS only (RHAE efficiency targets) —
no behavioral sequences to imitate.

## 2. Current action space audit (model/lm/cells.py)

| Cell | Coverage |
|---|---|
| `act.try{action,x?,y?}` | ANY action by id: 0=reset, 1–5/7 simple, 6=click. **Full simple-action plumbing EXISTS.** |
| `act.sweep{}` | Tries each untested simple action once, reports `id>changed-px` + LEVEL-UP. **Discovery primitive EXISTS.** |
| `act.click_at/pick_drop/repeat_click/ghost_follow` | ACTION6 click-based |
| Prompt (`perceive.describe`) | Includes `AVAIL[...]` — router SEES available action ids |
| `fixed_runner.py` | Dispatches any registered cell via `call_cell` — simple actions flow through, no runner change needed |

**Conclusion: the plumbing is complete. The gap is 100% training.**

## 3. Root cause of 0 keyboard levels

- traces_w2: 13.7K `act.try` uses; simple-action ids DO appear (1:2193, 2:1553, 3:4209, 4:4222, 5:518, 7:981)
  — but never precede level-ups (exploratory noise on click games; router learns "simple actions = wasted steps").
- traces_grammar: 0 simple actions. traces_fc: 4K simple uses, 0 sweep→exploit arcs.
- `act.sweep`: 81 uses in w2, 0 elsewhere — the discovery loop is absent from training.
- The router never learned: (a) simple actions are primary input on some games,
  (b) sweep→observe→exploit, (c) sustained purposeful simple-action sequences.

## 4. Design (general — action meanings discovered at test time, never hardcoded)

**4a. Cells (minimal extension).** Keep `act.try`, `act.sweep`. ADD:
- `act.key_seq{seq:[a,b,c,…]}` — execute N simple actions as ONE decision.
  Rationale: decisions are the scarce resource (maxdec 60, ~20s CPU each on box);
  keyboard L1s need long sequences (tr87 L1 baseline: 54 actions). One LM call →
  "press 3 fifty times" instead of 50 dithering decisions. Fully general (action ids
  only, zero game knowledge). Fixed-runner vocabulary, not per-submission Python.

**4b. Prompt.** AVAIL already shown. Action-effect memory: sweep OBS already reports
`id>px` effects; keep in history (last-2 blocks). If insufficient for long sequences,
add a durable `KEYFX` summary line to describe (adapt the *concept* from Oct-1
transition-memory prior art — not its service architecture).

**4c. Training (the real work).**
- Corpus: keyboard-discovery traces on NON-TRANSFER keyboard-responsive games:
  g50t, re86 (pure keyboard), tu93, ls20, bp35, cn04, dc22, ka59, sc25, sk48 (hybrids).
  NEVER tr87/wa30 (transfer, measure-only).
- Content: sweep → observe effects → hypothesize ("action 2 moves avatar right") →
  purposeful sequences via key_seq → level-ups. Include failed discoveries (contrastive).
- Generation: scripted competent explorer (extend trace_gen_fc.py: sweep-first, then
  exploit highest-effect action) and/or frontier-LM play. Mixed corpus (rehearsal —
  the forgetting lesson). Low LR (≤1e-4).

**4d. Test-time compute (structural weapon).** 42s/25 games vs 9h eval limit ≈ 350× headroom:
- Always sweep simple actions first on any game listing them (≤7 env steps, cheap).
- Multiple attempts/seeds per game at eval; keep best RHAE per game.
- Sequence search at test time: repeat effective actions, keep what progresses.
- The tiny model needs a good exploration policy + success recognition; compute does the rest.

**4e. Weights-only compliance.** `key_seq` is fixed-runner vocabulary. Researchers submit
pure tensors. The keyboard skill (when to sweep, which sequences to try) lives in weights.

## 5. Phase 2 plan

1. ✅ Bred-children dev-6 eval (running; gate).
2. Add `act.key_seq` to cells.py + registry + docstring; verify via fixed_runner on g50t with hand CALLs.
3. Generate keyboard-discovery traces (scripted sweep-first explorer).
4. Train keyboard-competent variant (low LR, mixed corpus).
5. Measure tr87/wa30 (measure-only). **First keyboard level completed = breakthrough — report immediately with true RHAE.**

## 6. Inventory mining (transfer-relevant; reusable, not rebuilt)

- `act.sweep{}` — the discovery primitive already exists; don't rebuild.
- **Transition-conditioned action-effect memory** (continuous-spatial NEXT_STEP.md, never
  dispatched): concept reusable for 4b; don't rebuild the service arch.
- Oct-1 cell families: candidate blackboard cells, not keyboard-specific.
- Sept-29 final_lm checkpoints: unmined, lower priority.
- **No keyboard-specific prior art found — genuinely new ground for the program.**
