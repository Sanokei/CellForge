# GAMMA-G2 — "Authored, Not Trained" (gen-2 submission, FINAL)

Researcher: gamma-g2 | Bankroll: 1311 | Stake: 400 | Date: 2026-10-16

## Thesis

Every gradient attempt last night caused catastrophic forgetting. This
submission uses ZERO gradient steps: pure DnD-style weight authoring.
The base model is only MEASURED (hidden-state directions), never updated.

## Composition (lineage)

`router_v3.pt` → `+ dnd1_deltas.json` (= `router_v3+dnd1.pt`)
→ `+ gamma_final_deltas.json` (this work)
= `/srv/cellforge-arena/gen2/submissions/gamma.pt`
(pure tensors {cfg, state}, 1,378,816 params, sha256 e2ef23af…)

## Shipped mechanisms (rank-2 LoRA on lm_head.weight only)

**M1 — <unk> suppression at cell-emission positions.** Observed failure
mode: at the cell-suffix position the model puts `<unk>` as its TOP token
(median +0.49 nats over the best real cell, p90 +2.50, max +3.72; measured
on 60 dev contexts) — `<unk>` is never a valid cell name, so every such
emission is an instant parse fail. Authored fix: suppress `<unk>` −4.0
nats at the suffix position, −2.0 nats at the cellname position, gated on
the top SVD directions of those natural emission contexts.

**M2 — probe rebalance (the transfer play).** At the suffix position:
sweep +0.5 / try +0.7 nats; repeat_click −0.3 / ghost_follow −0.3 nats.
Net on dev: dnd1's macro priors (+0.9) become +0.6 — answer routines still
win, but probes move from far-behind to near-parity, so on unfamiliar
contexts the model explores (act.sweep = "try each untested keyboard
action once, report effects"; act.try = single exploratory action)
instead of committing to the nearest-seen-game routine.

**Safety (anti-forgetting).** Both mechanisms gate ONLY on large-a
directions (top SVD dirs of natural emission contexts; typical
projections 12.87 / 11.99) → A rows ≤ 0.34 in abs → no think-injection
(verified: generations show no probe/macro tokens leaking into THINK).
blocks.0–3, wte, wpe, ln_f bit-identical to v3+dnd1. ||dW||/||W|| = 1.3%.

## Discarded attempts (measurement-first discipline)

- **rev C (rank-1 novelty gate):** discriminative novelty direction with
  small typical projection (a=0.269) needed huge A rows (b/a ≈ 3.7),
  which injected probe/macro tokens into mid-THINK generation whenever
  w·h was O(1) there → derailed CALL JSON → parse fails exploded
  (lf52: 10 → 45; dev 0.75% → 0.46%). Caught by the dnd1 control
  experiment. DISCARDED.
- **rev D (think-orthogonalized gate):** constraining the novelty
  direction orthogonal to top-8 think-position states still left
  max|w·h| = 3.3 on think states — the state-novelty signal is not
  separable from think geometry in a static linear gate. DISCARDED
  without applying.
- Lesson recorded: static LoRA gates on lm_head cannot be
  position-specific; only large-a (natural-context) gating directions
  are safe there.

## Self-check (guardrail), identical protocol both rows

`fixed_runner.py --checkpoint … --games lf52,s5i5,ls20,g50t,tu93,bp35
--seeds 0 --maxdec 60 --budget 3000`

| game | dnd1 (control)        | gamma (shipped)       |
|------|-----------------------|-----------------------|
| lf52 | 1.71% (pf 10)         | 1.82% (pf 14)         |
| s5i5 | 2.78% (pf 39)         | 2.78% (pf 23)         |
| ls20 | 0.00% (pf 49)         | 0.00% (pf 37)         |
| g50t | 0.00% (pf 16)         | 0.00% (pf 15)         |
| tu93 | 0.00% (pf 48)         | 0.00% (pf 38)         |
| bp35 | 0.00% (pf 41)         | 0.00% (pf 23)         |
| **mean** | **0.75%**         | **0.77%**             |

Guardrail (dev-6 ≥ 0.31%): PASSED with margin. No dev regression vs
dnd1 on any game; parse fails reduced on 4/6 games.

## Honest uncertainties

1. The probe rebalance is a weak, unconditional transfer play — no
   novelty detection (proved unsafe in rev C/D). Expected transfer
   effect is small; the five straight 0.00%s suggest a hard wall.
2. If transfer is 0 for all researchers, selection falls to the dev-6
   tiebreak, where this submission leads at 0.77%.
3. No transfer-game knowledge anywhere: no transfer game opened, named,
   or referenced in prompts, traces, deltas, or notes. Novelty contexts
   used during authoring were synthesized from dev-game text only.

## Files

- `/srv/cellforge-arena/gen2/submissions/gamma.pt` — shipped checkpoint
- `gamma_final_deltas.json` — shipped LoRA deltas (rank 2, lm_head only)
- `gamma_v2_deltas.json` — DISCARDED rev C deltas (kept as record)
- `author_v3.py` — final authoring script (rev E/F)
- `author_v2.py`, `diag_gate.py`, `diag_tails.py`, `diag_dial.py`,
  `probe_gen.py`, `probe_unk.py`, `trace_ep.py` — diagnostics
- `dev6_revF.log` / `dev6_revF.json` — shipped self-check raw output
- `dev6_dnd1_control.log` — dnd1 control raw output
- `STAKE.md` — stake declaration (400)
