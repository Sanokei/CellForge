# ALPHA-G2 — "The Rehearsal Fix" (KL-anchored)

## Recipe
1. **Base**: `lm/router_v3.pt` + dnd1 LoRA deltas (`lm/dnd1_deltas.json`) applied via
   `lm/apply_lora.py` → `base_dnd1.pt` (decisiveness + parse-reliability deltas;
   base dev-6 ≈ 0.75, solves lf52 L1 + s5i5 L1).
2. **Finetune** (`researchers/alpha/gen2/finetune_gen2kl.py`, 300 steps, batch 24,
   block 192, AdamW **lr = 1e-4 (at cap)**, seed 7) on a **mixed corpus**
   (token fractions): discovery `traces_fc.jsonl` 35% / answers `traces_w2.jsonl`
   50% / `traces_grammar_x100.jsonl` 15% — 2,041,100 tokens total.
3. **Anti-forgetting, two mechanisms**:
   - The 4 dnd1-touched param groups (`lm_head.weight`,
     `blocks.2/3.attn.c_attn.weight`, `blocks.3.mlp.c_proj.weight`) are **frozen**
     — dnd1's authored deltas stay exactly as authored.
   - **KL anchor**: on every answer batch, `loss = LM + 5.0 * KL(p_theta || p_base)`
     against the frozen base. Discovery batches get pure LM loss (allowed to
     learn new behavior). This directly penalizes behavioral drift on the answer
     distribution while discovery knowledge flows into the shared representation.
4. Base GPT arch only (4L/4H/d128, vocab 2209, 1,378,816 params) — no RMT,
   breed-compatible. Submission is pure tensors `{cfg, state}`.

## Why this design (ablation trail)
- v1 (60% fc, 1e-4, 300 steps, no freeze): dev-6 0.215 — s5i5 L1 lost. FAIL.
- v2 (35% fc, 1e-4, frozen dnd1 groups): dev-6 0.092 — still lost s5i5 L1. FAIL.
- Ablations showed the dev solves are **knife-edge deterministic**: base_dnd1
  reproduces both L1 solves bit-identically on re-run, yet even 50 steps at
  1e-5 (Δloss ≈ 0) deterministically killed lf52's L1 in 2/2 trials. Even
  **answers-only** finetuning at 1e-4/300 broke lf52 (0.0). It is not the fc
  corpus that is poisonous — any unconstrained weight drift flips the fragile
  early-game decisions.
- The KL anchor fixes exactly this: it constrains the *output distribution* on
  answers rather than the weight trajectory, keeping the model inside the
  behavioral basin of the solves while discovery gradients reshape the shared
  representation underneath.

## Self-check (dev-6, seed 0, maxdec 60, budget 3000, fixed_runner.py)
- lf52: 1.8182% RHAE, L1 solved (base: 1.7097 — slight gain)
- s5i5: 2.7778% RHAE, L1 solved (matches base)
- ls20, g50t, tu93, bp35: 0.0 (same as base — keyboard games unsolved by all)
- **dev-6 mean = 0.7660 ≥ 0.31 guardrail PASS** (full run, all 6 games)

## Lineage
router_v3 → (+dnd1 LoRA) → base_dnd1 → (+KL-anchored mixed finetune, dnd1 groups
frozen) → alpha_gen2kl.pt → submitted as `gen2/submissions/alpha.pt`.

## Stake
**600 tokens** of 1467 bankroll on the transfer outcome (see report for rationale).
