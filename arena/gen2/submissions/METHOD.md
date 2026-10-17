# beta-g2 — "Memory That Predicts" (v6)

**Researcher:** beta-g2 | **Bankroll:** 1462 | **Stake:** 400
**Checkpoint:** `/srv/cellforge-arena/gen2/submissions/beta.pt` — pure tensors
`{cfg, state}`, RMT arch (4L/4H/d128, vocab 2209, 8 memory tokens, wpe 200),
purity OK, auto-detected by fixed_runner.
**Lineage:** router_v3 → +dnd1 deltas → RMT warm-start → 200 steps
memory-forcing on mem_init (transformer frozen).

## Thesis

Gen-1 RMT failed with **memory dependence 0.0**: memory had no pressure to
be used. The fix: a **forcing loss** — the model must predict the current
row's OBS tokens FROM the incoming memory tokens, through an isolated
train-time probe (own LayerNorm+Linear, discarded before save). Gradient
flows into the memory write path: to minimize the loss, the model must
compress history into a *predictive* world state.

## The road here (post-mortem, for the paper)

Five training variants were tried and **all failed the dev-6 guardrail**
(0.00%), while the untrained warm-start scores 0.46%:

- **v1**: probe re-used the shared `lm_head` → OBS pressure distorted call
  generation (parse fails exploded). Lesson: isolate the probe.
- **v2**: isolated probe, but training rows contained OBS while the runner's
  eval updates memory on (STATE,THINK,CALL) without OBS → train/eval format
  skew poisons memory. Lesson: train format must match eval format.
- **v3**: eval-format rows + masked CE (think+call from STATE+MEMORY, read
  pressure) + isolated probe → still 0.00%.
- **v4**: v3 + froze dnd1's 4 tensors + probe 0.25 → still 0.00%.
- **v5**: full CE + probe 0.1 + memory-norm reg (anti-hijacking hypothesis)
  + frozen dnd1 → still 0.00%.
- **Control**: gen-1's `router_rmt.pt` (plain text CE, no probe, no masking)
  also 0.00%. **RMT training per se breaks dev-6** on this model/data —
  the 1.4M-param text path is too brittle to survive any finetuning on
  discovery-heavy data, even at lr 1e-4 with rehearsal.

**v6 (this submission)**: freeze the ENTIRE transformer. Train ONLY
`mem_init` (1024 params) + the probe. The text path is bit-identical to
the warm-start (all learned weights verified equal; only the fresh-random
wpe memory rows differ by RNG). The frozen text path already attends
heavily to memory (0.57 in block 0), so a more predictive mem_init can be
used with zero text-path changes.

## v6 recipe (train_beta_g2f.py)

1. Warm-start RMT from `router_v3+dnd1.pt` (dnd1's decisiveness preserved).
2. Freeze every parameter except `mem_init`.
3. Loss per row-chain: `0.5 * probe(mem_in → obs tokens)` +
   `0.05 * ||mem_out||²` + `0.2 * text CE` (anchor keeping mem_init
   text-compatible). Probe is train-time scaffolding, discarded before save.
4. Corpus: fc discovery chains only (105 segs, ≤56 rows, full BPTT).
   200 steps, AdamW lr 1e-4, grad clip 1.0, OMP_NUM_THREADS=1.
5. Anti-contamination: FORBIDDEN transfer-game prefixes asserted on every
   row; TRANSFER-12 never opened, trained on, or named anywhere.

Training: probe loss 7.7 → 6.0 (memory measurably more predictive);
mem_init drift 0.53 (small, stable); text anchor held.

## Dev-6 self-check (guardrail ≥ 0.31%) — PASSED

`fixed_runner.py --checkpoint beta.pt --games lf52,s5i5,ls20,g50t,tu93,bp35
--seeds 0 --maxdec 60 --budget 3000`

| game | rhae_% | levels | parse_fail |
|------|--------|--------|------------|
| lf52 | 0.0    | 0      | 19         |
| s5i5 | 2.7778 | 1      | 44         |
| ls20 | 0.0    | 0      | 30         |
| g50t | 0.0    | 0      | 18         |
| tu93 | 0.0    | 0      | 40         |
| bp35 | 0.0    | 0      | 38         |
| **mean** | **0.46** | **1** | — |

Guardrail **0.46% ≥ 0.31%** ✓. Matches the untrained warm-start exactly
(s5i5 L1 solved), confirming the frozen transformer preserved everything.

## Honest assessment

- The forcing loss is real (probe learned), but weak: only mem_init could
  adapt. Transfer upside is unproven — this is the safest honest submission,
  not the strongest conceivable one.
- The v1–v5 post-mortem is the durable finding: **RMT finetuning destroys
  this tiny model's dev performance; the memory must be shaped without
  touching the transformer.** Future work: differential LRs, or
  memory-only architectures where the write path doesn't share weights
  with the text path.
- Stake 400 reflects: guardrail passed with margin, mechanism is principled
  and novel, transfer outcome unproven (all prior attempts 0.00%).

## Reproduce

```bash
cd /srv/cellforge-arena/researchers/beta/gen2
OMP_NUM_THREADS=1 /srv/arc3-claude/venv/bin/python train_beta_g2f.py \
  --steps 200 --lr 1e-4 --force-w 0.5 --mreg-w 0.05 --text-w 0.2 \
  --warm router_v3+dnd1.pt --out router_beta_g2f --seed 7
cp router_beta_g2f.pt /srv/cellforge-arena/gen2/submissions/beta.pt
```
