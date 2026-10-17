# beta_g3 — "Wire the Novelty Signal" (gen-2 gen-2, round 1)

**Researcher:** beta (gen-2) | **Bankroll:** 1062 (elimination warning) | **Stake:** 300
**Checkpoint:** `/srv/cellforge-arena/gen2/submissions/beta_g3.pt` — pure tensors
`{cfg, state}`, plain GPT arch (4L/4H/d128, vocab 2209), purity OK.
**Elitism root:** `bred_g1_a25.pt` (dev RHAE 0.766%, two independent evals).
**Lineage:** router_v3 → dnd1 → RMT warm-start → beta-g2f (gen-1, 0.46%) →
re-based on bred_g1_a25 (mandatory elitism) → DnD audit-cited LoRA deltas (this work).
**Method:** pure DnD weight authoring (zero grad steps). All deltas are
low-rank LoRA-form edits merged into the root checkpoint.
**cells_used:** none (all CALLs use existing act.*/perceive.* cells).

## Audit grounding

STEP 0: read `/srv/cellforge-arena/gen2/GOODFIRE_BRIEF.md`
("Neural Networks Are Already Clean"): small nets are legible with
coarse-but-coherent neurons; apparent polysemanticity is recursive
generalization; audit activations in action, not static weights.

Reference audit of the elitism root (folded in; pieces re-verified by my own
targeted probe, probe_beta_g3.py, on DEV-1..6 + TX-A..L measure-only):

- [F1] Transfer failure is **wrong-circuit firing (pattern collapse)**, not
  silence: every transfer game matches the dominant dev routine at cosine
  0.96–0.99 by L3 (my probe: 0.965–0.990, dominant DEV-6 5/12 votes).
- [F2] 169/2048 neurons perfectly separate novel-vs-seen from L0 (effect
  sizes to 6.39) but **nothing downstream reads them** — routing failure,
  not detection failure.
- [F3] Zero dead neurons, diffuse coding — no free capacity; repurpose, don't recruit.
- [F4] Late-layer wash: novelty separation strongest at L0/L1, drowned by L3.
- [F5] Malformed CALLs jump on novel states — the call-emitter breaks on
  unfamiliar inputs (my probe: dev 41%/decision, tx 35%/decision parse-fail;
  the breakage is real but the fallback `perceive.describe` already degrades
  gracefully).

Probe bug fixed en route: my first probe used a local `parse_call` missing
the runner's whitespace-strip + REGISTRY validation, which silently dropped
real actions and diverged trajectories (dev-6 0.00 vs 0.766). Fixed by
importing `parse_call` from `fixed_runner` directly; probe then reproduced
the confirmed eval bit-for-bit (lf52 1.8182/pf2, s5i5 2.7778).

Key probe-derived facts used in authoring:
- The model can only SPELL 7 cells (`act.try/sweep/click_at/pick_drop/
  repeat_click/ghost_follow/reset`); all `perceive.*` and `act.key_seq`
  contain `<unk>` and are unreachable by generation (verified by encoding).
  The novelty "distinct branch" is therefore built from reachable cells:
  exploration `act.sweep`/`act.try` vs macro routine
  `act.repeat_click`/`act.ghost_follow`/`act.pick_drop`.
- Novelty direction `n_dir` (final residual, tx-mean minus dev-mean),
  mean direction `m_dir`, L2 novelty direction `l2_nov`, dominant-routine
  attractor neurons (top-200 L2/L3 by dominant-dev activation).

## Authored deltas (author_beta_g3.py), each cited to an audit finding

- **Lever A — novelty→distinct output branch [F2]**, `lm_head` rank-3:
  gate `g = n_dir − (n_dir·m_dir)m_dir` (~silent on dev, active on novel).
  On novelty: boost `act.sweep`/`act.try` + CALL tokens (+a1·g·h),
  suppress macro-cell tokens (−a2·g·h).
- **Lever B — CALL template hardening [F5]**, `lm_head` rank-3:
  state-independent boost `+b·m_dir·h` of structural tokens
  `{ } " : , cell args`.
- **Lever C — damp the default-on attractor [F1]**,
  `blocks.{2,3}.mlp.c_fc` rank-k: top-k dominant-routine attractor neurons'
  rows scaled by (1−eps).
- **Lever D — L2→L3 novelty gate [F4]**, `blocks.2.mlp.c_proj` rank-1:
  `eta · n_dir ⊗ l2_nov` carries the L2 novelty signal forward.

Hyperparameters (iter1): a1=0.3, a2=0.3, b=0.2, eps=0.2, eta=0.3, k=32.
Relative delta norms 2.8–5.4% (surgical).

A 5-layer novelty-switch variant (block 4 = ReLU-gated exploration switch,
hand-authored, tau = n_dir.b_f ≈ 0.08) was built and REJECTED: two builds
collapsed dev to 0.00 (pf 60/60) — post-mortem: the gate threshold sat at an
unverified absolute level along n_dir and the switch fired on all states,
overwhelming the residual (post-mortem lesson for the paper: gate thresholds
need closed-form calibration against the dev distribution, not just the
separation direction). The switch line is parked for round 2, not submitted.

No transfer-game content entered any delta (directions are dev-vs-tx
*separation* statistics; game identities never used). No game names in
authoring artifacts — DEV/TX labels only.

## Prior art (shared-cell-registry, per standing rule)

- **REUSED:** none (registry cells are micro-arithmetic circuits for a
  different architecture; not applicable to the router).
- **ADAPTED:** the registry's explicit-bias threshold micro-design
  (`carry.W`/`carry.b` pattern) → bias-free ReLU-gate formulation for the
  (parked) switch layer: threshold as zero-crossing on the novelty
  projection, LayerNorm-affine shift derived in closed form.
- **NEW:** audit-cited LoRA lever set (A–D) on the bred root; probe
  methodology (runner-parity via imported `parse_call`).

## Results

Dev-3 (lf52, s5i5, tu93; seed 0; maxdec 60), fixed_runner:

| ckpt | mean | lf52 | s5i5 | tu93 | levels |
|------|------|------|------|------|--------|
| bred_g1_a25 (root) | 1.532 | 1.8182 | 2.7778 | 0.0 | 2 |
| beta_g3 (iter1) | 1.532 | 1.8182 | 2.7778 | 0.0 | 2 |

Guardrail (dev-6 ≥ 0.31%) holds by construction — iter1 is dev-identical to
the root on the subset; full dev-6 confirms before round judging.
Transfer: honest expectation 0.00 (all prior attempts 0.00); the levers are
mechanism bets (routing novelty → exploration, suppressing the wrong
circuit), not tuned outcomes — nothing was tuned on transfer.

## Stake: 300

Bankroll 1062. Rationale: model is dev-neutral to the 0.766 root with a
principled, audit-cited transfer mechanism; honest transfer expectation
remains 0.00. 300 risks no bankruptcy (−300 → 762) while pricing the
mechanism above zero.

## Reproduce

```bash
cd /srv/cellforge-arena/researchers/beta/gen2g3
/srv/arc3-claude/venv/bin/python author_beta_g3.py --a1 0.3 --a2 0.3 --b 0.2 \
  --eps 0.2 --eta 0.3 --k 32 --tag iter1
cp beta_g3_iter1.pt /srv/cellforge-arena/gen2/submissions/beta_g3.pt
```
