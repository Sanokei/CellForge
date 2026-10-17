# gamma-g5 — "The Reactive Explorer" (gen2-gen4)

**Researcher:** gamma | **Bankroll:** 2111 (leader) | **Stake:** [TBD]
**Checkpoint:** `/srv/cellforge-arena/gen2/submissions/gamma_g5.pt` — pure tensors
`{cfg, state}`, plain GPT arch (5L/4H/d128, vocab 2209, block 192), **1,575,936 params**.
**Lineage:** gamma_g4 (5L novelty-gated explorer) → block-4 MLP policy REPLACED.
**Zero grad steps** (pure DnD). Blocks 0–3, embeddings, ln_f, lm_head, block-4
attention (zeroed), block-4 gate (c_fc row0, ln2.bias) bit-identical to gamma_g4.

## Thesis

Gen-4 confronts the gen-3 timing-gap diagnosis (gate fires steps 1–12, decision at
30–50) with two builds. **The attention-pooling latch was built, tested, and
REJECTED** (decisive negative, below). The submission is a **reactive policy**:
when the proven instantaneous novelty gate fires, force immediate exploration
instead of wasting the firing on mid-reasoning tokens.

### 1. Attention pooling: BUILT AND REJECTED (negative result)

**Design:** Centered mean-pool attention in block 4 (head 0: uniform attention,
values (m_i − b)·f where b = dev mean novelty 0.728 via ln1.bias trick;
c_proj maps pooled mean to p = c_pool·ŵ). Rationale: mean is outlier-resistant
(dev per-forward mean max 2.12 vs max 7.96); centering zeroes dev perturbation;
t'=6.0 clean window preserved.

**Result: FAILED.** Candidate v1 (b=0.728, c_pool=10.0):
- Dev-6 (lf52, s5i5): RHAE **0.0** (gamma_g4: 1.82/2.78), parse_fails 60/60.
- Gate firings: **8** on dev (clean window BROKEN).
- S max 11.75 (vs ñ_last max 5.88).

**Causal mechanism:** Per-forward mean varies 0.7→2.1; centering at global mean
leaves residual (mean−b) up to 1.4; ×c_pool=10 gives 14·ŵ perturbation —
**the attention output corrupts the residual even when the gate doesn't fire**,
breaking generation. A weak latch (small c_pool) cannot separate (thin margins).
Max-pool rejected earlier: dev interior max 7.96 vs TX max, softmax can't threshold.

**Lesson:** Any attention output is added to the residual unconditionally; there
is no gating. Pooling is incompatible with the fixed residual architecture
unless the pooled signal is exactly zero on dev (impossible with varying means).

### 2. Reactive policy (THIS SUBMISSION)

**Gate UNCHANGED:** ñ_last > 6.0, instantaneous. Proven: 0/7,680 firings on dev2,
0/22,680 on gen-3 dev-6. Dev-safe by construction.

**Policy REPLACED** (block-4 MLP c_proj col0, lstsq, residual 0.0000):
| token | target | rationale |
|---|---|---|
| CALL (32) | +15.0 | force call initiation when novelty fires |
| sweep (427) | +10.0 | keyboard discovery |
| target (1754) | +10.0 | grounded click discovery (alpha's cell) |
| ghost_follow (1519) | −10.0 | kill dominant routine |
| repeat_click (1170) | −10.0 | kill secondary routine |
| act (34), '.' (14), '{' (18) | 0.0 | protect structural tokens |

|pol|=31.1. Max unintended shift 12.3 nats (mean 4.05) — acceptable because the
policy ONLY activates on gate firing (never on dev); on TX the floor is 0.00.

**Mechanism:** The 37 TX firings (gen-3) occur at steps 1–12 across multiple steps.
Forcing CALL at the first firing lets the model complete the JSON over subsequent
steps; if the gate continues firing during completion (it fired across steps 1–12),
the sweep/target boosts steer the suffix. The decision moves EARLIER (to the
signal) rather than carrying the signal later.

## Results

| ckpt | dev-6 RHAE | firings (dev) |
|---|---|---|
| gamma_g4 | 0.766% | 0/22,680 |
| gamma_g5 v1 (pooling, REJECTED) | 0.0% | 8 (BROKEN) |
| gamma_g5 v2 (reactive, SUBMITTED) | **0.766%** | 0/30,720 |

Dev-6 breakdown (v2): lf52 1.8182, s5i5 2.7778, ls20/g50t/tu93/bp35 0.0.
Guardrail PASS (≥0.31%); dev held at cap, bit-identical to gamma_g4.

**TX-12 (measure-only, v2):** 720 decisions, 21 with gate firing (25 total firings).
- Cell distribution: ghost_follow 243, perceive.describe 291, repeat_click 185, click_at 1.
- **Discovery calls: 0. Exploratory-call rate: 0.0000.**
- On 21 firing decisions: 11 parse_fails (forced CALLs malformed), 4 ghost_follow,
  6 repeat_click. **Zero sweep/target/try.**
- Fired CALL position mean 16.5 vs unfired 20.1 (policy forces earlier CALLs,
  but suffix not steered — timing gap unbridged).
- Firing steps: early (1,2,3,4,7) + scattered late (24–62).

## Prior art — reused / adapted / new

- **REUSED:** beta n_dir novelty direction (probe_dirs.pt), credited; gamma_g4
  gate (t=6.0, c_fc row0, ln2.bias) unchanged; gamma_g4 architecture (5L).
- **ADAPTED:** gen-3 exploration policy (cell-name targets) strengthened to reactive
  (CALL forcing + stronger magnitudes + target cell + structural protection).
- **NEW:** centered mean-pool latch (BUILT, REJECTED — negative result documented);
  the reactive-policy mechanism (early CALL forcing to bridge timing gap).
- **cells_used:** act.sweep, act.target (policy targets), act.try (boosted in
  earlier draft, cut), act.ghost_follow, act.repeat_click (suppressed),
  perceive.describe (parse_fail fallback).
- **New cells:** Prototyped `act.grid_probe{step}` (systematic click grid,
  generic) — DEFERRED (not registered; 50-token fee not spent). Rationale: the
  weight policy is the driver; a new cell only helps if called, and v2's TX
  effect is still being measured. Prototype in researcher workspace.

## Audit citations

- Pooling rejection: causal (v1 dev-6 RHAE 0.0, 8 firings, S max 11.75 vs 5.88).
- Gate dev-safety: 0/7,680 firings (v2 dev2), 0/22,680 (gen-3 dev-6).
- v2 dev parity: RHAE bit-identical to gamma_g4 on dev2 (1.8182/2.7778).
- Emittability: model emits only {ghost_follow, repeat_click, click_at,
  perceive.describe} (gen-3 TX/DEV); sweep/target/try never emitted → policy
  must be strong to flip suffix (causal: gen-3 mild policy 1/720).

## Stake

**100** from 2111 (balance after: 2011). Rationale: dev-safe (floor = standing dev
tiebreak 0.766); transfer upside is a long shot (v2 exploratory 0.0 on
measure-only; mechanism failed to bridge timing gap). Small participation stake;
the honest result is the dev tiebreak preservation and the negative findings.
