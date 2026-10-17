# CellForge: tiny LM router + capability cells + learned glue — method log

Paper trail for the CellForge loop. Updated as generations run.

## Architecture (2026-10-15)

- **Tiny LM router** (nanoGPT-style decoder, ~1.6M params: 4 layers, 4 heads,
  d=128, word-level vocab from the trace corpus): does short English reasoning
  (what it sees, what it hypothesizes, what to try next) and emits one cell
  call per reasoning step.
- **Capability cells** (`model/lm/cells.py`): fixed tools with a JSON-ish glue
  protocol — `CALL {"cell": "<name>", "args": {...}}`.
  - Perception (free): `perceive.describe`, `perceive.objects`, `perceive.diff`,
    `perceive.hud`
  - Action (step the env): `act.try`, `act.sweep`, `act.click_at`,
    `act.pick_drop`, `act.repeat_click` (dial macro), `act.ghost_follow`
    (one ghost-chain link; chains by repeated calls), `act.reset`
- **Glue**: fixed protocol; the *routing decisions* are what's learned
  (distillation) or authored (researchers).
- **Harness**: real ARC-AGI framework (`arc_agi` 0.9.9 + `arcengine` 0.9.3 pip
  packages), official scorecard scoring. Every number reported as
  (model, version, harness).

## The loop: AUTHOR → COMPETE → SELECT → BREED

- **AUTHOR**: no pre-training. Frontier-LLM researchers probe dev games and
  author tiny agents' routing policies + weights directly from understanding.
  Understanding → weights, no gradient pre-training step.
- **COMPETE**: population plays dev games end-to-end via `compete.py`,
  scored by the framework scorecard on MULTIPLE seeds per game.
- **SELECT + BREED**: rank by mean score; top agents' ideas/weights breed the
  next generation; lineage recorded in `generations/`.
- **Researcher survival**: researcher score = best agent's score; bottom
  researcher eliminated each generation, replaced fresh (may study winners).
- **Anti-cheat** (structural): held-out set sealed; multi-seed scoring kills
  memorization; weight-purity perturbation checks; deterministic sandboxed
  agents, no network.

## Wave-2 findings folded in (2026-10-15)

- **VolatileMask3**: original volatility detection + VETO — never mask pixels
  whose changes discriminate actions (protects block trails, stamp toggles,
  HUD). Strict action-correlation variant regressed (missed lf52's progress
  bar where keyboard actions are no-ops).
- **Painter-demotion**: demote action classes that only ever deposit novel
  pixels (ls20: 1481→64 actions, 0.005→0.422; g50t 0→0.621).
- **Ghost chains** (lf52): click token → ghost marker appears → click ghost →
  teleport → next ghost. Solved in 8 chained clicks (one macro per link).
- **Dial macros** (s5i5): tokens are dials — (46,21)×7 then (24,45)×6, 13 clicks.
- **Caution**: lp85's 2.78 was a lucky 5-action solve — excluded from trace
  teaching corpus; do not overfit to luck.

## Trace corpora

- `model/lm/traces.jsonl` (v1, 1140 traces): scripted-teacher corpus —
  weak teacher (0.0 levels), basic routing structure only. Superseded.
- `model/lm/traces_w2.jsonl` (v2): blitz13 (all wave-2 fixes) action streams
  translated to cell-call traces with strategy-grounded reasoning. Excludes
  lp85. Corpus stats recorded at generation time.
- `model/lm/traces_grammar.jsonl` (12 VERIFIED grammar traces): s5i5 dials
  (13 clicks, score 2.78) + lf52 ghost chain (7 links, score 1.82), confirmed
  by direct probe. Lost once to a concurrent-write clobber (trace_gen's "w"
  vs probes' "a"); regenerated into this separate file.

## Tiny LM router results (2026-10-15)

- router_v2 (1,378,816 params, 4L/4H/d128, vocab 2209): trained on 32K v2
  traces, loss 0.33. Learned the routing GRAMMAR (valid cell calls, English
  reasoning) but solved 0 levels — the 12 solve traces were drowned out.
- router_v3: fine-tuned from v2 on the 12 grammar traces x100 (loss 0.068).
  **Solved s5i5 level 1 (score 2.78) and lf52 level 1 (score 1.82)**
  end-to-end through the arc_agi framework. 0 levels on keyboard games
  (ls20, g50t) — fine-tune overfit to click grammars. Parse-fail rate
  20-50% (format robustness gap); s5i5 solve was wasteful (174 actions).
- Eval prompt MUST match training format (STATE/THINK/CALL/OBS blocks;
  history as prior blocks). CALL parser needs balanced-brace matching +
  whitespace stripping (word tokenizer spaces out JSON punctuation).
- Honest assessment: narrow but real — a 1.4M-param LM doing English
  reasoning → cell routing → level solves. Not a general player yet.

## Cost tracking

- Box: Hetzner cx23-class, ≈ $0.0089/hr. $/task = secs/3600 × rate, reported
  per game in compete.jsonl. Last year's tiny bar: 24% at $0.24/task —
  we track both and maximize score, not satisfice.

## Staking mechanism (2026-10-15)

- Researchers gamble token bankrolls on their own models (ledger:
  `/srv/cellforge-arena/ledger/bankroll.json`, every change audited).
- Start 1000 tokens. Tokens buy compute (1 token / game-seed rollout) and
  are staked. Full official eval cost deducted from bankroll.
- Staking closes BEFORE eval seeds are drawn/announced (anti-cheat ordering).
- Payouts by generation rank: #1 → 3×; top 25% → 1.5×; above mean → 1.1×;
  below mean → −50%; bottom 25% → −100%.
- Elimination: bankruptcy, or bottom-25% twice running.
- Agent token budgets: researcher declares per-game think-token cap, funded
  from their bankroll (100 think tokens = 1 bankroll token); harness meters
  and cuts reasoning off at the cap. Reported: tokens allocated vs burned,
  score-per-kthink.

## Public leaderboard + mandatory breeding (2026-10-15)

- Every generation publishes `/srv/cellforge-arena/ledger/leaderboard.json`,
  mirrored to `/srv/cellforge-arena/common/leaderboard.json`
  (researcher-readable): per-researcher bankroll, stake, agent score, tokens
  burned, rank, status; agent leaderboard; champion lineage.
- Champion weights public (`model/champions/<gen>_<name>.py`); private notes
  stay private.
- Eliminated researchers are replaced by successors REQUIRED to start from
  the champion's bred weights/cells, documenting every mutation. Lineage in
  `model/generations/<gen>/lineage.json`. The gene pool only compounds.
- Generation cycle: AUTHOR → STAKE/BUDGET → COMPETE → PAYOUT → PUBLISH →
  ELIMINATE → REPLACE from champion lineage → next generation.

## Generations

### gen0 (2026-10-15/16) — CHAMPION: alpha (0.766 mean, 4 levels)

Selection: 6 games (lf52, s5i5, ls20, g50t, tu93, bp35) x 2 fresh seeds
(139878, 763613), drawn AFTER staking closed. maxdec=300.

| researcher | agent | mean | levels | stake | payout | balance | think |
|---|---|---|---|---|---|---|---|
| alpha | alpha-g0 | 0.766 | 4 | 500 | 1500 (3x, rank 1) | 1915 | 6038 |
| beta | beta-g0v2 | 0.000 | 0 | 250 | 125 (below mean, -50%) | 746 | 9622 |
| gamma | gamma-g0 | 0.000 | 0 | 1 | 0 (bottom 25%, -100%) | 861 | 10944 |

- Alpha solved lf52 L1 (1.82) and s5i5 L1 (2.78) on BOTH fresh seeds —
  seed-stable, not memorized. Key discovery: keyboard sweep BREAKS the lf52
  ghost chain (hidden selector state); open_sweep=False.
- $/task: alpha $0.000002, beta $0.000006, gamma $0.000005.
- No eliminations. Gamma: 1 consecutive bottom-25% (out if again in gen1).
- Champion weights public: model/champions/gen0_alpha.py. Lineage: founders.
- Purity: beta 9/9; alpha 3/5, gamma 5/6 on ft09 probe. Dead weights control
  branches ft09 never exercises (ghost probes, dial repeats, pick-drop) —
  code-used but trace-invisible; documented test limitation, not cheating.
- Think-token metering worked: alpha hit its 500/game cap (harness cut off),
  beta/gamma under theirs. Billed 100:1 from bankrolls.

- gen0: researchers alpha/beta/gamma authoring; 6 selection games × 2 seeds.
  (ledger in `model/generations/gen0/`)

## RHAE scoring — paper-exact spec (2026-10-16)

Switched all selection/scoring to TRUE RHAE per the ARC-AGI-3 technical
report (arXiv 2603.24621, sec 4.1). Scorer: model/rhae_scorer.py.
- Per completed level l (1-based): S_l = min(1.0, h_l / a_l)^2, where h_l =
  human baseline actions (framework-shipped baseline_actions; the paper
  defines it as second-best human — we take shipped values as given).
- Incomplete levels: 0. Environment: E_e = sum(l * S_l) / (n(n+1)/2) —
  linearly level-weighted, later levels count more.
- Total T = mean of E_e across environments, reported as %.
- Action accounting is authoritative: the frameworks scorecard tracks

## RHAE scoring — paper-exact spec (2026-10-16)

Switched all selection/scoring to TRUE RHAE per the ARC-AGI-3 technical
report (arXiv 2603.24621, sec 4.1). Scorer: model/rhae_scorer.py.
- Per completed level l (1-based): S_l = min(1.0, h_l / a_l)^2, where h_l =
  human baseline actions (framework-shipped baseline_actions; the paper
  defines it as second-best human — we take shipped values as given).
- Incomplete levels: 0. Environment: E_e = sum(l * S_l) / (n(n+1)/2) —
  linearly level-weighted, later levels count more.
- Total T = mean of E_e across environments, reported as %.
- Action accounting is authoritative: the framework's scorecard tracks
  actions_by_level itself. Every env.step counts, including failed attempts
  before a level completes ("total actions on first exposure").
- NOTE: the framework's built-in EnvironmentScoreCalculator uses cap 115
  (not the paper's 1.0); we recompute per paper spec from level_actions.

## Competitive framing — read this before citing any number (2026-10-16)

- The OFFICIAL ARC-AGI-3 leaderboard never reports public-set scores
  ("evaluating on it is emphatically not a valid measure of progress
  towards AGI"); official runs are on PRIVATE games, no harness, one fixed
  system prompt. Our lane is the COMMUNITY leaderboard — harness/agent
  research, which the paper explicitly values as economically valuable.
  Every public-set number we report is labeled as such, honestly.
- "Last year's 24%": fact-checked vs the ARC Prize 2025 Technical Report.
  NVARC won ARC Prize 2025 with 24.03% — on ARC-AGI-2 (static grids), NOT
  ARC-AGI-3 (launched Mar 2026), and it was the OVERALL top score, not a
  tiny category. Tiny-model fame last year: TRM (part of NVARC's approach)
  and CompressARC (76K params, 20% on ARC-AGI-1, no pretraining, 3rd place).
  "24 cents/task" could not be verified from any source. Our benchmark is
  ARC-AGI-3 RHAE: GPT-6 Astra 62.7%, Claude Opus 5 30.2% (Oct 2026).
- Bimodal-transfer warning (paper): handcrafted harnesses show EXTREME
  bimodal transfer — the paper's Duke harness hit 97.1% on a seen TR87
  variant and 0% on unseen BP35. Hand-authored game knowledge does NOT
  reliably transfer. Our structural defenses: sealed held-out set
  (held_v6.jsonl, never touched), multi-seed eval (fresh seeds per
  generation, drawn after staking closes), and selection pressure AGAINST
  hardcoded coordinates (general mechanisms beat fixed maps on new seeds).
  Generalization to unseen games is THE claim to prove; the paper must
  cite this limitation openly.
## Transfer RHAE — the progress metric (2026-10-16)

Per Sano: TRANSFER RHAE (first-contact score on unseen games) is now THE
progress metric; dev-set RHAE is secondary.
- Transfer set v1: model/transfer_set.json — 12 games never appearing in
  blitz traces, router training corpora, grammar traces, targeted mechanic
  probes, or gen-0 selection: ar25, cd82, lp85, m0r0, r11l, sb26, sp80,
  su15, tn36, tr87, vc33, wa30. (lp85 flagged: lineage-known lucky solve,
  no encoded mechanics. v1 scripted traces lack game labels — coverage
  unknown, noted as caveat.)
- Protocol: one run per game per agent, seed 0, no retries, no tuning.
  True RHAE via model/rhae_scorer.py.
- Measure-only: never tune on it, never add its games to training.
- RESULTS: alpha-g0 transfer RHAE = 0.00% (0 levels across all 12 games)
  vs dev RHAE 0.18% (25 games) / 0.77% (6 games). Textbook bimodal
  transfer — exactly the paper's warning, confirmed in our own numbers.
  The loop must optimize transfer, not dev.
- Note: after first contact these games are eval-seen; future transfer
  measurement needs fresh unseen games (held_v6.jsonl remains sealed for
  final scoring only).
## router_v3 true-RHAE re-score (2026-10-16)

router_v3 (1,378,816 params) on 6 gen-0 games, seed 0, maxdec 60:
- MEAN RHAE = 0.31%, 2 levels.
- lf52 L1 in 32 actions vs human baseline 32 → S=1.0 → env 1.8182%
  (exactly human-second-best action count).
- s5i5 L1 in 166 actions vs baseline 20 → (20/166)^2=0.0145 → env 0.04%.
  The champion's 13-action grammar solve scores 2.78% on the same level:
  RHAE punishes action count quadratically, so the LM's wasteful play
  (166 vs 13) is the efficiency gap gen-1 must distill away.
- 20-50% CALL parse fail rate confirmed (12-46 parse fails/game).
- Champion comparison on same 6 games: 0.77%. LM < champion on RHAE.
Result file: model/rhae_lm6.json.
## router_v3 transfer RHAE (2026-10-16)

First-contact run, 12 unseen games (transfer_set.json), seed 0, no retries:
- TRANSFER RHAE = 0.00%, 0 levels across all 12 games.
- Champion also 0.00% on the same set. Both systems exhibit total
  bimodal transfer: all capability is game-specific, nothing generalizes.
- Notable: parse-fail rates stayed high on unseen games (3-46/game) —
  the LM's call grammar is itself overfit to seen games' action patterns.
Result file: model/rhae_transfer_lm.json.

## Blackboard router v4: passive cell intake + bidding (2026-10-16)

Architecture pivot per Sano: perception moves OUT of the LM's action loop.
Every capability cell gets a passive observe() running EVERY decision step
(cheap numpy, zero env actions), returning (annotation, bid_0_to_1). Bids
>= 0.30, top-3, become INTAKE lines in the prompt. The LM never calls
perception cells -- perception arrives free, like peripheral vision. Only
act.* cells remain callable. Paper framing: Hearsay-II blackboard lineage
meets Global Workspace Theory; learned bids as evolved glue (RIMs /
Perceiver-style cross-attention intake as modern echoes). See
notes/blackboard_anecdote.md.

Why the pivot: router_v3 fumbled tool syntax 20-50% of the time, and its
vocab (reused from an older corpus) literally lacked the perceive.*
cell-name tokens -- it could not emit them, only <unk>. The LM learns
noticing, not tool use. Constrained decoding over the act.* CALL grammar
(callgen.py) makes structurally invalid CALLs impossible by construction.

Observers (model/lm/cells_intake/observers.py), each authoring its own
salience: scene (object list, bids on change), change (frame delta, bid
scales with changed px), counter (per-color count deltas), novelty (new
small components via the proven _new_small_comp flood-fill detector -- the
generic component labeler misses 1px markers; nearest-to-last-click
disambiguates marker constellations, mirroring the ghost_follow cell), hud
(top-3 rows). Anti-cheat invariant: annotations are generic observations
(positions, counts, changes) -- never game names, never level solutions, no
game-identity leaks. Context budget (160 tokens, ruthless): prompt = STATE
+ INTAKE (top-3, truncated) + THINK; no history -- the blackboard is Markov
by design.

Evolved glue: cells_intake/intake.py BidLogger logs which intakes were
shown each decision; on level-up the trailing window gets recency-weighted
credit. tune_bids.py converts credit into per-observer bid multipliers.
First pass (8 episodes, 2 level-ups): novelty 1.103, counter 1.000, scene
0.897 -- selection pressure on RHAE already tunes how loudly each cell bids.

Training (trace_gen_bb.py): MINIMAL-action grammars replayed through the
REAL observer loop so THINK demonstrably uses INTAKE content -- s5i5 dials
(13 actions x 6 seeds), lf52 ghost chains (8 actions x 6 seeds), ls20 blitz8
painter-demotion (64 actions). THINK echoes intake coordinates, teaching the
copy chain INTAKE -> THINK -> CALL. Efficiency weighting: episode rows
duplicated round(64/actions)x. Auxiliary fluency: 300 lines from
/srv/cellforge-private/realtext/valid.txt. Vocab rebuilt from the true
render (v3's omitted intake lines and 37/64 numbers); all act cell names +
0-63 + JSON grammar force-included. train_bb.py: 4L/4H/d128, 1,064,448
params, from scratch (v3 weights unusable -- vocab changed), 500 steps CPU.

Bugs fixed along the way (load-bearing for the method): (1) trace generator
returned the POST-action frame as prev_frame, blinding change/novelty
observers in training (frame==prev); (2) a cell-primer episode on s5i5
states taught conflicting actions for identical (state,intake) and poisoned
the dial policy -- removed; vocab coverage for unused cells comes from
mandatory tokens instead; (3) novelty observer initially used the generic
labeler which misses 1px markers.

Inventory (/srv/cellforge-inbox, /srv/cellforge-public, /srv/cellforge-
private, /opt/cellforge/code): directly portable was the realtext corpus
(fluency) and tinygrammar.py (unused for now). The logic-circuit threshold
nets are bit-level (no honest mapping to frame observers); memory/harness
modules are provenance infra, not game cells.


Results (paper-spec RHAE via rhae_scorer.environment_score, dev games,
seed 0, temp 0.2 -- decisive routing; temp 0.7 sampling noise breaks the
precise grammars):

- s5i5: L0 in exactly 13 actions (minimal dial grammar: 2 macro calls),
  RHAE 2.78 (level score 1.0). v3 took 166 actions for 0.04 -- efficiency
  distillation worked.
- lf52: L0 in exactly 8 actions (minimal ghost chain), RHAE 1.82 (level
  score 1.0).
- Parse-fail rate: 0% by construction (constrained CALL decoding; grammar
  restricted to vocab-available cells: click_at, ghost_follow, repeat_click,
  reset, sweep, try -- pick_drop excluded, x1/y1/x2/y2 not in vocab).

Load-bearing tests (intake ablations, temp 0.2):
- lf52 --drop novelty: 1 level -> 0 levels. PASS (novelty load-bearing).
- s5i5 --drop scene: 1 level -> 0 levels. PASS (scene load-bearing).
- lf52 --scramble novelty: STILL 1 level in 8 actions. REVEALS memorization:
  the deterministic L0 chain is memorized; the model completes it from the
  intake's presence signal without needing its content. Textbook
  bimodal-transfer warning -- dev-set solves can be memorized, which is why
  TRANSFER RHAE is the top metric, not dev RHAE.

Copy mechanism: ghost_follow coordinates get a +2.0 logit bonus toward the
novelty-reported marker (pointer-network-style inductive bias); 400
synthetic copy drills (random positions) teach the general copy operation
since lf52 L0 positions are seed-deterministic (only 8 unique positions --
memorization trap).

Honest limits:
- Keyboard games (ls20/g50t/tu93/bp35): minimal training (64 ls20 traces
  only); click_at/sweep/reset are vocab-available but behaviorally untrained.
- No recovery behavior: teacher never demonstrates mistakes, so the model
  loops when off-chain. Recovery traces are future work.
- Think text sometimes hallucinates intake content; the CALL is what scores.
- 160-token window honored: prompt = STATE + INTAKE (top-3) + THINK only.
- held_v6.jsonl sealed throughout. No game names or solutions in intakes.


## Fixed runner: reference-pattern lineage (2026-10-16)

The ARC-3 fixed runner (`model/fixed_runner.py`) is a deliberate sibling of
`/opt/cellforge/code/src/cellforge/reference/evaluator.py`, not a new invention.
Sano remembered correctly: weights-only scoring was already solved.

REUSED from the reference design:
- Versioned evaluator (`EVALUATOR_VERSION = "arc3-ref-eval/1"`; protocol changes
  => new version id, separate result epochs) — from evaluator.py + protocols.py
  ("Changing a protocol requires a new version id").
- Weights as DATA: the checkpoint is the "Graph equivalent" — pure tensors,
  sha256-hashed (`model_hash` in every result), versioned format id
  (`cellforge-arc3-ckpt/1`, cf. `cellforge-graph/1`).
- Per-case failure reasons, categorized and counted (cf. evaluator.py's
  `failure_reasons`); latency accounting (total + ms/decision, cf. ms/case);
  compute proxy (param count x decisions, cf. MACs).
- Submission contract (cf. construction/contracts.py): shapes match fixed spec,
  all finite, deterministic — the ARC-3 unit test for a weight submission.
- Protected surface (cf. reference/protocols.py): "Final-split seeds are never
  stored here: the operator supplies them on the evaluation server after an
  artifact is frozen." Our staking-closes-then-seeds rule is this, with lineage.

ARC-3-SPECIFIC (could not reuse):
- The "dataset" is interactive games, not static images: per-level action
  counts, not per-case predictions; the metric is RHAE, not accuracy.
- The fixed arch is the GPT router (4L/4H/d128, vocab 2209), not the cell/bundle IR.
- LoRA-delta submissions (rank <= 8) as a first-class submission type alongside
  full checkpoints — the DnD authoring interface.
- Wall-clock per decision as a first-class metric (Kaggle 9h budget).
## eval_bb.py checkpoint/vocab matching (2026-10-16)

Bug: eval_bb.py hardcoded vocab_bb.json (988 tokens) while instantiating the
model from the checkpoint's own cfg. A checkpoint trained under a different
vocab (router_v3.pt, vocab_size 2209) then sampled token IDs beyond the decode
table, raising IndexError (vocab index out of range) in generate_think_call,
the callgen.py decode step.

Fix (root cause, no clamping hacks): at startup, read the checkpoint cfg
vocab_size and select the vocab file whose length matches (vocab_bb.json is
the default; vocab.json and vocab_cur.json are alternates); hard-fail with a
clear message if none matches. Prompt encoding, CallGrammar, and decode then
all share the checkpoint's native vocabulary. Verified by exercising the exact
crash site offline: valid constrained CALL emitted, zero IndexError.

Note: running an intake-naive checkpoint (v3, never trained on INTAKE lines)
through the blackboard harness is a harness-ablation measurement, not the
blackboard system's transfer number (that is router_v4.pt with vocab_bb).
Eval temp 0.2 (track protocol; temp 0.7 sampling noise breaks precise
grammars).

## CPU eval throughput: torch thread thrash (2026-10-16)

Operational finding, relevant to the Kaggle 9h budget: on the 2-vCPU box,
torch's default 2-thread matmuls under 4-way process contention gave
0.51 s/forward for the 1.4M router; OMP_NUM_THREADS=1 gave 0.02 s/forward
(25x). End-to-end, eval_bb.py went from ~2.5 h/game to ~25 s/game (360x wall,
mostly the thrash). Lesson for the paper's compute story: for tiny models on
weak CPUs, single-threaded inference dominates; thread oversubscription is the
bottleneck, not FLOPs. All LM evals on this box should export
OMP_NUM_THREADS=1 (also speeds up the dnd1 and entry evals still running).
