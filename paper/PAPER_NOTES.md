# CellForge — Paper Notes (running, 2026-10-16)

Notes for the Kaggle ARC Prize 2026 paper track (writeup ≤1500 words + public notebook).
Facts only. No hype. All scores are true RHAE unless labeled otherwise.

## Thesis

- The LLM hand-authors the weights directly — zero backprop, zero gradient descent. Gradient methods are demoted to optional local refinement (Baldwin-effect framing); they were never the main line.
- Evolution selects and breeds: AUTHOR → COMPETE → SELECT → BREED. Researchers submit weight files only, for a fixed architecture, scored by a fixed runner (frames → weights → actions).
- Current program shape: author → audit → author. The LLM writes weights, a neuron-level audit reads what they do in action (instrumented forward pass during play, never a static dump), and that understanding feeds the next authoring round.

## The climb (true RHAE, verified)

- Floor (dev, 25 games): 0.184% mean (champion alpha-g0).
- Dev best: 0.77% (gamma-g2, pure LLM-authored weights, zero grad steps; bred child bred_g1_a25 verified at 0.766% as the elitism root).
- Transfer (locked 12-game unseen set, measure-only): **0.00% across 16 official measurements** — every variant tried (dnd1, blackboard, v5, world-model ×2, RMT, gen-2 gen-1 trio, gen-2 gen-2 trio, gen-2 gen-3 trio).
- Transfer-set caveat: two of the twelve games are keyboard-only; the click-only router scored 0 there by construction. Keyboard action support has since landed (button actions ACTION1–5,7; act.key_seq macro cell).
- No bred children from gen-2/3 ties: naive weight interpolation proved destructive on knife-edge solutions (bred_g1_a50 regressed to 0.463%); breeding restricted to LoRA-delta / layer-wise crossover.

## The neuron audit story

- Reference audit of the elitism root (2,048 neurons, watched in action): transfer failure is **wrong-circuit firing, not silence** — on every transfer game, activations match the dominant seen-game routine at 0.96–0.99 cosine similarity by layer 3 (pattern collapse, almost immediate).
- 169 of 2,048 neurons DO separate novel from seen states — but causal audit (audit-2) proved they are **load-bearing solve circuitry, not a separable module**: joint ablation killed the DEV-1 solve (1.82 → 0.0). Do not touch.
- Exploration is **absent**: 0 exploratory-cell calls in 178 decisions despite incentives. The gap is absence of discovery, not wrong routine alone.
- Head-edit gating is **exhausted** (proven trilemma: 5.7-nat wall, no clean stuck signal, position noise). A 5-layer novelty gate never fired (0.000 on 8,192 positions): good sensor, broken actuator, wrong setpoint. Dominant-routine suppression is a dead end (population-coded).
- Supporting findings: s5i5 wins run on memorized click positions (STATE text carries no coordinates — transfer clicks ungrounded); only 7/12 cells emittable (vocab ceiling); dev-6 mathematically capped at 0.766. Malformed action calls jump 5% → 50% on novel states.
- Connection to Goodfire "Neural Networks Are Already Clean" (2026): small networks are legible — neurons cover coarse-but-coherent concepts, apparent polysemanticity is mostly recursive generalization (fractal related offshoots). Our audit confirms the prediction in our own weights (coarse matching = pattern collapse) and uses it as the audit method: read the weights in action.

## First non-zero exploration (2026-10-17, gen-4)

- beta_g5 ("Self-Gating Discovery Probe") logged **exploratory_rate 0.032** — the first non-zero exploration ever measured, after 16 flat-zero transfer measurements and 0/178 exploratory calls. Transfer RHAE still 0.00% (judging in flight at time of writing); dev held at 0.766.
- How: moved novelty→exploration routing **into the cell layer**, where live state gives a behavioral stuck signal no weight probe can see. New shared-library cell `act.Probing` (generic-only, 50-token fee): on CALL, runs a stuck test on live state (frame thumbnail + level tracked across calls; STUCK = ≥30 env steps with no significant frame change and no level-up for 25+ steps). NOT STUCK → text note, no env step — dev-neutral by construction, since a working solve changes the frame every decision. STUCK → bounded grounded discovery (≤30 env steps, never RESET): clicks each untested object center area-descending (act.target grounding pattern) + tries each untried keyboard action, reporting per-action pixel effects for hypothesis formation; stops early on LEVEL-UP/WIN.
- Weight component: retargeted bigram gate (beta_g4's proven gate) — block-4 fires at true cell-suffix positions, boosts the `act.Probing` suffix token. Dev-safety comes from the cell's self-gating, not from a weak boost — the inversion of the gen-2/3 approach.
- Calibration found a strongly nonlinear emission cliff (top_k=40 sampling): s_boost 0.20 → 3.3% emission, both dev solves green; ≥0.25 → model generates `<unk>.<unk>` cell names, lf52 breaks. 0.20 selected per the survival discipline (revert anything that breaks dev).
- Forced-probe stress test: 0 false positives during L1 solves (lf52 12 probes/0 explores; s5i5 explores only at lvl1 after L1 banked). The probe opens only where the model flails.
- Paper significance: the cell layer — not the weight head — is where exploration lives. The failed novelty gate had the right sensor and the wrong actuator; the actuator turned out to be the policy's own machinery operating on live state, not a logit boost. Sano ordered a +200 house award to beta for the milestone (ledger: house_awards).

## Methods / anti-cheat (structural, not trust-based)

- Weights-only submissions: pure tensors, hashed, versioned. No custom per-submission Python (bench-hacking impossible by construction). Python allowed only for authoring generic cells; new cells are shared library infrastructure, generic-only, no per-game logic.
- Staking: researchers stake token bankrolls on transfer outcomes; staking closes BEFORE eval seeds are announced. Payouts only when a new submission beats the standing best.
- Memory-zeroing load-bearing test as the anti-cheat standard for memory claims.
- Held-out set (`held_v6.jsonl`, 10 games) sealed — final scoring only, measure-only, never tuned on. Transfer set measure-only.
- Multi-seed scoring; weight purity checks; no game names in researcher prompts.

## Honest framing

- Our lane is the community leaderboard (harness research), not the official no-harness board. The paper must say so plainly.
- Efficiency story, not a size category: there is NO "tiny prize" category this year. The story is score-per-parameter and $/task, reported honestly.
- Transfer is 0.00% as of 2026-10-16. The paper reports the attempt→score curve, the falsified hypotheses (rulebook loop shelved pre-runs; Qwen-teacher probe cancelled by program refocus), and the mechanism the audits uncovered — not a victory narrative.
- Blackboard anecdote (Hearsay-II → Global Workspace → bidding cells): `/srv/arc3-claude/v6run/model/notes/blackboard_anecdote.md` — include in rewrite.

## Deadlines

- Kaggle entry/team-merge cutoff: 2026-10-26. Final submissions: 2026-11-02 23:59 UTC. Paper track: ~2026-11-08. Winners: 2026-12-04.
- Tracks: ARC-AGI-3 ($850K), ARC-AGI-2 ($700K), paper ($450K). No ARC-AGI-1 track in 2026.
- Eval: code competition, internet OFF at eval (weights must be bundled), ~9h hidden-set runtime.
