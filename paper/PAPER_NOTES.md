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

## Gen-5: literature program (2026-10-17, Sano's order)

- Sano ordered literature help; two scout teams returned 12 papers with implementation sketches; all three researchers took up mechanisms. Full reports: `~/workspace/lit-scout-exploration-tiny-models.md`, `~/workspace/test-time-adaptation-scout.md`.
- **RAGEN-2 MI diagnostic (gamma):** first quantitative template-collapse measure — cross-scored THINK traces vs state prompts. Solved dev retrieval 0.150 (chance 0.025); own−best-other margin NEGATIVE (−0.195/−0.287 nats) — another state's prompt scores a trace higher than its own. Template collapse holds even on solved games. Reusable: `mi_collect.py`, `mi_score.py`.
- **Hebbian fast-weight gate (gamma, Chaudhary 2025):** built, validated, runner-integrated (`PlasticBlock4` in gpt.py + fixed_runner.py, opt-in via cfg.plastic). Dev-neutral (0.766 bit-identical), clean monotone dose-response. But: sensor fires at `<unk>` tokens, not novelty (bp35: 57/78 firings) — actuator built, sensor semantics suspect. Transfer 0.00. 2.5× latency cost.
- **Plumbing finding (beta):** gen-4's probe reports never reached the model at all — the runner discards act.* cell return values; only `perceive.describe` output becomes STATE text. Beta's describe patch fixes it. Discovery conversion still 0 (strict definition): seeded-hypothesis test perturbs action distribution (parse-fails 26→53) but doesn't convert to hypothesis-driven behavior. Channel open, gap narrowed not closed.
- **New cells:** `perceive.explore_map` (alpha, 8×8 per-level visitation tracker, fee 50), `act.learn` (beta, hypothesize→falsify in exactly 1 env step, verified live: `HYP h1 ... | TEST: CONFIRMED`), `perceive.reveal` (beta, free cell + passive verdict in describe output). Library now 18 cells. CDE surprise meter skipped (cells can't access THINK logits; needs runner change). ExpA routing: design accepted, implementation deferred. TTMM expert bank + Go-Explore/MAP-Elites: not taken up.
- **Judging:** all three transfer 0.00 (21 official measurements at zero), dev 0.766 each. 3-way tie → stakes returned, no payouts, no breeding. Bankrolls: γ2111 / α1427 (−50 fee) / β1112 (warning persists — tie = no tick; gen-4 act.Probing fee reconciled).
- Honest verdict: the cell layer now has exploration (Probing), hypotheses (learn), fast weights (plastic) — none solve transfer levels. Next levers: better sensors, TTMM experts, Go-Explore traces.

## Gen-6: predictive surprise loop (2026-10-17, Sano-approved via AdaJEPA arXiv:2606.32026)

- Full loop built with zero gradients: gamma's `perceive.predict` (JEPA-style latent predictor, authored/frozen kNN over 354 transition triples, 8×8 thumbnail latent, transition tuples from rollout traces — never answer traces; low-level pixel dynamics, not game rules) → surprise intake → Hebbian fast-weight gate (PlasticBlock4) modulated by real surprise; alpha's field-journal context + `plan.forecast` MPC-lite advisor; beta's act.learn for English revision.
- **Gamma closed the loop on transfer-12** (surprise loop wired in checkpoint cfg): calibrated sensor (523×/36×/5.8× LOO separation; correctly silent on stereotyped dynamics), gate bug found+fixed (dense per-position write → self-similar readout ~20× residual → parse-fail cascade; fixed with single anchored write + cap). Transfer run: exploratory 0.65, surprise 0.26, **0 levels**. Exploration without grounding is motion, not progress.
- **Alpha's critical negative:** loop intake breaks BOTH dev solves (lf52 1.82→0.0, s5i5 2.78→0.0) across 4 prompt variants — the 1.38M router's STATE→THINK→CALL template is too brittle for ANY prompt modification. "Intake-only = dev-safe" does NOT hold. Kills the whole prompt-intervention class on current weights; viable path is harness-side predictor use (TTMM expert switching) or training with intake from the start.
- **Beta's <unk>-soup diagnosis:** discovery intake is 44% <unk> — HYP/TEST/CONFIRMED/VIOLATED all out of vocabulary; verdict polarity invisible to the model. Explains gen-5's "perturbs but doesn't convert." Cheapest high-EV fix on the board: vocab-safe cell return formats.
- Beta's Go-Explore pipeline works (29/29 replay fidelity, 20 kept trajectories) but KL-anchored distillation of a zero-init block doesn't move behavior (output distribution identical). Merge bug fixed by coordinator (missing _HYPO_* constants → fired probes raised NameError on shared runner).
- **Judging:** all three transfer 0.00 (24 official measurements at zero), dev 0.766/0.7024/0.7660. 3-way tie → stakes returned, no payouts/breeding. Bankrolls: γ2311 / α1627 / β1312 (beta warning persists). Sano +200 effort awards to all three.
- Full report: arena/gen2/GEN6_REPORT.md.

## Gen-7: CONVERSION + the red-team (2026-10-17, Sano's standing directives)

- Theme: does exploration turn into levels? Answer: no. Transfer 0.00 × 27 official measurements (3-way tie, stakes returned). Dev 0.766 all three (alpha recovered from 0.7024).
- **Red-team instituted as a parallel track every generation** (Sano): adversarial verification of claims, not researchers. First HOLES.md delivered 10 ranked holes mid-generation: (1) exploratory_rate was INVERTED — measured stuckness (0.94 stuck at 0.98 "exploratory"), every claim on it void; (2) Hebbian gate behavior-neutral (0/3 flips, KL 0.002–0.006); (3) explore_map emits into the void (3rd instance of discarded-returns class); (4) discovery intake 44–80% OOV; (5) transfer = two memorized click triples + 52–58% parse fails, 0 probes/120; (6) ledger log stale, funds correct; (7) researcher Python outside submission hash; (8) surprise gate permanently open; (9) TTMM never executed; (10) hist[-2:] leaks across level-ups. Verified clean: seed freshness, artifact integrity, zero contamination. Standing rules adopted: no cell counts as working until its bytes appear in a live prompt; modules hash-pinned at submission; behavioral measures replace exploratory_rate. Holes #4 and #10 closed same generation.
- **Standing literature pipeline** (Sano): every generation opens with a scout on the bottleneck; briefs go to researchers as build targets AND red-team as checklists. Gen-7: 12 papers (conversion) + 7 Duck-harness briefs (Tufa Labs winner) + 6 Tycho briefs (NIMI-research) + 6 mission-C papers.
- **Beta — <unk>-soup kill (verified, merged):** in-vocab 0.5131→0.9944, coordinator-verified 0% OOV, dev-neutral (0.766), merged to shared cells.py. Honest verdict: readable ≠ followed — bottleneck moved can't-read → won't-follow (0/20 unforced target emissions; 20/20 with DISCIPL mask).
- **Alpha — harness-side conversion (honest negative):** TTMM switching, ε-deviation, MPC search all REJECTED with live falsifiers (diversity w/o conversion = noise; search theater). Verdict: harness amplifies understanding, cannot create it.
- **Gamma — template collapse (diagnostic):** MI margins negative everywhere (repair gate ≥0 not cleared); steering vector exists (4%→73% flips) but breaks solves; coverage-gated G2 (Tycho 6): open rate 65.5%→3.4%, 3× more action-relevant firings. perceive.predict registered (library: 19 cells).
- Bankrolls: γ2261 / α1627 / β1312 (beta warning persists). Sano: +200 effort awards to all three.
- The wall now has a name: TEMPLATE COLLAPSE. Next: directive-following (DISCIPL mask / HSL traces), steering hook, RPD-curated 1PNS corpus + MI gate.

## Methods / anti-cheat (structural, not trust-based)

- Weights-only submissions: pure tensors, hashed, versioned. No custom per-submission Python (bench-hacking impossible by construction). Python allowed only for authoring generic cells; new cells are shared library infrastructure, generic-only, no per-game logic.
- Staking: researchers stake token bankrolls on transfer outcomes; staking closes BEFORE eval seeds are announced. Payouts only when a new submission beats the standing best.
- Memory-zeroing load-bearing test as the anti-cheat standard for memory claims.
- Held-out set (`held_v6.jsonl`, 10 games) sealed — final scoring only, measure-only, never tuned on. Transfer set measure-only.
- Multi-seed scoring; weight purity checks; no game names in researcher prompts.

## Honest framing

- Our lane is the community leaderboard (harness research), not the official no-harness board. The paper must say so plainly.
- Efficiency story, not a size category: there is NO "tiny prize" category this year. The story is score-per-parameter and $/task, reported honestly.
- Transfer is 0.00% as of 2026-10-17 (27 official measurements). The paper reports the attempt→score curve, the falsified hypotheses (rulebook loop shelved pre-runs; Qwen-teacher probe cancelled by program refocus), and the mechanism the audits uncovered — not a victory narrative.
- Blackboard anecdote (Hearsay-II → Global Workspace → bidding cells): `/srv/arc3-claude/v6run/model/notes/blackboard_anecdote.md` — include in rewrite.

## Deadlines

- Kaggle entry/team-merge cutoff: 2026-10-26. Final submissions: 2026-11-02 23:59 UTC. Paper track: ~2026-11-08. Winners: 2026-12-04.
- Tracks: ARC-AGI-3 ($850K), ARC-AGI-2 ($700K), paper ($450K). No ARC-AGI-1 track in 2026.
- Eval: code competition, internet OFF at eval (weights must be bundled), ~9h hidden-set runtime.
