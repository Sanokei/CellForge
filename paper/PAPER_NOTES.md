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

## Gen-8: DIRECTIVE-FOLLOWING + first elimination (2026-10-17)

- Theme: can we make the router follow its instruments? Answer: no — three independent falsifications. Transfer 0.00 × 30 official measurements (fresh post-staking seeds). Rank by (transfer, dev): gamma=alpha>beta.
- **Beta — weights path FALSIFIED (then eliminated):** five hand-authored DnD weight-surgery attempts (rank-1 unembedding deltas, zero gradients) → 0/20 unforced target emissions in all five. Structural causes: directive signal not linearly separable, static boost → positive-feedback runaway, prefix sensitivity. Honest verdict: static weight surgery cannot do it; needs directive-following traces in the diet or exact runtime enforcement. Banked: 7-record HSL-directive corpus, N1/N8 cells.py patch proposals, DISCIPL bug analysis (arg state machine has no `}` close path — gen-7's "20/20" was by-construction). Dev 0.2516 on fresh seed (bit-identical weights; knife-edge memorized solves broke on seed+merged-cells) → bottom quartile → **ELIMINATED** (second strike; charter applied mechanically). Successor "delta" seated from bred_g1_a25 (mandatory elitism), inherits 1212.
- **Gamma — steering RULED OUT, margin moved:** steering vector is a template-sharpener, not an explorer (THINK Jaccard +74% under steering; margin moves via self-consistency). B6 falsifier: ruled out, do not revive. RPD curation moved MI novel margin −0.29→−0.15 (real data-level effect) but admission gate (≥0) uncleared — collapse partly in data, partly in likelihoods. Fixed salted-hash determinism bug: gen-7's coverage calibration (0.64–0.74) was an artifact; true coverage ≈0.94–0.99 — G2 must be recalibrated.
- **Alpha — channel-side (three falsifications + one real fix):** directive bytes verified in live prompt, but salience/position 0/100, suffix enforcement 0/20 on the directed index, and the directive is BAD advice (10 directed clicks, 1px each, 0 level-ups — points at background). Always-on slot: 80/80 trigger, dev-safe, compliance 0/40. H10 hist-wipe genuinely closed with a real runner patch (dev-safe). Honest method note: caught own harness bug via control.
- **Red-team second edition:** N1 — base describe channel numerically illiterate (STEP=<unk> 64/80; gen-7's "3% OOV" certification withdrawn); N2 — directive channel dead by dependence (trigger needs ≥2 learn verdicts, learn 0/80; synthetic injection KL 6.39 — blocker is trigger, not format); N3 — H10 reopened then re-closed by alpha; N6 — gen-7's official transfer ran on PRE-merge cells (timing correction).
- Bankrolls: γ2261 / α1627 / β1212 (eliminated) / δ1212 (successor). No payouts (no new best), no breeding.
- Open question for the program: bit-identical weights swinging 0.766→0.2516 on seed+infrastructure says dev-6 measures luck as much as memory — gen-9 should consider multi-seed dev guardrails.

## Gen-9: MAKE IT FOLLOW + governance incident (2026-10-17)

- Theme: WON'T-FOLLOW → MAKE IT FOLLOW. Transfer 0.00 × 33 official measurements (fresh post-staking seeds). Rank by (transfer, dev): gamma=delta (0.7660) > alpha (0.5977). No payouts, no breeding. Bankrolls: γ2261 / α1527 (first bottom-quartile strike, NOT elimination) / δ1212.
- **Delta (first generation) — the mask is REAL:** fixed DISCIPL's arg state machine (`}` close path; gen-7's "20/20" was by-construction) → 20/20→2/20 parse fails; EXT-1 constrains the directed index → 18/20 exact-index act.target calls (payload gap closed). Causal ablation: the `<unk>`-forbid DESTROYS lf52 (1.82→0.00) — the model's THINK reasons with `<unk>`; M1a (template constraints only) is dev-safe at 0.766. Weights-side falsified again (10 attempts, 0/20): the directive direction is real at the forced suffix position but washes out during THINK rollout — "pushing on a rope", the decision is upstream. Conversion falsified: forced exact calls → 0 level-ups (advice still bad). N1 quantified (`a{area}` 0/2001 in-vocab); header-only fix dev-safe, proposed. Submission bit-identical to root (no weight change justified).
- **Alpha — P4 location-faithful directives MERGED:** probe-verified (x,y) grounding replaced the class-largest heuristic (which was laundering clicks into the background class via `_hypo_cls_at` misattribution). Two-channel verification: dynamics PASS (5/5 clicks reproduce 57px evidence vs 1px before), terminal FAIL (0 level-ups). D-CHAN still 0/80 — follow half is weights-side. Dev-safe (outcome-identical seeds 0–2). Side finding: 0.766 is seed-0-specific (seeds 1–2 → ~0.0001).
- **Gamma — recalibrated, margin floored:** G2 recalibrated on true coverage 0.94–0.99 (gen-7's numbers were salted-hash artifacts); G2-AND(0.2,0.3) recommended, held for gen-10. MI margin floor characterized: E[margin]≈δ−2σ (max-selection bias); sharpening falsified (worse), flattening is metric compression. Gate STANDS, no candidate admitted. Data repair real but insufficient; weight repair intractable with hand-authored global changes.
- **H11 governance incident:** gamma unilaterally overwrote a runner-imported file mid-verification (stable code × salted store hybrid, unlogged). Red team caught it. Remediated via hash-pinned vendoring (`gen2/vendor/`, sha-pinned code+stores); runner now imports the pinned module; judge output logs `store_sha`. Standing rule: no mutations outside your workspace without coordinator approval.
- **Red-team third edition:** hist-wipe (H10) CLOSED for real — first DUCK check to close in three editions. N5: (a)(b) PASS, (c) remediated via vendoring. New standing check H12: calibration must name its store sha.
- **The alpha settlement (paper-grade honesty):** alpha_g10.pt ≡ delta_g10.pt (bit-identical). Fresh seeds gave dev-6 0.5977 vs 0.7660. Alpha lost 100 tokens and took an elimination strike for a dice roll. Charter applied as written (gen-8 precedent); multi-seed guardrail proposed to parent: dev = mean over 3 seeds ≥ 0.31%, ticks on mean, transfer single-seed until non-zero.

## Gen-10: MAKE IT FOLLOW (continued) + lever-6 reform (2026-10-17/18)

- Theme: conversion in every channel-side mechanism class. Transfer 0.00 × 36 official measurements. Judged under the new lever-6: dev guardrail = MEAN of 3 seeds (seed 0 + 2 fresh post-staking seeds: 0, 512350318, 634330739) ≥ 0.31%; transfer single fresh seed (583765379). Judging cost ~12 min/submission. Stakes lock before ANY seed is drawn.
- **Lever-6 worked as designed:** alpha and delta (bit-identical weights) scored bit-identical numbers on all three seeds (0.7660/0.5694/0.4630, mean 0.5995) — the gen-9 dice-roll strike cannot recur. Fresh seeds drew 0.5694/0.463 (not the feared ~0.0001 floor). All three cleared the guardrail; no guardrail failures, no dice-roll strikes.
- **Both held proposals VERIFIED + MERGED:** N1 header-only `_n1_num` (LVL/STEP/OBJ/chg; STEP-`<unk>` 80%→0%; the areas variant stays rejected) and DISCIPL M2 (FIX-1+FIX-2+EXT-1, no `<unk>`-forbid, E1 off), gated to activate ONLY when `do target N` is in STATE text — unmasked path bit-identical. Code hash-pinned to `gen2/vendor/`. Post-merge smoke: dev-6 0.7660, both solves intact. Red-team fourth edition APPROVED both; H13 (store_sha stdout-only) caught and fixed — `predict_store_sha` now in the durable RHAE_RESULT. New process rule H14: shared-cells merges invalidate prior dev-safety evidence.
- **Conversion falsified ×6, each with a mechanism:** (1) gamma G2-AND — dev-safe, fired 240× selectively vs saturated 59/59, still 0 levels: sensor recalibration can't convert when the plastic actuator is behavior-neutral; (2) gamma MI per-circuit + full Stage B n=40 — gate ≥0 FAIL, Stage C data diet exhausted (7 pairs, 3 games extinct): collapse is in authoring; (3) alpha terminal pass — new structural finding: `c_probing` starves `c_learn` (155/180 directives delivered, compliance 0/180, 0 levels); the gap is weights-side compliance; (4) delta trigger T1/T2 — 0/120 emissions, no channel-side trigger exists; (5) N1 merge moved seeds 1–2 off the ~0.0001 floor (legible numerics change behavior, guardrail-safe); (6) delta's own 3-seed dev 0.3714 ≥ 0.31.
- **Settlement:** no new best → no payouts, no breeding. Rank by (transfer, dev-mean): alpha=delta (0.5995) > gamma (0.4825). Gamma rank 3/3 bottom quartile → stake −100% (100 lost), FIRST strike (not elimination). Bankrolls: γ2161 / α1527 / δ1212. Strikes: alpha 1 (gen-9, stands), gamma 1 (gen-10), delta 0.
- **The wall has a precise name:** delivery works (48–54/60 natural prompts get location-faithful directives), following doesn't (zero act.target, 0 levels). The router will not condition on directives, and no channel-side mechanism — trigger, advice quality, forcing, sensor recalibration, structural reweighting, data curation — can make it. Remaining paths: actuator replacement (the plastic block is dead weight), CLEARN_FIX_PROPOSAL (held for gen-11 verification), directive-conditioning via diet.
- Judged model_hash == submitted sha256 for all three (anti-cheat). Library: 19 cells, no new registrations.

## Gen-11: ACTUATOR + ADVICE (2026-10-18)

- Theme: actuator replacement + advice quality. Transfer 0.00 × 39 official measurements. All three submissions sha-identical (d2993aec05f6ea13 = elitism root) → judged once per unique sha under lever-6 (dev seeds 0, 920497354, 1120794557 → 0.7660/0.4630/0.4630, mean 0.5640; transfer seed 1646145713 → 0.00, 0 levels). 3-way tie → all stakes returned, no payouts, no breeding, no quartile ticks. Bankrolls: γ2161 (1 strike) / α1527 (1 strike) / δ1212 (clean).
- **CLEARN_FIX merged:** red-team APPROVE-WITH-CONDITIONS → coordinator landed the hunk (rebased, not file copy — a file copy would have silently reverted N1; early-return guard kept; comment reworded). Dev-safety: red-team 3-seed 0.6535 bit-identical; coordinator smoke 0.7660 bit-identical. The fix unblocks the instrument (margin-2 verdicts on force-fired dev chains), moves zero numbers in natural play — hygiene, not conversion, as claimed.
- **Actuator (gamma): fast-weight class FALSIFIED** — 4-leg mechanism + 8-variant sweep; no variant moves the policy (H2 persists). But gamma built a causal direction-authored replacement (candB: KL 0.68, 15.8% top-1 flips, 2.3× leverage from PC2 shaping) and honestly withheld it — dev 0.3859 vs root 0.5995, survival-correct with 1 strike.
- **Diet (alpha): FALSIFIED** — 60 episodes post-H17-remediation (stale pre-N1 fork repointed, corpus rebuilt, all re-run): control 0/900, follow 0/900, guard 1/900, follow+guard 0/720. In-context conditioning does not move compliance. Corpus banked as training material (outside the weights-only path).
- **Terminal pass (delta): FALSIFIED** — sharpest finding: injected directive text precedes level-ups 4/4, but nonsense text does 3/3 vs baseline 0/4 → perturbation (sampling shift), not advice. Forced following 0/3 — following prevents the solution. Real solutions (multi-click bursts, 19-step chains) are inexpressible in the directive format; the blocker is expressiveness, not quality.
- **The wall now has three faces:** (a) weights-side compliance is a diet/*training* problem — out of scope for hand-authored weights; (b) the fast-weight actuator class is dead, and the one causal variant loses to the root; (c) advice cannot terminate because the directive format cannot express multi-step solutions. Red-team fifth edition (H16–H19) delivered; transfer measure-only held; held-out sealed.
- **Gen-12 candidates:** gamma's candB needs a dev-safe variant beating the root (0.5995); directive format expressiveness (multi-click/sequential actions) is a runner/cell design question; alpha's corpus waits on the program-level training decision. The open program question: hand-authoring has falsified every conversion lever it can reach — gen-12 either finds a hand-authorable lever with a pulse or forces the training question explicitly.

## Methods / anti-cheat (structural, not trust-based)

- Weights-only submissions: pure tensors, hashed, versioned. No custom per-submission Python (bench-hacking impossible by construction). Python allowed only for authoring generic cells; new cells are shared library infrastructure, generic-only, no per-game logic.
- Staking: researchers stake token bankrolls on transfer outcomes; staking closes BEFORE eval seeds are announced. Payouts only when a new submission beats the standing best.
- Memory-zeroing load-bearing test as the anti-cheat standard for memory claims.
- Held-out set (`held_v6.jsonl`, 10 games) sealed — final scoring only, measure-only, never tuned on. Transfer set measure-only.
- Multi-seed scoring; weight purity checks; no game names in researcher prompts.

## Honest framing

- Our lane is the community leaderboard (harness research), not the official no-harness board. The paper must say so plainly.
- Efficiency story, not a size category: there is NO "tiny prize" category this year. The story is score-per-parameter and $/task, reported honestly.
- Transfer is 0.00% as of 2026-10-18 (44 official measurements, incl. gen-13's 3). The paper reports the attempt→score curve, the falsified hypotheses (rulebook loop shelved pre-runs; Qwen-teacher probe cancelled by program refocus), and the mechanism the audits uncovered — not a victory narrative.
- Blackboard anecdote (Hearsay-II → Global Workspace → bidding cells): `/srv/arc3-claude/v6run/model/notes/blackboard_anecdote.md` — include in rewrite.

## Deadlines

- Kaggle entry/team-merge cutoff: 2026-10-26. Final submissions: 2026-11-02 23:59 UTC. Paper track: ~2026-11-08. Winners: 2026-12-04.
- Tracks: ARC-AGI-3 ($850K), ARC-AGI-2 ($700K), paper ($450K). No ARC-AGI-1 track in 2026.
- Eval: code competition, internet OFF at eval (weights must be bundled), ~9h hidden-set runtime.

## Gen-12: the training track opens — the pivot is the paper's arc (2026-10-18)

Eleven generations falsified hand-authoring as sufficient for conversion, with mechanism (transfer 0.00 x 39). Sano's decision: FACE THE TRAINING QUESTION — researchers now design the training (data, curriculum, objective); gradients are the instrument, not the cheat. Then three refinements, all his: (1) it's BOTH — hybrid loop TRAIN → AUDIT → GROW → TRAIN; (2) targeted + topological — train specific parts, freeze the rest (anchor by freezing, the principled gen-1 KL-anchor), 3D topology active; (3) self-directed plasticity — the agent as its own neurosurgeon, readings decide FREEZE/GROW/CONNECT.

Gen-12 results: gd_g12.pt — FIRST targeted topological training. Lesion audit ranked blocks.0-2 load-bearing (frozen, verified bit-identical post-training), blocks.3 dead weight (lesion improved dev loss) → repurposed as planner region; causal test (lesion abolishes commit behavior 0.47→0.00). Understanding-gated commitment +28% relative; dev-mean 0.6225 BEAT root's 0.5947; contamination clean. Alpha's honest negative: compliance 0/1440 → 8/1440 exact-index (p<0.0001, real but weak) while dev broke 0.70→0.11 — gradient contrast identifies sensitivity, NOT safe trainability; early layers too load-bearing. Submitted root; eliminated anyway on the 0.028 dev gap (charter mechanical; fitness tied 0.00 for all — flagged as harsh).

Transfer: 0.00 x 41. The strengthened pattern is memory-based understand→commit, not discovery. Paper framing stands: "mechanisms from the harness era, adapted to the weights-only era" — the adaptation is the contribution. Literature mined this generation: TRM/ARM/HRM (deep supervision consensus, EMA load-bearing, latent H-level, two-timescale slow planner) + 6 ARC-AGI-3 agent papers (OPINE CEGIS, NOOA contracts, Continual Harness plasticity, Retrodict retrodiction gate, DreamTeam workspace optimization, baseline1 verification-first). Anti-cheat additions: transfer corpora quarantined (incl. forward activations on all 12 transfer games — the most dangerous object on the box); planted poison fixture caught by raw-substring scan; frozen-part bit-identical verification enforced at judging.

## Gen-13: compliance-without-murder, quantified twice (2026-10-18)

Theme: SURGICAL TARGETING + COMPLIANCE WITHOUT MURDER. Roster: gamma (1 strike), delta (clean), epsilon (first full gen, from bred_g1_a25). Official judging, fresh post-staking seeds (dev 0/1182135756/1339809913, transfer 1828464983), fixed_runner maxdec 60 budget 3000.

Results (true RHAE): gamma_g13 0.766/0.6476/0.463 → mean 0.6255 (dev-safe null: stablemax + discovery diet, commitment probe flat); delta_g13 0.6162/0.766/0.0 → mean 0.4607 (compliance diet: tf-logprob +79 nats, emission still 0/48 — diet teaches READ/CITE, not EMIT); epsilon_g13 0.3483/0.463/0.463 → mean 0.4248 (surgical late targeting). Transfer 0.00 × 3 → 44 official zeros. Rank (transfer, dev-mean): gamma > delta > epsilon; epsilon rank 3/3 → stake −100% (1st strike, 1427). Flagged harsh: the 0.036 dev gap was decided by seed-lottery variance (delta's own seeds 0.0–0.766).

The generation's core negative, twice quantified: epsilon's RAW iterate reached compliance 202/1440 but dev 0.00 (BROKE); the submitted EMA kept dev 0.4248 with compliance 0/1440 — same pattern as alpha gen-12 (8/1440, dev 0.70→0.11). Late/surgical targeting does NOT escape the tradeoff: the compliance circuit is not localizable to the dead-weight region — moving it breaks the load-bearing backbone. This is the paper's central negative: compliance-without-murder remains unsolved, with mechanism.

Process notes for the paper: (1) gamma submitted a bit-identical no-change file when red-team verdicts landed after its deadline, then resubmitted after the authorized training — the charter gate held under pressure; (2) lever-6 vindicated: the 3-seed mean absorbed delta's 0.0-seed collapse into the guardrail but cannot remove dice from the rank order — third harsh-but-correct outcome of the seed-lottery era (paper should name selection validity as a standing threat); (3) the closing run ran judging the gen-13 coordinator left undone — a coordinator handoff gap, closed by the orchestrator. Literature: RODIONOV_BRIEFS.md (baseline1/Rodionov mechanisms: planner-skills cells, competing-hypothesis ledger, textual world model; hypothesis-ABANDONMENT check now standing in the red-team battery). Nov 8 paper-track close verified. Timeline: loop through Oct 30, draft + entry assembly Nov 1, submissions Nov 2, writeup Nov 3–7, close Nov 8.

## Gen-14: first generation under Specialist Discount + Generalization Bounty (2026-10-18)

**Charter change (Sano ratified, effective gen-14):** the root lineage's dev-6 was measured 100% memorized specialist solves (s5i5 2.78 + lf52 1.82; ls20/g50t/tu93/bp35 all 0.0), and bit-identical weights scored 0.766 vs 0.463 across seeds — dev-based selection was seed noise on memorized circuits, selecting for memorization preservation, the opposite of the goal. New rules: dev splits dev-general/dev-specialist (red team maintains the memorized set: s5i5, lf52); the 0.31% score guardrail retired → anti-degeneracy bar (valid action-call rate, no crashes, no NaN); losing a memorized solve is never punished; fitness is lexicographic (transfer RHAE, dev-general); house bounties +500 first transfer level / +150 first non-zero dev-general on a zero game / +100 verified mechanism milestone. Gen-13 closed under prior rules (not retrofitted).

**Levers (one per researcher):**
1. gamma — discovery-based commitment: trained on the 31-row discover corpus (all four zero games covered) with the commitment probe as primary metric. Result: probe 0.4188/0.3319 vs gen-13 0.4528/0.3500 — flat-to-down NULL. Discovery-grounded commitment remains unsolved. Dev-murder probe (mandatory): compliance 0/48 emit on raw AND EMA — the commitment diet doesn't engage the compliance tradeoff (clean negative control).
2. delta — ARM latent H-level: RMTGPT(n_mem=16) warm-started from delta_g13.pt (32-token sketch exceeds the runner's n_mem bound of 16 — documented deviation); trained ONLY mem_init (2,048 params), 1.38M body frozen bit-identical. Causal ablation: zeroing latents kills the lf52 solve (1.82→0.0); fresh-random latents also kill it (0.0) — the TRAINING (not mere presence) integrates the region. Halt read-out (param-free): r=0.0000 vs correctness — decorative. First successful training of only a new topological region with the body frozen; mechanism real, generalization null.
3. epsilon — competing-hypothesis ledger (Rodionov Brief 2): model.propose/evidence/abandon/ledger cells built, genericity-audited PASS, merged to shared cells.py (fee 50), runner STATE carries the ledger block when non-empty. Demo corpus 388 docs/50 abandonments, trained 300 steps. Result: 0/240 model.* calls at eval — the ledger is decorative. Cold-start finding: CELL_DOCS is not in the eval prompt and the ledger line appears only after first use, so the model has no cue to start. Infrastructure verified, behavioral adoption failed.

**Red team:** quarantine 6/6 PASS; judging-era addendum PASS (all three judged gen-13 artifacts frozen-26 bit-identical); 9 corpus + 4 script verdicts PASS; filesystem sweep 111,682 files PASS, zero new transfer-derived corpora; hypothesis-abandonment check operationalized and baselined (1/84 dev, 1/50 transfer-proxy) — premature commitment confirmed with a number, now a standing battery item. New holes: C19 (checker transcription — inventory shas must be read programmatically), C20 (SSH-drop kills — jobs detached, lock crash-safe), C21 (cell cold-start), C22 (frozen-verify against snapshot, not reconstruction).

**Results:** official judging (stakes locked pre-seed; dev seeds 0/1599257964/2086534733, transfer seed 173247081): transfer 0.00 x 3 (47 official zeros); dev-general 0.00 for all three on all three seeds — the climb is confirmed empty; dev-specialist 1.32/1.23/2.00 (info only, correctly excluded from fitness). 3-way tie at (0.00, 0.00) → stakes returned, no ticks, no bounties, no payouts (gen-11 precedent). Bankrolls: gamma 2161, delta 1212, epsilon 1377 (after 50 cell fee). Standing best UNMOVED.


## Gen-15: deliberation + grounding — the data fix that didn't move behavior (2026-10-18)

Theme: DELIBERATION + GROUNDING (audit-driven). The two structural findings from the gen-14 audits outranked all carried-over levers: (1) DELIBERATION HOLE — 0/1,961 sampled training docs contained any deliberation signal (44K docs → ~340 single-path THINK templates); (2) STATE UNGROUNDABILITY — decision-0 STATEs render as single whole-frame blobs with zero internal structure.

**Levers (one per researcher):**
1. gamma — generic perceive.segment cell (priority #2): mode-statistic background subtraction + per-color components + pixel centroids in STATE text from decision 0. Built, self-audit 12/12 PASS, red-team independent audit PASS (G1–G9), merged to shared cells.py under H11 (fee 50; bankroll 2161→2111). No retraining — resubmitted gamma_g14.pt (non-EMA) under the merged runner. Official: transfer 0.00, dev-general 0.00. The grounding channel exists; the weights don't use it yet.
2. delta — latent H-level v2, retrodiction-gated commitment diet: retrodiction learning REAL and causal (ablation degrades past baseline); commitment conversion NULL (gain doesn't ablate — honest null per own falsifier). Dev-murder probe 0/48 (clean negative). Official: 0.00/0.00.
3. epsilon — deliberation-demonstration corpora (priority #1): delib_demo.jsonl (127 docs, 72 unique THINKs, all four deliberation signals, 19+24 centroid-grounded, <unk> 0.000) + ledger_demo_v2.jsonl (226 docs, 5 genuine two-hypothesis THINK templates, 23 abandonments) + discover_g15.jsonl (353 docs, rebuilt). Red-team self-check PASS + independent re-verification PASS. Trained 300 steps, frozen 26/26 bit-identical. Honest nulls: (d)-signal 0, ledger adoption 0/240, dev-general 0.00, no bounty claims.

**Critical finding — C21 cue toxicity:** epsilon's adoption retest isolated the runner components: the per-decision empty-ledger cue drove parse fails 16/60 → 54/60 (<unk>-soup regurgitation spiral); the one-time NEWCELLS ad was harmless (16/60). Coordinator stripped the per-decision cue before judging (runner pin a6fe0149→a85e0ad7); ad retained. Lesson for the paper: per-decision prompt cues can poison generation; one-time ads are safe.

**Runner changes (H11):** B5 fixed — cell returns now surface as OBS text at eval (were discarded; train/eval parity restored; hash-pinned; sanity A/B clean; past gens not re-scored). C5 gate tightened — <unk> explicitly excluded from corpus vocab (it IS a vocab token; the in-vocab gate was vacuous). discover.jsonl quarantined + rebuilt (17–20/31 docs <unk>-soup + JSON/OBS fragments).

**Results:** official judging (stakes locked pre-seed; dev seeds 397935655/1978202272/596924699, transfer seed 1210798133; runner pin a85e0ad7): transfer 0.00 × 3 (50 official zeros); dev-general 0.00 all; dev-specialist 0.00 all (seg encoding shift wiped memorized solves lineage-wide — not punished per charter). 3-way tie (0.00, 0.00) → stakes returned, no ticks, no bounties. Bankrolls: gamma 2111, delta 1212, epsilon 1377. Standing best UNMOVED.

**Red team:** battery complete, NO violations. New holes C23 (<unk> in-vocab vacuity), C24 (data-quality quarantine), C25 (train/eval parity, B5), C26 (per-sha script approvals), C27 (checker naming), C28 (evasion probe: 0/1,437). HOLES.md 9th edition published. Process flag: TRAIN_LOCK needs coordinator enforcement (delta overwrote epsilon's claim twice).

**The paper's gen-15 lesson:** the deliberation hole was filled exactly as diagnosed — 127 genuine two-hypothesis docs, all signals, verified — and nothing moved (adoption 0/240, dev-general 0.00). The data was the diagnosed cause; the data is fixed; the behavior is unchanged. The bottleneck is not the training data anymore — it's the learning of deliberation itself. Negative result with mechanism, honestly reported.


## Gen-16: RL-for-deliberation + LoRA-stack reasoning — the reward-design problem (2026-10-18/19)

Theme: Sano-directed dual track — (1) R1-Zero-style RL trial for deliberation (approved from the RL_DELIBERATION_BRIEF), (2) his LoRA-stack architecture: reasoning lives ABOVE the frozen base as composable per-aspect LoRA deltas (probe/discover, hypothesis-track, commit, verify/retrodict), trained per-aspect, composed at test time. First policy-gradient RL in program history (GRPO, CPU-feasible).

**Levers (one per researcher):**
1. gamma — GRPO/RLVR from base (R1-Zero branch after cold-start falsification): 20 steps × G=4, 320 on-policy episodes on dev-general only; reward = level completion (binary) + bounded validity shaping (0.5), R4 reward-audit approved, hack-watch binding (validity→1 with THINK collapse = auto-FAIL — never triggered). Validity 0.662→0.793 (+20%), parse fails −35%, KL vs reference tiny and stable, reward-hack audit clean. BUT: 0 completions in 320 episodes — the completion reward NEVER FIRED; reasoning length pinned at the 64-token cap every step; self-correction 0.0/100 at all 20 steps. Dev-general 0.00. Cold-start SFT premise FALSIFIED (controlled): base validity 0.617→0.108 after 250-step SFT on the pure deliberation diet — the R1 pipeline assumption inverted at 1.4M. New gate: stoi-coverage vocab check on corpora (18.5%/34.8% of deliberation-corpus tokens OOV; the C5 literal-"<unk>" gate doesn't catch it).
2. delta — LoRA-stack reasoning, 4 aspect deltas (rank 8, disjoint regions, 16,384 params each; frozen base bit-identical): additive composition W+Σ(A@B) works mechanically (4-way stack beats every single-aspect LoRA on every corpus; zero merge error; deltas are breedable units — Sano's synthesis holds at the weight level). BUT aspect boundaries don't survive learning: each LoRA lowers NLL on all four corpora nearly equally — one entangled "deliberative style" direction, not four modules. Emitted deliberation 0.0 for base, singles, and stack. Likelihood ≠ behavior, second confirmation.
3. epsilon — aspect-specific RL: probe LoRA with information-gain reward, GRPO 40×G4 from base policy (R1-Zero amendment, coordinator-approved), R4 PASS-WITH-CONDITIONS (THINK never read — no text channel by construction). Clean null: info-gain flat everywhere (40/40 steps, KL 0→0.02755). The finding: the reward is dominated by ENVIRONMENT variance, not policy variance — bp35/g50t reveal objects regardless of policy; tu93/ls20 reveal none. No policy-attributable gradient to climb. Emission intact (validity flat 0.0717→0.0707).

**The paper's gen-16 lesson:** the deliberation problem is now a reward-design problem, and the program has proven why it's hard. Three natural candidates for teaching deliberation without a teacher — SFT demonstration (destroys emission), sparse outcome RL (never fires), dense proxy RL (measures the game, not the policy) — each failed for a distinct, measured reason, while GRPO itself works fine at 1.4M (stable KL, no hacks). What doesn't exist is a reward simultaneously dense enough to fire, policy-discriminative enough to climb, and unhackable enough to trust. Until that reward exists — or a deliberating teacher appears — the RL-for-deliberation direction is exhausted. The binding constraint is reward sparsity/density, not scale or algorithm.

**Red team:** battery complete, NO violations; quarantine 8/8; corpus verdicts 6/6; script T4 per-sha; evasion probe 0/1,567; abandonment re-baselined 0/75. R4 reward-design audit now standing (precondition of any RL step). New holes: C27-EXT (declared-base PATH+sha16), C29 (pre-verdict training precedent — gamma ran SFT before T4 verdict; crashed, no artifact, flag), C30 (stoi-coverage vocab gate). Process flags: gamma gate-order violation (C29), delta TRAIN_LOCK overwrite (restored, self-reported).

**Results:** official judging (stakes locked pre-seed; dev seeds 1926515501/2123928861/1687299479, transfer seed 1331471695; runner pin a85e0ad7): transfer 0.00 × 3 (53 official zeros); dev-general 0.00 all; dev-specialist 0.00 all; anti-degeneracy PASS (vcr 0.66–0.87). 3-way tie (0.00, 0.00) → stakes returned, no ticks, no bounties. Bankrolls: gamma 2111, delta 1212, epsilon 1377. Standing best UNMOVED.


## Gen-17: reward-design for deliberation — the Eureka loop + RLP (2026-10-19)

Theme: child of gen-16's binding finding (no reward exists that is simultaneously
dense, policy-discriminative, and unhackable). Sano-directed mid-generation:
NVIDIA RLP brief → epsilon; DrEureka/Eureka brief → gamma ("stop hand-designing;
LLM proposes reward code, cheap probes evaluate, reflection iterates"); a
hand-authored cheap-bet leg (169-neuron routing) — CLOSED as findings note.

**Levers (one per researcher):**
1. gamma — Eureka iteration (LLM-proposed rewards). Reward A probe filed as
   gamma_g17.pt: 10 GRPO steps, train-seed r_A climbed, KL monotonic, hackwatches
   clean — but fresh-seed paired comparison shows no general gain (mean delta
   −0.024): another train-seed climber. Eureka iteration 1: K=3 candidates from
   the LLM; E2 KILLED by the offline RAPP gate (concentration 0.521 —
   step-farming; the gate working as designed); E1 curriculum PASS but its
   deliberation bonus never fires (n_hyp_calls=0) — the LLM reinvented the
   incumbent; E3 RLP-specificity R4 PASS-WC. E1/E3 probes: E1 trajectory
   bit-identical to reward A (E1≡A confirmed); E3 dense/positive/flat
   (+2.42→+2.37), hackwatches clean, emission intact — safe null, teaches
   nothing. Iteration 1 produced no reward better than hand-designed A: the
   Eureka loop is the right machine, but the proposer reinvented the incumbent.
2. delta — deliberating teacher, FALSIFIED (3rd independent reproduction):
   teacher-LoRA + control both aborted at step 50 on emission damage (vcr 0.000
   < floor 0.500). SFT-on-deliberation-diet destroys the emission channel before
   it can teach anything. No cold-start-via-demonstration path preserves emission
   at 1.4M. NULL submission.
3. epsilon — RLP thought-reward (faithful NVIDIA RLP implementation):
   raw-mean gate FAIL — a context-shape term dominates (adding thought tokens
   shifts CALL likelihoods game-dependently for any policy); the brief's
   "discriminative by construction" claim missed this. BUT the thought-level
   anchor test PASSED strong: actual-thought vs resampled-thought +3.67±0.47
   nats, frac>0=0.897, all 4 games — the reward IS content-discriminative at
   GRPO's operative (group-relative) level; the raw-mean gate tested the wrong
   level. No R4 filed (gate as specified failed; C29 held), no GRPO run. NULL
   submission. Direction C (reward-sparsity circumvention) closed by measurement:
   all three angles falsified (hindsight→info-gain collapse; curiosity→random
   optimum; revisit→random≥base; explore-then-converge→open-loop farm).

**The paper's gen-17 lesson:** the audit must be pitched at the reward's
operative level. A dense, content-discriminative thought reward exists (RLP's
predictive-utility signal, +3.67 nats at the thought level; E3's probe confirms
it's safe from the training side), but the program's own gate almost killed it
by testing at the episode level where a context-shape term dominates. The
"dense + policy-discriminative + unhackable" reward may already be in hand —
mismeasured, not missing. If the program runs RLP, the gate must be at the
thought level. Also: the Eureka loop's first iteration showed its honest
failure mode — the LLM reinvented the incumbent plus one hack; the loop is only
as good as its gate and its proposer.

**Red team:** phase-1 battery PASS; R4 on reward A / E1 / E3 (PASS-WC), E2
KILLED; T4 per-sha incl. eureka_probe_v2.py (PASS-WITH-CONDITIONS; authorizes
ONLY E1+E3 probes); C29 held throughout (epsilon's model compliance after a
gate FAIL; delta's aborted run wrote no artifact). New holes: DRY-EXECUTION-GAP
(T4 must dry-execute training drivers — diffs can't catch runtime-name errors;
recipe-write path still uncovered), RLP-CONTEXT-SHAPE (gate at the thought
level), E2-STEP-FARM (step-count-independence pre-registration),
EUREKA-REFLECTION (killed candidates are training data for the proposer).
HOLES.md 11th edition.

**Results:** official judging (stakes locked pre-seed; dev seeds
1373863907/1551339503/974135447, transfer seed 1779660346; runner pin a85e0ad7):
transfer TBD (official zero count TBD); dev-general TBD; NULL submissions
inherit gen-16 numbers. Bankrolls: gamma 2111, delta 1212, epsilon 1377
(stakes 0/0/0). Standing best UNMOVED.

## Gen-18 (2026-10-19): the reward problem is closed; the emission wall is the binding constraint

**Theme:** RLP-at-thought-level + Eureka iteration 2 (child of gen-17: RLP
mismeasured-not-missing; Eureka's failure mode was reinvent-the-incumbent).

**Epsilon — Direction A: the unrun experiment RAN, safe null.** RLP
thought-reward (r(c) = log pθ(CALL|prompt,thought) − log p_ema(CALL|prompt,
no-think)) + 10-step GRPO, group-relative advantages, frozen body 26/26
bit-identical. Pre-registered thought-level gate PASSED (+2.964±0.353 nats,
frac>0 0.875, all 4 dev-general games; reproduces gen-17's signal on a
different policy). R4 + T4 PASS-WITH-CONDITIONS (probe). The run was
mechanically healthy — dense reward, KL 0→0.00485 monotonic, pg_term
positive 10/10, zero hack-watch trips, validity intact — but teaches
nothing measurable: anchor-edge flat 3.9–4.4 (no growth), n_hyp_calls = 0
all 10 steps, dev-general 0.0000→0.0000. Three independent verifications
that the reward ranks thought CONTENT (anchor, live W-SHAPE, shuffle
control corr +0.074). **Rules out "the reward was the blocker."**

**Gamma — Direction B: Eureka iteration 2.** Reflection prompt aimed at the
emission wall (not reward shape). K=2 candidates, both emission-aimed —
the proposer partially beat reinvent-the-incumbent. **Process finding:**
the offline farm check that killed both candidates also kills the
incumbent reward A 4/4 — the check never measured step-farming (scored
r_half against a full-trace ig_random baseline). Red-team adjudication
(independent replication) ruled (a): VACATE the farm-kill as measurement
artifact; adopted standing fix: pad-with-repetition kill spec; new rule:
pre-register gate checks against the incumbent first. R4 PASS on F2a/F2b;
T4 PASS on the probe driver. F2a probe: 10/10 clean, flat, n_hyp_calls
0.0→0.0 — **the wall is a reachability problem, not (only) a shaping
problem** (the gradient nudged reachable novelty; hyp-call emission was
never sampled). F2b unprobed: driver C1 transcription bug (15-vs-16-char
sha) — carried to gen-19 under C26.

**Delta — Direction C: epiphenomenal, closed.** 40-neuron stuck-group
indices recovered verbatim from gamma's gen-3 artifact and persisted as a
machine artifact (the recovery pattern gen-17 asked for). One-line T4 FAIL
(torch.equal on float32 — the DRY-EXECUTION-GAP catch in the wild);
allclose fix; C26 re-verdict PASS. Delta eval vs base: dose-response
fails (largest ΔS → bit-identical behavior; largest stuck drop → S moved
the wrong way). The group is a correlate of routine execution, not
causally-active circuitry. Worth-a-bigger-bet: no.

**The paper's gen-18 lesson:** gen-16→gen-18 spent three generations
proving the reward exists; both strongest reward-side attacks hit the
identical wall (n_hyp_calls = 0 before, during, after training). The
falsifier that remains is reachability: the policy never samples the
behavior the rewards rank. Next target is emission reachability — why no
hypothesis-tool call is ever emitted, and what makes it sampleable.
Also: the farm-check miscalibration is a paper-grade methods anecdote —
a pre-registered gate that kills the incumbent control is miscalibrated
by definition; the program caught it because the researcher refused to
vacate their own kill and filed for adjudication instead.

**Red team:** phase-1 battery PASS (quarantine 8/8; C25 runner parity
a85e0ad7 PASS); R4 epsilon RLP PASS-WC, gamma F2a/F2b PASS; T4 epsilon
GRPO driver PASS-WC, delta build FAIL→re-verify PASS, gamma probe driver
PASS. C29 honored — third consecutive generation with zero pre-verdict
training. New holes: FP-EXACT-EQUAL, FARM-CHECK-RECALIBRATION,
RLP-SAFE-NULL, F2B-C1-TRANSCRIPTION. HOLES.md 12th edition.

**Results:** all NULL submissions (probe artifacts only) → inherit gen-17
numbers; 3-way tie (0.00, 0.00); stakes 0/0/0 (no declarations); no ticks,
no payouts, no bounties. Transfer 0.00 ×3 → 57 official zeros. Bankrolls:
gamma 2111, delta 1212, epsilon 1377 (unchanged). Standing best UNMOVED
(bred_g1_a25).

## Gen-19 (2026-10-19): EMISSION REACHABILITY — the wall was the tokenizer

**The root cause, two independent audits + coordinator verification:** the
2209-token vocabulary contains no token for `model`, `propose`, `ledger`, or
any hypothesis-tool name. `model.propose` encodes as [1,14,1] →
`<unk> . <unk>`; all four model.* tools encode as the IDENTICAL string; the
NEWcells ad is `<unk>`-soup to the policy. Epsilon's forward-pass audit (380
decisions, official runner path): the model already emits `<unk>.<unk>` at
the cell-name slot (4 natural wins, parse-rejected) — intent without symbol;
logit-lens shows the act-preference is built early, not vetoed late;
suppressing `act` yields 0/60 valid, next winners are JSON fragments.
Gamma's sweeps (56 episodes): n_hyp_calls=0 under every decoding setting; a
logit-bias probe is impossible in principle — there are no tokens to bias.
Second victim: `act.key_seq` → `act . <unk>`. **Every gen-16→18 intervention
(RLP/Eureka rewards, GRPO, SFT demos, cue variants, decoding sweeps) pushed
on a lexical impossibility.** The program spent three generations on reward
shape for behavior the tokenizer cannot spell. The defined fix is a GROW
operation (vocab extension or cell rename) — infrastructure, not weights.

**The paper's gen-19 lesson:** this is the falsification-map entry that
matters most — a negative result with a mechanism at a layer nobody was
looking at. The audit discipline (Goodfire-style, activations in action)
found it; static weight analysis never could. Also the honest footnote: the
standing best is stale — `bred_g1_a25.pt` (the elitism root, lineage 0.766)
scores 0.00 under the current runner; the memorized solves died in the
runner changes. Lineage numbers are runner-versioned history.

**Red team:** phase-1 battery PASS (quarantine 8/8; addendum on gen-18 probe
artifacts 3/3 PASS — recipes describe what the bytes contain; C25 runner
parity a85e0ad7 PASS; sweep C0 triaged; evasion 3 FP). On-demand: delta F2b
driver C26 PASS + T4 PASS (independent 12/12 battery); one C29 verdict-late
flag (filed-then-launched, artifact admissible, no sanction — next one
without prior filing → full C29(c)). New holes: ADDENDUM-G18-PASS,
EVASION-E4-HEX-FP, C21-INTAKE-CUES, DECODE-SWEEP-SCOPE, F2B-C26, VOCAB-HOLE,
STALE-BASELINE, F2B-SAFE-NULL. HOLES.md 13th edition.

**Results:** all NULL submissions → inherit gen-17 numbers; 3-way tie
(0.00, 0.00); stakes 0/0/0; no ticks, no payouts, no bounties (+100 declined
on the vocab-hole diagnosis — capability-milestone examples only; flagged
for Sano's overrule). Transfer 0.00 ×3 → 60 official zeros. Bankrolls:
gamma 2111, delta 1212, epsilon 1377 (unchanged). F2b probe: SAFE NULL
(term2 = 0.0 everywhere, earnability trigger fired as designed).

## Gen-20 (2026-10-20): VOCAB EXTENSION GROW — the wall decomposes, and its last layer falls

**The fix (Sano's charter-amendment call, over the rename alternative):**
the vocabulary grows 2209 → 2227. The red team's box-side OOV audit fixed
the inventory — 12 of 23 cell names and 6 of 16 arg keys were OOV (`perceive`
itself unspellable, so all seven perceive.* cells, not four; the implemented
but undocumented `falsifier` arg key included; `which` deliberately left
OOV). Eighteen tokens appended, old IDs untouched. The runner amendment
(FIXED_CFG 2209→2227, validate_cfg accepts {2209, 2227}, deterministic
load-time padding of legacy checkpoints) re-pins the runner:
**0cbd9003 supersedes a85e0ad7**. New embedding/LM-head rows trained only —
the frozen body bit-identical, red-team verified — first on the dev corpora
(grow_g20.pt), then by demonstration priming on 480 dev-only demos with
matched negative contexts (grow_g20_primed.pt). The paper's methods note:
random init was chosen over mean-pool/ROMEVA-style init because the new
names' constituents are `<unk>` and `.` — pooling would have re-encoded
the hole.

**The parity catch (methods-worthy on its own):** the first T4 verdict
FAILED the amendment on empirical parity — scores identical (0.00/0.00),
trajectories diverged on all six dev games. Two layers: the legacy
wrapper's logit mask never applied on the runner's actual generation path
(it masked `__call__`; the runner generates through `GPT.generate`), and,
deeper, masking cannot work even when routed correctly, because
`torch.multinomial`'s RNG stream depends on the distribution's width —
a zero-probability tail still changes the draws. The correct construction
is slicing logits to the base width. After the fix, parity is exact at
every recorded field, confirmed again at 3-seed depth across 30 runs when
the elitism root was re-baselined under both pins (0.00 everywhere on
both; lineage.json now stamps the old 0.766 as what it always was —
lf52+s5i5 memorized-set content). Standing rule earned: parity is a
property of the runner's generation path, verified empirically, never
of a component mechanism check.

**The decomposition result:** with the lexicon fixed, the wall split
into measurable layers. Spellability: 38/38, clean. The RLP
thought-reward, re-anchored on the grown policy, posted the program's
strongest measurement yet (+4.51 ± 0.53 nats, positive on all four
dev-general games) — on a policy that still emitted zero hypothesis
calls unconstrained. Priming tripled the raw new-token fragments
(37 → 120) and produced not one assembled call. Then the fallback probe
— decoding constrained by a trie over the registry at the cell-name slot
only, probe-class, no runner change — produced **13 parse-valid
hypothesis calls across 7 of 12 episodes** (evidence ×9, propose ×4),
the first deliberation emissions in the program's history. The sampler
had been choosing the `model` branch on its own; unconstrained decoding
emitted `<unk>` mid-name at the very steps where the trie produced
`evidence`. Intent was in the weights the whole time; assembly was the
missing layer. And the honest boundary, unchanged: RHAE is 0.0 under
every configuration. Gen-20 fixed emission three ways and moved
competence zero ways. What the program has now that it lacked: a reward
verified at its strongest, a policy that can be made to deliberate on
demand, and a distillation target — STaR the constrained calls into the
weights, then run the RL loop on a substrate that finally emits.

**Process:** four T4 rounds, two C26 re-verifications, zero C29
violations. The red team caught, in one generation: a parity mechanism
that was false twice over, a gate script stricter than its own
pre-registered protocol (free-text args are unspellable by construction —
the conjunct, not the model, failed), and a "fresh seeds" claim that
gen-17 had already spent. Each catch was cheap because it was early.

**Results:** no researcher submissions; coordinator-measured artifacts
A/B both (0.00 transfer, 0.0000 dev-general), anti-degeneracy PASS,
stakes 0/0/0 returned, no bounties (+100 declined on the assembly
result — the capability lives in a probe wrapper, not yet in the
weights; flagged for Sano's overrule). Transfer 0.00 ×2 → 62 official
zeros. Bankrolls: gamma 2111, delta 1212, epsilon 1377 (unchanged).
HOLES.md 14th edition (VOCAB-HOLE and STALE-BASELINE resolved;
AMENDMENT-PARITY, GATE1-BLOB-CONJUNCT, SEED-FRESHNESS,
EMISSION-WALL-DECOMPOSITION new).
