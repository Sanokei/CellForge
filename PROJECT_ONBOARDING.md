# CellForge ARC-AGI-3 — Agent Onboarding

Read this first. It gets you productive on the current program in one pass.
Last updated: 2026-10-16 (EDT). Numbers are true RHAE unless labeled otherwise.

## 1. What this is

CellForge is Sano's program to breed effective AGI out of tiny hand-authored models.
The current front is **ARC Prize 2026, ARC-AGI-3 track** ($850K): interactive games where an
agent must explore, learn rules, and act efficiently. Our entry is a ~1.4M-parameter
decoder-only LM router that reasons in English and routes to capability cells.
Objective: MAXIMIZE score. Target: 60% RHAE. Never satisfice to a threshold.

## 2. The thesis (non-negotiable)

- The **LLM itself writes the weights** — hand-authored, zero gradient steps. Backprop is
  "kind of cheating"; allowed only as tiny local refinement (Baldwin effect).
- **Evolution selects and breeds.** Loop: AUTHOR → AUDIT → AUTHOR → COMPETE → SELECT → BREED.
- Researchers are DnD weight-authors competing under staking incentives; the assistant
  orchestrates and owns anti-cheat.
- Small networks are legible (Goodfire "already clean"): audit by watching activations
  **in action during play**, never static weight dumps.

## 3. Deadlines (verified)

- Kaggle entry/team-merge cutoff: **2026-10-26** (Sano handles account; single-source date)
- Final submissions: **2026-11-02 23:59 UTC**. Paper track ~Nov 8–9. Winners Dec 4.
- Code competition, **internet OFF at eval** (bundle all weights), ~9h hidden-set limit.

## 4. Infrastructure

- **cellforge-1**: Hetzner id 167976488, `168.119.121.253`, 2 vCPU, **NO GPU**, 4GB RAM.
  SSH: `ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519 root@168.119.121.253`
  Always use the venv python: `/srv/arc3-claude/venv/bin/python` (system python lacks numpy).
  `OMP_NUM_THREADS=1` gives ~25x torch speedup. Quote carefully: outer SSH command is
  single-quoted — write files locally and scp them instead of inline heredocs with quotes.
- Key paths on box:
  - `/srv/arc3-claude/v6run/model/fixed_runner.py` — THE runner (frames→weights→actions).
    Flags: `--checkpoint --games --seeds (required) --maxdec --out --budget --rmt --zero-mem-at`.
  - `/srv/arc3-claude/v6run/model/lm/cells.py` — cell registry (shared library lives here).
  - `/srv/cellforge-arena/` — arena: `GEN2_CHARTER.md`, `gen2/` reports, `ledger/`, `gen2/bred/lineage.json`.
  - `/srv/cellforge-arena/gen2/GEN2_STREAM.jsonl` — live event stream (rounds, submissions, judges).
- This repo mirrors the program: `arena/` (box arena), `arc-agi-3/` (v6run code), `paper/`.

## 5. The loop right now

Gen-2 researcher loop (charter: `arena/GEN2_CHARTER.md`): researchers alpha/beta/gamma author
weights, compete with **fitness = transfer RHAE** (dev guardrail ≥0.31%), stake token
bankrolls (staking closes BEFORE seeds are drawn), breed via LoRA-delta/layer-wise crossover
only (naive interpolation banned — proven destructive). Accelerating round clock with
payout-only-on-new-best. Cell registration costs a fee; no cell marketplace; usefulness
dividends from the house. Beta is on elimination warning.

## 6. Current state (2026-10-16)

- Dev best: **0.77%** (gamma-g2, pure authored). Transfer: **0.00%** (16 official measurements).
- Audit findings: transfer = wrong-circuit firing (cosine 0.96–0.99 vs dev routine by L3);
  169 novelty neurons are LOAD-BEARING solve circuitry (do not touch); exploration is ABSENT
  (0 calls/178 decisions); weight-head gating is EXHAUSTED (trilemma proven).
- Live paths: cell-layer exploration (act.target + act.Clicking in 14-cell library),
  attention pooling for the timing gap, discovery-trace distillation as backup.
- Full history: `arena/gen2/GEN1_REPORT.md`, `GEN2_REPORT.md`, `GEN3_REPORT.md`.

## 7. Measurement law

- **RHAE exact**: per completed level min(1.0, h/a)² (h = second-best human actions);
  environment = level-weighted mean (w=l); total = mean across envs, 0–100%.
  Scorer: `arc-agi-3/model/rhae_scorer.py`. Label every number RHAE or LEGACY, never mix.
- `transfer_set.json` (12 games): MEASURE-ONLY, never trained on, never tuned on.
- `held_v6.jsonl` (10 games): **SEALED** — assistant only, final scoring, one touch.
- A full 25-game eval takes ~42s. Eval fast, iterate fast.

## 8. Standing rules (do not break)

- No backprop/gradients/optimizers anywhere. Hand-authored weights only.
- Researchers submit weight files; Python allowed for creating generic cells only
  (no per-game logic, no game names in prompts/intakes/memory — ever).
- No pushes to the default branch, no PRs, no deploys without Sano's explicit word.
  Commits authored as Sano, no AI credit.
- Anti-cheat is the orchestrator's accountability: sealed held-out, post-staking seeds,
  weight purity checks, token metering harness-side.
- Grep prior art before building (Sano's rule); report reused/adapted/new.

## 9. Paper

Kaggle paper track: writeup ≤1,500 words + public notebook + cover image.
Running notes: `paper/PAPER_NOTES.md`. Honest framing: community-leaderboard lane,
efficiency story ($/task), no tiny-prize category exists — never claim one.
