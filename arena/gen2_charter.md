# Gen-2 Charter — WEIGHTS-ONLY ARENA (effective 2026-10-16)

## The correction

Gen-0/gen-1 researchers bench-hacked: optimizing the score with clever Python
instead of through the method. The method (handwritten neural weights) IS the
product. A score achieved through Python cleverness is worthless to the paper
and the prizes; a score achieved through weights-only is the entire claim.

## The new game

1. Researchers submit ONLY weight files: a `.pt` checkpoint for the FIXED tiny
   architecture (4L/4H/d128, vocab 2209, 1,378,816 params, `cellforge-arc3-ckpt/1`),
   or LoRA deltas (rank <= 8) against `router_v3.pt`. NO custom Python per
   submission. No solve.py, no cells, no strategy code.
2. ONE fixed runner (orchestrator's): `model/fixed_runner.py` — frames in,
   weights in, actions out, true RHAE out. It purity-checks every submission
   (weights_only load, shapes match, all finite, deterministic) and rejects
   anything executable.
3. The researcher implementation IS the DnD weight-author: the frontier LLM,
   conditioned on probe findings + failure modes + current weights, authors
   weight deltas directly. Researchers differ in authoring strategies, not code.
4. Anti-cheat by construction: bench-hacking is impossible when Python isn't in
   the action space. Remaining: held-out sealed, seeds drawn post-staking-close
   (lineage: reference/protocols.py "protected surface"), multi-seed, weight-only
   purity test.
5. Staking/bankroll/breeding continue unchanged, keyed on weights-only RHAE.
   Breeding = recombining champion weight tensors (Sakana-style merging, literally).
6. Legacy: gen0_alpha.py, alpha-g1, beta-g1, gamma-g1, router_v3.pt are frozen
   as LEGACY baselines (see leaderboard `legacy_baselines`). They do not compete.

## Lineage

This arena is the ARC-3 sibling of `/opt/cellforge/code/src/cellforge/reference/`:
researchers author weights-as-data (cf. `ir/graph.py`), the fixed runtime scores
them (cf. `reference/evaluator.py`), components carry contracts
(cf. `construction/contracts.py`), final seeds are operator-held
(cf. `reference/protocols.py`). See METHODS.md "Fixed runner: reference-pattern
lineage" for the reuse map.
