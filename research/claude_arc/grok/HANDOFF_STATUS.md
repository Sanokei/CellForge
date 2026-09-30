# Historical experiment, partial sweep

This is a legacy gradient-free neural-weight evolution experiment, not an authored/count-fitted runtime component. It is excluded from the owner's pure-runtime path. Codex launched no additional weight searches.

The 180-run planned sweep was still running during handoff verification. `HANDOFF_VERIFIED.json` freezes counts from completed JSONL records and their source hash. Delayed-generalization counts apply only to those completed records; they do not establish a final rate. Stop-at-fit arm C is derived from A rather than independently executed.

The live results and logs are deliberately excluded from this commit. Run `python research/claude_arc/grok/verify_handoff.py` from the repository to regenerate aggregate evidence for all four handoff directories. This reads only existing public-training/game and synthetic receipts, without running solvers, fetching data or inspecting sealed evaluations.
