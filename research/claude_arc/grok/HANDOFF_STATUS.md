# Historical experiment, partial sweep

This is a legacy gradient-free neural-weight evolution experiment, not an authored/count-fitted runtime component. It is excluded from the owner's pure-runtime path. Codex launched no additional weight searches.

The 180-run planned sweep has now completed. `HANDOFF_VERIFIED.json` records counts and the source hash. Local-task delayed-generalization counts are A 6/30, B(0.001) 5/30, B(0.004) 6/30, B(0.016) 5/30, random D 0/30. Parsimony does not show an advantage. Stop-at-fit arm C is derived from A rather than independently executed. These are synthetic neural-weight results, not ARC evaluation or evidence of AGI.

The live results and logs are deliberately excluded from this commit. Run `python research/claude_arc/grok/verify_handoff.py` from the repository to regenerate aggregate evidence for all four handoff directories. This reads only existing public-training/game and synthetic receipts, without running solvers, fetching data or inspecting sealed evaluations.
