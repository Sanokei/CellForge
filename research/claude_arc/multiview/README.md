# Multi-view handoff status

Views and fitted rules use counting and exact demonstration consistency, with no gradients or runtime LLM. Run `python test_multiview.py` for eight synthetic checks.

Both caches are complete: 772 development and 228 validation public-training tasks. Development selects E existing@g3 + mv@g3 (120 solves, 14 false positives). On validation this arm solves 31 with five false positives versus baseline 29 with five false positives. This is a modest single-split gain; it does not establish broad transfer or AGI. Do not tune an arm using validation outcomes.

The split hashes filenames **including `.json`**, unlike dsl_port, which hashes bare task IDs (184 validation tasks). These results cannot be compared as matched splits. Existing caches are preserved; changing the split would require a new experiment and exposure accounting.

`HANDOFF_VERIFIED.json` records aggregate counts and the source hash. Cached times estimate arm runtime; separately timed runs are needed for an efficiency claim.
