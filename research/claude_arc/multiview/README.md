# Multi-view handoff status

Views and fitted rules use counting and exact demonstration consistency, with no gradients or runtime LLM. Run `python test_multiview.py` for eight synthetic checks.

The validation cache completed 228 public-training tasks. Baseline solves 29 with five false positives; existing cells plus multiview at g4 solves 31 with five false positives. This is a fixed-arm comparison, not evidence for a development-selected disinhibition gain. No development cache exists yet. Do not choose an arm using these validation outcomes.

The split hashes filenames **including `.json`**, unlike dsl_port, which hashes bare task IDs (184 validation tasks). These results cannot be compared as matched splits. Existing caches are preserved; changing the split would require a new experiment and exposure accounting.

`HANDOFF_VERIFIED.json` records aggregate counts and the source hash. Cached times estimate arm runtime; separately timed runs are needed for an efficiency claim.
