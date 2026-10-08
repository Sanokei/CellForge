# Cloud control-integrity repair

Base: GitHub master `02beaf18251fca5ed1f1eb55e92f00a3e3820e88`.
Work begins on an exact-blob-verified partial source snapshot, not a full Git
clone. The owner requested continued cloud implementation, testing and GitHub
publication. Current scope is source/test integrity; no model-author calls,
new research campaign, benchmark access, template edits or STOP override.

The capability checkpoint identifies a concrete defect: `structcode.evaluate`
discards errors from zero/random control executions. Repair that acceptance
path to retain per-arm/seed errors and fail closed, and make revision feedback
distinguish invalid controls from evidence that tensors do not matter.
Keep all seeds, random distributions, existing templates, protected tests,
floors and historical receipts unchanged. Add isolated regression coverage.

The existing raw-LUT acceptance fixture may conflict with correct fail-closed
behavior because random values can produce invalid negative grid colors.
Report any failure honestly; do not alter the fixture or sampler to hide it.
Focused passes will not imply ONE_STANDARD qualification or a learned-weight
capability gain.

## Verified correction

The owner approved changing the specific obsolete raw-LUT test. It now asserts
rejection and retained random-control errors; all other existing tests/floors
remain unchanged. New additive cases provide a legitimate total-domain positive
fixture and arm/seed/error/feedback/admission regressions. The final focused
command ran 28 tests, OK; an independent rerun also passed 28. The required
standard command, with all 19 tracked declarations materialized, ran 25 tests:
seven contract/milestone failures and three missing-evaluator metadata errors.
No frozen receipt was modified. Full commands, outputs and limitations:
`reports/CONTROL_ERROR_INTEGRITY_20261008.md`.
