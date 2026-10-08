# Qualification failure taxonomy

The unchanged required command currently reports **25 tests, seven failures and
three errors**. Exact output: `nca-tranche-terminal-20261008/qualification-tests.log`.
This diagnosis reads declarations and source only; it does not evaluate new data,
change qualification rules or replace historical evidence.

## Seven failures

1. **Budget declaration/receipt mismatch.** `test_one_standard.py:113–118`
   counts proposals per arm/seed and requires the declared budget. The first failure
   is `weights-local-maker-initial-20261006`, arm llm, seed11: zero versus two.
   Five of the 18 discovered declarations have at least one such mismatch.
   These empty frozen declarations are not a faithful cumulative attempt ledger;
   the failure alone does not prove the underlying host made zero calls or spent
   an unequal amount. Do not fabricate proposals to reconcile them.
2. **Insufficient registered seeds.** Lines107–110 require at least three distinct
   seeds. Fifteen declarations fall short. First failure: analytical reference,
   one seed. The two-seed NCA feasibility tranche deliberately falls short too.
3. **No improving three-generation history.** Lines192–203 find no populated
   generation history. This capability/evidence milestone is genuinely unmet.
4. **No crossover children.** Lines185–190 find no crossover genome entries.
5. **Insufficient niches/founders.** Lines160–168 require four niches each with
   three LLM-authored founders; all discovered founder arrays are empty.
6. **No registered mutation advantage.** Lines175–183 find no paired populated
   genome arrays whose LLM mean accuracy exceeds random. The new founder result
   is independently null and its mutation timed out; it cannot supply this evidence.
7. **Insufficient distinct genomes.** Lines170–172 find zero registered genome
   hashes versus 100 required. This means zero in these declarations, not proof
   that no literal tensor artifacts exist anywhere in the repository.

## Three errors, one repeated schema boundary defect

`tests/standard_checks.py:73–74` directly imports `s['evaluator']`.
All 18 discovered declarations omit that field. The first sorted declaration is
`weights-belief-analytical-reference-20261006/STANDARD.json`, an explicitly
nonqualifying analytical reference with empty population arrays. The same missing
key interrupts accuracy recomputation, LLM-versus-random recomputation and router
checks (`test_standard_semantic.py:301–316`). These are unhandled schema errors,
not three distinct inference failures. Later checks/runs are not fully inspected
by those failing test methods, so the observed error count is not exhaustive.

The basic required-field check (`test_one_standard.py:94–97`) does not require
`evaluator`, although semantic consumers do. Merely supplying that name is
insufficient: a real evaluator must implement the correct frozen data, tensor and
router semantics, and other semantic fields/receipts remain absent. Attaching a
fake evaluator or current host runtime to historical declarations would be false
provenance. The protected checks cannot be edited under current authorization.

## Additional harness limitations

- Founder/lineage/origin tests iterate empty arrays and pass vacuously. They do
  not offset failing milestone tests.
- Both test suites discover only existing `weights-*/STANDARD.json` files.
  A weights directory missing its declaration is silently absent from this
  discovery; the code does not enforce its own docstring's every-directory rule.
- `scientific_acceptance:false` is not a semantic-check exemption. Ignoring such
  files or renaming them out of discovery would route around the current gate.
- Preparation writers intentionally emit empty declarations, e.g.
  `native_parallel_batch.py:164`, `native_composition_batch.py:169`,
  `native_transfer_curriculum.py:189`, `nca_feasibility_preflight.py:219`,
  `nca_author_intake.py:241`. NCA declarations bind immutable preparation state;
  subsequent attempt/terminal receipts truthfully record the later two calls.
  Their unchanged zero-author wording is historical freeze-time context, not
  a current aggregate claim.
- The separately diagnosed dictionary-zero ablation bug remains protected and
  unresolved. See `PROTECTED_CONTROL_GATE_DIAGNOSTIC_20261008.md`.

## Permitted next work and explicit limits

No implementation-only fix can honestly make the existing qualification run
pass. Existing snapshots and hash-bound source must stay unchanged. A future,
separately approved experiment can ship an additive, fully bound terminal
qualification export with an authentic evaluator and actual attempt/provenance
ledger; it must retain invalid/timeout/duplicate attempts and must fail when the
scientific thresholds are unmet. That export would require a prospectively
specified compatibility contract; it is not fabricated here. Changing protected
schema handling or discovery needs specific owner approval. Adding seeds or
inventing crossover/genomes is not a code repair.

Three additive expected-defect regressions in
`tests/test_qualification_schema_diagnostics.py` pass in 0.001 seconds: all three
semantic paths reproduce the missing-evaluator exception, empty population checks
pass vacuously, and discovery omits a declaration-less directory. These synthetic
fixtures establish diagnostic behavior only. No existing file was changed.

The exact per-declaration counts, seed lists, mismatch arms and source SHA256s
are in `qualification-taxonomy-20261008/MANIFEST_INVENTORY.json`. This inventory
covers the partial exact-source workspace's 16 historical top-level declarations
plus the two new NCA preparation declarations; it makes no whole-repository or
excluded evaluation-dataset coverage claim.
