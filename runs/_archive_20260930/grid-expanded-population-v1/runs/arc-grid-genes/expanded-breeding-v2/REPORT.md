# Seven-slot breeding operators and frozen parents

Implemented homologous uniform crossover and one-slot uniform discrete mutation for the expanded genome. Each crossover slot inherits one of the corresponding parent alleles; the child records both parent artifact hashes, seed and donor-per-slot choices. Mutation records slot, old/new allele and no-change outcome. All edits replace typed graph definitions from the existing shared authored bank. There are no gradient estimators or weight updates.

Validation checks seven declared slots, allele/definition agreement, the strict microcell hierarchy, available bank, donor hash, immutable parameters and shared definitions, entries, state and attempts. The primary no-gradient marker is required. Canonical genotype identity uses the experiment's frozen base hash and validated alleles; lineage-only metadata changes do not rejuvenate a living genotype. The population runner must retain the same frozen base for canonical identity throughout a run.

Tests cover16 reproducible crossover children and16 mutations, every inherited slot, exact parent hashes, parent immutability, neutral metadata identity and rejection of slot, parameter, runtime and gradient-marker tampering. Five genetics/expanded-interface tests passed together; the two genetics tests were repeated after adding the explicit marker guard. These are construction and contract checks, not selection efficacy measurements.

Frozen inventory contains base, two complementary parents and four actual children at seeds0..3, with eight verified ledger entries. All have273 stored parameters/state1515; active compute varies with the chosen alleles. Parents:

- Crop/fill: `457826479d56a0d29ced282768d1c13c3ad3b0b0b7c39b86c6b847334cf9b755`
- Object/adjacency/scaling: `5795d37ce7de9c3791f0b67829ee9e6ff8478ecb149b311f130b0cecf8d10071`

Builder/operator source hashes, artifact hashes, donor slots and canonical identities are committed in `INVENTORY.json` and `ledger.jsonl`. V1 remains unchanged; V2 commits the operator source after the no-gradient marker guard was added. Graph behavior and parent hashes are unchanged by that guard.

No benchmark, query scoring, audit, public evaluation, final data, gradient steps, downloads or paid calls. The earlier110/120 authored capacity result is separate; these offspring have not received a fitness or query score. Native throughput and peak RSS for the children remain unmeasured. Next connect these operators to a matched living-population runner with independently seeded parent/proposal streams, train-only fitness, frozen winners and disjoint original report batches. Independent frontier superiority remains unproven.
