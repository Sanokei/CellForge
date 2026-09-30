# Validity review: maker repair loop and Red Queen (literature check, 2026-09-30)

Question from the owner: does the breed-then-repair idea (docs/MAKER_REPAIR_LOOP.md) hold up against published evidence, and is
the Red Queen idea better for competition? Summaries below come from abstracts and search results, not full-paper reading.

## Evidence for the maker repair loop
- LLM as mutation operator with an external executable fitness works across several systems: FunSearch, AlphaEvolve, CodeEvolve
  (arXiv 2510.14150), ShinkaEvolve, Darwin Gödel Machine. CodeEvolve reports that matched-condition gains come from the
  *interaction* of islands, inspiration-based crossover, meta-prompting and refinement, not any single operator; with a 30B open
  model it beat AlphaEvolve on one benchmark at roughly an order of magnitude lower cost than a frontier ensemble.
- Semantic crossover works better than splicing text: give the LLM several parents as 'inspiration' and let it recombine, because
  spliced code breaks syntax and semantics. (Relevant: our typed-slot `import_def` crossover avoids that problem structurally.)
- Repair needs an EXTERNAL signal. 'Large Language Models Cannot Self-Correct Reasoning Yet' (arXiv 2310.01798): without external
  feedback, self-correction often does not help and can degrade results. With execution feedback, self-debugging improves code generation.
- What in the feedback matters (arXiv 2606.31511, frozen 0.5B-1.5B code models, 290 failing tasks): concrete failing examples paired
  with the code gave +18 unlocks over a bare retry and +15 over a generic placebo; instruction wording alone gave +3 (not significant).
  But blind resampling also gave +18 over a bare retry, so part of any repair gain is just extra samples.
- Modular NEAT / memetic precedent: local repair after recombination (Lamarckian) is an established hybrid; Baldwinian fitness after
  adaptation smooths the landscape.

## Evidence against / limits
- LLMs do not reliably write dense numeric weights. NeuroGen (arXiv 2505.12470) and D2NWG (arXiv 2402.18153) generate parameters only after
  training on weight codebooks; chat models adapted without that produced short, inconsistent codebooks. Design implication: the maker
  should write structure, small lookup tables, slot choices, wiring and code for small cells, not thousands of free-form numbers.
- Intrinsic reflection is weak. Diagnostics must be concrete executable counterexamples (input, expected output, predicted output, the
  cells that fired), not prose critique.
- Published gains use strong LLMs with large evaluation budgets. Whether the loop pays off with our small evaluation budget and CPU
  fitness is untested. ARC-specific support: LLM-written program libraries reach 26-29% ARC-AGI-2 but use the LLM at inference.
- Iterative refinement can reduce hypothesis diversity (arXiv 2606.31543 reports degradation from prescriptive refinement): keep novelty
  rejection and niche archives so repair does not collapse the population.

## Red Queen / coevolution
- Works: adversarial archives produce robustness (Digital Red Queen: each warrior beats all predecessors; behavioural convergence;
  MAP-Elites used per generation to keep diversity). Self-play co-evolution yields more robust defenders than training against a static attacker.
- Fails: competitive overfitting / cycling. Both sides hyper-specialise against each other, internal win rates look balanced while
  out-of-distribution generalisation collapses, invisible without an external evaluation set. DRQ itself is a single lineage in a sandbox.
- For ARC-style static tasks there is no direct published evidence yet; the mechanism is plausible as a curriculum/selection pressure.

## Assessment
- The two ideas are complements, not rivals: Red Queen supplies *selection pressure* (what to be good at, via generators that are not
  too easy); maker repair supplies *variation* (how to fix a specific failure). Repair has the stronger direct evidence for improving a
  given artifact; Red Queen has evidence for robustness, and for competition its value depends on an external validation set to
  detect competitive overfitting.
- Best synergy: arms-race generators produce hard counterexamples, and those are exactly the concrete falsifying examples the maker
  repairs from (the feedback type shown to work).
- Required controls: (a) breeding only, (b) breeding + random/resampling repair at equal evaluations, (c) breeding + maker repair,
  (d) each with and without Red Queen generators, always scored on an untouched validation slice of real tasks.
- A null result is acceptable and reportable.
