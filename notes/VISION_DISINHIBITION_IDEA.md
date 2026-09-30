# Idea check: perception, "controlled craziness" and novel problem solving (2026-09-30)

Owner idea: Leivada and Boeckx, 'Schizophrenia and cortical blindness: protective effects and implications for language' (Frontiers in Psychology 2014): maybe giving
the system vision/language-token perception, which makes it a little 'crazy', is an important ingredient for solving novel problems.
Sources: search summaries of that paper, schizotypy-creativity reviews, predictive-coding accounts of psychosis (not full-paper reads).

## What the research actually says
- The paper: essentially no case reports of schizophrenia in congenital/early blindness; in a population study none of 66 children with cortical blindness developed psychosis.
  The protection seems specific to CORTICAL (not peripheral) blindness, tied to occipital cortex, LGN and pulvinar. The authors link it to reduced flexibility in language and in
  dynamic body representation, which would protect against thought disorder. It is an argument about what shields against psychosis, not evidence that vision causes
  creativity or novel problem solving.
- Schizotypy and creativity: positive schizotypy goes with looser semantic associations, lower latent inhibition and more divergent thinking in some studies, but a meta-analysis
  found only a weak relationship, and reviews stress that divergent thinking without convergence/coherence is the failure mode.
- Predictive coding: psychosis is modelled as aberrant precision weighting (priors vs sensory evidence mis-weighted), i.e. a failure of reality testing.

## What transfers (and what does not)
- Does NOT transfer: 'make it crazy'. The evidence points to loose generation PLUS intact reality testing, not to disorder.
- Transfers as engineering:
  1. Loose generation, strict verification: widen and diversify hypotheses (distant associations, lower prefiltering), keep the demo-pair check (our oracle) as the reality test. This matches the
     ARC literature finding that success is generation-bound and that diversity matters (modality-diverse search, arXiv 2606.31543).
  2. Multi-view perception: our solver sees grids as pixels and a few object summaries. The analogue of 'also seeing it another way' is several parallel representations (pixel grid,
     object graph, row/column projections, colour histograms, symmetry/periodicity spectra, relation graph) feeding one hypothesis generator, with cross-view proposals.
  3. An evolvable 'disinhibition' gain: how many features/hypotheses are kept before verification, annealed or evolved, and (for ARC-AGI-3) an evolvable precision weight between learned
     action-effect priors and fresh observations.
- Test: arms with/without multi-view cells and with/without the disinhibition gain, equal budget, validation bucket of real training tasks, false-positive rate reported. Null result is acceptable.
