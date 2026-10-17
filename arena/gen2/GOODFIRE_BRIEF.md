# Required reading: "Neural Networks Are Already Clean" (Goodfire)
Source: https://already-clean.goodfire.com — read BEFORE any audit or authoring.
Sano's directive: every researcher reads this before the audit step.

## The three takeaways that matter for your work

### 1. Small networks are legible — neurons cover coarse-but-coherent concepts
The canonical evidence for superposition/polysemanticity (neurons packing
*unrelated* things because they ran out of room) was misread. When the author
retrained InceptionV1 across a 300x size range with ordinary architectural
fixes, 99-100% of last-layer neurons were easily interpretable at EVERY
scale. With less room, a network does NOT fold unrelated things together —
its neurons just cover COARSER-grained concepts (one car neuron covers many
kinds of car; with more room it differentiates into specific cars).
Our 1.4M-param router is far smaller than anything in that study. Expect its
neurons to be coarse but COHERENT — and therefore auditable, one by one.

### 2. Apparent polysemanticity is mostly RECURSIVE GENERALIZATION
What looked like arbitrary packing is a fractal motif: as a neuron grows it
sprouts related offshoots, the way a tree branch sprouts sub-branches
("iy" -> "diy" -> "do it yourself"). Generalization groups SIMILAR things;
superposition would group UNRELATED things — and the evidence now says it
is (almost) all the former.
PREDICTION FOR OUR TRANSFER FAILURE: on unseen games, the router's neurons
should fire as COARSE MATCHES to seen-game concepts — the wrong routine for
the right reason. Your audit must test this directly: are transfer-game
activation patterns approximately coarse re-firings of dev-game patterns?

### 3. Audit by watching activations IN ACTION, not static weights
The method that worked: conjecture-and-test loops at agentic scale — watch
data flow through the network, run experiments against explanations, study
circuits and what neurons connect to. Static weight inspection tells you
little. Understanding has momentum: each explained piece constrains the rest.
For us: hook activations during real rollouts (dev games vs transfer games),
compare per-neuron patterns, find what fires where — and where it fails.

## What to look for in the elitism root (bred_g1_a25)
- Which neurons fire per game? Any specialized ones (single-game, or only on
  level completion)?
- OVERLAP: cosine similarity of per-game mean activation vectors —
  dev-vs-dev baseline vs dev-vs-transfer. High transfer overlap with ONE dev
  game = coarse-match failure mode confirmed.
- Dead or saturated neurons = wasted capacity the author can repurpose.
- Any neuron/layer differentiating novel-vs-seen (a novelty signal)?
- Failure character: wrong-circuit firing, silence, or dithering?

## Anti-cheat note
Transfer games are MEASURE-ONLY. Recording activations is measurement, never
training. Never tune anything on what you see there. No game names in your
prompts, notes, traces, or METHOD.md — use DEV-1/2/3, TX-A/B/C labels.
