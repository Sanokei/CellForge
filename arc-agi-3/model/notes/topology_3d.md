# 3D Weight Topology — FUTURE WORK (not the ship path)

Sano's idea, recorded 2026-10-16. Beautiful; a weeks-long research bet. Does NOT
displace the Kaggle ship path (entry due 2026-11-02, paper ~Nov 8).

## The idea

Cells as weight clusters positioned in 3D space. Connectivity between cells
determined by spatial proximity; wiring cost penalized like the brain.
Position becomes an inductive bias: things that fire together, wire together —
because they're neighbors.

## Lineage

- Spatial embeddings (word2vec → transformer positional encodings → learned latent spaces)
- Wiring-cost minimization in neuroscience (Cherniak; Bullmore & Sporns — the brain
  minimizes wiring length subject to functional constraints)
- Neuromorphic / physical neural nets (position as a real constraint, not a metaphor)
- Our own Oct-1 work: families as typed clusters; the registry as a flat namespace.
  3D topology is the registry grown a geometry.

## Why it's for the paper, not the entry

- Requires a training regime where position is differentiable or evolvable (weeks).
- The current blackboard (bids as soft wiring) is the functional prototype of the same
  idea: bids ARE learned proximity. A future iteration could literally embed bid-space
  in 3D and let geometry do what attention does.
- Paper framing: "from bidding to geometry — attention as the shadow of wiring cost."

## Concrete first experiment (post-deadline)

Take the trained blackboard router; extract the bid matrix (cell × step contexts);
embed in 3D (UMAP/t-SNE); test whether functional families cluster spatially without
supervision. If yes: the brain-like geometry is already latent in the bids, and making
it explicit is the next architecture.

---

## STATUS UPDATE (2026-10-16): ACTIVE TIMEBOXED EXPERIMENT

Sano was right — the research exists. Moved from future-work to timeboxed experiment
(days, not weeks). Transfer RHAE decides: if the spatial variant wins, it's the ship
candidate; if not, honest null results become a paper section.

### Citations (verified)

1. **HyperNEAT** (Stanley et al. 2009): substrates are N-dimensional — the 3D substrate
   is CPPN(x1,y1,z1,x2,y2,z2)=w. Published, established. The lineage Sano remembered.
2. **"Training Neural Networks by Optimizing Neuron Positions"** (arXiv:2506.13410,
   Martinos/Harvard 2025): neurons embedded in Euclidean space, positions optimized
   BY GRADIENT DESCENT, weights = inverse spatial distance. Parameter-EFFICIENT
   (fewer params), competitive on MNIST, holds up past 80% sparsity. The implementable
   one — no evolutionary paradigm change required.
3. **"Harnessing cortical geometry, wiring, and function as inductive biases for RNNs"**
   (arXiv:2606.14975, June 2026): real cortical coordinates constraining recurrent
   nets beat baselines.

### Experiment

- Spatial-embedding variant of the tiny router: every neuron gets a learnable 3D
  coordinate; connection strength = f(distance) (inverse-distance per the Harvard
  paper) + wiring-cost penalty in the loss. Fewer learnable weight params, same or
  better capacity.
- Compare vs baseline router on TRANSFER RHAE (unseen games).
- Rationale: parameter efficiency + sparsity tolerance serve the tiny-prize story
  (fewer params, cheaper inference); the biological inductive bias is the paper's
  differentiator. Ship-path discipline holds: transfer RHAE decides.

### The earlier "first experiment" (bid-matrix embedding) stays valid as analysis,
### but the implementable bet is now the gradient-optimized spatial router itself.
