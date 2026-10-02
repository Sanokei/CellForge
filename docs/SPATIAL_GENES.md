# Spatial gene library

`scripts/build_grid_spatial_genes.py` builds 108 individually selectable grid
graphs: 104 new spatial configurations plus four existing repeat/tile genes.
Every transformation executes through existing tensor IR operations. There
are no new interpreter primitives, runtime Python solvers, fitted weights,
benchmark reads, or demonstration-dependent routing.

| Family | Count | Behavior |
| --- | ---: | --- |
| Symmetry completion | 7 | Horizontal/vertical mirror, half turn, main/anti diagonal, quarter-turn orbit, dihedral orbit |
| Periodic continuation | 19 | Row, column, shared 2D periods 2–6; smallest valid period inference 1–6; independent row/column period inference |
| Drawing | 33 | Rays, same-color connections, and boundary-reflected paths: eight directions plus orthogonal, diagonal, and all-direction unions |
| Stencil Kronecker | 15 | Factors 2–4 with diagonal, anti diagonal, cross, border, checker stencils |
| Independent axis scaling | 30 | Each axis factor 1–4, excluding identity: nearest-neighbor enlargement and top-left strided sampling |
| Existing repeat/tile | 4 | Whole-grid tiling and isotropic pixel enlargement at factors 2 and 3 |

## Semantics

Completion and drawing treat color zero as empty. Original foreground is
preserved; new foreground collisions select the highest numbered color.
Symmetry uses the actual input dimensions. Diagonal and quarter-turn
completion requires a square; rectangular inputs return identity for those
operations. Dihedral completion still applies rectangular mirror/half-turn
symmetries.

Period inference fills missing pixels within the original canvas. Candidate
periods group observed foreground by coordinate residues. Any residue with
multiple colors rejects the candidate. Inferred shared periods prefer the
smallest value; independent periods rank by tile area, sum of periods, row
period, then column period. A rejected fixed period returns identity. Entirely
unobserved residues remain zero. A legitimate zero inside a motif cannot be
distinguished from a missing pixel. These rules infer a compatible completion,
not a uniquely proven hidden pattern or a demonstration-learned output size.

Rays propagate simultaneously from every foreground pixel and stop at the
first foreground obstacle. Connections fill pixels between collinear endpoints
of the same color (horizontal, vertical, or 45 degrees); intervening foreground
is preserved. They do not implement arbitrary-slope rasterization or ordered
waypoint navigation. Bounce paths reflect at rectangle boundaries, pass through
foreground, and continue for 29 steps by default. Singleton axes stay fixed.
Individual builders accept a shorter step budget from 1 to 29.

Kronecker genes replace every input pixel with its color multiplied by the
chosen binary stencil. For an even factor the cross uses the lower/right
central row/column. Enlargements beyond the 30×30 interface return identity.
Sampling requires both dimensions to be divisible by their respective factors;
otherwise it returns identity. Sampling is top-left selection, not majority
pooling. Kronecker and scaling produce background zeros normally and do not
preserve the old canvas as a foreground overlay.

## Build and reuse

From the repository's installed Python environment:

```powershell
python scripts/build_grid_spatial_genes.py --out runs/spatial-gene-library-v1
python scripts/build_grid_spatial_genes.py --gene connect_diagonal --out runs/my-connect-gene
python -m pytest tests/test_grid_spatial_genes.py -q
```

Output directories must be fresh. `INVENTORY.json` records each graph's name,
relative artifact path, content hash, and graph statistics. Load an individual
artifact with `Graph.load`. New graphs expose `spatial.gene`, a reusable cluster
with `input`, `input_mask`, `input_height`, and `input_width` inputs and `colors`,
`height`, and `width` outputs. Existing repeat/tile artifacts retain their
original `repeat.gene` contract. Import clusters through the existing
`import_def` transaction helper. Duplicate expert wrappers support the current
profile; they do not represent two independent capabilities.

All new graphs validate with strict microcells and the ladder-micro profile.
The fixtures exercise exact outputs on both reference and compiled runtimes,
including conflicts, empty inputs, rectangular dimensions, overflow, singleton
bounces, and nondivisible sampling. This establishes authored behavior, not ARC
coverage, admission to an evolved population, or a breeding advantage.
