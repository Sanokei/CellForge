# Authored grid rule genes

`scripts/build_grid_rule_genes.py` adds 39 configurable families through generic
CellForge IR. `add_rule_cluster(graph, rule, ..., prefix=...)` installs a reusable
cluster with inputs `input`, `input_height`, `input_width` and outputs `colors`,
`height`, `width`. `build_rule_gene(rule, ...)` wraps it in a standalone
`grid-graph/1`, `ladder-micro/1` artifact. Clusters can be imported or composed
with existing gene tooling. This adds capabilities, not automatic admission to
an existing frozen population or demonstration-based rule selection.

```powershell
python scripts/build_grid_rule_genes.py --rule component_size --value 3 --out runs/rule-size3
python scripts/build_grid_rule_genes.py --rule enclosed --source 0 --target 3 --iterations 900 --out runs/rule-enclosed
python scripts/build_grid_rule_genes.py --rule local_table --table '{"0":5,"2":6}' --out runs/rule-table
```

Existing output directories are preserved. Each exported artifact includes a
model card, configuration and graph hash. Construction reads no benchmark data.

| Family | Meaning |
| --- | --- |
| touching / not_touching | Source has / lacks a marker neighbor |
| neighbor_count | Exactly `value` marker neighbors |
| isolated / interior / boundary | Zero / all / fewer than all neighbors are source |
| border / not_border | Source lies / does not lie on the valid rectangle's edge |
| row / column | Zero-based coordinate equals `value` |
| top / bottom / left / right | Corresponding valid grid edge |
| row_parity / column_parity / checkerboard | Coordinate parity equals `value` (0 or 1) |
| color_size | Total source pixel count equals `value` |
| majority / minority | Source is most / least frequent present color globally |
| row_majority / row_minority | Same frequency test within each row |
| column_majority / column_minority | Same frequency test within each column |
| neighbor_majority / neighbor_minority | Same test among valid neighbors, excluding self |
| local_table | Map exact marker neighbor count to output color; omitted counts persist |
| flood_marker | Recolor source components reachable from a marker neighbor |
| flood_border | Recolor source components reachable from the grid border |
| enclosed | Recolor source components unreachable from the grid border |
| component_size / component_parity | Source component area equals `value` / has parity `value` |
| component_touching / component_not_touching | Entire source object touches / does not touch marker |
| component_border / component_not_border | Entire source object touches / does not touch grid border |
| component_largest / component_smallest | Select one source component by area |
| contained_bbox | Source pixel lies strictly inside the global marker bounding box |

All families preserve dimensions and other colors. Parameters default to
source=2, target=3, marker=4, value=1, connectivity=4, iterations=64.
Connectivity accepts 4 or 8; grid edges always use the four rectangle edges.
Frequency ties select the lowest present color, including zero. Component-area
ties select the first row-major component. Object connectivity concerns only
the source color; it does not join multicolor objects.

Flood and component propagation check an additional step. If any labels or
reachability still change, the whole transform returns identity. Set iterations
to 900 for convergence on any supported 30x30 grid, at higher graph cost.
Component tests construct a 900x900 membership matrix; parity adds another
900x451 comparison. These are substantial intermediate arrays despite small
parameter storage. `contained_bbox` is geometric pixel containment, not a
topological enclosure test; use `enclosed` for the latter.

Updates are simultaneous. A local table does not cascade its replacements.
Flood propagation follows the original source mask, not newly recolored pixels.
No rule learns its colors or predicate from demonstrations. Tests use independent
examples, backend agreement, validity/padding invariants, connectivity and
nonconvergence cases; no ARC coverage or fitness improvement has been measured.
