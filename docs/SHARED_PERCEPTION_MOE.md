# Shared perception, routed experts: one artifact for the ARC Prize 2026 tracks

Date: 2026-09-30. Design note only. No code was run and no new result is reported. Every number below is quoted from a repo file that is named next to it. Status words: "exists" means a file in the repo does it today; "proposal" means it does not.

## 1. Goal and non-goals

**Goal.** Build one codebase and one artifact family, made of small hand-authored cells joined into circuits (bundles), then experts, then a graph-owned router. The same perception cells feed the ARC-AGI-2 static grid experts, the ARC-AGI-3 interactive explorer, and the Paper Prize experiments. The constraints are fixed by the owner: no gradients, no LLM at runtime, CPU and low power, and variation by `import_def` crossover plus discrete selection (`docs/BREEDING.md`, `docs/ARC_PRIZE_2026_PLAN.md`).

**What "one agent for all prizes" means.** One repository, one shared library of perception cells and circuits, one evolution harness, and per-track expert bundles that import the shared parts. This shared artifact and codebase is the claim.

**What it does not mean.**
- Kaggle still needs a separate notebook and submission for ARC-AGI-2 and for ARC-AGI-3. The Paper Prize needs a separate paper (`docs/ARC_PRIZE_2026_PLAN.md`, deadlines section: Nov 2 submissions, Nov 8 papers). One model file cannot be submitted to all three.
- It is not a claim of general intelligence, and not a prediction of any private score. The plan file withdraws the earlier "low single digits to 10%" aspiration.

**Is it an LLM?** No. The language graph is a count-model-level system. The unified-byte artifact `f0b43dbe` (`runs/unified-byte/artifact/MODEL_CARD.json`) has 30,791 stored parameters, and its inherited experts use only the last eight projected bytes, per the card's own `limitations` field. The count-model reference in `notes/CLAUDE_RESEARCH.md` reaches about 1.27 bits/byte with counts plus a word model and cache, and 1.22 with the gradient-free adaptive mixer; the same note records the founder at 2.51 on a 24-document subset, worse than a trigram count model and worse than xz. The task brief describes generation from the language graph as degenerate; I did not find a file that measures this, so I treat it as owner-reported and unverified.

To call anything here LLM-like we would need, at minimum: (a) a text-in/text-out interface; (b) LM quality measured on held-out data against the count references (bits/byte, with `mc-byte-loglik/1` multiple-choice scores kept as a separate metric, per `docs/BREEDING.md`); (c) instruction following measured on a sealed set. `docs/BREEDING.md` already says document benchmarks stay on hold until plain-text QA works. None of (a) to (c) exists.

**"A really smart corvid"** is an aspiration with a testable reading: strong few-shot reasoning and tool-like exploration on small problems, with little stored knowledge. It is not a current property. The only evidence available is the ARC numbers in section 6, and they do not support the word "smart" yet. The tests that would: ARC-AGI-3 levels cleared and action efficiency, and ARC-AGI-2 evaluation-style solves with few false positives.

## 2. Architecture: layers of reuse

```
 raw input: grid pair list (ARC-AGI-2)  |  64x64 frame stream (ARC-AGI-3)  |  text/tool stub
                              |
                   [ ROUTER: modality + confidence ]   <- graph-owned, top-1 (later top-k) dispatch
                   /                |                       \
   +---------------+   +---------------------+   +------------------+
   | GRID EXPERTS  |   |  EXPLORER EXPERT    |   | LANGUAGE/TOOL    |
   | geometry      |   |  frontier + action  |   | expert (stub)    |
   | fitted local  |   |  prioritiser +      |   | count LM,        |
   | tiling        |   |  reset/backtrack    |   | arithmetic, OCR  |
   | crop-by-crit. |   +---------------------+   +------------------+
   | object-relat. |          ^        ^
   +---------------+          |        |
        ^      ^          action   state id
        |      |          proposals
   -----+------+-----------------+--------------------   CIRCUITS (shared)
   object-graph builder -> relation table -> state-graph memory
        ^                     ^                   ^
   -----+---------------------+-------------------+----   CELLS (shared)
   segmentation | colour/role abstraction | symmetry/period | counting
   state hashing + volatile mask | relation extraction (contains/adjacent/aligned/same-shape)
```

### Cells (shared perception, authored)
Each cell is deterministic, parameter-light and has one typed job:
- **Segmentation**: grid or frame to connected components (by colour, by non-background, with 4 or 8 connectivity), plus hole and pocket regions and separator-delimited cells. Existing Python prototypes: `research/claude_arc/cells.py` (`components`, `obj_largest`, `grid_cell_pick`), `research/claude_arc/objects.py` (`seg_objects`, `seg_holes`, `seg_pockets`, `seg_cells_of_separators`), and `research/claude_arc3/cells.py` (flood fill for frames).
- **Colour and role abstraction**: background detection (`bg_color`), colour to role (background / same as center / other / outside) as in `fitted.py` (`role_features`).
- **Symmetry and periodicity**: `sym_complete`, `mirror_*`, `_period_fill`, `_diag_period_fill` (`objects.py`).
- **Counting**: `_counts`, `count_colors_bar` (`cells.py`, `objects.py`).
- **State hashing with volatile-region masking**: `VolatileMask`, `state_key` in `research/claude_arc3/cells.py`.
- **Relation extraction**: per-region property and relation rows (`region_table` in `objects.py`). Contains/adjacent/aligned/same-shape are named in the brief. `objects.py` has adjacency-flavoured properties, but I did not verify a dedicated "same-shape" or "aligned" relation cell, so treat those as **proposals**.

### Circuits (bundles of cells)
1. **Object-graph builder**: grid to object list (id, bbox, mask, colour histogram, shape key, holes).
2. **Relation table**: object list to a fixed-width table of pairwise relations. This is the ARGA-style abstraction (arXiv 2210.09880, `notes/CLAUDE_RESEARCH.md` pass 7) that the gap study says we lack.
3. **State-graph memory**: state id, action edge table, untested-action lists, dead states. Prototype: `Graph` in `research/claude_arc3/explorer.py`.

### Experts (bundles that import circuits)
- **ARC-AGI-2 grid experts**: geometry and mirroring, fitted local rules (`fitted.py`), tiling and orientation (`fitted2.py`), crop-by-criterion (`fitted2.py`, `objects2.py` `fit_crop_obj`), and object-relational transforms (`objects.py` `fit_regions`, `fit_move`, `fit_lines`; `objects2.py` `fit_stamp`, `fit_rays`).
- **ARC-AGI-3 explorer expert**: state graph, frontier (untested state-action pairs), action prioritisation by salience, BFS to the nearest frontier, RESET edge. Prototype: `explorer.py`.
- **Language/tool expert (stub)**: the existing count-model/byte graph, arithmetic, OCR and deduction tools listed in `docs/SYSTEM12_DESIGN.md`. They remain separate components whose claims are bounded there.

### Router
A graph-owned dispatch cell. Inputs: modality flag (grid-pair list vs frame stream vs text), cheap demo statistics (same-shape, object count, colour count), and expert confidence. Confidence for grid experts is demonstration consistency with an explicit false-positive flag; `research/claude_arc/solve.py` already does demo verification and ranks two attempts. The count-fitted selector evidence (3/789 vs 0/789 with adaptation off, `docs/ARC_PRIZE_2026_PLAN.md`) is the existing router-shaped result.

### Typed port contracts

| Port | Type | Produced by | Consumed by |
|---|---|---|---|
| `grid` | int tensor [H,W], values 0..9, H,W <= 30 (ARC-AGI-2) or 64x64 frame (ARC-AGI-3) | input adapter | segmentation, hashing |
| `objects` | list of {id, bbox, mask, colour hist, shape key, hole count}, capped length | object-graph builder | relation table, experts |
| `relations` | fixed-width rows (i, j, relation code, scalar) | relation table | object-relational experts |
| `state_id` | 64-bit hash of the masked frame | hashing cell | state-graph memory |
| `action_proposal` | ranked list of (action, optional x,y) with a priority key | explorer expert | host loop |
| `grid_out` | two ranked attempts, per the `grid-graph/1` contract | grid experts | submission writer |

Each contract includes a declared size cap, so validation and runtime fixtures can check them the way `docs/MAKER_OPERATOR.md` already requires for grid state.

### Authored vs fitted vs evolved

| Kind | What | Provenance label |
|---|---|---|
| Authored | cell logic, circuits, expert structure, router features | hand-written, no fit |
| Fitted by counting at inference | local-rule tables, colour maps, crop criteria, tile orientations, edge tables of the state graph | fitted on the demonstrations of one task or on the current game episode; no gradients |
| Evolved offline | which experts occupy which typed slots, wiring, discrete constants, exploration priority orderings | selected on development fitness only |

The repo already distinguishes these provenance classes: fitted cells are "not purely hand-authored static cells" (`docs/ARC_PRIZE_2026_PLAN.md`), and count-fitted language seeds must not silently become ARC cells (`docs/MAKER_OPERATOR.md`).

### Where recurrent state lives
- Grid track: the `state` tensor in the `grid_step` entry, updated once per demonstration and frozen at the query (`src/cellforge/runtime/grid.py`). Reset for each call.
- Explorer track: the state-graph memory. It is per-level episodic memory, reset on level change (`explorer.py`). Across games there is no memory in this design. That is a deliberate limit: cross-game learning would need a plasticity rule (a hypothesis in `notes/CLAUDE_RESEARCH.md` pass 4).

## 3. Mapping to the existing CellForge IR

**Exists today.**
- Cells, bundles, nested and shared definitions, top-1 lazy `dispatch`, recurrent state carried through entries: `src/cellforge/ir/graph.py` (the module docstring lists `op:`, `param:`, `dispatch`, definition instances).
- `import_def` crossover/transplant (`docs/BREEDING.md`), transactions with validation and fixtures (`docs/MAKER_OPERATOR.md`).
- A grid entry: `meta.task == "grid"` requires `grid_step` (`graph.py` validate), run by `GridPredictor` (`src/cellforge/runtime/grid.py`).
- Parameter-free ops include `patchify`, `shift`, `cumsum`, `cummax`, `csr_lookup`, `onehot_argmax`, `slice`, `pad` (`src/cellforge/runtime/ops.py`). Dense, fixed-shape tensor maths is the native style.
- Bank/slot evidence: seven typed grid genes, seven-slot homologous crossover with lineage checks (`docs/ARC_PRIZE_2026_PLAN.md` and the reports it links).

**Does not exist.**
- **Object-graph ops.** All of the segmentation, object, relation and fitted-rule code in `research/claude_arc/` is ordinary Python, not CellForge graphs. The plan file says the Claude baseline is "an external reference, not a CellForge graph".
- **Dynamic-size grids.** `pack_grid` pads everything to 30x30 with a mask; there is no 64x64 frame path.
- **Interactive action loop.** `GridPredictor.__call__` takes complete demonstrations and one query. Nothing in `graph.py` or `grid.py` expresses observe, choose action, observe. The ARC-AGI-3 explorer (`research/claude_arc3/explorer.py`) uses the `arcengine` Python package and a Python dict graph.

**Minimal IR extensions (all proposals).**
1. `op:label_components` (masked, bounded count K, fixed output [K, H, W] masks plus a count scalar). Fixed K keeps the IR dense and validatable.
2. `op:region_props` (mask stack to property table) and `op:relation_table` (property table to pairwise relation tensor).
3. `op:state_hash` with an explicit volatile mask input, output a 64-bit id as two floats or an integer port type. This needs an integer/hash port type, which the float-only tensors do not have.
4. A host-loop entry `act_step`: `(frame, state) -> (action_logits or ranked list, state)` mirroring `grid_step`. The host calls it per step. Graph-owned memory rides in `state`. The edge table is large and sparse, so a sparse state (as `csr_lookup` already uses) is preferable to a dense vector.
5. Variable-size grid support by extending the `pack_grid` contract to 64x64 and making the mask a first-class input of every segmentation op.

If extension 3 or 4 proves awkward, the fallback is a validated host-side Python adapter that wraps the graph, with the edge table outside the graph. That would weaken the "graph-owned" claim for ARC-AGI-3, and the paper must say so.

## 4. Evolution plan

All operators are discrete. No gradients and no fitness-weighted gradient estimators, per the owner rule (`docs/BREEDING.md`, plan file).

- **Fixed typed-slot genome.** Slots: segmentation variant, background rule, relation set, per-expert rule family, exploration priority order, router feature set. Validity is preserved by construction. This follows the 20-gene fixed-width design in arXiv 2609.27242 (`notes/CLAUDE_RESEARCH.md` pass 3), and the repo's seven-slot bank.
- **Homologous recombination.** Swap like-typed slots between contemporaries with `import_def`.
- **Symbiogenesis operator.** Merge two viable graphs into one and select at the new level, then prune. The research note records the earlier A1+A2 child as an unnamed instance.
- **Niche archive and novelty rejection.** Niche keys: shape relation, object count, colour count. Reject a candidate whose outputs on all probe inputs duplicate an existing member. `research/claude_arc/evo/archive.py` is the existing prototype.
- **Red-Queen arms race on generators.** Generators are dense, re-arc-style; new candidates must beat the accumulated archive. Prototype: `research/claude_arc/evo/redqueen.py`. The Hodel re-arc and ARC-DSL downloads still need the owner's approval (pass 7).
- **Baldwin-style fitness.** Score a genome after in-task adaptation (demonstration-fitted tables or state). Selects genomes that adapt fast.
- **Controls, at equal evaluation budget:** random proposals with the same validity filter, and no-crossover (mutation only). The repo already reports that mutation can tie crossover (`docs/ARC_PRIZE_2026_PLAN.md`: matched seven-slot study, 22/24 vs 22/24, random 20/24), and that the earlier population pilot showed no advantage. Treat crossover benefit as unproven.
- **Sealed splits.** Develop on public training; public evaluation is already exposed for this library and must not be used for tuning. Freeze before any independent final access.

Claims stay modest: on a 2-CPU, 4 GB server, populations will be small, and the plan file already records one negative matched result. The evolution story is worth telling only with its controls.

## 5. Evaluation protocol

**Predeclared metrics.**
- ARC-AGI-2: tasks solved (exact match on all test inputs, two attempts), false-positive rate (demo-consistent but wrong on query, both as a fraction of consistent tasks and of all tasks, the two forms used in the plan file table), reported separately for the development split and the validation split of public training. Training performance stays in internal provenance, per the paper guidance noted in the plan file.
- ARC-AGI-3: levels cleared per game and the action-efficiency score. The exact score computation is listed as unconfirmed in `docs/ARC_PRIZE_2026_PLAN.md`; verify before declaring any threshold.
- Cost: CPU seconds per task or per episode, and peak RSS (the repo already records both).

**Shared-perception ablation.** Build the ARC-AGI-2 object-relational expert and the ARC-AGI-3 explorer twice:
- (S) *shared*: both import the same segmentation, colour-role and hashing cells from one definition (one hash).
- (D) *duplicated*: each track carries its own independently authored copy, with no `import_def` link.

Hold everything else fixed (expert logic, budgets, seeds, splits). Report metrics above plus unique and expanded parameters and definition counts. Also run a cross-transplant check: replace a track's perception circuit with one evolved on the other track and measure the change.

**What would falsify reuse.** The hypothesis is that shared cells cost nothing and transfer useful structure. It is falsified if any of these holds with predeclared margins: (1) (S) is worse than (D) on either track beyond the bootstrap interval, meaning one perception stack cannot serve both; (2) the cross-transplanted circuit loses most of the target track's performance, meaning cells specialise by track; (3) the definition counts and runtimes of (S) and (D) are equal, meaning reuse saved nothing. A result of "no measurable difference in score but fewer definitions" supports a weaker engineering claim only, and the paper should say that.

## 6. Risks and honest expectations

- **ARC-AGI-2 transfer.** Measured on the read-only library: public training 97/1,000 (9.7%) with the colour-role rules, demo-consistent 111, false positives 14; public evaluation 0/120 with no demo-consistent chain (`notes/CLAUDE_RESEARCH.md`, update 2026-09-30 later; receipts `research/claude_arc/results_*.json`). The earlier version in the plan file table (v3: 95/1,000) used the same evaluation and also scored 0/120. The research note's 97 figure is not in the plan file table and I did not re-count it.
- **Why the gap.** The gap study (`research/claude_arc/gap_study.json`): evaluation grids average 373 cells vs 182, 5.3 vs 3.9 colours, and a conflict-free local rule exists for 7.4% of same-shape tasks vs 22.4%.
- **Curated-against-shallow-search.** Pass 8 records an unverified snippet that ARC-AGI-2's evaluation set is curated to filter out tasks solvable by shallow transformation search. It would explain the 0/120 at depth 2. I could not verify the claim. Expect a pure ARC-AGI-2 solver to land near zero on the hidden set (same note).
- **ARC-AGI-3 is the better fit, with caveats.** A training-free graph explorer reports a preview placement of third on the private board (arXiv 2512.24156, pass 8), but the 2026 Kaggle public notebooks report public scores around 0.09 to 0.23 (repo README claims, unverified). These are reported figures from other people's work. No ARC-AGI-3 result of ours is recorded in the files I read; `research/claude_arc3/` exists but I read no results from it. The explorer needs the `arc-agi` package (owner approval) and an owner-run Kaggle notebook.
- **Population size.** On a 2-CPU machine, evolution will run small populations with few generations. Prior runs are negative or tied against controls (section 4).
- **Shared perception could simply not help.** Section 5 lists the outcome that would say so.
- **IR fit.** The interactive loop and sparse state may not fit the float-tensor IR well. A host-side adapter is the honest fallback.
- **Exposure.** The public evaluation set is exposed for this library, and library revisions fitted on training are not independently unexposed.

## 7. Four-week build order (Sept 30 to Nov 8)

Owners: "Claude subagents" (authoring and analysis in `research/`), "Codex" (IR, runtime, packaging, controls, per the owner's usual division; confirm). The owner runs Kaggle uploads and approves downloads.

| Week | Milestone | Owner |
|---|---|---|
| 1 (Sept 30 to Oct 6) | Verify ARC-AGI-3 rules, score formula, `arc-agi` package approval. Specify the five IR extensions as a written contract with fixtures. Port segmentation, hashing and colour-role cells behind one typed interface. | Owner (approvals); Claude subagents (cell port); Codex (IR spec) |
| 2 (Oct 7 to Oct 13) | Implement `label_components` and `state_hash` ops and `act_step` entry. First internet-disabled Kaggle ARC-AGI-2 notebook around Oct 12 (plan file target). Explorer baseline running locally on public games. | Codex; owner (notebook run) |
| 3 (Oct 14 to Oct 25) | Object-graph and relation circuits; object-relational expert bundle. Run the S vs D ablation and evolution arms with random and no-crossover controls at equal budget. Entry and team-merger deadline Oct 26. | Claude subagents (experts, relations); Codex (harness, controls) |
| 4 (Oct 26 to Nov 2) | Freeze artifacts, seeds, fallbacks; reproducibility and runtime margin; owner-reviewed final submissions for both competitions by Nov 2. No private-score tuning. | Codex (packaging); owner (submit) |
| Nov 3 to Nov 8 | Paper: architecture, ablation, controls, negative evidence, exposure and provenance classes. Submit by Nov 8. | Claude subagents (draft); owner (review, submit) |

If the week-2 IR extensions slip, ship the ARC-AGI-3 track as a host-adapter explorer and say so. The paper's shared-perception claim then rests on the ARC-AGI-2 side and the Python cell library, and the ablation is reported as partial.
