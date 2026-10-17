# Cell-Family Digest — Oct-1 Program (two-loop-20261001 + cell populations)

Date: 2026-10-16. Purpose: prior art for the blackboard/bidding + decoder-only RMT router.
Sano's thesis stands: perception + reasoning as one set of weights, weights evolving
to glue themselves to other weights. This is the prior iteration, not a restart.

## What the Oct-1 program built

**Shared cell registry** (`shared-cell-registry/LATEST.json`): 3 live families, 12 live cells,
competition_iteration 1, one representative per family, execution receipts per task.

The three families (all perfect on their dev sets):
- **decimal-subtraction** (rep bb59131b…): 9/9 — borrow chains, propagate, clear-borrow, underflow, leading-zero
- **spatial-cardinal** (rep 50b21880…): 40/40 on spatial development tasks
- **signed-temporal-deduction** (rep 6fae589d…): 8/8 — three-hop, mixed conjunction, contradiction-no-explosion, delayed negative

**Cell authoring** (`cell_population.py`): "independent literal-weight cells with charged,
traceable development selection." Cells are finite literal numerical tensors
(`validate_literal` enforces finiteness, no slot-keys). Selection across fresh/archive/anchor
partitions with per-task outcome manifests.

**The glue that existed** — numerical crossover with inheritance receipts:
- `_inheritance(recipient, donor, receipt)`: two-parent birth requires a receipt binding
  BOTH parent tensor hashes; whole homologous tensor transfers replayable across families;
  `inherit_neighbors` with `router_row_from_donor` — a router row inherited from the donor,
  i.e. routing glue passed genetically.
- Red-queen opponent history (`OpponentHistory`) for competitive co-evolution.
- MoE scheduler work (`author-moe-v2`), planning-runtime, hard-reasoning-runtime,
  reasoning-cell-bootstrap-stage — composition machinery beyond single cells.

**Competition dynamics**: families competed via the shared typed registry — representatives
selected per family, execution receipts recorded per environment. Input-type gating:
a cell facing an incompatible input type records `status: "inapplicable"` instead of
executing. **This is salience in spirit** — a hard 0/1 bid. The blackboard's continuous
bid_0_to_1 is the soft generalization of this gate.

## What stalled (why architecture_complete=false)

The registry payload says it plainly:
> scope: "imported earned survivors, actual typed broad replay; **native evolution not
> yet dispatched through this registry**", `architecture_complete: false`,
> `scientific_acceptance: false`.

- The registry was a REPLAY of earned champions, not a live evolution loop.
  Competition froze at iteration 1 — no second generation ever ran through it.
- Families were siloed by input type: cross-family receipts are almost all "inapplicable."
  No cross-family composition was ever demonstrated on a shared task — the glue existed
  as machinery (crossover, router rows) but never as a selected behavior.
- Some arms failed outright (g1 summary: `llm_count`/`neighbor_count` cell, all outcomes false).
- Document-crossover/audit logs in /srv/cellforge-private are empty shells — the audit
  trail was never written.

## REUSE vs REPLACE

**REUSE** (port into the blackboard):
- Typed capability families as cell species — decimal-subtraction, spatial-cardinal,
  signed-temporal-deduction are ready-made reasoning cells with proven dev behavior.
- Execution receipts / input-type gating → the bid heuristic: a cell's salience rule
  starts as its applicability gate, then softens to a learned bid.
- Inheritance receipts (hash-bound two-parent crossover) → the evolved-glue mechanism:
  bids and router rows as heritable, auditable tensors.
- Literal-weight discipline (finite, no slot-keys) → keep for all authored cells.
- Representative-per-family selection → keep as the elitism pattern.

**REPLACE** (what the new architecture fixes):
- Static typed registry → blackboard with per-step bidding. Gating was hard 0/1 and
  per-environment; bids are continuous and per-step, tuned by live RHAE pressure.
- Replay-only competition (iteration 1, frozen) → the CellForge loop: AUTHOR → STAKE →
  COMPETE → PAYOUT → ELIMINATE, every generation, with transfer RHAE as the metric.
- Siloed families (cross-family = "inapplicable") → decoder-only RMT router where cells
  compose through learned bids in one token stream, one loss. Composition becomes the
  selected behavior instead of the untested machinery.
- No memory story → 8 RMT memory tokens with the zeroing load-bearing test.

## Concrete next steps for the LM specialist

1. Port the 3 family representatives as blackboard cells: wrap each in
   `observe(state) -> (annotation, bid)` where bid starts as the old applicability gate
   (1.0 if input type matches the family's domain, 0.0 otherwise), then learns.
2. Seed the bid-evolution design with the receipt pattern: every bid update writes an
   inheritance-style receipt (which intakes preceded which efficiency gains).
3. Do NOT port the registry replay machinery — the CellForge loop replaces it.

## One-line summary

Oct-1 proved cells could be authored, typed, gated, and crossed with auditable receipts —
but the competition never ran past iteration 1 and families never composed. The blackboard
+ RMT router is the same thesis with the competition actually switched on.
