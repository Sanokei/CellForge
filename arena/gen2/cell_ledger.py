#!/usr/bin/env python3
"""Cell coopetition ledger logic — gen2-gen2 charter amendment 4. JUDGE ONLY.

Rules:
- Proposing a cell for the shared library costs a flat FEE from the author's
  bankroll, debited at proposal, NON-REFUNDABLE even if rejected.
- No marketplace: cells are never bought/sold/traded (nothing to implement —
  the absence is the rule).
- Usefulness dividend (house-minted): after generation judging, each ACCEPTED
  cell used by a model with transfer_rhae STRICTLY above the generation mean
  pays its author: DIVIDEND_TIERS[0] for the 1st qualifying adoption,
  [1] for the 2nd, [2] for the 3rd; cap 3 paid adoptions per cell per gen.
- Lineage: record_cells_used logs which registered cells each submitted model
  used (judge-verified from rollout logs, >= MIN_INVOCATIONS invocations).
"""
import json
import os
import datetime

ARENA = "/srv/cellforge-arena"
LEDGER = os.path.join(ARENA, "ledger", "bankroll.json")
LIB = os.path.join(ARENA, "gen2", "cell_library.json")

FEE = 50
DIVIDEND_TIERS = [100, 50, 25]
MIN_INVOCATIONS = 5


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _save(p, d):
    with open(p, "w") as f:
        json.dump(d, f, indent=1)


def register_cell(author, cell_id, description):
    """Proposal: debit fee (non-refundable), cell enters as 'proposed'.
    The coordinator audits for generic-only before accept_cell()."""
    L = _load(LEDGER)
    if author not in L["researchers"]:
        return False, f"unknown researcher {author}"
    bal = L["researchers"][author]["balance"]
    if bal < FEE:
        return False, f"{author} balance {bal} < fee {FEE}"
    lib = _load(LIB)
    if cell_id in lib["cells"]:
        return False, f"cell {cell_id} already registered"
    L["researchers"][author]["balance"] = bal - FEE
    L["log"].append({
        "ts": _now(), "gen": "gen2-gen2", "researcher": author,
        "type": "cell_fee", "amount": -FEE, "balance_after": bal - FEE,
        "note": f"registration fee for cell {cell_id} (non-refundable)",
    })
    lib["cells"][cell_id] = {
        "author": author, "ts": _now(), "status": "proposed",
        "description": description,
    }
    _save(LEDGER, L)
    _save(LIB, lib)
    return True, f"cell {cell_id} proposed by {author}; fee {FEE} debited"


def accept_cell(cell_id):
    """Coordinator calls this after generic-only audit passes."""
    lib = _load(LIB)
    c = lib["cells"].get(cell_id)
    if not c:
        return False, f"unknown cell {cell_id}"
    c["status"] = "accepted"
    c["accepted_ts"] = _now()
    _save(LIB, lib)
    return True, f"cell {cell_id} accepted into shared library"


def reject_cell(cell_id, reason):
    lib = _load(LIB)
    c = lib["cells"].get(cell_id)
    if not c:
        return False, f"unknown cell {cell_id}"
    c["status"] = "rejected"
    c["reject_reason"] = reason
    _save(LIB, lib)
    return True, f"cell {cell_id} rejected (fee not refunded): {reason}"


def record_cells_used(researcher, model_path, cell_ids, gen="gen2-gen2"):
    """Judge-verified usage lineage. cell_ids: list of accepted cell ids."""
    L = _load(LEDGER)
    L["log"].append({
        "ts": _now(), "gen": gen, "researcher": researcher,
        "type": "cells_used", "amount": 0,
        "balance_after": L["researchers"][researcher]["balance"],
        "cells": list(cell_ids),
        "note": f"model {model_path} used cells {list(cell_ids)} "
                f"(judge-verified >= {MIN_INVOCATIONS} invocations)",
    })
    _save(LEDGER, L)
    return True, f"recorded {len(cell_ids)} cells for {researcher}"


def settle_dividends(gen, transfer):
    """transfer: {researcher: transfer_rhae}. House-minted dividends for
    accepted cells used by models STRICTLY above the generation mean."""
    L = _load(LEDGER)
    lib = _load(LIB)
    rs = list(transfer.keys())
    mean = sum(transfer[r] for r in rs) / max(1, len(rs))
    # collect qualifying adoptions per cell
    uses = {}
    for r in rs:
        if transfer[r] <= mean:
            continue
        for e in L["log"]:
            if (e.get("gen") == gen and e.get("researcher") == r
                    and e.get("type") == "cells_used"):
                for cid in e.get("cells", []):
                    if lib["cells"].get(cid, {}).get("status") == "accepted":
                        uses.setdefault(cid, []).append(r)
    paid = []
    for cid, adopters in uses.items():
        author = lib["cells"][cid]["author"]
        for i, r in enumerate(adopters[:len(DIVIDEND_TIERS)]):
            amt = DIVIDEND_TIERS[i]
            bal = L["researchers"][author]["balance"] + amt
            L["researchers"][author]["balance"] = bal
            L["log"].append({
                "ts": _now(), "gen": gen, "researcher": author,
                "type": "cell_dividend", "amount": amt, "balance_after": bal,
                "note": f"house dividend: cell {cid} used by above-mean {r} "
                        f"(transfer {transfer[r]:.4f} > mean {mean:.4f}, tier {i+1})",
            })
            paid.append({"cell": cid, "author": author, "used_by": r,
                         "amount": amt})
    _save(LEDGER, L)
    return paid, mean


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "register":
        ok, msg = register_cell(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "accept":
        ok, msg = accept_cell(sys.argv[2])
    elif cmd == "reject":
        ok, msg = reject_cell(sys.argv[2], sys.argv[3])
    elif cmd == "used":
        ok, msg = record_cells_used(sys.argv[2], sys.argv[3], sys.argv[4].split(","))
    elif cmd == "settle":
        transfer = json.loads(sys.argv[3])
        paid, mean = settle_dividends(sys.argv[2], transfer)
        ok, msg = True, f"mean={mean:.4f} paid={paid}"
    else:
        ok, msg = False, "cmds: register|accept|reject|used|settle"
    print(("OK " if ok else "FAIL ") + msg)
