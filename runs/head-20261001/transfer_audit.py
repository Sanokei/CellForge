"""Transfer audit of a bank (server, CPU only, no maker calls): how much of the ARC score is reusable weights and how much is per-task authoring?

For every task T of a split the bank's own routing rule (lowest leave-one-out demonstration error) picks a cell. We re-run that routing with the cells authored
FOR T removed (`authored_for == T`), so only weights written for OTHER tasks (or for generated niches) can win. Outcomes per task:
  solved_full        the whole bank solves T (what the headline score counts)
  solved_foreign     T is still solved when every cell authored for T is removed: the stored weights transferred
  own_only           solved only because a cell was authored for T from T's demonstrations (test-time authoring, not transfer)
  niche_only         solved by a skill-niche cell written for generated tasks (never saw T)

  python transfer_audit.py runs/weights-head-structbank1 --split report
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, "/srv/cellforge-inbox/head-20261001")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import head_driver as D  # noqa: E402
import skillcore as S  # noqa: E402
import structlib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--split", choices=("report", "selection"), default="report")
    a = ap.parse_args()
    work = a.root.parent / (a.root.name + "-audit")  # the audited run directory is never written to
    work.mkdir(exist_ok=True)
    for name in ("REGISTRY.json", L.LIB):
        (work / name).write_bytes((a.root / name).read_bytes())
    L.register(S, json.loads((work / L.LIB).read_text()))
    d = D.Driver(work, fake=True)
    rows = d.rows_rep if a.split == "report" else d.rows_sel
    cells = d.cells()
    router_full = S.default_router(cells, legacy=d.legacy_router)
    full = d.all.score(cells, router_full, rows=rows)["outcomes"]
    out = {"solved_full": 0, "solved_foreign": 0, "own_only": 0, "niche_only": 0, "unsolved": 0}
    per_task = {}
    foreign_pick = {}
    for row in rows:
        tid = row["id"]
        rest = [c for c in cells if c.get("authored_for") != tid]
        res = d.all.score(rest, S.default_router(rest, legacy=d.legacy_router), rows=[row])
        foreign_solved = res["outcomes"][tid]
        picked = next((c for c in rest if c["id"] == res["picks"][tid]), None)
        if full[tid]:
            out["solved_full"] += 1
            if foreign_solved:
                out["solved_foreign"] += 1
                kind = "niche_only" if picked is not None and picked.get("niche") != "proposed" else "foreign_structure"
                if kind == "niche_only":
                    out["niche_only"] += 1
                foreign_pick[tid] = {"kind": kind, "cell": picked["id"] if picked else None, "authored_for": picked.get("authored_for") if picked else None}
            else:
                out["own_only"] += 1
        else:
            out["unsolved"] += 1
        per_task[tid] = {"full": full[tid], "foreign": foreign_solved}
    out["foreign_structure"] = out["solved_foreign"] - out["niche_only"]
    out["tasks"] = len(rows)
    out["split"] = a.split
    (work / "TRANSFER_AUDIT.json").write_text(json.dumps({"summary": out, "per_task": per_task, "foreign_picks": foreign_pick}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
