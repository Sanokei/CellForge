"""Test-time structure authoring: for every task no existing cell reproduces, houses write a structure + literal tensors from the task's demonstrations ONLY.

Protocol (frozen before the report split is touched; prompt and acceptance live in struct_pilot.py / structcode.py and were fixed on the selection split):
  - Which tasks get authored: those where NO cell in the bank reproduces every demonstration leave-one-out (a criterion that uses demonstrations only).
  - The maker sees the demonstration pairs and the test INPUTS, never a test output. Houses are tried in rotation; the first accepted structure wins
    (accepted = static whitelist + every demonstration reproduced + zeroed and random tensors fail + at most 256 tensor entries).
  - Decisions never read a test output. `killed` in STRUCT_RESULTS.jsonl is a measurement taken afterwards; it is never used to choose, retry or tune anything.
  - An accepted structure becomes a library entry (STRUCTURES.json) and a bank cell; the router then picks cells by demonstration error as for every other cell.
  - Run once per split; do not rerun after reading outcomes.

  python struct_solve.py runs/weights-head-structbank1 --seed-from runs/weights-head-tourney1 --split report --houses 4 --max-houses 2 --workers 6
"""
import argparse
import json
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/srv/cellforge-inbox/head-20261001")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import head_arc_cells as H  # noqa: E402
import head_driver as D  # noqa: E402
import skillcore as S  # noqa: E402
import struct_pilot as P  # noqa: E402
import structlib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--seed-from", type=pathlib.Path)
    ap.add_argument("--split", choices=("selection", "report"), required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = every task that needs authoring")
    ap.add_argument("--houses", type=int, default=4)
    ap.add_argument("--max-houses", type=int, default=2, help="houses tried per task before giving up")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    P.BOX_TIMEOUT = L.LIBRARY_TIMEOUT  # accepted under the same timeout the library will run with
    d = D.Driver(a.root, fake=True)
    if a.seed_from and not d.registry:
        old = H.load(a.seed_from / "REGISTRY.json")
        d.registry = [c for c in old if (c["kind"] == "cell" and c.get("origin", "").startswith("maker")) or c.get("arc_selection_gain")]
        d.save()
    lib_path = a.root / L.LIB
    entries = json.loads(lib_path.read_text()) if lib_path.exists() else []
    L.register(type("NS", (), {"TEMPLATES": S.TEMPLATES}), entries)
    results_path = a.root / "STRUCT_RESULTS.jsonl"
    done = {json.loads(line)["task"] for line in results_path.read_text().splitlines() if line.strip()} if results_path.exists() else set()
    rows = sorted(d.rows_rep if a.split == "report" else d.rows_sel, key=lambda r: r["id"])
    cells = d.cells()
    todo = [r for r in rows if r["id"] not in done and min(d.all.demo_error(c, r) for c in cells) > 0]
    if a.limit:
        todo = todo[:a.limit]
    print("split %s: %d tasks, %d already authored, %d need authoring (no cell reproduces every demonstration), houses %d (max %d per task)" % (
        a.split, len(rows), len(done), len(todo), a.houses, a.max_houses), flush=True)
    houses = P.HOUSES[:a.houses]
    lock = threading.Lock()
    tally = {"accepted": 0, "killed": 0, "done": 0}

    def work(item):
        i, row = item
        tid, rec, tried = row["id"], None, []
        order = houses[i % len(houses):] + houses[:i % len(houses)]
        for h in order[:a.max_houses]:
            rec = P.attempt(a.root, h, tid, row["task"])
            tried.append(h)
            if rec["accepted"]:
                break
        with lock:
            last = rec["attempts"][-1] if rec and rec["attempts"] else {}
            out = {"task": tid, "split": a.split, "houses_tried": tried, "accepted": bool(rec and rec["accepted"]), "killed": bool(rec and rec["killed"]),
                   "calls": sum(len(r["attempts"]) for r in [rec] if r), "positions": last.get("positions", 0)}
            if rec and rec["accepted"]:
                entry = L.make_entry(last["source"], last["shapes"], house=rec["house"], task=tid, doc=last.get("rationale", "")[:200])
                entries.append(entry)
                L.register(type("NS", (), {"TEMPLATES": S.TEMPLATES}), [entry])
                d.registry.append(L.cell_for(S, entry, last["params"], niche="proposed", pattern=tid, house=rec["house"], authored_for=tid, generation=1,
                                             general=False, origin="maker gpt-6.1-sol low, structure proposal (test-time authoring from demonstrations) house %s" % rec["house"]))
                L.save(a.root, entries)
                d.save()
            with results_path.open("a") as f:
                f.write(json.dumps(out) + "\n")
            tally["done"] += 1
            tally["accepted"] += out["accepted"]
            tally["killed"] += out["killed"]
            print("%-9s %-6s accepted=%-5s killed=%-5s | done %d/%d accepted %d killed %d" % (tid, tried[-1], out["accepted"], out["killed"], tally["done"], len(todo),
                                                                                           tally["accepted"], tally["killed"]), flush=True)
        return out

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, list(enumerate(todo))))
    rec = d.measure("structbank:" + a.split, authored=tally["done"], accepted=tally["accepted"], killed_measured=tally["killed"])
    print("MEASURED held-out %d (val %d, report %d), dev %d | authored %d accepted %d killed %d" % (rec["arc_held_out"], rec["arc_val"], rec["arc_report"], rec["arc_dev"],
                                                                                              tally["done"], tally["accepted"], tally["killed"]), flush=True)


if __name__ == "__main__":
    main()
