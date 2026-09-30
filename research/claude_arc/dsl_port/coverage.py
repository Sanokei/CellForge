"""Offline per-family coverage table from re-arc generated examples (DEVELOPMENT families only).
For each generator family: E episodes, each = 3 generated demo pairs + 1 generated test input; run the dsl_port families
(fit on the 3 demos) and record which families produce an EXACT test output (any candidate) and which attempt-1 is right.
Offline curriculum/selection tool only: generators exist only for known ARC-1 task ids (not usable at test time).
usage: python coverage.py cache/coverage.jsonl [--episodes 4] [--tcap 4] [--workers 2] [--limit N]
"""
import os, sys, json, glob, time, argparse
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import objfam
from evaluate import GEN, split_of


def one(a):
    fid, E, tcap = a
    assert split_of(fid) == "dev"
    ex = json.load(open(os.path.join(GEN, fid + ".json")))
    res = {"task": fid, "episodes": [], "sec": 0.0}
    t0 = time.time()
    for e in range(E):
        if len(ex) < 3 * E + E:
            break
        demos = [(np.array(q["input"]), np.array(q["output"])) for q in ex[3 * e:3 * e + 3]]
        t = ex[3 * E + e]
        ti, to = np.array(t["input"]), np.array(t["output"])
        cands = objfam.run_all(demos, [ti], tcap)
        cands.sort(key=lambda c: c.cost)
        fams = sorted({c.family for c in cands if c.preds[0].shape == to.shape and np.array_equal(c.preds[0], to)})
        wrongf = sorted({c.family for c in cands if not (c.preds[0].shape == to.shape and np.array_equal(c.preds[0], to))})
        top = cands[0] if cands else None
        res["episodes"].append({"solved_by": fams, "wrong_from": wrongf, "n": len(cands),
                                "top1_ok": bool(top and top.preds[0].shape == to.shape and np.array_equal(top.preds[0], to))})
    res["sec"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--episodes", type=int, default=4); ap.add_argument("--tcap", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ids = sorted(os.path.basename(f)[:-5] for f in glob.glob(GEN + "/*.json"))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task"] for l in open(a.out) if l.strip()}
    ids = [i for i in ids if i not in done][:a.limit]
    with open(a.out, "a") as fo, Pool(min(2, a.workers)) as p:
        for i, r in enumerate(p.imap_unordered(one, [(i, a.episodes, a.tcap) for i in ids], chunksize=1)):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            if i % 25 == 0:
                print(i, len(ids), flush=True)


if __name__ == "__main__":
    main()
