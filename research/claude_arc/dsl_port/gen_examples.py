"""Offline curriculum generator wrapper (re-arc, MIT, Michael Hodel; NOT copied into the repo).
Runs in its OWN process because re-arc ships its own `dsl.py`/`utils.py` that clash with third_party/arc_dsl.
Only DEVELOPMENT-bucket families are generated (crc32(id)%5 != 0). These generators exist only for known ARC-1 task
ids: this is an offline selection/curriculum tool, not a test-time component.
usage: python gen_examples.py N_EXAMPLES [ids...]   (default: all dev ids that have a generator and are in ARC-2 training)
env: REARC=<path to re-arc clone>, ARC2=<arc2 data root>, GEN_TIMEOUT=<sec per family, default 20>
"""
import os, sys, json, glob, re, zlib, time, random, signal, threading

HERE = os.path.dirname(os.path.abspath(__file__))
SP = ("C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/"
      "48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad")
REARC = os.environ.get("REARC", SP + "/re-arc")
ARC2 = os.environ.get("ARC2", SP + "/arc2/data")
OUT = os.path.join(HERE, "cache", "gen")


def is_dev(n):
    return zlib.crc32(n.encode()) % 5 != 0


def main():
    n = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    src = open(REARC + "/generators.py").read()
    have = set(re.findall(r"def generate_(\w+)", src))
    train = {os.path.basename(f)[:-5] for f in glob.glob(ARC2 + "/training/*.json")}
    ids = sys.argv[2:] or sorted(i for i in have & train if is_dev(i))
    ids = [i for i in ids if is_dev(i)]       # hard guard: never generate validation-bucket families
    sys.path.insert(0, REARC)
    os.chdir(REARC)
    import generators
    budget = float(os.environ.get("GEN_TIMEOUT", "20"))
    for i in ids:
        fo = os.path.join(OUT, i + ".json")
        if os.path.exists(fo):
            continue
        random.seed(zlib.crc32(i.encode()))
        gen = getattr(generators, "generate_" + i)
        ex, seen, t0, tries = [], set(), time.time(), 0
        while len(ex) < n and time.time() - t0 < budget and tries < 20 * n:
            tries += 1
            try:
                e = gen(0, 1)
                a, b = e["input"], e["output"]
                if a == b or a in seen or len(a) > 30 or len(a[0]) > 30 or len(b) > 30 or len(b[0]) > 30:
                    continue
                seen.add(a)
                ex.append({"input": [list(r) for r in a], "output": [list(r) for r in b]})
            except Exception:
                pass
        json.dump(ex, open(fo, "w"))
        print(i, len(ex), round(time.time() - t0, 1), flush=True)


main()
