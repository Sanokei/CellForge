"""Conformance check for docs/ONE_STANDARD.md.

Usage: python scripts/check_standard.py [--runs runs] [--fail]

Scans runs/**/ledger.jsonl and reports, per run, how many genomes are
  - CONFORMING: carry literal weight tensors (params/weights/tensors with numeric values) and an author/origin field
  - GENE-LIST (non-conforming): genome is a list of named alleles/slots chosen from a library (selection, not weights)
  - OTHER: unrecognized shape
Exit code 1 with --fail if any run under the current standard (runs whose name matches --since-prefix) is non-conforming.
Read-only.
"""
import argparse
import collections
import json
import os


def classify(g):
    if not isinstance(g, dict):
        return "other"
    for k in ("params", "weights", "tensors", "edits"):
        v = g.get(k)
        if isinstance(v, (dict, list)) and v:
            return "weights"
    if "alleles" in g or "slots" in g:
        return "gene-list"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fail", action="store_true")
    ap.add_argument("--since-prefix", default="weights-")
    a = ap.parse_args()
    per = collections.defaultdict(collections.Counter)
    for dp, _, fn in os.walk(a.runs):
        if "ledger.jsonl" not in fn:
            continue
        run = os.path.relpath(dp, a.runs).replace("\\", "/")
        with open(os.path.join(dp, "ledger.jsonl"), encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    p = json.loads(line)["payload"]
                except Exception:
                    continue
                g = p.get("genome") if isinstance(p, dict) else None
                if g is not None:
                    per[run][classify(g)] += 1
    tot = collections.Counter()
    for run in sorted(per):
        tot.update(per[run])
        print("%-55s %s" % (run[:55], dict(per[run])))
    print("TOTAL", dict(tot))
    bad = [r for r in per if r.startswith(a.since_prefix) and per[r]["weights"] == 0]
    if a.fail and bad:
        print("NON-CONFORMING current-standard runs:", bad)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
