"""Recompute BFCL-derived scalar-call weights on fresh generated requests.

Uses only schemas from saved report inputs; never their answers. This evaluator
is posthoc for the v2 pilot, so it does not establish predeclared replication.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bfcl_skill as B


def make_tasks(seed, n):
    path = Path(__file__).resolve().parents[1] / 'runs' / 'weights-hard-tool-use-v2' / 'REPORT_TASKS.json'
    report = json.loads(path.read_text())
    result = []
    for i in range(n):
        row = {'function': report[i % len(report)]['input']['tools']}
        pair = B.generated(row, seed * 1000 + i, tier=1)
        result.append((pair['input'], json.dumps(pair['output'], sort_keys=True)))
    return result


def predict(tensors, request):
    try:
        return json.dumps(B.bfcl_call2(tensors, request), sort_keys=True)
    except Exception:
        return 'INVALID_CALL'
