"""Library of accepted maker-written structures. Each entry becomes a skillcore template whose code runs in the structcode sandbox.

Identity: a structure is named by the hash of its source and declared shapes (`x<10 hex>`). Its tensors are renamed with a per-structure prefix (`s<6 hex>_`),
so the key set of every structure is unique and `skillcore.template_of` can never mistake one structure for another when weights are unpacked from the
flat tensor dictionary (two structures that both call their tensor `colors` would otherwise collide).

The library file `STRUCTURES.json` sits next to the evaluator; `skillcore._load_proposed_structures` registers it when `skillcore` is loaded, so an evaluator
in a clean process needs nothing else (it also needs structcode.py and this file beside it).
"""
import hashlib
import json
import pathlib
import sys
import threading

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import structcode as X  # noqa: E402

LIB = "STRUCTURES.json"
LIBRARY_TIMEOUT = 2.0
MAX_STRIKES = 3
_box, _lock, _strikes = None, threading.Lock(), {}


def reset():
    """Forget the worker process and the strike counts (call in a forked child so processes never share one pipe)."""
    global _box
    _box = None
    _strikes.clear()


def _sandbox():
    global _box
    if _box is None:
        _box = X.Sandbox(timeout=LIBRARY_TIMEOUT)
    return _box


def sha_of(source, shapes):
    return hashlib.sha256(json.dumps([source, shapes], sort_keys=True).encode()).hexdigest()


def template_name(sha):
    return "x" + sha[:10]


def prefix_of(sha):
    return "s" + sha[:6] + "_"


def make_entry(source, shapes, house=None, task=None, doc=""):
    sha = sha_of(source, shapes)
    return {"sha": sha, "name": template_name(sha), "source": source, "shapes": {k: list(v) for k, v in shapes.items()}, "house": house, "task": task, "doc": doc}


def renamed(entry, params):
    prefix = prefix_of(entry["sha"])
    return {prefix + k: v for k, v in params.items()}


def register(S, entries):
    """Register every entry as a template in S.TEMPLATES (idempotent). Entries that fail the static whitelist are skipped, never run."""
    added = 0
    for e in entries:
        name = e["name"]
        if name in S.TEMPLATES or not X.validate(e["source"])[0] or sha_of(e["source"], e["shapes"]) != e["sha"]:
            continue
        prefix, source = prefix_of(e["sha"]), e["source"]

        def fn(p, g, d=None, source=source, prefix=prefix, name=name):
            if _strikes.get(name, 0) >= MAX_STRIKES:  # circuit breaker: a structure that keeps timing out on other tasks' grids stops costing seconds per call
                raise ValueError("disabled after repeated timeouts")
            params = {k[len(prefix):]: v for k, v in p.items()}
            with _lock:
                out, error = _sandbox().call(source, params, np.asarray(g).tolist(), list(d or []))
            if error:
                if "timeout" in error:
                    _strikes[name] = _strikes.get(name, 0) + 1
                raise ValueError(error)
            return out

        S.TEMPLATES[name] = {"shapes": {prefix + k: tuple(v) for k, v in e["shapes"].items()}, "fn": fn, "aware": X.uses_demos(source), "niche": "proposed",
                             "doc": e.get("doc", "")}
        added += 1
    return added


def load(S, directory):
    path = pathlib.Path(directory) / LIB
    return register(S, json.loads(path.read_text())) if path.exists() else 0


def save(directory, entries):
    path = pathlib.Path(directory) / LIB
    seen, out = set(), []
    for e in entries:
        if e["sha"] not in seen:
            seen.add(e["sha"])
            out.append(e)
    path.write_text(json.dumps(out))
    return len(out)


def cell_for(S, entry, params, **meta):
    return S.make_cell(entry["name"], renamed(entry, params), **meta)
