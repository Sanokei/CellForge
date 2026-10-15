"""Sandbox and activation for agent-written development-harness plugins (G3).

A diagnostics plugin is a Python file defining::

    TRACE_NODES = ["node_id", ...]        # optional: nodes whose activations it wants
    def report(ctx: dict) -> dict          # pure function of arrays

``ctx`` keys: X [n,L], y [n], logits [n,C], pred [n], traces {node: array [n,...]},
train_X, train_y (author-visible dev_train arrays), and ``ref_acc`` (the
reference evaluator's accuracy on X, y, computed independently).  The plugin
returns a JSON-serializable dict (<= 20 KB) and may include
``"heatmaps": {name: 2-D list}`` which the host renders to PNGs.

Checks before activation (all must pass):
  1. static: only numpy/math/json/collections/itertools/functools imports; no
     open/eval/exec/compile/__import__/getattr-of-dunder, no os/sys/subprocess/
     socket/pathlib/shutil names;
  2. executed in a separate process with a timeout, no environment secrets, in an
     empty temporary directory, with only npz-passed arrays;
  3. differential: if the plugin returns "acc" it must equal ref_acc exactly, and
     determinism (two runs give identical JSON);
  4. output size and JSON validity.
Untrusted code: this process-level isolation is a development-time check, not a
hostile-code sandbox (see the handoff: enforceable capabilities are required for
live adversarial use).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ALLOWED_IMPORTS = {"numpy", "math", "json", "collections", "itertools", "functools"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                   "os", "sys", "subprocess", "socket", "pathlib", "shutil", "importlib", "ctypes"}
MAX_OUT = 20_000

RUNNER = textwrap.dedent('''
    import json, sys, numpy as np
    src, data, out = sys.argv[1:4]
    ns = {"__name__": "plugin"}
    exec(compile(open(src).read(), src, "exec"), ns)
    z = np.load(data, allow_pickle=False)
    ctx = {k: z[k] for k in z.files if not k.startswith("trace__")}
    ctx["traces"] = {k[7:]: z[k] for k in z.files if k.startswith("trace__")}
    ctx["ref_acc"] = float(ctx["ref_acc"])
    res = ns["report"](ctx)
    json.dump(res, open(out, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
''')


class PluginError(ValueError):
    pass


def static_check(src: str) -> list[str]:
    tree = ast.parse(src)
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    bad.append(f"import {a.name}")
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                bad.append(f"from {n.module} import")
        elif isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES:
            bad.append(f"name {n.id}")
        elif isinstance(n, ast.Attribute) and (n.attr.startswith("__") or n.attr in ("system", "popen", "load", "loadtxt", "fromfile", "save", "savez", "tofile")):
            bad.append(f"attribute {n.attr}")
    return bad


def declared_trace_nodes(src: str) -> list[str]:
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "TRACE_NODES" for t in n.targets):
            return list(ast.literal_eval(n.value))
    return []


def run_plugin(src_path: Path, ctx: dict, timeout: int = 60) -> dict:
    src = Path(src_path).read_text(encoding="utf-8")
    bad = static_check(src)
    if bad:
        raise PluginError("static check failed: " + ", ".join(sorted(set(bad))[:8]))
    with tempfile.TemporaryDirectory() as td:
        arrays = {k: np.asarray(v) for k, v in ctx.items() if k != "traces"}
        for k, v in ctx.get("traces", {}).items():
            arrays["trace__" + k] = np.asarray(v)
        np.savez(Path(td) / "data.npz", **arrays)
        (Path(td) / "runner.py").write_text(RUNNER)
        (Path(td) / "plugin.py").write_text(src, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
               "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.run([sys.executable, "runner.py", "plugin.py", "data.npz", "out.json"], cwd=td,
                                  env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise PluginError(f"plugin exceeded {timeout}s")
        if proc.returncode != 0:
            raise PluginError("plugin crashed: " + (proc.stderr or "")[-400:])
        raw = (Path(td) / "out.json").read_text()
    if len(raw) > MAX_OUT:
        raise PluginError(f"plugin output {len(raw)} bytes exceeds {MAX_OUT}")
    return json.loads(raw)


def make_ctx(run, graph, X, y, nodes: list[str]):
    from ..runtime.clf import evaluate_clf, predict_logits
    lg, it = predict_logits(graph, X, trace=bool(nodes), trace_filter=(lambda p: any(("/" + n) in p for n in nodes)) if nodes else None)
    r = evaluate_clf(graph, X, y)
    tx, ty = run.clf_data("dev_train")
    traces = {}
    for k, v in (it.trace or {}).items():
        traces[k.replace("/", "_").replace("[", "_").replace("]", "")] = v
    return {"X": X, "y": y, "logits": lg, "pred": np.argmax(np.where(np.isfinite(lg), lg, -np.inf), axis=1),
            "train_X": tx, "train_y": ty, "ref_acc": r["acc"], "traces": traces}


def check_plugin(path: Path, run, n: int = 300) -> dict:
    """Full acceptance check on the current main graph and a dev sample."""
    src = Path(path).read_text(encoding="utf-8")
    try:
        bad = static_check(src)
        if bad:
            return {"ok": False, "stage": "static", "error": ", ".join(sorted(set(bad))[:8])}
        nodes = declared_trace_nodes(src)
        g = run.load_graph("main")
        X, y = run.clf_data("dev_eval")
        ctx = make_ctx(run, g, X[:n], y[:n], nodes)
        a = run_plugin(path, ctx)
        b = run_plugin(path, ctx)
    except (PluginError, SyntaxError, ValueError) as e:
        return {"ok": False, "stage": "execute", "error": str(e)[:500]}
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        return {"ok": False, "stage": "determinism", "error": "two runs differ"}
    if "acc" in a and abs(float(a["acc"]) - ctx["ref_acc"]) > 1e-12:
        return {"ok": False, "stage": "differential",
                "error": f"plugin acc {a['acc']} != reference {ctx['ref_acc']} (metric corruption?)"}
    return {"ok": True, "output_keys": sorted(a), "output_bytes": len(json.dumps(a)), "trace_nodes": nodes}


def activate(run, path: Path, note: str) -> dict:
    res = check_plugin(path, run)
    if not res["ok"]:
        return res
    hdir = run.root / "harness" / "plugins"
    hdir.mkdir(parents=True, exist_ok=True)
    ver = len(list(hdir.glob("diag_clf_v*.py"))) + 2  # built-in diagnostics is v1
    dest = hdir / f"diag_clf_v{ver}.py"
    dest.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
    (run.root / "harness" / "ACTIVE_DIAG.json").write_text(json.dumps({"file": dest.name, "version": ver}))
    run.ledger.append("harness_activation", {"plugin": dest.name, "version": ver, "note": note[:500],
                                             "check": res, "harness_hash": run.harness_hash()}, "recorder")
    return {"ok": True, "activated": dest.name, "version": ver, "harness_hash": run.harness_hash(), "check": res}


def active_plugin(run):
    p = run.root / "harness" / "ACTIVE_DIAG.json"
    if not p.exists():
        return None
    return run.root / "harness" / "plugins" / json.loads(p.read_text())["file"]
