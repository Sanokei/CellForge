"""cellforge command line."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _run(args):
    from .construction.run import Run
    return Run(args.run)


def _print(obj):
    print(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------ commands
def cmd_doctor(args):
    import PIL
    from .data.render import font_dir
    fonts = sorted(p.name for p in font_dir().glob("*.ttf")) if font_dir().exists() else []
    _print({"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__,
            "font_dir": str(font_dir()), "fonts": fonts})


def cmd_init_run(args):
    from .construction.run import Run
    from .reference.protocols import PROTOCOLS
    r = Run.init(args.run, args.protocol, arm=args.arm, strict=args.strict, profile=args.profile)
    proto = PROTOCOLS[args.protocol]
    if proto.get("kind") in ("lm", "clf"):
        _print({"run": str(r.root), "branches": r.branches(), "anchor": r.anchor_history()})
        return
    spec = proto["anchor"]["spec"]
    from dataclasses import asdict
    cur = {"version": "curriculum/1", "stages": {
        "dev_main": {"spec": asdict(spec), "n": 400, "seed": 5000,
                     "note": "same distribution as anchor, disjoint seed"}}}
    (r.root / "harness/curriculum.json").write_text(json.dumps(cur, indent=1))
    _print({"run": str(r.root), "branches": r.branches(), "anchor": r.anchor_history()})


def cmd_status(args):
    r = _run(args)
    hist = r.anchor_history()
    b = r.branches()
    out = {"branches": {k: {"hash": v[:16], "anchor": {m: hist.get(v, {}).get(m) for m in ("cer", "exact_match")}}
                        for k, v in b.items()},
           "ledger_entries": r.ledger.verify(),
           "accepted": len(r.ledger.find("model_transaction")),
           "rejected": len(r.ledger.find("model_transaction_rejected"))}
    _print(out)


def cmd_task_new(args):
    from .memory.retrieve import query
    r = _run(args)
    d = r.root / "tasks" / args.id
    if d.exists():
        raise SystemExit(f"task {args.id} exists")
    d.mkdir(parents=True)
    base = r.resolve(args.branch)
    packet = None
    if not args.no_memory:
        packet = query(r.memory, {"text": args.hypothesis + " " + args.scope}, r.memory_policy())
        (d / "memory_packet.json").write_text(json.dumps(packet, indent=1))
    task = {"id": args.id, "role": args.role, "branch": args.branch, "base": base,
            "harness_hash": r.harness_hash(), "scope": args.scope, "hypothesis": args.hypothesis,
            "dev_eval_budget": args.budget, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "memory_packet": "memory_packet.json" if packet else None,
            "retrieved_context_manifest_sha256": packet["retrieved_context_manifest_sha256"] if packet else None}
    (d / "task.json").write_text(json.dumps(task, indent=1))
    r.ledger.append("task_commissioned", task, "orchestrator")
    _print(task)


def _load_proposal(path):
    return json.loads(Path(path).read_text())


def cmd_dev_eval(args):
    from .construction.transactions import validate_proposal
    from .harness import diagnostics
    r = _run(args)
    tdir = r.root / "tasks" / args.task
    task = json.loads((tdir / "task.json").read_text())
    usage = tdir / "usage.jsonl"
    used = len(usage.read_text().splitlines()) if usage.exists() else 0
    if args.validate_only:
        from .construction.transactions import validate_proposal as _vp
        prop = _load_proposal(args.proposal)
        prop["base"] = task["base"]
        v = _vp(r.load_graph(task["base"]), prop, r.sample_imgs(), r.limits(), kind=r.kind, profile=r.profile, resolver=r.load_graph)
        with (tdir / "validations.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "ok": v["ok"]}) + "\n")
        _print({"ok": v["ok"], "error": v.get("error"), "stats": v["graph"].stats() if v["ok"] else None})
        return
    if used >= task["dev_eval_budget"]:
        raise SystemExit(f"dev-eval budget exhausted ({used}/{task['dev_eval_budget']})")
    if args.set == "anchor":
        raise SystemExit("the reference anchor is not available to dev-eval; use a curriculum stage")
    g = r.load_graph(task["base"])
    if args.proposal:
        prop = _load_proposal(args.proposal)
        prop["base"] = task["base"]  # makers always edit their task base
        v = validate_proposal(g, prop, r.sample_imgs(), r.limits(), kind=r.kind, profile=r.profile, resolver=r.load_graph)
        if not v["ok"]:
            with usage.open("a") as f:
                f.write(json.dumps({"time": time.time(), "ok": False}) + "\n")
            _print({"ok": False, "error": v["error"], "dev_evals_used": used + 1})
            return
        g = v["graph"]
    if r.kind == "clf":
        from .harness import diagnostics_clf
        X, y = r.clf_data("dev_eval" if args.set == "dev_main" else args.set)
        n = min(args.n, len(y))
        out = tdir / "evals" / f"{used:03d}"
        rep = diagnostics_clf.run(g, X[:n], y[:n], out_dir=out)
        from .harness import sandbox
        plug = sandbox.active_plugin(r)
        if plug is not None and "error" not in rep:
            try:
                nodes = sandbox.declared_trace_nodes(plug.read_text(encoding="utf-8"))
                pout = sandbox.run_plugin(plug, sandbox.make_ctx(r, g, X[:n], y[:n], nodes))
                heat = pout.pop("heatmaps", {}) if isinstance(pout, dict) else {}
                from PIL import Image
                import numpy as _np
                for hname, arr in heat.items():
                    a = _np.asarray(arr, dtype=float)
                    if a.ndim == 2 and a.size <= 200_000:
                        lo, hi = a.min(), a.max()
                        img = ((a - lo) / (hi - lo + 1e-12) * 255).astype(_np.uint8)
                        k = max(1, int(256 // max(img.shape)))
                        Image.fromarray(_np.kron(img, _np.ones((k, k), dtype=_np.uint8))).save(out / f"heat_{hname}.png")
                        pout.setdefault("heatmap_files", []).append(f"heat_{hname}.png")
                rep["plugin"] = {"file": plug.name, "output": pout}
            except Exception as e:  # plugin failures never break dev-eval
                rep["plugin_error"] = str(e)[:300]
        rep["dev_set"] = f"dev_eval first {n} signals"
        rep["stats"] = g.stats()
        rep["profile"] = r.profile
        rep["dev_evals_used"] = used + 1
        with usage.open("a") as f:
            f.write(json.dumps({"time": time.time(), "ok": "error" not in rep, "n": n, "acc": rep.get("acc")}) + chr(10))
        _print(rep)
        return
    if r.kind == "lm":
        from .harness import diagnostics_lm
        import re as _re
        texts = r.lm_texts("dev_eval" if args.set == "dev_main" else args.set)
        n = min(args.n, len(texts))
        lex = set(_re.findall(r"[a-z]+", " ".join(r.lm_texts("dev_train"))))
        out = tdir / "evals" / f"{used:03d}"
        rep = diagnostics_lm.run(g, texts[:n], lex, out_dir=out)
        rep["dev_set"] = f"dev_eval first {n} docs"
        rep["stats"] = g.stats()
        rep["profile"] = r.profile
        rep["dev_evals_used"] = used + 1
        with usage.open("a") as f:
            f.write(json.dumps({"time": time.time(), "ok": "error" not in rep, "n": n, "bpc": rep.get("bpc")}) + "\n")
        _print(rep)
        return
    imgs, labels, man, ddir = r.dev_set(args.set)
    n = min(args.n, len(labels))
    out = tdir / "evals" / f"{used:03d}"
    rep = diagnostics.run(g, imgs[:n], labels[:n], out_dir=out,
                          viz_nodes=[s for s in (args.viz or "").split(",") if s],
                          trace_nodes=[s for s in (args.trace or "").split(",") if s])
    rep["dev_set"] = f"{args.set} ({ddir.name}), first {n}"
    rep["stats"] = g.stats()
    rep["outputs"] = str(out)
    rep["dev_evals_used"] = used + 1
    with usage.open("a") as f:
        f.write(json.dumps({"time": time.time(), "ok": "error" not in rep, "set": args.set, "n": n,
                            "cer": rep.get("cer")}) + "\n")
    if not args.full:
        rep.pop("trace_summaries", None) if len(json.dumps(rep.get("trace_summaries", {}))) > 4000 else None
    _print(rep)


def cmd_component_test(args):
    from .construction.contracts import run_contract
    from .construction.transactions import TransactionError, apply_ops
    r = _run(args)
    task = json.loads((r.root / "tasks" / args.task / "task.json").read_text())
    g = r.load_graph(task["base"])
    if args.proposal:
        try:
            g, _ = apply_ops(g, _load_proposal(args.proposal)["ops"])
            g.validate(r.limits())
        except (TransactionError, ValueError) as e:
            _print({"ok": False, "error": str(e)[:500]})
            return
    contract = json.loads(Path(args.contract).read_text())
    if r.profile in ("ladder/1", "ladder-micro/1"):
        from .ir.profiles import ALLOWED, _node_kind, check_micro
        name = contract["component"]
        try:
            if r.profile == "ladder-micro/1":
                check_micro(g)
            d = g.defs.get(name)
            if d is not None and d["kind"] == "bundle":
                role = d.get("role")
                if role not in ("cluster", "expert", "router"):
                    raise ValueError(f"component {name} must have role cluster|expert|router")
                for n in d["nodes"]:
                    k = _node_kind(g, n["use"])
                    if k not in ALLOWED[role]:
                        raise ValueError(f"{role} {name} may not contain {k} ({n['id']})")
        except (ValueError, Exception) as e:
            _print({"ok": False, "error": "profile violation (checked by component-test): " + str(e)[:400]})
            return
    res = run_contract(g, contract)
    with (r.root / "tasks" / args.task / "contract_tests.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.time(), "contract": args.contract, "ok": res["ok"],
                            "passed": res.get("passed"), "total": res.get("total")}) + "\n")
    _print(res)


def cmd_branch(args):
    r = _run(args)
    h = r.resolve(args.source)
    r.set_branch(args.name, h)
    r.ledger.append("branch", {"name": args.name, "from": args.source, "hash": h}, "orchestrator")
    _print(r.branches())


def cmd_accept(args):
    r = _run(args)
    prop = _load_proposal(args.proposal)
    out = r.accept(prop, args.task, branch=args.branch)
    _print(out)


def cmd_inspect(args):
    from .ir.graph import Graph
    if args.artifact:
        g = Graph.load(args.artifact)
    else:
        g = _run(args).load_graph(args.graph)
    if args.view == "stats":
        _print(g.stats())
        return
    if args.view == "hierarchy":
        def show(name, depth, seen):
            d = g.defs[name]
            if d["kind"] == "cell":
                W1 = g.params[d["params"]["W1"]]
                w2 = g.params[d["params"]["W2"]].shape if "W2" in d["params"] else None
                print("  " * depth + f"cell {name} act={d.get('act','identity')} W1{list(W1.shape)}"
                      + (f" W2{list(w2)}" if w2 else "") + f" params={list(d['params'].values())}")
                return
            print("  " * depth + f"bundle {name} inputs={d['inputs']} outputs={list(d['outputs'])}")
            for n in d["nodes"]:
                u = n["use"]
                if u.startswith("op:") or u.startswith("param:") or u == "dispatch":
                    extra = f" args={n.get('args')}" if n.get("args") else ""
                    shp = f" shape={list(g.params[u[6:]].shape)}" if u.startswith("param:") else ""
                    print("  " * (depth + 1) + f"{n['id']}: {u}{shp} in={n.get('in', {})}{extra}")
                    if u == "dispatch":
                        for b in n["args"]["branches"]:
                            show(b, depth + 2, seen)
                else:
                    print("  " * (depth + 1) + f"{n['id']}: -> {u} in={n.get('in', {})}")
                    if u not in seen or args.expanded:
                        seen.add(u)
                        show(u, depth + 2, seen)
                    else:
                        print("  " * (depth + 2) + "(shown above)")
        for e in ("encode", "step"):
            print(f"== entry {e}")
            show(g.entries[e], 0, set())
        print("meta:", json.dumps({k: v for k, v in g.meta.items()}))
        return
    if args.view == "json":
        _print({"defs": g.defs, "entries": g.entries, "meta": g.meta,
                "params": {k: list(v.shape) for k, v in g.params.items()}})
        return
    if args.view == "param":
        np.set_printoptions(linewidth=200, precision=3, suppress=True, threshold=5000)
        for p in args.names.split(","):
            print(p, g.params[p].shape)
            print(g.params[p])


def cmd_export(args):
    from .ir.graph import Graph
    r = _run(args)
    g = r.load_graph(args.graph)
    out = Path(args.out)
    h = g.save(out)
    card = {"model_hash": h, "stats": g.stats(), "vocab": g.meta["vocab"],
            "pixel_contract": g.meta.get("pixel_contract"), "task": g.meta.get("task", "ocr"), "primitive_version": g.meta.get("primitive_version"),
            "bootstrap": g.meta.get("bootstrap"), "run_protocol": r.cfg["protocol"],
            "inference_dependencies": ["numpy", "cellforge.runtime (ops/interp/decoder)", "cellforge.ir.graph"],
            "not_included": ["prompts", "labels", "development data", "memory", "author identities", "keys"],
            "anchor_result_development_feedback": r.anchor_history().get(h)}
    (out / "MODEL_CARD.json").write_text(json.dumps(card, indent=1, default=str))
    r.ledger.append("export", {"graph": h, "out": str(out)}, "recorder")
    _print(card)


def cmd_predict(args):
    from PIL import Image
    from .ir.graph import Graph
    from .runtime.decoder import decode
    g = Graph.load(args.artifact)
    img = np.asarray(Image.open(args.image).convert("L"), dtype=np.float64) / 255.0
    pc = g.meta["pixel_contract"]
    if img.shape != (pc["height"], pc["width"]):
        raise SystemExit(f"image must be {pc['height']}x{pc['width']} (got {img.shape})")
    preds, info = decode(g, img[None])
    _print({"prediction": preds[0], "failure": info["failure"][0]})


def cmd_evaluate(args):
    from .ir.graph import Graph
    from .reference.evaluator import evaluate
    g = Graph.load(args.artifact)
    res = evaluate(g, args.dataset, keep_per_case=False)
    res.pop("predictions", None)
    res.pop("per_case_edits", None)
    _print(res)


def cmd_memory(args):
    from .memory.retrieve import check_evidence, query
    r = _run(args)
    if args.mcmd == "query":
        req = json.loads(Path(args.request).read_text()) if args.request else {"text": args.text}
        _print(query(r.memory, req, r.memory_policy()))
    elif args.mcmd == "record":
        body = json.loads(Path(args.body).read_text())
        artifacts = []
        for p in body.pop("attach", []):
            artifacts.append(r.memory.put_artifact(p, Path(p).suffix.lower()))
        body["artifacts"] = body.get("artifacts", []) + artifacts
        _print(r.memory.record(args.type, body, producer=args.producer))
    elif args.mcmd == "inspect":
        for e in r.memory.all_events()[-args.last:]:
            print(e["seq"], e["type"], e["producer"], json.dumps(e["payload"])[:300])
    elif args.mcmd == "check-evidence":
        pk = json.loads(Path(args.packet).read_text())
        g = r.load_graph("main")
        _print(check_evidence(r.memory, pk, set(g.defs)))


def cmd_harness(args):
    from .harness.sandbox import activate, check_plugin
    r = _run(args)
    if args.activate:
        _print(activate(r, Path(args.proposal), args.note))
    else:
        _print(check_plugin(Path(args.proposal), r))


def cmd_report(args):
    from .reporting.report import run_report
    _print(run_report(_run(args)))


def cmd_smoke(args):
    from .providers.replay import smoke
    _print(smoke(Path(args.workdir)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cellforge")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor").set_defaults(fn=cmd_doctor)
    p = sp.add_parser("init-run"); p.add_argument("--run", required=True); p.add_argument("--protocol", default="digits-mono/1")
    p.add_argument("--arm", default="primary"); p.add_argument("--strict", action="store_true")
    p.add_argument("--profile", default="open/1", choices=["open/1", "ladder/1", "ladder-micro/1", "flat/1"]); p.set_defaults(fn=cmd_init_run)
    p = sp.add_parser("status"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_status)
    p = sp.add_parser("task-new"); p.add_argument("--run", required=True); p.add_argument("--id", required=True)
    p.add_argument("--role", default="maker"); p.add_argument("--branch", default="main")
    p.add_argument("--scope", required=True); p.add_argument("--hypothesis", default="")
    p.add_argument("--budget", type=int, default=25); p.add_argument("--no-memory", action="store_true")
    p.set_defaults(fn=cmd_task_new)
    p = sp.add_parser("dev-eval"); p.add_argument("--run", required=True); p.add_argument("--task", required=True)
    p.add_argument("--proposal"); p.add_argument("--set", default="dev_main"); p.add_argument("--n", type=int, default=400)
    p.add_argument("--viz"); p.add_argument("--trace"); p.add_argument("--full", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.set_defaults(fn=cmd_dev_eval)
    p = sp.add_parser("component-test"); p.add_argument("--run", required=True); p.add_argument("--task", required=True)
    p.add_argument("--proposal"); p.add_argument("--contract", required=True); p.set_defaults(fn=cmd_component_test)
    p = sp.add_parser("branch"); p.add_argument("--run", required=True); p.add_argument("--name", required=True)
    p.add_argument("--source", default="main"); p.set_defaults(fn=cmd_branch)
    p = sp.add_parser("accept"); p.add_argument("--run", required=True); p.add_argument("--task", required=True)
    p.add_argument("--proposal", required=True); p.add_argument("--branch", default="main"); p.set_defaults(fn=cmd_accept)
    p = sp.add_parser("inspect"); p.add_argument("--run"); p.add_argument("--graph", default="main")
    p.add_argument("--artifact"); p.add_argument("--view", default="hierarchy",
                                                 choices=["hierarchy", "expanded", "stats", "json", "param"])
    p.add_argument("--names", default=""); p.add_argument("--expanded", action="store_true"); p.set_defaults(fn=cmd_inspect)
    p = sp.add_parser("export"); p.add_argument("--run", required=True); p.add_argument("--graph", default="ref_best")
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_export)
    p = sp.add_parser("predict"); p.add_argument("--artifact", required=True); p.add_argument("--image", required=True)
    p.set_defaults(fn=cmd_predict)
    p = sp.add_parser("evaluate"); p.add_argument("--artifact", required=True); p.add_argument("--dataset", required=True)
    p.set_defaults(fn=cmd_evaluate)
    p = sp.add_parser("memory"); p.add_argument("mcmd", choices=["query", "record", "inspect", "check-evidence"])
    p.add_argument("--run", required=True); p.add_argument("--request"); p.add_argument("--text", default="")
    p.add_argument("--type", default="observation"); p.add_argument("--body"); p.add_argument("--producer", default="maker")
    p.add_argument("--packet"); p.add_argument("--last", type=int, default=30); p.set_defaults(fn=cmd_memory)
    p = sp.add_parser("harness-check"); p.add_argument("--run", required=True); p.add_argument("--proposal", required=True)
    p.add_argument("--activate", action="store_true"); p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_harness)
    p = sp.add_parser("report"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_report)
    p = sp.add_parser("smoke"); p.add_argument("--workdir", default="runs/_smoke"); p.set_defaults(fn=cmd_smoke)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
