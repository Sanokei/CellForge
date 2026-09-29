"""Run state: graph versions, branches, tasks, acceptance and records.

Layout of runs/<run>/::

  run.json            protocol, limits, arm, resource budget
  graphs/<hash>/      every accepted graph version (immutable)
  branches.json       {branch: hash}; "ref_best" is the separately evaluated best
  ledger.jsonl        hash-chained record (recorder writes measurements)
  tasks/<id>/         task packets, maker proposals, notes, dev-eval outputs
  dev/<set>/          development datasets (harness-owned, versioned)
  anchor/             reference development anchor set (protocol-owned)
  harness/            versioned curriculum, memory policy, active plugins
  memory/             evidence store
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..data.render import RenderSpec, generate, load_dataset, save_dataset
from ..ir.graph import Graph, Limits, canonical_json, sha256_bytes
from ..governance.ledger import Ledger
from ..memory.retrieve import DEFAULT_POLICY, policy_hash
from ..memory.store import MemoryStore
from ..reference.evaluator import evaluate
from ..reference.protocols import PROTOCOLS
from .bootstrap import bootstrap, bootstrap_lm
from .transactions import validate_proposal

HARNESS_VERSION_FILE = "harness/HARNESS_VERSION"


class Run:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.cfg = json.loads((self.root / "run.json").read_text())
        self.ledger = Ledger(self.root / "ledger.jsonl")
        self.memory = MemoryStore(self.root / "memory")

    # ------------------------------------------------------------- create
    @classmethod
    def init(cls, root: str | Path, protocol: str, arm: str = "primary", strict: bool = False,
             budget: dict | None = None, profile: str = "open/1") -> "Run":
        root = Path(root)
        if (root / "run.json").exists():
            raise FileExistsError(f"run {root} exists")
        root.mkdir(parents=True)
        proto = PROTOCOLS[protocol]
        kind = proto.get("kind", "ocr")
        cfg = {"protocol": protocol, "kind": kind, "profile": profile, "arm": arm,
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "limits": asdict(Limits(strict_microcells=strict)),
               "budget": budget or {"maker_tasks": 200, "dev_evals_per_task": 25}}
        (root / "run.json").write_text(json.dumps(cfg, indent=1))
        for d in ("graphs", "tasks", "dev", "harness/plugins", "memory"):
            (root / d).mkdir(parents=True, exist_ok=True)
        (root / "harness/memory_policy.json").write_text(json.dumps(DEFAULT_POLICY, indent=1))
        (root / "harness/curriculum.json").write_text(json.dumps({"version": "curriculum/1", "stages": {}}, indent=1))
        run = cls(root)
        if kind == "lm":
            raw = Path(proto["anchor_file"]).read_bytes()
            (root / "anchor").mkdir()
            (root / "anchor/texts.json").write_bytes(raw)
            man = {"split": "anchor", "source": proto["anchor_file"], "texts_sha256": sha256_bytes(raw)}
            (root / "anchor/manifest.json").write_text(json.dumps(man, indent=1))
            mh = sha256_bytes((root / "anchor/manifest.json").read_bytes())
            g = bootstrap_lm(proto["charset"], proto["ctx_len"])
        else:
            a = proto["anchor"]
            imgs, labels, man = generate(a["spec"], a["n"], a["seed"], "anchor")
            mh = save_dataset(root / "anchor", imgs, labels, man, png_count=0)
            g = bootstrap(proto["charset"], proto["max_len"])
        h = run.store_graph(g)
        run.set_branch("main", h)
        run.ledger.append("run_init", {"protocol": protocol, "arm": arm, "anchor_manifest_sha256": mh,
                                       "bootstrap": g.meta["bootstrap"], "profile": profile,
                                       "graph_hash": h}, "recorder")
        res = run.reference_eval(h, reason="bootstrap")
        run.set_branch("ref_best", h)
        return run

    # ------------------------------------------------------------- graphs
    def store_graph(self, g: Graph) -> str:
        h = g.hash()
        d = self.root / "graphs" / h
        if not d.exists():
            g.save(d)
        return h

    def load_graph(self, ref: str) -> Graph:
        return Graph.load(self.root / "graphs" / self.resolve(ref))

    def resolve(self, ref: str) -> str:
        b = self.branches()
        if ref in b:
            return b[ref]
        matches = [p.name for p in (self.root / "graphs").iterdir() if p.name.startswith(ref)]
        if len(matches) != 1:
            raise KeyError(f"cannot resolve graph ref {ref!r}")
        return matches[0]

    def branches(self) -> dict:
        p = self.root / "branches.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def set_branch(self, name: str, h: str):
        b = self.branches()
        b[name] = h
        (self.root / "branches.json").write_text(json.dumps(b, indent=1))

    def limits(self) -> Limits:
        return Limits(**self.cfg["limits"])

    # ------------------------------------------------------------ harness
    def harness_hash(self) -> str:
        files = sorted((self.root / "harness").rglob("*"))
        from ..harness import diagnostics
        h = {"diagnostics": diagnostics.DIAGNOSTICS_VERSION}
        for f in files:
            if f.is_file():
                h[str(f.relative_to(self.root))] = sha256_bytes(f.read_bytes())
        return sha256_bytes(canonical_json(h))

    def memory_policy(self) -> dict:
        return json.loads((self.root / "harness/memory_policy.json").read_text())

    def dev_set(self, name: str):
        cur = json.loads((self.root / "harness/curriculum.json").read_text())
        st = cur["stages"].get(name)
        if st is None:
            raise KeyError(f"no dev stage {name!r}; stages: {sorted(cur['stages'])}")
        spec = RenderSpec(**st["spec"])
        key = sha256_bytes(canonical_json({"spec": st["spec"], "n": st["n"], "seed": st["seed"]}))[:16]
        d = self.root / "dev" / f"{name}-{key}"
        if not (d / "manifest.json").exists():
            imgs, labels, man = generate(spec, st["n"], st["seed"], f"dev:{name}")
            save_dataset(d, imgs, labels, man, png_count=min(64, st["n"]))
        return load_dataset(d) + (d,)

    @property
    def kind(self) -> str:
        return self.cfg.get("kind", "ocr")

    @property
    def profile(self) -> str:
        return self.cfg.get("profile", "open/1")

    def lm_texts(self, name: str) -> list[str]:
        if name == "anchor":
            return json.loads((self.root / "anchor/texts.json").read_text())
        return json.loads(Path(PROTOCOLS[self.cfg["protocol"]]["dev_files"][name]).read_text())

    def sample_imgs(self, n: int = 8):
        if self.kind == "lm":
            return self.lm_texts("dev_eval")[:4]
        imgs, _, _ = load_dataset(self.root / "anchor")
        return imgs[:n].astype(np.float64) / 255.0

    # ---------------------------------------------------------- recording
    def context_hashes(self, graph_hash: str) -> dict:
        return {"model_hash": graph_hash, "development_harness_hash": self.harness_hash(),
                "execution_backend": "cellforge-ref-interp/1", "task_protocol": self.cfg["protocol"],
                "dataset_manifest_sha256": sha256_bytes((self.root / "anchor/manifest.json").read_bytes()),
                "memory_policy_sha256": policy_hash(self.memory_policy()),
                "memory_snapshot_sha256": self.memory.snapshot_hash()}

    def reference_eval(self, h: str, reason: str) -> dict:
        g = self.load_graph(h)
        if self.kind == "lm":
            from ..reference.lm_eval import evaluate_lm
            res = evaluate_lm(g, self.lm_texts("anchor"))
            res.pop("per_doc_bits"); res.pop("per_doc_tokens")
            res["model_hash"] = h
        else:
            res = evaluate(g, self.root / "anchor", keep_per_case=False)
            res.pop("predictions", None)
        payload = {"reason": reason, "result": res, "stats": g.stats(), "context": self.context_hashes(h)}
        self.ledger.append("anchor_eval", payload, "recorder")
        return res

    def anchor_history(self) -> dict:
        return {e["payload"]["context"]["model_hash"]: e["payload"]["result"] for e in self.ledger.find("anchor_eval")}

    # ------------------------------------------------------------- accept
    def accept(self, proposal: dict, task_id: str, branch: str = "main", evaluate_anchor: bool = True) -> dict:
        base_hash = self.resolve(branch)
        if proposal.get("base") not in (base_hash,):
            # allow proposals against an explicit older base only via rebase by the orchestrator
            return self._reject(proposal, task_id, f"stale base: branch {branch} is at {base_hash[:12]}")
        g = self.load_graph(base_hash)
        v = validate_proposal(g, proposal, self.sample_imgs(), self.limits(), kind=self.kind, profile=self.profile)
        if not v["ok"]:
            return self._reject(proposal, task_id, v["error"])
        new_hash = self.store_graph(v["graph"])
        entry = self.ledger.append("model_transaction", {
            "task_id": task_id, "author": proposal.get("author"), "base": base_hash, "new": new_hash,
            "pure_refactor": v["pure_refactor"], "equivalence": v.get("equivalence"),
            "rationale": str(proposal.get("rationale", ""))[:2000],
            "ops_sha256": sha256_bytes(canonical_json(proposal["ops"])),
            "op_types": [o.get("op") for o in proposal["ops"]], "branch": branch}, "recorder")
        (self.root / "graphs" / new_hash / "proposal.json").write_text(json.dumps(proposal, indent=1))
        self.set_branch(branch, new_hash)
        out = {"ok": True, "new_hash": new_hash, "pure_refactor": v["pure_refactor"], "ledger_seq": entry["seq"]}
        if evaluate_anchor and not v["pure_refactor"]:
            res = self.reference_eval(new_hash, reason=f"accept:{task_id}")
            keys = ("bpc", "top1", "failures", "n_tokens") if self.kind == "lm" else (
                "exact_match", "cer", "failures", "n_cases", "total_edits", "total_ref_chars")
            out["anchor"] = {k: res[k] for k in keys}
            best = self.anchor_history().get(self.resolve("ref_best"))
            better = (lambda a, b: a["bpc"] < b["bpc"]) if self.kind == "lm" else (
                lambda a, b: (a["cer"], -a["exact_match"]) < (b["cer"], -b["exact_match"]))
            if best is None or better(res, best):
                self.set_branch("ref_best", new_hash)
                out["new_ref_best"] = True
        elif v["pure_refactor"]:
            # behavior-equivalent: inherit measurement, record lineage
            self.memory.record("lineage", {"summary": f"pure refactor {base_hash[:12]} -> {new_hash[:12]}",
                                           "old": base_hash, "new": new_hash, "equivalence": v.get("equivalence"),
                                           "task_id": task_id, "op_types": [o.get("op") for o in proposal["ops"]]},
                               producer="recorder")
        return out

    def _reject(self, proposal, task_id, why):
        self.ledger.append("model_transaction_rejected", {"task_id": task_id, "author": proposal.get("author"),
                                                          "base": proposal.get("base"), "error": why[:1000]},
                           "recorder")
        return {"ok": False, "error": why}
