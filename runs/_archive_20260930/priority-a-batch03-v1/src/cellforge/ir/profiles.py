"""Structural profiles.

``open/1``   v3 open recursion: any nesting, no role requirements.
``ladder/1`` owner-specified hierarchical MoE:
    cell    -> primitive (kind cell)
    cluster -> bundle of ops/params/cells only
    expert  -> bundle of ops/params/clusters only
    router  -> bundle of ops/params/clusters only; must output ``scores``
    root    -> the step entry: ops/params/clusters (shared trunk), router
               instances and dispatch nodes; at least one dispatch whose
               branches are >=2 experts and whose scores come from a router.
  Raw cells may appear only inside clusters.  Experts may not nest experts.
``ladder-micro/1`` ladder/1 plus every cell capped at 40 parameter slots
             (params in param: nodes of clusters/experts are not capped, but count).
``flat/1``   only the entry bundles plus cells; no dispatch.
"""
from __future__ import annotations

from .graph import Graph, GraphError, parse_ref


def _node_kind(g: Graph, use: str) -> str:
    if use.startswith("op:"):
        return "op"
    if use.startswith("param:"):
        return "param"
    if use == "dispatch":
        return "dispatch"
    d = g.defs[use]
    return "cell" if d["kind"] == "cell" else d.get("role", "bundle")


ALLOWED = {
    "cluster": {"op", "param", "cell"},
    "expert": {"op", "param", "cluster"},
    "router": {"op", "param", "cluster"},
    "root": {"op", "param", "cluster", "router", "dispatch"},
}


def check_ladder(g: Graph) -> dict:
    root_names = {g.entries[e] for e in g.entries}
    n_experts = set()
    for name, d in g.defs.items():
        if d["kind"] == "cell":
            continue
        role = "root" if name in root_names else d.get("role")
        if role not in ALLOWED:
            raise GraphError(f"ladder: bundle {name} needs role cluster|expert|router (has {d.get('role')!r})")
        if role == "router" and "scores" not in d["outputs"]:
            raise GraphError(f"ladder: router {name} must output 'scores'")
        for n in d["nodes"]:
            k = _node_kind(g, n["use"])
            if k not in ALLOWED[role]:
                raise GraphError(f"ladder: {role} {name} may not contain {k} ({n['id']} uses {n['use']})")
    dispatches = 0
    for name in root_names:
        d = g.defs[name]
        ids = {n["id"]: n for n in d["nodes"]}
        for n in d["nodes"]:
            if n["use"] != "dispatch":
                continue
            br = n["args"]["branches"]
            if len(br) < 2 or any(g.defs[b].get("role") != "expert" for b in br):
                raise GraphError(f"ladder: dispatch {n['id']} must select among >=2 experts")
            kind, tgt, _ = parse_ref(n["in"]["scores"])
            if kind != "node" or _node_kind(g, ids[tgt]["use"]) != "router":
                raise GraphError(f"ladder: dispatch {n['id']} scores must come from a router instance")
            dispatches += 1
            n_experts.update(br)
    if dispatches == 0:
        raise GraphError("ladder: step root must contain at least one expert dispatch")
    return {"profile": "ladder/1", "dispatches": dispatches, "experts": sorted(n_experts)}


def check_flat(g: Graph) -> dict:
    roots = {g.entries[e] for e in g.entries}
    for name, d in g.defs.items():
        if d["kind"] == "bundle" and name not in roots:
            raise GraphError(f"flat: extra bundle definition {name} not allowed")
        if d["kind"] == "bundle":
            for n in d["nodes"]:
                if n["use"] == "dispatch":
                    raise GraphError("flat: dispatch not allowed")
    return {"profile": "flat/1"}


def check_micro(g: Graph, cap: int = 40) -> None:
    """Every cell <= cap slots, and every param: tensor used directly in a bundle
    <= cap elements, so no weights hide outside micro cells."""
    for name, d in g.defs.items():
        if d["kind"] == "cell":
            slots = sum(int(g.params[p].size) for p in d["params"].values())
            if slots > cap:
                raise GraphError(f"ladder-micro: cell {name} has {slots} slots > {cap}")
        else:
            for n in d["nodes"]:
                if n["use"].startswith("param:") and g.params[n["use"][6:]].size > cap:
                    raise GraphError(f"ladder-micro: param node {name}/{n['id']} exceeds {cap} elements")


def check_profile(g: Graph, profile: str) -> dict:
    if profile == "ladder/1":
        return check_ladder(g)
    if profile == "ladder-micro/1":
        check_micro(g)
        return check_ladder(g) | {"profile": "ladder-micro/1"}
    if profile == "flat/1":
        return check_flat(g)
    return {"profile": "open/1"}
