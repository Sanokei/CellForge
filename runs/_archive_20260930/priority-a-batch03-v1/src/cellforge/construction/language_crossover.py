"""Exact module transfer for evolutionary language candidates; no selection."""
from __future__ import annotations

from .transactions import import_def
from ..ir.graph import Graph


def crossover_module(recipient: Graph, donor: Graph, target: str, source: str,
                     prefix: str) -> tuple[Graph, dict]:
    """Namespace donor ancestry and reconnect existing recipient call sites.

    Contract checks are structural. Runtime shape/state probes remain a required
    fitness admission gate; equal port names alone do not establish semantics.
    Parents and unused recipient definitions are preserved exactly.
    """
    if not prefix or any(name.startswith(prefix) for name in recipient.defs):
        raise ValueError("a fresh nonempty namespace is required")
    for key in ("vocab", "text_codec", "ctx_len", "state_dim"):
        if recipient.meta.get(key) != donor.meta.get(key):
            raise ValueError(f"incompatible language contract: {key}")
    a, b = recipient.defs[target], donor.defs[source]
    for key in ("kind", "role", "inputs"):
        if a.get(key) != b.get(key):
            raise ValueError(f"incompatible module contract: {key}")
    if a["kind"] != "bundle" or set(a["outputs"]) != set(b["outputs"]):
        raise ValueError("module transfer requires matching bundle output ports")
    child = recipient.copy()
    copied = set(import_def(child, donor, source, prefix))
    replacement = prefix + source
    sites = []
    for name, definition in child.defs.items():
        if name in copied or definition["kind"] != "bundle":
            continue
        for node in definition["nodes"]:
            if node["use"] == target:
                node["use"] = replacement
                sites.append([name, node["id"], "call"])
            elif node["use"] == "dispatch":
                branches = node["args"]["branches"]
                if target in branches:
                    node["args"]["branches"] = [replacement if x == target else x for x in branches]
                    sites.append([name, node["id"], "dispatch"])
    if not sites:
        raise ValueError("target has no recipient call sites")
    lineage = {"protocol": "language-module-crossover/1",
               "recipient_hash": recipient.hash(), "donor_hash": donor.hash(),
               "target": target, "source": source, "prefix": prefix,
               "copied_definitions": sorted(copied), "rewired_sites": sites,
               "runtime_admission_required": True}
    child.meta["evolution_module_lineage"] = lineage
    child.validate()
    return child, dict(lineage, child_hash=child.hash())
