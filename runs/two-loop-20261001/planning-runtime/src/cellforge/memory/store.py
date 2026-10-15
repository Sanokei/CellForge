"""Evidence memory: content-addressed artifacts + append-only evidence events.

Three separate structures (see MEMORY.md):
  neural graph   -> runs/<run>/graphs/ (not here)
  evidence graph -> events.jsonl (append-only, hash-chained) + artifacts/
  retrieval index-> rebuildable view (memory/index.py); losing it loses nothing

Final-evaluation material is refused at write time: the store rejects the
``final`` namespace and any artifact whose bytes match a registered final
manifest hash.  The real boundary is physical (final data lives only on the
evaluation server); this check is defence in depth.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from ..governance.ledger import Ledger

CLAIM_STATES = {"proposed", "measured-on-development", "reproduced-under-specified-conditions",
                "contradicted", "superseded", "inconclusive"}
EVENT_TYPES = {"experiment", "observation", "claim", "correction", "decision", "lineage", "note",
               "harness_change", "tombstone"}
SAFE_EXT = re.compile(r"^\.[a-z0-9]{1,8}$")


class MemoryError_(ValueError):
    pass


class MemoryStore:
    def __init__(self, root: str | Path, final_denylist: set[str] | None = None):
        self.root = Path(root)
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        self.events = Ledger(self.root / "events.jsonl")
        self.final_denylist = set(final_denylist or ())
        deny = self.root / "final_denylist.txt"
        if deny.exists():
            self.final_denylist |= {l.strip() for l in deny.read_text().splitlines() if l.strip()}

    # ------------------------------------------------------------ artifacts
    def put_artifact(self, src: str | Path | bytes, ext: str) -> str:
        if not SAFE_EXT.match(ext):
            raise MemoryError_(f"unsafe artifact extension {ext!r}")
        data = src if isinstance(src, bytes) else Path(src).read_bytes()
        h = hashlib.sha256(data).hexdigest()
        if h in self.final_denylist:
            raise MemoryError_("artifact matches a final-evaluation hash; refused")
        p = self.root / "artifacts" / f"{h}{ext}"
        if not p.exists():
            p.write_bytes(data)
        return f"{h}{ext}"

    def artifact_path(self, ref: str) -> Path:
        if "/" in ref or "\\" in ref or ".." in ref or not re.match(r"^[0-9a-f]{64}\.[a-z0-9]{1,8}$", ref):
            raise MemoryError_(f"unsafe artifact reference {ref!r}")
        p = self.root / "artifacts" / ref
        if not p.exists():
            raise MemoryError_(f"artifact {ref} not available (tombstoned or missing)")
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != ref.split(".")[0]:
            raise MemoryError_(f"artifact {ref} fails integrity check")
        return p

    # --------------------------------------------------------------- events
    def record(self, etype: str, body: dict, producer: str, namespace: str = "dev",
               live: bool = True) -> dict:
        if etype not in EVENT_TYPES:
            raise MemoryError_(f"unknown event type {etype}")
        if namespace != "dev":
            raise MemoryError_("author memory accepts only the dev namespace")
        if etype == "claim" and body.get("state") not in CLAIM_STATES:
            raise MemoryError_(f"claim state must be one of {sorted(CLAIM_STATES)}")
        if "trusted" in body:
            body = dict(body, caller_asserted_trusted=body.pop("trusted"))  # not authentication
        for ref in body.get("artifacts", []):
            self.artifact_path(ref)
        text = json.dumps(body)
        for h in self.final_denylist:
            if h in text:
                raise MemoryError_("event references a final-evaluation hash; refused")
        return self.events.append(etype, {"namespace": namespace, "live": live, **body}, producer)

    def all_events(self) -> list[dict]:
        return self.events.entries()

    def snapshot_hash(self) -> str:
        seq, h = self.events.head()
        return h

    def tombstone(self, ref: str, reason: str, producer: str):
        p = self.root / "artifacts" / ref
        if p.exists():
            p.unlink()
        return self.events.append("tombstone", {"namespace": "dev", "artifact": ref, "reason": reason}, producer)

    def export_bundle(self, out: str | Path):
        shutil.copytree(self.root, out, dirs_exist_ok=True)
