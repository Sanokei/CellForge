"""Append-only, hash-chained run ledger.

Each entry commits to the previous entry's hash, so silent edits or deletions
are detectable by ``verify``.  Hash chaining establishes integrity, not truth:
the recorder (trusted code paths such as reference evaluation) is the only
writer of measurement entries, and the chain is mirrored to the evaluation
server after each run segment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..ir.graph import canonical_json, sha256_bytes

GENESIS = "0" * 64


class LedgerError(ValueError):
    pass


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def entries(self) -> list[dict]:
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def head(self) -> tuple[int, str]:
        e = self.entries()
        return (e[-1]["seq"], e[-1]["hash"]) if e else (-1, GENESIS)

    def append(self, etype: str, payload: dict, producer: str) -> dict:
        seq, prev = self.head()
        body = {"seq": seq + 1, "prev": prev, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "type": etype, "producer": producer, "payload": payload}
        body["hash"] = sha256_bytes(canonical_json(body))
        with self.path.open("a") as f:
            f.write(json.dumps(body, sort_keys=True) + "\n")
        return body

    def verify(self) -> int:
        prev = GENESIS
        for i, e in enumerate(self.entries()):
            h = e.pop("hash")
            if e["seq"] != i or e["prev"] != prev or sha256_bytes(canonical_json(e)) != h:
                raise LedgerError(f"ledger chain broken at seq {i}")
            prev = h
        return len(self.entries())

    def find(self, etype: str) -> list[dict]:
        return [e for e in self.entries() if e["type"] == etype]
