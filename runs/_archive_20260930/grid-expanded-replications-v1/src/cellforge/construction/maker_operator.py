"""Auditable manual-maker transactions; no model service or benchmark access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from .transactions import apply_ops
from ..ir.profiles import check_profile
from ..runtime.interp import Interpreter


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _guard(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in str(key).lower() for word in ("audit", "final", "sealed", "test", "heldout")):
                raise ValueError("maker diagnostics cannot contain reserved/test fields")
            _guard(item)
    elif isinstance(value, list):
        for item in value:
            _guard(item)


class MakerOperator:
    """One submitted proposal equals one budget unit, including rejected edits.

    Diagnostics are explicitly development TRAIN-pair evidence. The caller is
    responsible for their truthful provenance; this schema cannot detect labels
    disguised as permitted fields. Runtime fixtures validate typed ports/shapes,
    not correctness against hidden answers. Each valid changed candidate may be
    scored by the operator's aggregate-only development callback.
    """

    def __init__(self, ledger, *, budget, profile="ladder-micro/1"):
        if type(budget) is not int or budget < 1:
            raise ValueError("positive attempt budget required")
        if profile not in ("ladder/1", "ladder-micro/1", "flat/1"):
            raise ValueError("explicit supported graph profile required")
        self.path = Path(ledger)
        self.budget = budget
        self.profile = profile
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []

    def _append(self, row):
        row["time"] = datetime.now(timezone.utc).isoformat()
        row["previous_receipt"] = self.rows[-1]["receipt_hash"] if self.rows else None
        row["receipt_hash"] = hashlib.sha256(_json(row).encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_json(row) + "\n")
        self.rows.append(row)
        return row

    def submit(self, parent, *, diagnostics, ops, provenance, fixtures,
               resolver=None, evaluate=None, accept=None):
        if any(row["kind"] == "freeze" for row in self.rows):
            raise ValueError("maker population is frozen")
        attempt = sum(row["kind"] == "attempt" for row in self.rows) + 1
        if attempt > self.budget:
            raise ValueError("attempt budget exhausted")
        # JSON serializability is an API prerequisite; every serializable edit
        # submission below receives a receipt even if its schema is rejected.
        row = json.loads(_json({"kind": "attempt", "attempt": attempt,
            "parent_hash": parent.hash(), "diagnostics_shown": diagnostics,
            "ops": ops, "provenance": provenance, "status": "invalid"}))
        candidate = None
        try:
            if not isinstance(provenance, dict) or not provenance.get("actor") or not provenance.get("model"):
                raise ValueError("actor and model provenance required")
            if provenance.get("method") not in ("manual-llm-maker", "non-llm-mutation-control"):
                raise ValueError("explicit maker/control method required")
            if diagnostics.get("partition") != "development" or diagnostics.get("evidence") != "train-pairs":
                raise ValueError("only development train-pair diagnostics allowed")
            _guard(diagnostics)
            if not isinstance(ops, list):
                raise ValueError("transaction list required")
            candidate, _ = apply_ops(parent, ops, resolver=resolver)
            candidate.validate()
            row["profile"] = check_profile(candidate, self.profile)
            if not fixtures:
                raise ValueError("typed runtime fixtures required")
            for fixture in fixtures:
                outputs = Interpreter(candidate).run_bundle(fixture["entry"],
                    {key: np.asarray(value) for key, value in fixture["inputs"].items()}, "/maker")
                if set(outputs) != set(fixture["output_shapes"]):
                    raise ValueError("runtime output ports differ from typed contract")
                for key, shape in fixture["output_shapes"].items():
                    array = np.asarray(outputs[key])
                    if array.shape != tuple(shape) or not np.isfinite(array).all():
                        raise ValueError("runtime output shape/nonfinite failure")
            row["candidate_hash"] = candidate.hash()
            row["stats"] = candidate.stats()
            if candidate.hash() == parent.hash():
                row["status"] = "no-change"
            elif evaluate is None:
                row["status"] = "validated-unscored"
            else:
                metrics = evaluate(candidate)
                _guard(metrics)
                row["development_metrics"] = metrics
                row["status"] = "accepted" if accept is not None and accept(metrics) else "rejected"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            candidate = None
        receipt = self._append(row)
        return candidate, receipt

    def freeze(self, graph):
        if any(row["kind"] == "freeze" for row in self.rows):
            raise ValueError("already frozen")
        graph.validate()
        check_profile(graph, self.profile)
        return self._append({"kind": "freeze", "model_hash": graph.hash()})

    def sealed_once(self, graph, evaluator):
        frozen = [row for row in self.rows if row["kind"] == "freeze"]
        if not frozen or frozen[-1]["model_hash"] != graph.hash():
            raise ValueError("exact frozen graph required")
        if any(row["kind"] == "sealed-call" for row in self.rows):
            raise ValueError("sealed evaluator already called")
        # Consume access before invoking evaluator, including failed calls.
        self._append({"kind": "sealed-call", "model_hash": graph.hash()})
        return evaluator(graph)  # aggregate receipt stays outside maker diagnostics
