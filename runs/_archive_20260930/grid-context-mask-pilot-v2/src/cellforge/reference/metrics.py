"""Reference OCR metrics (protected surface; changes need external approval).

Definitions (protocol ``ref-metrics/1``):
* Normalization: none.  Strings are compared exactly after decoding.
* Exact match: prediction == reference.  Failed cases (timeout, non-finite,
  missing EOS, special token mid-sequence) count as non-matches.
* CER = total Levenshtein edits / total reference characters.  A failed case
  contributes ``len(reference)`` edits (scored as an empty prediction plus a
  failure flag).  A blank reference contributes 0 to the denominator and its
  edits (insertions) still count in the numerator.  CER may exceed 1.
* Denominators are reported with every aggregate.
"""
from __future__ import annotations

PROTOCOL = "ref-metrics/1"


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(preds: list, refs: list[str]) -> dict:
    if len(preds) != len(refs):
        raise ValueError("prediction/reference count mismatch")
    edits = 0
    chars = 0
    exact = 0
    failures = 0
    per = []
    for p, r in zip(preds, refs):
        chars += len(r)
        if p is None:
            failures += 1
            e = len(r)
        else:
            e = levenshtein(p, r)
            exact += int(p == r)
        edits += e
        per.append(e)
    n = len(refs)
    return {
        "protocol": PROTOCOL,
        "n_cases": n,
        "exact_match": exact / n if n else 0.0,
        "exact_count": exact,
        "cer": edits / chars if chars else float(edits > 0),
        "total_edits": edits,
        "total_ref_chars": chars,
        "failures": failures,
        "per_case_edits": per,
    }
