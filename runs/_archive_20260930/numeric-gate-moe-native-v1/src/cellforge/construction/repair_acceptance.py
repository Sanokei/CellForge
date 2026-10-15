"""Development-only archive admission; no scoring or hidden-data access."""
import math


def repair_admission(parent, candidate):
    """Require fresh niche gain and per-task archive/anchor nonregression.

    Both arguments contain fresh, archive and anchor score sections. Each
    section identifies a development manifest, source and exact task outcomes.
    Callers must truthfully supply provenance; this cannot detect disguised
    final data or prove a purported real anchor is real.
    """
    reasons = []
    for label in ("fresh", "archive", "anchor"):
        left, right = parent[label], candidate[label]
        for section in (left, right):
            if section.get("partition") != "development":
                raise ValueError("development-only admission evidence required")
            if not section.get("manifest") or not section.get("outcomes"):
                raise ValueError("manifest and nonempty per-task outcomes required")
            if any(type(v) is not bool for v in section["outcomes"].values()):
                raise ValueError("exact boolean task outcomes required")
            rate = section["false_positive_given_consistency"]
            if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1):
                raise ValueError("invalid false-positive rate")
        if (left["manifest"] != right["manifest"] or left.get("source") != right.get("source")
                or left["outcomes"].keys() != right["outcomes"].keys()):
            raise ValueError("paired admission populations differ")
        if label == "anchor" and left.get("source") != "public-training-real":
            raise ValueError("real public-training anchor required")
        losses = [k for k in left["outcomes"] if left["outcomes"][k] and not right["outcomes"][k]]
        if losses:
            reasons.append({"gate": label + "_regression", "tasks": losses})
        old, new = left["false_positive_given_consistency"], right["false_positive_given_consistency"]
        # Undefined consistency denominators cannot prove rate nonregression.
        if old is None or new is None or new > old:
            reasons.append({"gate": label + "_false_positive", "parent": old, "candidate": new})
    if sum(candidate["fresh"]["outcomes"].values()) <= sum(parent["fresh"]["outcomes"].values()):
        reasons.append({"gate": "strict_fresh_gain"})
    if candidate.get("caps_passed") is not True or candidate.get("fixtures_passed") is not True:
        reasons.append({"gate": "caps_or_fixtures"})
    return {"accepted": not reasons, "reasons": reasons,
            "scope": "paired development admission; not final-data generalization"}
