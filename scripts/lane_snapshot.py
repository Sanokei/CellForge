"""Reduce server lane directories to a small JSON that tests/test_hard_lane_focus.py reads. Read-only on the server.

  python scripts/lane_snapshot.py                  # ssh to the server in HARD_LANES.json, write reports/lane-snapshot/latest.json
  python scripts/lane_snapshot.py reduce BASE B64  # internal: runs on the server (or locally on a fixture directory)

Formats understood (see HARD_LANES.json "format"):
  generations: GENERATION-n.json {decisions:[{seed,niche,arm,candidate,solved,accepted,reasons}]}, CHARGES-n.jsonl,
               AUTHOR-n/response.json (list of proposals with rationale), PROMPTS-n.json, STATUS.json, STOP, SERVICE.pid
  arc:         ARC_HISTORY.json, ARC_SCORE.json, SERVICE_STATUS.json
"""
import base64
import zlib
import datetime
import glob
import json
import os
import subprocess
import sys

SELF = globals().get("__file__") or "lane_snapshot.py"  # undefined when the server runs this from stdin
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SELF)))
OUT =os.path.join(ROOT, "reports", "lane-snapshot", "latest.json")


def iso(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).isoformat()


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def n_of(path):
    return int(os.path.basename(path).split("-")[1].split(".")[0])


def dev_total(run, niche, fallback):
    if fallback:  # explicit head-center value wins over inference
        return fallback
    tasks = load(os.path.join(run, "TASKS.json"))
    if tasks and isinstance(tasks.get("development"), dict) and niche in tasks["development"]:
        return len(tasks["development"][niche])
    decl = load(os.path.join(run, "DECLARATION.json"))
    if decl and isinstance(decl.get("development"), list) and decl["development"]:
        return len(decl["development"])
    return fallback


def charges(run, n):
    out = {"llm": 0, "random": 0}
    path = os.path.join(run, "CHARGES-%d.jsonl" % n)
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row.get("arm") in ("llm", "random"):
                    out[row["arm"]] += int(row.get("positions", 0))
    return out


def responses(run, n):
    data = load(os.path.join(run, "AUTHOR-%d" % n, "response.json"))
    if not isinstance(data, list):
        return []
    return [{"seed": r.get("seed"), "niche": r.get("niche"), "rationale": str(r.get("rationale", "")),
             "charged_positions": r.get("charged_positions"), "prompt_sha256": r.get("prompt_sha256")}
            for r in data if isinstance(r, dict)]


def reduce_generations(run, cfg):
    gens = []
    for path in sorted(glob.glob(os.path.join(run, "GENERATION-*.json")), key=n_of):
        n = n_of(path)
        data = load(path) or {}
        rows = [dict({k: d.get(k) for k in ("seed", "niche", "arm", "solved", "accepted", "reasons")},
                     candidate=d.get("candidate") or d.get("id") or d.get("child_id"))  # services name the candidate hash differently
                for d in data.get("decisions", [])]
        total = max([dev_total(run, r["niche"], cfg.get("dev_total")) or 0 for r in rows] or [0]) or cfg.get("dev_total")
        gens.append({"n": n, "utc": iso(path), "rows": rows, "total": total, "charges": charges(run, n),
                     "responses": responses(run, n)})
    return gens


def oracle_values(node, out):
    """Collect every oracle_solved_diagnostic_only (best expert per task: the ceiling any router edit can reach)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "oracle_solved_diagnostic_only" and isinstance(value, int):
                out.append(value)
            else:
                oracle_values(value, out)
    elif isinstance(node, list):
        for value in node:
            oracle_values(value, out)
    return out


def reduce_arc(run, cfg):
    score = load(os.path.join(run, "ARC_SCORE.json")) or {}
    out = {"history": load(os.path.join(run, "ARC_HISTORY.json")) or score.get("history", []),
           "claimed": score.get("claimed", {}) and {k: score["claimed"].get(k) for k in ("solved", "total")},
           "controls": score.get("controls", {}), "oracle_solved": None, "pool_cells": None}
    states = sorted(glob.glob(os.path.join(run, "STATE-*.json")), key=n_of)
    if states:
        state = load(states[-1]) or {}
        found = oracle_values(state, [])
        out["oracle_solved"] = max(found) if found else None
        out["pool_cells"] = len((state.get("pool") or {}).get("cells", {})) if isinstance(state.get("pool"), dict) else None
    return out


def reduce_driver(run, cfg):
    """Head-center skill-bank driver: HISTORY.jsonl (one JSON per stage) -> compact niche / breeding / mutation records."""
    rows = []
    path = os.path.join(run, "HISTORY.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    keep = ("stage", "utc", "niche_cells", "best_pass_rate", "arc_dev", "arc_val", "arc_report", "arc_held_out", "baseline_dev", "baseline_val",
            "baseline_report", "children_valid", "children_admitted", "children_rejected_no_arc_gain", "mutations_tried", "mutations_admitted",
            "random_edits_that_would_gain", "random_chain_valid", "targets", "cells")
    niches = {}
    reg_path = os.path.join(run, "REGISTRY.json")
    if os.path.exists(reg_path):  # breadth = distinct niches the registry actually holds cells for (a seeded run inherits earlier niches)
        for cell in json.load(open(reg_path, encoding="utf-8")):
            if cell.get("niche") and cell.get("kind") == "cell":
                niches[cell["niche"]] = niches.get(cell["niche"], 0) + 1
    return {"records": [{k: r.get(k) for k in keep if k in r} for r in rows if not (r.get("stage", "").count(":gen"))], "registry_niches": niches}


def reduce_lane(base, cfg):
    base = cfg.get("base", base)
    run = os.path.join(base, "runs", cfg["run"])
    if not os.path.isdir(run):
        return {"id": cfg["id"], "run": cfg["run"], "exists": False}
    status = load(os.path.join(run, "STATUS.json")) or load(os.path.join(run, "SERVICE_STATUS.json")) or {}
    pid = status.get("pid")
    if pid is None and os.path.exists(os.path.join(run, "SERVICE.pid")):
        try:
            pid = int(open(os.path.join(run, "SERVICE.pid")).read().strip())
        except ValueError:
            pid = None
    stop = None
    for candidate in (os.path.join(run, "STOP"), os.path.join(base, "control", cfg["run"].replace("weights-", ""), "STOP")):
        if os.path.exists(candidate):
            stop = open(candidate, encoding="utf-8", errors="replace").read(400)
    files = [os.path.join(run, f) for f in os.listdir(run) if os.path.isfile(os.path.join(run, f))]
    out = {"id": cfg["id"], "run": cfg["run"], "exists": True, "format": cfg["format"],
           "status": dict({k: status.get(k) for k in ("status", "waiting_for", "utc", "generation", "round")}, status=status.get("status") or status.get("state")),
           "stop": stop, "pid": pid,
           "pid_alive": os.path.exists("/proc/%s" % pid) if pid and os.path.isdir("/proc") else None,
           "last_activity_utc": max(iso(f) for f in files) if files else None}
    out["generations"] = []
    if cfg["format"] == "driver":
        out["driver"] = reduce_driver(run, cfg)
    elif cfg["format"] == "arc":
        out["arc"] = reduce_arc(run, cfg)
    elif cfg["format"] == "generations":
        out["generations"] = reduce_generations(run, cfg)
    return out  # format "opaque": existence/status/activity only


def safe_lane(base, cfg):
    try:
        return reduce_lane(base, cfg)
    except Exception as error:  # one malformed lane must not blind the snapshot; tests/hard_lane_checks.py fails on reduce_error
        return {"id": cfg["id"], "run": cfg["run"], "exists": True, "generations": [], "reduce_error": "%s: %s" % (type(error).__name__, error)}


def reduce_all(base, lanes):
    runs = os.path.join(base, "runs")
    return {"schema": "lane-snapshot/1",
            "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "base": base, "run_dirs": sorted(d for d in os.listdir(runs) if d.startswith("weights-") and os.path.isdir(os.path.join(runs, d)))
            if os.path.isdir(runs) else [],
            "lanes": {cfg["id"]: safe_lane(base, cfg) for cfg in lanes}}


def write_status(snap):
    """reports/lane-snapshot/HEAD_STATUS.md: which focus checks are red right now, readable without server access."""
    sys.path.insert(0, ROOT)
    from tests import hard_lane_checks as H
    cfg = H.load(H.CFG_PATH)
    lines = ["# HEAD STATUS (auto-written by scripts/lane_snapshot.py)", "", "captured %s; directive %s" % (snap["captured_utc"], cfg["directive_utc"]), ""]
    red = 0
    for check in H.ALL:
        found = check(snap, cfg)
        red += bool(found)
        lines.append("- %s %s" % ("RED  " if found else "green", check.__name__[len("check_"):]))
        lines += ["    - " + p for p in found]
    lines += ["", "Lanes:"]
    for lane_cfg, lane in H.lane_pairs(snap, cfg):
        gens = (lane or {}).get("generations", [])
        credit = len(H.credited_generations(lane, lane_cfg, cfg)) if lane and lane.get("exists") else 0
        lines.append("- %s [%s] status=%s generations=%d credited=%d/%d" % (
            lane_cfg["id"], lane_cfg["role"], ((lane or {}).get("status") or {}).get("status"), len(gens), credit, cfg["policy"]["min_hard_generations"]))
    lines.insert(2, "**%d checks RED**" % red if red else "**all focus checks green**")
    with open(os.path.join(os.path.dirname(OUT), "HEAD_STATUS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return red


def pull():
    cfg = load(os.path.join(ROOT, "HARD_LANES.json"))
    base = cfg["server"]["base"]
    arg = "z:" + base64.b64encode(zlib.compress(json.dumps(cfg["lanes"]).encode())).decode()  # the ssh command line truncates near 8 KB
    with open(os.path.abspath(SELF), "rb") as src:
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", cfg["server"]["ssh"],
                               "python3 - reduce %s %s" % (base, arg)], stdin=src, capture_output=True, timeout=300)
    if proc.returncode:
        sys.exit("ssh reduce failed: " + proc.stderr.decode(errors="replace")[-600:])
    snap = json.loads(proc.stdout)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=1, sort_keys=True)
    print("wrote", OUT, "(%d focus checks red; see HEAD_STATUS.md)" % write_status(snap))
    for lane in snap["lanes"].values():
        gens = lane.get("generations", [])
        print("%-24s exists=%s status=%s gens=%d stop=%s" % (lane["id"], lane["exists"], (lane.get("status") or {}).get("status"),
                                                              len(gens), bool(lane.get("stop"))))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "reduce":
        raw = sys.argv[3]
        lanes = json.loads(zlib.decompress(base64.b64decode(raw[2:])) if raw.startswith("z:") else base64.b64decode(raw))
        print(json.dumps(reduce_all(sys.argv[2], lanes)))
    else:
        pull()
