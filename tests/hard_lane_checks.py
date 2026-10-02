"""Focus checks for the hard lanes (HARD_LANES.json). Head-center controlled: do not weaken, skip, delete or work around.

Why these exist: every lane built so far was solved by the maker in ONE generation (4/4, 8/8, 9/9, 24/24, 40/40) and the loop then ran
5-28 more generations of neutral rewrites ("feedback reports 9/9 and no errors") while the lane with real headroom (ARC 0/100) re-ran with
an unchanged weights hash, and lanes that got hard went idle ("declared_budget_complete", "waiting for new declaration").

Each check takes (snap, cfg, now) where snap is reports/lane-snapshot/latest.json (scripts/lane_snapshot.py) and returns a list of problem
strings (empty = pass). Only generations after cfg["directive_utc"] are policed, so history cannot be rewritten and is never forgiven in
the credit count: a lane must EARN credit after the directive.

A generation is CREDITED only if all hold: the maker had headroom (champion below saturation_fraction of the dev set), its proposal is
not a neutral rewrite, charges are equal across arms and non-zero, and its candidate was not submitted before.
"""
import datetime
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "HARD_LANES.json")
SNAP_PATH = os.path.join(ROOT, "reports", "lane-snapshot", "latest.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def utc(text):
    return datetime.datetime.fromisoformat(text)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def lane_pairs(snap, cfg, role=None):
    for lane_cfg in cfg["lanes"]:
        if role is None or lane_cfg["role"] == role:
            yield lane_cfg, snap["lanes"].get(lane_cfg["id"])


def after_directive(gen, cfg, lane_cfg=None):
    """Policed generations: after the directive, and after the lane's `forgiven_through_utc` (a lane stopped by head-center is judged on what it
    does AFTER the stop, not on history nobody can change)."""
    start = max(utc(cfg["directive_utc"]), utc((lane_cfg or {}).get("forgiven_through_utc") or cfg["directive_utc"]))
    return utc(gen["utc"]) > start


def llm_rows(gen):
    return [r for r in gen["rows"] if r["arm"] == "llm"]


def total_of(gen, lane_cfg):
    return lane_cfg.get("dev_total") or gen.get("total")


def headroom(lane, lane_cfg, cfg):
    """{generation n: True if the maker could still gain}. Champion = best ADMITTED llm solve in earlier generations; a generation whose
    candidates sit at saturation but were rejected for 'strict_development_gain' also means the parent was already saturated."""
    frac = cfg["policy"]["saturation_fraction"]
    out, champion = {}, 0
    for gen in lane["generations"]:
        total = total_of(gen, lane_cfg)
        thr = frac * total if total else None
        rows = llm_rows(gen)
        parent_saturated = thr is not None and any(
            r["solved"] is not None and r["solved"] >= thr and not r["accepted"] and "strict_development_gain" in (r["reasons"] or [])
            for r in rows)
        out[gen["n"]] = (thr is None or champion < thr) and not parent_saturated
        admitted = [r["solved"] for r in rows if r["accepted"] and r["solved"] is not None]
        if admitted:
            champion = max(champion, max(admitted))
    return out


def neutral(gen, cfg):
    pat = re.compile(cfg["policy"]["neutral_rationale_regex"], re.I)
    return [r for r in gen["responses"] if pat.search(r["rationale"] or "")]


def charges_ok(gen):
    c = gen["charges"]
    return c.get("llm", 0) > 0 and c.get("llm") == c.get("random")


def seen_candidates(lane, upto_n=None):
    seen = {}
    for gen in lane["generations"]:
        if upto_n is not None and gen["n"] >= upto_n:
            break
        for r in gen["rows"]:
            if r.get("candidate"):  # lanes that do not report a candidate hash cannot be checked for resubmission
                seen[(r["seed"], r["niche"], r["arm"], r["candidate"])] = gen["n"]
    return seen


def credited_generations(lane, lane_cfg, cfg):
    """Generation numbers (generations format) or iteration numbers (arc format) that earn hard credit after the directive."""
    if lane_cfg["format"] == "arc":
        hist = (lane.get("arc") or {}).get("history", [])
        base = lane_cfg.get("baseline_iterations", 0)
        return [h["iteration"] for i, h in enumerate(hist) if i >= max(base, 1) and hist[i - 1]["weights_hash"] != h["weights_hash"]]
    room, out = headroom(lane, lane_cfg, cfg), []
    for gen in lane["generations"]:
        if not after_directive(gen, cfg, lane_cfg):
            continue
        prior = seen_candidates(lane, gen["n"])
        fresh = all(prior.get((r["seed"], r["niche"], r["arm"], r["candidate"])) is None for r in gen["rows"] if r.get("candidate"))
        valid = bool(llm_rows(gen)) and all(r.get("candidate") for r in llm_rows(gen))  # an invalid / cancelled generation proposes nothing: no credit
        if room[gen["n"]] and not neutral(gen, cfg) and charges_ok(gen) and fresh and valid:
            out.append(gen["n"])
    return out


def check_snapshot_fresh(snap, cfg, now=None):
    age = ((now or now_utc()) - utc(snap["captured_utc"])).total_seconds() / 60
    limit = cfg["policy"]["snapshot_max_age_minutes"]
    return [] if age <= limit else ["lane snapshot is %.0f min old (limit %d): run python scripts/lane_snapshot.py" % (age, limit)]


def check_snapshot_complete(snap, cfg, now=None):
    """A lane the reducer could not read is a lane the guards cannot see: that is a failure, never a pass."""
    return ["lane %s could not be reduced (%s): fix scripts/lane_snapshot.py" % (lid, lane["reduce_error"])
            for lid, lane in snap["lanes"].items() if lane.get("reduce_error")]


def check_lanes_exist(snap, cfg, now=None):
    return ["lane %s: run directory %s does not exist on the server yet" % (lane_cfg["id"], lane_cfg["run"])
            for lane_cfg, lane in lane_pairs(snap, cfg, "hard") if not lane_cfg.get("optional") and (not lane or not lane.get("exists"))]


def check_no_unregistered_lanes(snap, cfg, now=None):
    """Every runs/weights-* directory must be registered in HARD_LANES.json, so opening a fresh small lane is not a way to look busy."""
    known = {lane_cfg["run"] for lane_cfg in cfg["lanes"]}
    return ["run directory %s is not registered in HARD_LANES.json: write notes/CODEX_TO_HEAD_<date>.md with its dev-task count and "
            "generation-1 baseline" % d for d in snap.get("run_dirs", []) if d not in known]


def check_closed_lanes_stay_closed(snap, cfg, now=None):
    out = []
    for lane_cfg, lane in lane_pairs(snap, cfg, "closed"):
        late = [g["n"] for g in (lane or {}).get("generations", []) if after_directive(g, cfg, lane_cfg)]
        late = late[lane_cfg.get("grace_generations", 0):]
        if late:
            out.append("closed lane %s ran generations %s after the directive: it saturated, a further generation is a neutral rewrite"
                       % (lane_cfg["id"], late))
    return out


def check_no_generation_without_headroom(snap, cfg, now=None):
    out = []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        if not lane or not lane.get("generations"):
            continue
        room = headroom(lane, lane_cfg, cfg)
        burned = [g["n"] for g in lane["generations"] if after_directive(g, cfg, lane_cfg) and not room[g["n"]]]
        if burned:
            out.append("lane %s ran generations %s with no headroom (champion already saturated): stop the lane and escalate difficulty"
                       % (lane_cfg["id"], burned))
    return out


def check_hard_lane_dev_set_not_trivial(snap, cfg, now=None):
    need, out = cfg["policy"]["min_dev_tasks_for_hard_lane"], []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        if lane_cfg["format"] != "generations" or not lane or not lane.get("generations"):
            continue
        total = total_of(lane["generations"][-1], lane_cfg)
        need = lane_cfg.get("min_dev_tasks", need)
        if not total or total < need:
            out.append("lane %s has %s dev tasks (< %d): a trivial dev set saturates in one generation" % (lane_cfg["id"], total, need))
    return out


def check_no_neutral_authoring(snap, cfg, now=None):
    out = []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        for gen in (lane or {}).get("generations", []):
            if after_directive(gen, cfg, lane_cfg) and neutral(gen, cfg):
                out.append("lane %s generation %d: maker rationale is a neutral rewrite (%r)"
                           % (lane_cfg["id"], gen["n"], neutral(gen, cfg)[0]["rationale"][:90]))
    return out


def check_equal_nonzero_charges(snap, cfg, now=None):
    out = []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        for gen in (lane or {}).get("generations", []):
            if after_directive(gen, cfg, lane_cfg) and not charges_ok(gen):
                out.append("lane %s generation %d: charges %s are not equal and non-zero across arms" % (lane_cfg["id"], gen["n"], gen["charges"]))
    return out


def check_candidates_never_resubmitted(snap, cfg, now=None):
    out = []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        for gen in (lane or {}).get("generations", []):
            if not after_directive(gen, cfg, lane_cfg):
                continue
            prior = seen_candidates(lane, gen["n"])
            again = [r["candidate"][:8] for r in gen["rows"]
                     if r.get("candidate") and prior.get((r["seed"], r["niche"], r["arm"], r["candidate"])) is not None]
            if again:
                out.append("lane %s generation %d resubmitted identical candidates %s" % (lane_cfg["id"], gen["n"], again))
    return out


def check_arc_weights_move(snap, cfg, now=None):
    """The ARC lane may fail to gain, but its weights hash must change between iterations (an iteration with the same hash did nothing)."""
    win, out = cfg["policy"]["arc_stall_window"], []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        if lane_cfg["format"] != "arc" or not lane:
            continue
        hashes = [h["weights_hash"] for h in (lane.get("arc") or {}).get("history", [])]
        if len(hashes) >= win and len(set(hashes[-win:])) == 1:
            out.append("lane %s: last %d ARC iterations share weights hash %s: iterations ran without changing a single weight"
                       % (lane_cfg["id"], win, hashes[-1][:10]))
    return out


def check_arc_edits_can_gain(snap, cfg, now=None):
    """If the best expert per task already equals the lane's solved count, a router/assembly edit has nothing to gain: ARC headroom then
    requires NEW cell tensors (an ARC skill niche no pool cell covers). Skipped for lanes with no oracle ceiling (pure cell authoring)."""
    out = []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        arc = (lane or {}).get("arc") or {}
        oracle, solved = arc.get("oracle_solved"), (arc.get("claimed") or {}).get("solved")
        if lane_cfg["format"] == "arc" and oracle is not None and solved is not None and oracle <= solved:
            out.append("lane %s: oracle ceiling %s <= solved %s of %s over %s pool cells: no router edit can gain; author new cell tensors "
                       "for ARC skill niches the pool does not cover" % (lane_cfg["id"], oracle, solved, (arc.get("claimed") or {}).get("total"), arc.get("pool_cells")))
    return out


def check_no_plateau(snap, cfg, now=None):
    """Persistence is required, but not blind repetition: `plateau_patience` credited generations without a single admitted LLM child means the
    strategy (cell structure, what the maker is shown, edit vocabulary) must change. Write notes/CODEX_TO_HEAD_<date>.md naming the change;
    head-center then sets the lane's strategy_epoch_utc in HARD_LANES.json, which restarts the count."""
    out, patience = [], cfg["policy"]["plateau_patience"]
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        if not lane or not lane.get("exists") or lane_cfg["format"] != "generations":
            continue
        epoch = utc(lane_cfg.get("strategy_epoch_utc") or cfg["directive_utc"])
        credited = set(credited_generations(lane, lane_cfg, cfg))
        since = [g for g in lane["generations"] if g["n"] in credited and utc(g["utc"]) > epoch]
        if len(since) >= patience and not any(r["accepted"] for g in since for r in llm_rows(g)):
            out.append("lane %s plateaued: %d credited generations since %s with no admitted child; change the strategy, not the generation count"
                       % (lane_cfg["id"], len(since), epoch.isoformat()[:16]))
    return out


def driver_lanes(snap, cfg):
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        if lane_cfg["format"] == "driver" and lane and lane.get("exists"):
            yield lane_cfg, lane, (lane.get("driver") or {}).get("records", [])


def check_niche_breadth(snap, cfg, now=None):
    """Easy skills get ONE generation, so a skill lane is measured by how many DISTINCT niches it has earned, not by repeats."""
    out = []
    for lane_cfg, lane, records in driver_lanes(snap, cfg):
        earned = {r["stage"].split(":", 1)[1] for r in records
                  if r["stage"].startswith("niche:") and ((r.get("niche_cells") or 0) >= 1 or (r.get("best_pass_rate") or 0) >= 0.9)}
        earned |= {n for n, count in (lane["driver"].get("registry_niches") or {}).items() if count >= 1}
        got = earned & set(lane_cfg["niches"])
        if len(got) < lane_cfg["min_niches"]:
            out.append("lane %s earned %d/%d distinct niches %s (earned: %s)" % (lane_cfg["id"], len(got), lane_cfg["min_niches"], lane_cfg["niches"], sorted(got)))
    return out


def check_report_split_progress(snap, cfg, now=None):
    """The REPORT split (never used for selection) must beat its baseline by the lane's declared margin."""
    out = []
    for lane_cfg, lane, records in driver_lanes(snap, cfg):
        need = lane_cfg.get("min_report_gain")
        if need is None or not records:
            continue
        last = [r for r in records if "arc_report" in r][-1:] or [{}]
        gain = (last[0].get("arc_report", 0)) - (last[0].get("baseline_report", 0))
        if gain < need:
            out.append("lane %s: report-split gain %d < %d (report %s vs baseline %s on tasks never used for selection)"
                       % (lane_cfg["id"], gain, need, last[0].get("arc_report"), last[0].get("baseline_report")))
    return out


def check_breeding_must_help_arc(snap, cfg, now=None):
    """Children and mutations are admitted only on strict ARC dev+validation gain. Admissions without that gain are the neutral-generation pattern."""
    out = []
    for lane_cfg, lane, records in driver_lanes(snap, cfg):
        prev = None
        for r in records:
            sel = (r.get("arc_dev", 0) or 0) + (r.get("arc_val", 0) or 0)
            admitted = (r.get("children_admitted") or 0) + (r.get("mutations_admitted") or 0)
            if r["stage"].startswith(("breed", "mutate")) and admitted and prev is not None and sel <= prev:
                out.append("lane %s %s admitted %d cells with no gain on the ARC selection tasks (%d -> %d)" % (lane_cfg["id"], r["stage"], admitted, prev, sel))
            if "arc_dev" in r:
                prev = sel
    return out


def check_driver_alive_or_finished(snap, cfg, now=None):
    out, now = [], now or now_utc()
    for lane_cfg, lane, records in driver_lanes(snap, cfg):
        state = ((lane.get("status") or {}).get("status") or "").lower()
        stamp = (lane.get("status") or {}).get("utc") or lane.get("last_activity_utc")
        quiet = (now - utc(stamp)).total_seconds() / 60 if stamp else 1e9
        if quiet > cfg["policy"]["max_idle_minutes"] and lane.get("pid_alive") is not True and state not in ("rounds_complete", "plateau", "user_stop", "season_complete"):
            out.append("lane %s: driver quiet for %.0f min with state %r: restart it; it is neither running nor in a terminal state (plateau, rounds_complete, season_complete, user_stop)" % (lane_cfg["id"], quiet, state))
    return out


def check_hard_lanes_earn_credit(snap, cfg, now=None):
    need, out = cfg["policy"]["min_hard_generations"], []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        if lane_cfg.get("optional") or lane_cfg["format"] == "driver":
            continue
        got = len(credited_generations(lane, lane_cfg, cfg)) if lane and lane.get("exists") else 0
        if got < need:
            out.append("lane %s has %d/%d credited hard generations since the directive (headroom, non-neutral, equal charges, fresh candidate)"
                       % (lane_cfg["id"], got, need))
    return out


def check_hard_lanes_not_stalled(snap, cfg, now=None):
    """A hard lane that has not earned its credit may not stop, wait for a declaration, hit a deadline or sit dead: it keeps generating."""
    need, words, idle = cfg["policy"]["min_hard_generations"], cfg["policy"]["stall_status_words"], cfg["policy"]["max_idle_minutes"]
    now, out = now or now_utc(), []
    for lane_cfg, lane in lane_pairs(snap, cfg, "hard"):
        if lane_cfg.get("optional") or lane_cfg["format"] == "driver" or not lane or not lane.get("exists") or len(credited_generations(lane, lane_cfg, cfg)) >= need:
            continue
        status = ((lane.get("status") or {}).get("status") or "").lower()
        why = []
        if lane.get("stop"):
            why.append("STOP file present")
        if any(w in status for w in words):
            why.append("status=%s" % status)
        if (lane.get("status") or {}).get("waiting_for"):
            why.append("waiting_for=%s" % lane["status"]["waiting_for"])
        last = lane.get("last_activity_utc")
        if lane.get("pid_alive") is False and (not last or (now - utc(last)).total_seconds() / 60 > idle):
            why.append("no live process and no activity for >%d min" % idle)
        if why:
            out.append("lane %s is stalled below its credit (%s): difficulty is a reason to keep authoring weight edits from the failing "
                       "cases, not to stop or switch to infrastructure" % (lane_cfg["id"], ", ".join(why)))
    return out


ALL = [check_snapshot_fresh, check_snapshot_complete, check_lanes_exist, check_no_unregistered_lanes, check_closed_lanes_stay_closed, check_no_generation_without_headroom,
       check_hard_lane_dev_set_not_trivial, check_no_neutral_authoring, check_equal_nonzero_charges,
       check_candidates_never_resubmitted, check_arc_weights_move, check_arc_edits_can_gain, check_no_plateau, check_niche_breadth, check_report_split_progress,
       check_breeding_must_help_arc, check_driver_alive_or_finished, check_hard_lanes_earn_credit,
       check_hard_lanes_not_stalled]


def waste_report(snap, cfg):
    """Informational: pre-directive generations that ran with no headroom, per lane."""
    rows = []
    for lane_cfg, lane in lane_pairs(snap, cfg):
        if lane and lane.get("generations"):
            room = headroom(lane, lane_cfg, cfg)
            rows.append((lane_cfg["id"], len(lane["generations"]), sum(1 for v in room.values() if not v)))
    return rows
