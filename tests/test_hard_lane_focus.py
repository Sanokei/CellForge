"""Hard lanes must keep generating real weight edits; easy lanes get one generation. Head-center controlled: do not weaken, skip, delete or work around.

Run: python scripts/lane_snapshot.py && python -m unittest tests.test_hard_lane_focus -v
Part 1 (MetaTests): synthetic lane snapshots prove every check passes an honest hard lane and catches each way of looking busy without
                    working (saturated re-runs, neutral rewrites, idling, no-op ARC iterations, trivial dev sets, unequal charges...).
Part 2 (RealRuns):  the same checks on the live server snapshot. They FAIL (never skip) while a required hard lane is missing, stalled,
                    or short of credited generations. Stopping a hard lane because it is difficult is a red test, not a plan.
"""
import copy
import datetime
import json
import os
import tempfile
import unittest

from scripts import lane_snapshot as L
from tests import hard_lane_checks as H

CFG = H.load(H.CFG_PATH)
CFG_ARC = copy.deepcopy(CFG)  # the real manifest closed the router-only ARC lane; the ARC checks are still proven on a manifest that keeps one
for _lane in CFG_ARC["lanes"]:
    if _lane["id"] == "arc-moe":
        _lane["role"] = "hard"
        _lane["baseline_iterations"] = 4
CFG_FREE = copy.deepcopy(CFG)  # the real manifest pins tool-use to 48 dev tasks; these synthetic tests need the generation's own size
for _lane in CFG_FREE["lanes"]:
    _lane.pop("dev_total", None)
D0 = H.utc(CFG["directive_utc"])
NOW = D0 + datetime.timedelta(hours=3)
HONEST = "Failing dev case dev-7 predicts 0 where 1 is expected; raise the weight on the third antecedent and lower the bias."
NEUTRAL = "Supplied development feedback reports 9/9 successes and no errors. Propose a harmless rescale."


def stamp(minutes):
    return (D0 + datetime.timedelta(minutes=minutes)).isoformat()


def gen(n, minute, solved=10, accepted=False, total=30, reasons=None, rationale=HONEST, charges=(12, 12), cand=None):
    rows = [{"seed": s, "niche": "n", "arm": arm, "candidate": cand or ("c%d-%d-%s" % (n, s, arm)),
             "solved": solved if arm == "llm" else 3, "accepted": accepted and arm == "llm",
             "reasons": [] if accepted and arm == "llm" else (reasons or ["strict_development_gain"])} for s in (1, 2, 3) for arm in ("llm", "random")]
    return {"n": n, "utc": stamp(minute), "rows": rows, "total": total, "charges": {"llm": charges[0], "random": charges[1]},
            "responses": [{"seed": s, "niche": "n", "rationale": rationale, "charged_positions": 4, "prompt_sha256": "p"} for s in (1, 2, 3)]}


def lane(lid, gens=(), status="authoring", stop=None, alive=True, last=None, arc=None, waiting=None):
    return {"id": lid, "exists": True, "status": {"status": status, "waiting_for": waiting}, "stop": stop, "pid_alive": alive,
            "last_activity_utc": last or (gens[-1]["utc"] if gens else stamp(10)), "generations": list(gens), "arc": arc}


def snapshot(**lanes):
    return {"schema": "lane-snapshot/1", "captured_utc": NOW.isoformat(), "lanes": lanes}


def climbing(lid, count=10, start=10):
    """Honest hard lane: partial solves that improve slowly, never saturating."""
    return lane(lid, [gen(i + 1, 10 + 5 * i, solved=start + i, accepted=(i % 3 == 0)) for i in range(count)])


def driver_records(niches=("association", "sequence", "glyph", "tiling", "neighbor", "object", "crop", "ray"), report=12, baseline_report=0, breed_gain=True):
    recs, dev, val = [], 0, 1
    for n in niches:
        dev += 1
        val += 1
        recs.append({"stage": "niche:" + n, "niche_cells": 1, "best_pass_rate": 1.0, "arc_dev": dev, "arc_val": val, "arc_report": report // 2,
                     "baseline_report": baseline_report})
    gained = (dev + 1) if breed_gain else dev
    recs.append({"stage": "breed:round1", "children_valid": 20, "children_admitted": 1, "children_rejected_no_arc_gain": 19, "arc_dev": gained, "arc_val": val,
                 "arc_report": report, "baseline_report": baseline_report})
    return recs


def driver_lane(lid, records, status="running", minutes_ago=2, alive=None):
    stamp_ = (NOW - datetime.timedelta(minutes=minutes_ago)).isoformat()
    return {"id": lid, "exists": True, "status": {"status": status, "utc": stamp_}, "stop": None, "pid_alive": alive, "last_activity_utc": stamp_,
            "generations": [], "driver": {"records": records}}


def arc_history(hashes, oracle=9, solved=0):
    return {"history": [{"iteration": i, "solved": 0, "weights_hash": h} for i, h in enumerate(hashes)],
            "claimed": {"solved": solved, "total": 100}, "controls": {}, "oracle_solved": oracle, "pool_cells": 6}


def honest():
    hashes = ["h%d" % i for i in range(4)] + ["x%d" % i for i in range(10)]
    return snapshot(**{"arc-moe": lane("arc-moe", status="running", arc=arc_history(hashes)),
                       "reasoning-multistep": climbing("reasoning-multistep"), "tool-use": climbing("tool-use"),
                       "language-skills": driver_lane("language-skills", driver_records()), "ocr-skills": driver_lane("ocr-skills", driver_records()),
                       "arc-skills": driver_lane("arc-skills", driver_records()),
                       "arithmetic-subtraction": lane("arithmetic-subtraction", [gen(1, -30, 9, True, 9)], "user_stop", "STOP", False),
                       "reasoning-horn": lane("reasoning-horn", [], "declared_epoch_budget_complete", None, False),
                       "spatial-quorum": lane("spatial-quorum", [gen(1, -30, 40, True, 40)], "user_stop", "STOP", False),
                       "signed-deduction": lane("signed-deduction", [gen(1, -30, 8, True, 8)], "user_stop", "STOP", False)})


def problems(snap, check):
    return check(snap, CFG, NOW)


class MetaTests(unittest.TestCase):
    def test_honest_run_passes_every_check(self):
        snap = honest()
        for check in H.ALL:
            self.assertEqual(problems(snap, check), [], check.__name__)

    def test_manifest_requires_arc_reasoning_and_tool_use_as_hard_lanes(self):
        hard = {l["id"] for l in CFG["lanes"] if l["role"] == "hard"}
        self.assertTrue({"reasoning-multistep", "tool-use", "language-skills", "ocr-skills", "arc-skills"} <= hard)
        self.assertEqual(len({l["id"] for l in CFG["lanes"]}), len(CFG["lanes"]))
        self.assertGreaterEqual(CFG["policy"]["min_hard_generations"], 8)
        self.assertGreaterEqual(CFG["policy"]["min_dev_tasks_for_hard_lane"], 24)

    def test_saturated_lane_rerun_is_caught_and_not_credited(self):
        snap = honest()
        gens = [gen(1, 5, 30, True, 30)] + [gen(i, 5 + 5 * i, 30, False, 30) for i in range(2, 12)]
        snap["lanes"]["tool-use"] = lane("tool-use", gens)
        self.assertTrue(H.check_no_generation_without_headroom(snap, CFG_FREE, NOW))
        self.assertEqual(len(H.credited_generations(snap["lanes"]["tool-use"], {"id": "tool-use", "format": "generations"}, CFG_FREE)), 1)
        self.assertTrue(H.check_hard_lanes_earn_credit(snap, CFG_FREE, NOW))

    def test_parent_already_saturated_is_caught_even_in_first_generation(self):
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, 5 * i, 24, False, 24) for i in range(1, 10)])
        self.assertTrue(H.check_no_generation_without_headroom(snap, CFG_FREE, NOW))

    def test_neutral_rewrite_is_caught_and_not_credited(self):
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, 5 * i, rationale=NEUTRAL) for i in range(1, 12)])
        self.assertTrue(problems(snap, H.check_no_neutral_authoring))
        self.assertEqual(H.credited_generations(snap["lanes"]["tool-use"], {"id": "tool-use", "format": "generations"}, CFG), [])
        self.assertTrue(problems(snap, H.check_hard_lanes_earn_credit))

    def test_closed_lane_must_stay_closed(self):
        snap = honest()
        snap["lanes"]["arithmetic-subtraction"]["generations"].append(gen(2, 40, 9, False, 9))
        self.assertIn("arithmetic-subtraction", " ".join(problems(snap, H.check_closed_lanes_stay_closed)))

    def test_skill_lane_with_too_few_distinct_niches_is_caught(self):
        snap = honest()
        snap["lanes"]["arc-skills"] = driver_lane("arc-skills", driver_records(niches=("tiling", "neighbor")))
        self.assertIn("arc-skills", " ".join(problems(snap, H.check_niche_breadth)))

    def test_breadth_counts_niches_the_registry_holds_even_in_a_seeded_run(self):
        snap = honest()
        lane_ = driver_lane("arc-skills", driver_records(niches=("tiling",)))
        lane_["driver"]["registry_niches"] = {"tiling": 60, "neighbor": 30, "object": 32, "crop": 16, "ray": 48}
        snap["lanes"]["arc-skills"] = lane_
        self.assertEqual(problems(snap, H.check_niche_breadth), [])

    def test_repeating_one_niche_does_not_count_as_breadth(self):
        snap = honest()
        recs = driver_records(niches=("tiling",)) + [{"stage": "niche:tiling", "niche_cells": 1, "arc_dev": 1, "arc_val": 3, "arc_report": 2, "baseline_report": 0}] * 5
        snap["lanes"]["arc-skills"] = driver_lane("arc-skills", recs)
        self.assertTrue(problems(snap, H.check_niche_breadth))

    def test_report_split_must_beat_its_baseline(self):
        snap = honest()
        snap["lanes"]["arc-skills"] = driver_lane("arc-skills", driver_records(report=3, baseline_report=0))
        self.assertIn("report-split gain", " ".join(problems(snap, H.check_report_split_progress)))

    def test_breeding_without_arc_gain_is_the_bloat_pattern_and_is_caught(self):
        snap = honest()
        snap["lanes"]["arc-skills"] = driver_lane("arc-skills", driver_records(breed_gain=False))
        self.assertIn("no gain on the ARC selection tasks", " ".join(problems(snap, H.check_breeding_must_help_arc)))

    def test_quiet_dead_driver_is_caught_but_a_live_or_recent_one_is_not(self):
        snap = honest()
        snap["lanes"]["arc-skills"] = driver_lane("arc-skills", driver_records(), status="breeding", minutes_ago=90, alive=False)
        self.assertTrue(problems(snap, H.check_driver_alive_or_finished))
        snap["lanes"]["arc-skills"] = driver_lane("arc-skills", driver_records(), status="breeding", minutes_ago=3, alive=None)
        self.assertEqual(problems(snap, H.check_driver_alive_or_finished), [])
        for terminal in ("plateau", "rounds_complete", "user_stop", "season_complete"):  # a finished Cell Wars season is a clean stop, not a dead driver
            snap["lanes"]["arc-skills"] = driver_lane("arc-skills", driver_records(), status=terminal, minutes_ago=90, alive=False)
            self.assertEqual(problems(snap, H.check_driver_alive_or_finished), [], terminal)

    def test_a_generation_with_no_valid_llm_candidate_earns_no_credit(self):
        snap = honest()
        gens = [gen(i, 5 * i) for i in range(1, 12)]
        for r in gens[3]["rows"]:
            if r["arm"] == "llm":
                r["candidate"] = None  # cancelled / all-invalid maker generation
        snap["lanes"]["reasoning-multistep"] = lane("reasoning-multistep", gens)
        credited = H.credited_generations(snap["lanes"]["reasoning-multistep"], {"id": "reasoning-multistep", "format": "generations"}, CFG_FREE)
        self.assertNotIn(4, credited)
        self.assertEqual(len(credited), 10)

    def test_optional_small_lane_is_policed_for_honesty_but_not_required(self):
        cfg = copy.deepcopy(CFG)
        for lane_cfg in cfg["lanes"]:
            lane_cfg.pop("forgiven_through_utc", None)
        snap = honest()
        snap["lanes"]["resource-planning"] = lane("resource-planning", [gen(1, 5, 7, True, 8), gen(2, 10, 7, False, 8)])
        for check in H.ALL:
            self.assertEqual(check(snap, cfg, NOW), [], check.__name__)
        snap["lanes"]["resource-planning"] = lane("resource-planning", [gen(1, 5, 8, True, 8), gen(2, 10, 8, False, 8, rationale=NEUTRAL)])
        self.assertTrue(H.check_no_generation_without_headroom(snap, cfg, NOW))
        self.assertTrue(H.check_no_neutral_authoring(snap, cfg, NOW))
        del snap["lanes"]["resource-planning"]
        self.assertEqual(H.check_lanes_exist(snap, cfg, NOW), [])

    def test_forgiven_through_excuses_history_but_not_what_happens_later(self):
        cfg = copy.deepcopy(CFG)
        for lane_cfg in cfg["lanes"]:
            if lane_cfg["id"] == "tool-use":
                lane_cfg["forgiven_through_utc"] = stamp(60)
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(1, 20, rationale=NEUTRAL), gen(2, 40, rationale=NEUTRAL)])
        self.assertEqual(H.check_no_neutral_authoring(snap, cfg, NOW), [])
        snap["lanes"]["tool-use"]["generations"].append(gen(3, 90, rationale=NEUTRAL))
        self.assertIn("generation 3", " ".join(H.check_no_neutral_authoring(snap, cfg, NOW)))

    def test_unreadable_lane_fails_instead_of_hiding(self):
        snap = honest()
        snap["lanes"]["tool-use"] = {"id": "tool-use", "exists": True, "generations": [], "reduce_error": "KeyError: 'arm'"}
        self.assertIn("tool-use", " ".join(problems(snap, H.check_snapshot_complete)))

    def test_plateau_needs_a_declared_strategy_change(self):
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, 5 * i, solved=10, accepted=False) for i in range(1, 14)])
        self.assertIn("plateaued", " ".join(problems(snap, H.check_no_plateau)))
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, 5 * i, solved=10, accepted=(i == 7)) for i in range(1, 14)])
        self.assertEqual(problems(snap, H.check_no_plateau), [])
        cfg = copy.deepcopy(CFG)
        for lane_cfg in cfg["lanes"]:
            if lane_cfg["id"] == "tool-use":
                lane_cfg["strategy_epoch_utc"] = stamp(5 * 12)
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, 5 * i, solved=10, accepted=False) for i in range(1, 14)])
        self.assertEqual(H.check_no_plateau(snap, cfg, NOW), [])

    def test_unregistered_candidate_field_names_are_read(self):
        with tempfile.TemporaryDirectory() as base:
            run = os.path.join(base, "runs", "weights-y")
            os.makedirs(run)
            decisions = [{"seed": 1, "arm": "llm", "id": "abc", "solved": 3, "accepted": False, "reasons": []}]
            json.dump({"decisions": decisions}, open(os.path.join(run, "GENERATION-1.json"), "w"))
            out = L.reduce_lane(base, {"id": "y", "run": "weights-y", "format": "generations"})
            self.assertEqual(out["generations"][0]["rows"][0]["candidate"], "abc")

    def test_opening_an_unregistered_lane_is_caught(self):
        snap = honest()
        snap["run_dirs"] = [l["run"] for l in CFG["lanes"]]
        self.assertEqual(problems(snap, H.check_no_unregistered_lanes), [])
        snap["run_dirs"].append("weights-brand-new-small-lane")
        self.assertIn("weights-brand-new-small-lane", " ".join(problems(snap, H.check_no_unregistered_lanes)))

    def test_closed_lane_grace_tolerates_only_the_declared_in_flight_generations(self):
        cfg = copy.deepcopy(CFG)
        for lane_cfg in cfg["lanes"]:
            if lane_cfg["id"] == "spatial-quorum":
                lane_cfg["grace_generations"] = 1
        snap = honest()
        snap["lanes"]["spatial-quorum"]["generations"] += [gen(2, 2, 40, False, 40)]
        self.assertEqual(H.check_closed_lanes_stay_closed(snap, cfg, NOW), [])
        snap["lanes"]["spatial-quorum"]["generations"] += [gen(3, 6, 40, False, 40)]
        self.assertIn("spatial-quorum", " ".join(H.check_closed_lanes_stay_closed(snap, cfg, NOW)))

    def test_neutral_regex_separates_real_neutral_rewrites_from_real_honest_rationales(self):
        neutral = ["Supplied development feedback reports 9/9 successes and no errors. Propose subtracting b and incoming borrow.",
                   "Fresh bias revision preserves the parent's Boolean decisions. Supplied parent evidence already solves 40/40; no strict development gain is possible or claimed.",
                   "Positive scaling by 1.5 preserves the parent's step decisions. Supplied DEV evidence is already 24/24; no gain is claimed.",
                   "The supplied development feedback reports 8/8 correct with no errors."]
        honest_ = ["Development feedback shows completion in 12 steps versus an 11-step shortest plan for dev-three-deliveries. Author a positive acquire free-cargo coefficient. No evaluation or improvement is claimed.",
                   "Development traces show repeated acquisition causing capacity and energy failures. Unevaluated; no improvement claimed.",
                   "Failing dev case dev-7 predicts 0 where 1 is expected; raise the weight on the third antecedent.",
                   "Solved cases stay solved; dev-12 fails with no errors raised but a wrong colour at the border, so widen the stencil."]
        import re
        pat = re.compile(CFG["policy"]["neutral_rationale_regex"], re.I)
        self.assertEqual([bool(pat.search(t)) for t in neutral], [True] * len(neutral))
        self.assertEqual([bool(pat.search(t)) for t in honest_], [False] * len(honest_))

    def test_generations_before_directive_are_not_policed_but_earn_nothing(self):
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, -100 + i, rationale=NEUTRAL) for i in range(1, 12)])
        self.assertEqual(problems(snap, H.check_no_neutral_authoring), [])
        self.assertTrue(problems(snap, H.check_hard_lanes_earn_credit))

    def test_stall_by_status_stop_file_waiting_or_dead_process_is_caught(self):
        for kw in ({"status": "declared_epoch_budget_complete"}, {"stop": "pause"}, {"waiting": "new_capability_epoch_declaration"},
                   {"status": "saturation_pause"}, {"alive": False, "last": stamp(-600)}):
            snap = honest()
            gens = [gen(i, 5 * i) for i in range(1, 4)]
            kw = dict(kw)
            kw.setdefault("last", gens[-1]["utc"])
            snap["lanes"]["tool-use"] = lane("tool-use", gens, **kw)
            self.assertTrue(problems(snap, H.check_hard_lanes_not_stalled), kw)

    def test_a_lane_that_earned_its_credit_may_finish(self):
        snap = honest()
        snap["lanes"]["tool-use"]["status"]["status"] = "declared_epoch_budget_complete"
        self.assertEqual(problems(snap, H.check_hard_lanes_not_stalled), [])

    def test_slow_progress_is_fine_failing_forever_is_not_stopping(self):
        """Ten generations with no gain at all still earn credit: the owner wants persistence, not a promise of success."""
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, 5 * i, solved=10, accepted=False) for i in range(1, 12)])
        self.assertEqual(problems(snap, H.check_hard_lanes_earn_credit), [])

    def test_arc_iterations_with_unchanged_weights_are_caught_and_not_credited(self):
        snap = honest()
        snap["lanes"]["arc-moe"] = lane("arc-moe", status="running", arc=arc_history(["h"] * 20))
        self.assertTrue(H.check_arc_weights_move(snap, CFG_ARC, NOW))
        self.assertTrue(H.check_hard_lanes_earn_credit(snap, CFG_ARC, NOW))

    def test_arc_router_only_iterations_with_zero_ceiling_are_caught(self):
        """The real failure: the pool's best expert solves 0/100, so a router rewrite can never gain."""
        snap = honest()
        snap["lanes"]["arc-moe"] = lane("arc-moe", status="running", arc=arc_history(["h%d" % i for i in range(14)], oracle=0, solved=0))
        self.assertIn("no router edit can gain", " ".join(H.check_arc_edits_can_gain(snap, CFG_ARC, NOW)))
        snap["lanes"]["arc-moe"] = lane("arc-moe", status="running", arc=arc_history(["h%d" % i for i in range(14)], oracle=5, solved=5))
        self.assertTrue(H.check_arc_edits_can_gain(snap, CFG_ARC, NOW))
        snap["lanes"]["arc-moe"] = lane("arc-moe", status="running", arc=arc_history(["h%d" % i for i in range(14)], oracle=None, solved=0))
        self.assertEqual(H.check_arc_edits_can_gain(snap, CFG_ARC, NOW), [])

    def test_arc_credit_counts_only_changed_hashes_after_baseline(self):
        arc_cfg = {"id": "arc-moe", "format": "arc", "baseline_iterations": 2}
        lane_arc = lane("arc-moe", arc=arc_history(["a", "b", "b", "c", "c", "d"]))
        self.assertEqual(H.credited_generations(lane_arc, arc_cfg, CFG), [3, 5])

    def test_resubmitted_candidate_is_caught(self):
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(1, 5, cand="same"), gen(2, 10, cand="same")])
        self.assertTrue(problems(snap, H.check_candidates_never_resubmitted))

    def test_unequal_or_zero_charges_are_caught(self):
        for charges in ((12, 6), (0, 0)):
            snap = honest()
            snap["lanes"]["tool-use"] = lane("tool-use", [gen(1, 5, charges=charges)])
            self.assertTrue(problems(snap, H.check_equal_nonzero_charges), charges)

    def test_trivial_dev_set_is_caught(self):
        snap = honest()
        snap["lanes"]["tool-use"] = lane("tool-use", [gen(i, 5 * i, 3, False, 8) for i in range(1, 12)])
        self.assertTrue(H.check_hard_lane_dev_set_not_trivial(snap, CFG_FREE, NOW))

    def test_missing_required_lane_and_stale_snapshot_are_caught(self):
        snap = honest()
        del snap["lanes"]["tool-use"]
        self.assertTrue(problems(snap, H.check_lanes_exist))
        old = honest()
        old["captured_utc"] = (NOW - datetime.timedelta(hours=9)).isoformat()
        self.assertTrue(problems(old, H.check_snapshot_fresh))


class ReducerContract(unittest.TestCase):
    """scripts/lane_snapshot.py must read the real lane file formats (fixture mirrors server files)."""

    def test_generation_lane_is_reduced_from_server_formats(self):
        with tempfile.TemporaryDirectory() as base:
            run = os.path.join(base, "runs", "weights-x")
            os.makedirs(os.path.join(run, "AUTHOR-1"))
            decisions = [{"seed": 1, "niche": "n", "arm": a, "candidate": "c" + a, "solved": s, "accepted": a == "llm", "reasons": []}
                         for a, s in (("llm", 7), ("random", 2))]
            json.dump({"state": {}, "decisions": decisions}, open(os.path.join(run, "GENERATION-1.json"), "w"))
            open(os.path.join(run, "CHARGES-1.jsonl"), "w").write("\n".join(json.dumps({"arm": a, "positions": 5}) for a in ("llm", "random")))
            json.dump([{"seed": 1, "niche": "n", "rationale": "fix dev-3", "charged_positions": 5}], open(os.path.join(run, "AUTHOR-1", "response.json"), "w"))
            json.dump({"status": "authoring", "pid": 1, "generation": 1}, open(os.path.join(run, "STATUS.json"), "w"))
            open(os.path.join(run, "STOP"), "w").write("pause")
            out = L.reduce_lane(base, {"id": "x", "run": "weights-x", "format": "generations", "dev_total": 30})
            self.assertTrue(out["exists"])
            self.assertEqual(out["stop"], "pause")
            g = out["generations"][0]
            self.assertEqual((g["n"], g["total"], g["charges"]), (1, 30, {"llm": 5, "random": 5}))
            self.assertEqual([r["solved"] for r in g["rows"]], [7, 2])
            self.assertEqual(g["responses"][0]["rationale"], "fix dev-3")

    def test_odd_charge_rows_and_opaque_lanes_do_not_break_the_snapshot(self):
        with tempfile.TemporaryDirectory() as base:
            run = os.path.join(base, "runs", "weights-odd")
            os.makedirs(run)
            json.dump({"decisions": []}, open(os.path.join(run, "GENERATION-1.json"), "w"))
            open(os.path.join(run, "CHARGES-1.jsonl"), "w").write(json.dumps({"context": 401, "positions": 3}) + chr(10))
            snap = L.reduce_all(base, [{"id": "odd", "run": "weights-odd", "format": "generations"},
                                        {"id": "opq", "run": "weights-odd", "format": "opaque"}])
            self.assertEqual(snap["lanes"]["odd"]["generations"][0]["charges"], {"llm": 0, "random": 0})
            self.assertEqual(snap["lanes"]["opq"]["generations"], [])
            self.assertEqual(snap["run_dirs"], ["weights-odd"])

    def test_driver_history_is_reduced_with_a_per_lane_base(self):
        with tempfile.TemporaryDirectory() as base:
            run = os.path.join(base, "runs", "weights-skills")
            os.makedirs(run)
            rows = [{"stage": "niche:ray", "niche_cells": 48, "arc_dev": 8, "arc_val": 13, "arc_report": 10, "other": 1}, {"stage": "niche:ray:gen1", "submitted": 48},
                    {"stage": "breed:round1", "children_admitted": 1, "arc_dev": 8, "arc_val": 14, "arc_report": 10}]
            open(os.path.join(run, "HISTORY.jsonl"), "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
            json.dump({"state": "running", "utc": "2026-10-01T00:00:00+00:00"}, open(os.path.join(run, "STATUS.json"), "w"))
            out = L.reduce_lane("/nonexistent", {"id": "x", "run": "weights-skills", "base": base, "format": "driver"})
            self.assertEqual([r["stage"] for r in out["driver"]["records"]], ["niche:ray", "breed:round1"])
            self.assertNotIn("other", out["driver"]["records"][0])

    def test_arc_lane_is_reduced_and_missing_lane_reported(self):
        with tempfile.TemporaryDirectory() as base:
            run = os.path.join(base, "runs", "weights-a")
            os.makedirs(run)
            json.dump([{"iteration": 0, "solved": 0, "weights_hash": "h"}], open(os.path.join(run, "ARC_HISTORY.json"), "w"))
            json.dump({"claimed": {"solved": 0, "total": 100}, "controls": {}}, open(os.path.join(run, "ARC_SCORE.json"), "w"))
            json.dump({"status": "x"}, open(os.path.join(run, "SERVICE_STATUS.json"), "w"))
            json.dump({"pool": {"cells": {"a": {}, "b": {}}}, "contexts": [{"arms": {"llm": {"oracle_solved_diagnostic_only": 0}},
                                                                           "other": [{"oracle_solved_diagnostic_only": 4}]}]},
                      open(os.path.join(run, "STATE-2.json"), "w"))
            out = L.reduce_lane(base, {"id": "a", "run": "weights-a", "format": "arc"})
            self.assertEqual((out["arc"]["oracle_solved"], out["arc"]["pool_cells"]), (4, 2))
            self.assertEqual(out["arc"]["history"][0]["weights_hash"], "h")
            self.assertFalse(L.reduce_lane(base, {"id": "b", "run": "weights-none", "format": "arc"})["exists"])


class RealRuns(unittest.TestCase):
    """Live snapshot of the Hetzner lanes. FAIL, never skip."""

    @classmethod
    def setUpClass(cls):
        cls.snap = H.load(H.SNAP_PATH) if os.path.exists(H.SNAP_PATH) else None

    def run_check(self, check):
        self.assertIsNotNone(self.snap, "no reports/lane-snapshot/latest.json: run python scripts/lane_snapshot.py")
        self.assertEqual(check(self.snap, CFG), [], check.__name__)


def _add(check):
    setattr(RealRuns, "test_real_" + check.__name__[len("check_"):], lambda self, c=check: self.run_check(c))


for _check in H.ALL:
    _add(_check)

if __name__ == "__main__":
    unittest.main()
