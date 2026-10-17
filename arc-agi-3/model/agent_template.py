"""Template researcher-authored agent. Copy to researchers/<name>/agent.py and
make it yours: probe the games, encode what you learn as routing logic and
WEIGHTS. Every number in WEIGHTS must actually steer decide() (purity-checked).
"""
NAME = "template-g0"

WEIGHTS = {
    "sweep_min_untested": 2,   # untested actions before a sweep is worth it
    "stuck_limit": 4,          # fruitless decisions before reset
    "ghost_max_area": 120,     # max px area to count as ghost marker
    "dial_repeats": 5,         # repeat clicks when dial-like
}

CELLS = ["perceive.describe", "perceive.objects", "perceive.diff",
         "act.sweep", "act.try", "act.click_at", "act.pick_drop",
         "act.repeat_click", "act.ghost_follow", "act.reset"]


def decide(state, history, rng):
    W = WEIGHTS
    avail = state["avail"]
    kbd = [a for a in avail if a not in (0, 6)]
    lvl = state["level"]

    # count how often we've seen this level
    seen_level = sum(1 for h in history if True)
    tested = set()
    for h in history:
        if h[1] == "act.try" and isinstance(h[2], dict):
            tested.add(h[2].get("action"))

    untested = [a for a in kbd if a not in tested]
    if state["steps_used"] == 0:
        return ("New level. Scanning first.",
                "perceive.describe", {})
    if len(untested) >= W["sweep_min_untested"]:
        return (f"{len(untested)} actions untested. Sweeping to learn effects.",
                "act.sweep", {})
    if 6 in avail and state["n_objects"]:
        o = state["objects"][0]
        return (f"Click available. Trying biggest object at ({o['cx']},{o['cy']}).",
                "act.click_at", {"x": o["cx"], "y": o["cy"]})
    # stuck detection: last decisions changed nothing
    recent = history[-W["stuck_limit"]:]
    if len(recent) >= W["stuck_limit"] and state["steps_used"] > 0:
        return ("No progress. Resetting the level.", "act.reset", {})
    if untested:
        return (f"Probing action {untested[0]}.", "act.try",
                {"action": untested[0]})
    a = rng.choice(kbd) if kbd else 1
    return (f"Exploring with action {a}.", "act.try", {"action": a})
