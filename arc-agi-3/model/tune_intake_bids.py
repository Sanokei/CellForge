"""Learn intake bid multipliers from BidLog JSONL (dev games ONLY).

Usefulness[name] = total recency-weighted level-up credit / episodes.
Multiplier = clip(usefulness / median_usefulness, 0.25, 4.0).
Cells with zero credit across >= MIN_EPISODES episodes are pruned
(multiplier 0.0): the evolved glue cuts cells that never help.

Transfer-set discipline: the log must only contain dev-game episodes.
Never run this on transfer-game logs.

Usage: python tune_intake_bids.py <intake_log.jsonl> [--out bids.json]
       [--min-episodes 3]
"""
import argparse
import json
from collections import defaultdict

import sys
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
from intake import ALL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="intake_bids.json")
    ap.add_argument("--min-episodes", type=int, default=3)
    a = ap.parse_args()
    credit = defaultdict(float)
    episodes = 0
    for line in open(a.log):
        d = json.loads(line)
        if d["event"] == "levelup":
            for n, c in d["credit"].items():
                credit[n] += c
        elif d["event"] == "episode":
            episodes += 1
    if episodes == 0:
        print("no episodes logged")
        return
    use = {n: c / episodes for n, c in credit.items()}
    med = sorted(use.values())[len(use) // 2] if use else 1.0
    med = med or 1.0
    names = [c.name for c in ALL]
    bids = {}
    for n in names:
        u = use.get(n, 0.0)
        if u == 0.0 and episodes >= a.min_episodes:
            bids[n] = 0.0  # pruned: never credited
        else:
            bids[n] = round(max(0.25, min(4.0, u / med)), 3)
    out = {"episodes": episodes, "usefulness": use, "bids": bids}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
