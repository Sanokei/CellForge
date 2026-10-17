"""Learn bid multipliers from BidLogger JSONL: which intakes precede level-ups.

Usefulness[name] = total recency-weighted credit / episodes.
Multiplier = clip(usefulness / median_usefulness, 0.25, 4.0).
Writes bids.json {name: multiplier}. This is the evolved glue: selection
pressure (level-ups, i.e. RHAE) tunes how loudly each cell bids.

Usage: python tune_bids.py <intake_log.jsonl> [--out bids.json]
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="bids.json")
    a = ap.parse_args()
    credit = defaultdict(float)
    episodes = 0
    shown_count = defaultdict(int)
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
    bids = {n: round(max(0.25, min(4.0, u / med)), 3) for n, u in use.items()}
    with open(a.out, "w") as f:
        json.dump({"episodes": episodes, "usefulness": use, "bids": bids}, f, indent=1)
    print(json.dumps({"episodes": episodes, "usefulness": use, "bids": bids}, indent=1))


if __name__ == "__main__":
    main()
