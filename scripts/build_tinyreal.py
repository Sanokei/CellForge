"""Build tinyreal/1 splits from TinyStories (MIT). Runs on the evaluation server.

Normalization: lowercase; newlines/tabs -> space; keep [a-z ,.!?'"]; map curly
quotes to straight; collapse runs of spaces; drop stories containing other
letters after this mapping is applied to <2% of characters is NOT required:
stories with any other character outside the set are dropped entirely.
Keep stories of 150..600 normalized characters (whole stories only).
Split by sha256(story) parity buckets so the private final pool is disjoint.
"""
import hashlib, json, re, sys
src, pub, priv = sys.argv[1:4]
KEEP = set(" ,.!?'\"abcdefghijklmnopqrstuvwxyz")
raw = open(src, encoding="utf-8").read().split("<|endoftext|>")
stories, dropped = [], 0
for s in raw:
    s = s.strip().lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    if not (150 <= len(s) <= 600) or any(c not in KEEP for c in s):
        dropped += 1
        continue
    stories.append(s)
buckets = {"dev_train": [], "dev_eval": [], "anchor": [], "final_pool": []}
for s in stories:
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16) % 100
    b = "dev_train" if h < 40 else "dev_eval" if h < 43 else "anchor" if h < 48 else "final_pool"
    buckets[b].append(s)
for k in ("dev_train", "dev_eval", "anchor"):
    json.dump(buckets[k][: {"dev_train": 4000, "dev_eval": 300, "anchor": 500}[k]], open(f"{pub}/{k}.json", "w"))
json.dump(buckets["final_pool"], open(f"{priv}/final_pool.json", "w"))
print({k: len(v) for k, v in buckets.items()}, "kept", len(stories), "dropped", dropped)
