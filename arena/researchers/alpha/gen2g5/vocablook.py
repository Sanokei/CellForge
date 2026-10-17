import json, sys
vocab, stoi = {}, {}
d = json.load(open('/srv/arc3-claude/v6run/model/lm/vocab.json'))
print("vocab.json type:", type(d))
if isinstance(d, dict):
    print("keys:", list(d.keys())[:10])
    # try common layouts
    for k in list(d.keys())[:5]:
        print(k, "->", str(d[k])[:80])
elif isinstance(d, list):
    print("len:", len(d), "first 30:", d[:30])
