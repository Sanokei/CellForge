import json, sys
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
from tok import load_vocab, encode
vocab, stoi = load_vocab('/srv/arc3-claude/v6run/model/lm/vocab.json')
d = json.load(open('/srv/arc3-claude/v6run/model/lm/vocab.json'))
v = d['vocab']
alpha = [t for t in v if t.isalpha() and len(t) >= 2]
print("ALL alpha tokens (%d):" % len(alpha))
print(alpha)
print()
# how do candidate leaf-ish strings tokenize?
for s in ["act.probe", "act.Probing", "act.scanning", "act.Sweeping", "act.target",
          "act.follow", "act.try", "probe", "Probing", "scanning", "Sweeping"]:
    ids = encode(s, stoi)
    toks = [vocab[i] for i in ids]
    print(f"{s!r:20} -> {toks}")
