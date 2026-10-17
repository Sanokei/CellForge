import json
d = json.load(open('/srv/arc3-claude/v6run/model/lm/vocab.json'))
v = d['vocab']
print("total tokens:", len(v))
# print all tokens containing only letters (possible leaf names), limit 2209
alpha = [t for t in v if t.isalpha() and len(t) >= 2]
print("alpha tokens:", len(alpha))
print(alpha[:120])
print("...")
# tokens with underscore (like ghost_follow)
us = [t for t in v if '_' in t]
print("underscore tokens:", us)
