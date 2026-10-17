import json
from collections import Counter, defaultdict
rows = json.load(open('/srv/cellforge-arena/researchers/alpha/gen3/explore_log.json'))
print('total decisions:', len(rows))
by_game = defaultdict(Counter)
for r in rows:
    by_game[r['game']][r['cell']] += 1
for g in sorted(by_game):
    print(g, dict(by_game[g]))
