import json
d = json.load(open('/srv/cellforge-inbox/two-loop-20261001/shared-cell-registry/LATEST.json'))
print("families:", json.dumps(d.get('families'), indent=1)[:2000])
print("live_cells type:", type(d.get('live_cells')))
lc = d.get('live_cells')
if isinstance(lc, dict):
    print("live cell names:", list(lc.keys()))
elif isinstance(lc, list):
    print("live cells:", [c.get('name', c) if isinstance(c, dict) else c for c in lc][:20])
print("representatives:", str(d.get('representatives'))[:500])
