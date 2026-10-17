import sys, json, logging
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
import eval_lm
from eval_lm import parse_call
# monkeypatch to capture generations
import eval_lm as E
orig_parse = E.parse_call
fails = []
def patched(text):
    name, args = orig_parse(text)
    if name is None:
        fails.append(text)
    return name, args
E.parse_call = patched
sys.argv = ['eval_lm.py', 'router_v3.pt', 'lf52', 's5i5', '--seed', '0', '--maxdec', '40']
eval_lm.main()
with open('/tmp/v3_fails.jsonl','w') as f:
    for t in fails: f.write(json.dumps(t)+'\n')
print('FAILS', len(fails))
