"""Serial continuation after the current reasoning author finishes."""
import json
from pathlib import Path
import subprocess
import sys
import time

base = Path(__file__).resolve().parent
runs = base.parent / 'runs'
reasoning = runs / 'weights-hard-multistep-reasoning-v2'
while True:
    status = json.loads((reasoning / 'STATUS.json').read_text())
    if status.get('status') == 'rounds_finished_not_goal_complete' and status.get('generations', 0) >= 17:
        break
    latest = sorted(reasoning.glob('GENERATION-*.json'), key=lambda p: int(p.stem.split('-')[-1]))
    if latest:
        record = json.loads(latest[-1].read_text())
        if record['generation'] >= 10 and all(sum(d['solved'] >= .95*d['total'] for d in record['decisions'] if d['arm']=='llm' and d['seed']==s) for s in (1701,1702,1703)):
            if status.get('status') == 'rounds_finished_not_goal_complete':
                break
    time.sleep(15)
for module, name, rounds in [('anchor_assign_skill','weights-anchor-assign4',16),
                             ('corridor_skill','weights-corridor-marker2',8)]:
    subprocess.run([sys.executable, str(base/'run_bfcl_skill.py'),str(runs/name),
                    '--module',module,'--source',str(base/(module+'.py')),
                    '--rounds',str(rounds)],check=True)
