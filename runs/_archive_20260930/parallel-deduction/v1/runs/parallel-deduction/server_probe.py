"""Frozen synthetic development evaluator with a separate reference closure."""
import hashlib
import json
import random
import secrets
import time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.deduction import HornDeductionPredictor

root = Path(__file__).resolve().parent
snapshot_root = root.parents[1]
model = Graph.load(root/'artifact-v1')
# Model is frozen before sampling variation; these are development cases only.
seed = secrets.randbits(128)
rng = random.Random(seed)
cases = []
for i in range(80):
    n = rng.randrange(4,29)
    symbols = ['s'+str(rng.getrandbits(50)) for _ in range(n)]
    rules = [{'if':[symbols[j]],'then':symbols[j+1]} for j in range(n-1)]
    if i%3 == 0:
        rules.append({'if':[symbols[-1]],'then':symbols[0]})
    facts = [symbols[0]] if i%2 else []
    if i%5 == 0:
        rules[0] = {'if':[],'then':symbols[1]}
    rng.shuffle(rules)
    cases.append({'facts':facts,'rules':rules,'query':symbols[-1]})
for i in range(40):
    symbols = ['v'+str(rng.getrandbits(50)) for _ in range(16)]
    facts = rng.sample(symbols,rng.randrange(0,6))
    rules = [{'if':rng.sample(symbols,rng.randrange(0,5)),'then':rng.choice(symbols)} for _ in range(24)]
    cases.append({'facts':facts,'rules':rules,'query':rng.choice(symbols)})
frozen = json.dumps({'seed':seed,'cases':cases},sort_keys=True).encode()
(root/'frozen_development_cases.json').write_bytes(frozen)

def reference(case):
    known = set(case['facts'])
    while True:
        nxt = known | {r['then'] for r in case['rules'] if set(r['if'])<=known}
        if nxt == known:
            return case['query'] in known
        known = nxt

predict = HornDeductionPredictor(model)
started = time.time()
passed = ablation_passed = permutation_passed = positive = 0
failures = []
for index,case in enumerate(cases):
    expected = reference(case)
    positive += expected
    result = predict(**case)['entailed']
    ablation = predict(case['facts'],[],case['query'])['entailed']
    permuted = predict(list(reversed(case['facts'])),list(reversed(case['rules'])),case['query'])['entailed']
    passed += result == expected
    ablation_passed += ablation == expected
    permutation_passed += permuted == expected
    if result != expected:
        failures.append({'index':index,'expected':expected,'actual':result})
sources = {str(p.relative_to(snapshot_root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((snapshot_root/'src').rglob('*.py'))}
report = {'protocol':'positive-horn-transfer-dev/1','split':'development-only',
 'model_hash':model.hash(),'cases_sha256':hashlib.sha256(frozen).hexdigest(),
 'seed':seed,'n':len(cases),'positive_n':positive,'negative_n':len(cases)-positive,
 'correct':passed,'no_rules_ablation_correct':ablation_passed,'order_permutation_correct':permutation_passed,
 'failures':failures,'elapsed_seconds':time.time()-started,'source_sha256':sources,
 'limitations':'bounded authored positive Horn deduction; no learned understanding or frontier comparison; evaluator authored by same team'}
(root/'REPORT.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
