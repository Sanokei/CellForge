"""Frozen development-only transfer probe. Reference answers never enter predictor."""
import hashlib
import json
import random
import secrets
import sys
import time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.arithmetic import DecimalAdditionPredictor

root = Path(__file__).resolve().parent
sys.set_int_max_str_digits(12000)
seed = secrets.randbits(128)
rng = random.Random(seed)
cases = []
for length in (1,2,7,31,129,513,1024):
    for _ in range(8):
        a = ''.join(str(rng.randrange(10)) for _ in range(length))
        b = ''.join(str(rng.randrange(10)) for _ in range(rng.randrange(1,length+1)))
        cases.append((a,b))
cases.extend([('9'*n,'1') for n in (1,7,129,1024,4096)])
cases.extend([('000','0'),('0','0')])
frozen = json.dumps({'seed':seed,'cases':cases},sort_keys=True).encode()
(root/'frozen_development_cases.json').write_bytes(frozen)
predict = DecimalAdditionPredictor(Graph.load(root/'artifact-v1'))
started = time.time()
passed = control_passed = 0
failures = []
for index,(a,b) in enumerate(cases):
    expected = str(int(a)+int(b))
    actual = predict(a,b)['sum']
    passed += actual == expected
    control_passed += (a.lstrip('0') or '0') == expected
    if actual != expected:
        failures.append({'index':index,'operand_lengths':[len(a),len(b)],'actual_sha256':hashlib.sha256(actual.encode()).hexdigest()})
invalid_rejected = 0
for a,b in [('-1','1'),('','1'),('1.5','1'),('１２','3'),('9'*4097,'1')]:
    try:
        predict(a,b)
    except ValueError:
        invalid_rejected += 1
sources = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((root/'src').rglob('*.py'))}
report = {'protocol':'decimal-addition-transfer-dev/1','split':'development-only',
 'model_hash':predict.graph.hash(),'cases_sha256':hashlib.sha256(frozen).hexdigest(),
 'seed':seed,'n':len(cases),'correct':passed,'identity_control_correct':control_passed,
 'failures':failures,'invalid_rejected':invalid_rejected,'invalid_n':5,
 'elapsed_seconds':time.time()-started,'source_sha256':sources,
 'limitations':'algorithmic addition only; no language model integration or frontier comparison; evaluator authored by same research team'}
(root/'REPORT.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
