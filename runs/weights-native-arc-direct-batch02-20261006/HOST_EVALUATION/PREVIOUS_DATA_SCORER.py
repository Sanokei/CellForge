"""One frozen direct-prediction baseline. Data comparison only; no author code."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import native_proposal_import as N


def decode_prediction(raw):
    if len(raw) > 65536:
        raise ValueError('response exceeds byte limit')
    value = N.decode(raw)
    if type(value) is not dict or set(value) != {'task_id', 'predictions', 'rationale'} or value['task_id'] != 'bd283c4a':
        raise ValueError('exact task and response fields required')
    if type(value['rationale']) is not str or len(value['rationale'].encode('utf-8')) > 4096:
        raise ValueError('rationale exceeds bound')
    predictions = value['predictions']
    if type(predictions) is not list or len(predictions) != 1:
        raise ValueError('exactly one query prediction required')
    rows = predictions[0]
    if type(rows) is not list or not 1 <= len(rows) <= 30:
        raise ValueError('grid height out of bounds')
    if any(type(row) is not str or not 1 <= len(row) <= 30 or any(v not in '0123456789' for v in row) for row in rows):
        raise ValueError('bounded ASCII digit rows required')
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('rectangular grid required')
    return [[[int(v) for v in row] for row in rows]]


def reference(grid):
    h, w = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row)
    flat = [c for c in sorted(counts, key=lambda c: (-counts[c], c)) for _ in range(counts[c])]
    return [[flat[c * h + h - 1 - r] for c in range(w)] for r in range(h)]


def comparison(predicted, expected):
    shape = len(predicted) == len(expected) and all(len(a) == len(b) for a, b in zip(predicted, expected))
    differing = sum(a != b for pa, ea in zip(predicted, expected) for a, b in zip(pa, ea)) if shape else None
    return {'exact': predicted == expected, 'shape_matches': shape, 'differing_pixels': differing}


def score():
    N.stop(ROOT)
    protocol = N.read_signed(ROOT / 'DIRECT_GRID_PROTOCOL.json')
    freeze = N.read_signed(ROOT / 'DIRECT_GRID_RESPONSE_FROZEN.json')
    dispatch = N.read_signed(ROOT / 'DIRECT_GRID_DISPATCH.json')
    if freeze['protocol_hash'] != protocol['hash'] or freeze['dispatch_hash'] != dispatch['hash']:
        raise ValueError('frozen provenance differs')
    raw = (ROOT / 'DIRECT_GRID_RESPONSE.raw.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != freeze['raw_sha256']:
        raise ValueError('frozen raw response changed')
    predictions = decode_prediction(raw)
    # Expected query labels are accessed only after response freeze and validation.
    raw_task = (REPO / 'data/arcagi2-local-20261006/training/bd283c4a.json').read_bytes()
    if hashlib.sha256(raw_task).hexdigest() != protocol['data_sha256']:
        raise ValueError('public-training source changed')
    task = json.loads(raw_task)
    if len(task['test']) != len(predictions):
        raise ValueError('query count differs')
    outcomes = {}
    for name, grids in [('direct_codex', predictions), ('identity', [q['input'] for q in task['test']]),
                        ('host_inferred_program', [reference(q['input']) for q in task['test']])]:
        queries = [comparison(out, q['output']) for out, q in zip(grids, task['test'])]
        outcomes[name] = {'task_exact': all(q['exact'] for q in queries), 'queries': queries}
    return N.signed({'kind': 'direct_grid_baseline_result/1', 'task_id': 'bd283c4a',
                     'protocol_hash': protocol['hash'], 'response_freeze_hash': freeze['hash'],
                     'grader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'data_sha256': protocol['data_sha256'], 'outcomes': outcomes,
                     'author_calls': 1, 'credit_usage': None, 'model_backend': None, 'reasoning': 'xhigh',
                     'genome_positions': 0, 'weight_result': False,
                     'scope': 'Previously exposed public-training search task; not held-out transfer.'})


if __name__ == '__main__':
    result = score()
    if sys.argv[1:] == ['--verify']:
        if result != N.read_signed(ROOT / 'HOST_EVALUATION/DIRECT_GRID_RESULT.json'):
            raise ValueError('independent replay differs')
        print('Independent direct-grid replay matched:', result['hash'])
    elif not sys.argv[1:]:
        N.write_once(ROOT / 'HOST_EVALUATION/DIRECT_GRID_RESULT.json', result)
        print(json.dumps(result, indent=2))
    else:
        raise ValueError('only --verify is supported')
