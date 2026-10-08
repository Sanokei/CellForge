"""Task-bound reuse of the preceding strict data-only grid comparator."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import native_proposal_import as N

TASKS = ('1a07d186', '3194b014', 'f5b8619d')


def decode_prediction(raw, task_id):
    if task_id not in TASKS or len(raw) > 65536:
        raise ValueError('unbound task or response exceeds byte limit')
    value = N.decode(raw)
    if type(value) is not dict or set(value) != {'task_id', 'predictions', 'rationale'} or value['task_id'] != task_id:
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


def comparison(predicted, expected):
    shape = len(predicted) == len(expected) and all(len(a) == len(b) for a, b in zip(predicted, expected))
    differing = sum(a != b for pa, ea in zip(predicted, expected) for a, b in zip(pa, ea)) if shape else None
    return {'exact': predicted == expected, 'shape_matches': shape, 'differing_pixels': differing}


def score(task_id):
    if task_id not in TASKS:
        raise ValueError('unbound task')
    N.stop(ROOT)
    protocol = N.read_signed(ROOT / 'PROTOCOL.json')
    row = next(r for r in protocol['tasks'] if r['task_id'] == task_id)
    freeze = N.read_signed(ROOT / (task_id + '.FROZEN.json'))
    dispatch = N.read_signed(ROOT / 'DISPATCH.json')
    if freeze['task_id'] != task_id or freeze['protocol_hash'] != protocol['hash'] or freeze['dispatch_hash'] != dispatch['hash']:
        raise ValueError('frozen provenance differs')
    if freeze['author'] != next(a for a in dispatch['authors'] if a['task_id'] == task_id):
        raise ValueError('author binding differs')
    if hashlib.sha256((ROOT / ('AUTHOR_' + task_id + '.json')).read_bytes()).hexdigest() != row['packet_sha256']:
        raise ValueError('author packet changed')
    raw = (ROOT / (task_id + '.raw.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest() != freeze['raw_sha256']:
        raise ValueError('frozen raw response changed')
    predictions = decode_prediction(raw, task_id)
    # Query labels are used only here, after raw response freeze and validation.
    raw_task = (REPO / 'data/arcagi2-local-20261006/training' / (task_id + '.json')).read_bytes()
    if hashlib.sha256(raw_task).hexdigest() != row['data_sha256']:
        raise ValueError('public-training source changed')
    task = json.loads(raw_task)
    if len(task['test']) != len(predictions) or len(predictions) != row['query_count']:
        raise ValueError('query count differs')
    outcomes = {}
    for name, grids in [('direct_codex', predictions), ('identity', [q['input'] for q in task['test']])]:
        queries = [comparison(out, q['output']) for out, q in zip(grids, task['test'])]
        outcomes[name] = {'task_exact': all(q['exact'] for q in queries), 'queries': queries}
    return N.signed({'kind': 'direct_grid_batch_task_result/1', 'task_id': task_id,
                     'protocol_hash': protocol['hash'], 'response_freeze_hash': freeze['hash'],
                     'grader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'data_sha256': row['data_sha256'], 'outcomes': outcomes,
                     'author_calls': 1, 'credit_usage': None, 'model_backend': None, 'reasoning': 'xhigh',
                     'genome_positions': 0, 'weight_result': False,
                     'scope': 'Previously exposed public-training search task; not held-out transfer.'})


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != '--verify'):
        raise ValueError('usage: score_batch.py TASK_ID [--verify]')
    result = score(sys.argv[1])
    path = ROOT / 'HOST_EVALUATION' / (sys.argv[1] + '.RESULT.json')
    if len(sys.argv) == 3:
        if result != N.read_signed(path):
            raise ValueError('independent replay differs')
        print('Independent direct-grid replay matched:', sys.argv[1], result['hash'])
    else:
        N.write_once(path, result)
        print(json.dumps(result, indent=2))
