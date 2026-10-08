"""Bounded search-demonstration feasibility only; no maker, query or dev scores."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
FIRST = REPO / 'runs/weights-native-agent-one-20261006'
sys.path.insert(0, str(FIRST / 'HOST_EVALUATION/runtime'))
import skillcore as S
import native_proposal_import as N

ROOT = REPO / 'runs/weights-native-spatial-plan-20261006-fix1'
SOURCE_HASH = '911a5a1d0aca72a2aa3f5e3b25e8d579cbf5b4fcd158d09ce6dc02ebd19d1e95'
source_path = REPO / 'data/arcagi2-local-20261006/SOURCE.json'
source_bytes = source_path.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_HASH
source = json.loads(source_bytes)
request = json.loads((FIRST / 'REQUEST_MANIFEST.json').read_bytes())
code = (FIRST / 'HOST_EVALUATION/runtime/skillcore.py').read_bytes()
assert hashlib.sha256(code).hexdigest() == request['skillcore_sha256']
demos, hashes = {}, {}
for item in source['panels']['search_fitness']:
    raw = (source_path.parent / 'training' / (item['id'] + '.json')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    # Deliberately discard all query pairs; inspection uses training pairs only.
    demos[item['id']] = json.loads(raw)['train']
    hashes[item['id']] = N.C.digest(demos[item['id']])

ROOT.mkdir()  # Never overwrite a prior plan, probe or packet.
selection = {'schema': 'cellforge-search-structure-selection/1', 'status': 'plan only; no model call',
    'partition': 'search_fitness', 'inspected_demonstration_hashes': hashes,
    'selected_task_id': 'f5b8619d', 'templates': ['ray_cast', 'tile_orient'],
    'source_manifest_sha256': SOURCE_HASH, 'skillcore_sha256': request['skillcore_sha256'],
    'rationale': 'Shown outputs fill source-bearing columns with color 8 while preserving source pixels, then repeat the entire resulting grid in two rows and two columns. Existing ray casting followed by whole-grid tiling can express this; uniform per-color mapping cannot.',
    'original_task_diagnostic': 'snap_to_line can express 1a07d186, but zero tensors choose background 0 and orphan removal; check this disqualifying control before further model work.',
    'diagnostic_plan': 'One source-derived hand-coded chain and its all-zero control on the three selected demonstrations; one all-zero snap_to_line cell on the three original demonstrations. No parameter search or query scoring.',
    'authored_positions_per_future_attempt': 167, 'random_positions_per_future_attempt': 167,
    'maximum_future_author_calls_recommended': 1, 'automatic_dispatch': False,
    'diagnostic_weights_are_llm_authored': False, 'scientific_acceptance': False}
N.write_once(ROOT / 'SELECTION.json', N.signed(selection))


def zeros(template):
    def tensor(shape):
        return [tensor(shape[1:]) for _ in range(shape[0])] if shape else 0.0
    return {key: tensor(shape) for key, shape in S.TEMPLATES[template]['shapes'].items()}


def evaluate(cell, examples):
    outcomes = []
    for example in examples:
        try:
            actual = S.run(cell, example['input'], [])
            outcomes.append({'correct': actual == example['output'],
                             'prediction_hash': N.C.digest(actual), 'runtime_error': None})
        except Exception as error:
            outcomes.append({'correct': None, 'prediction_hash': None, 'runtime_error': type(error).__name__})
    return outcomes


def full_authored_positions(cell):
    # positions(chain)==0 describes inherited assembly, not newly written leaves.
    return (sum(full_authored_positions(part) for part in cell['parts'])
            if cell['kind'] == 'chain' else S.positions(cell))


# Hand-coded mechanism probes, never founders or candidate evidence.
ray, tile = zeros('ray_cast'), zeros('tile_orient')
ray['src'] = [0.0] + [1.0] * 9
ray['dirs'][0] = ray['dirs'][4] = 1.0
ray['stop'][2] = ray['paint'][8] = 1.0
tile['per'][1] = tile['pec'][1] = 1.0
for row in tile['orient']:
    for orientation in row:
        orientation[0] = 1.0
chain = S.typed_chain_cell(S.make_cell('ray_cast', ray), S.make_cell('tile_orient', tile))
zero_chain = S.typed_chain_cell(S.make_cell('ray_cast', zeros('ray_cast')),
                                S.make_cell('tile_orient', zeros('tile_orient')))
snap = S.make_cell('snap_to_line', zeros('snap_to_line'))
proof = {'schema': 'cellforge-search-structure-feasibility/1', 'llm_authored': False,
    'candidate_admission': False, 'hand_coded_chain': chain, 'zero_chain': zero_chain,
    'hand_coded_chain_results': evaluate(chain, demos['f5b8619d']),
    'zero_chain_results': evaluate(zero_chain, demos['f5b8619d']),
    'zero_snap_to_line_results': evaluate(snap, demos['1a07d186']),
    'query_predictions': 0, 'development_or_report_predictions': 0, 'parameter_search': False,
    'weight_positions': full_authored_positions(chain),
    'legacy_inherited_assembly_positions': S.positions(chain),
    'interface': S.cell_reference(chain)['interface'],
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
N.write_once(ROOT / 'HOST_FEASIBILITY.json', N.signed(proof))
assert all(row['correct'] is True for row in proof['hand_coded_chain_results'])
assert all(row['correct'] is False for row in proof['zero_chain_results'])
assert all(row['correct'] is True for row in proof['zero_snap_to_line_results'])
assert proof['weight_positions'] == 167

packet = {'schema': 'cellforge-native-spatial-candidate-packet/1', 'status': 'candidate plan; no author dispatched',
    'task_id': 'f5b8619d', 'partition': 'search_fitness', 'demonstrations': demos['f5b8619d'],
    'composition': ['ray_cast', 'tile_orient'], 'literal_positions': 167,
    'templates': {name: {'shapes': S.TEMPLATES[name]['shapes'], 'semantics': S.TEMPLATES[name]['doc']}
                  for name in ('ray_cast', 'tile_orient')},
    'instruction': 'Propose all literal tensor values for both cells. The host will compose ray_cast then tile_orient. Return exactly one response object matching the accompanying schema. Numeric values must be finite and in [-9,9]. No code or task-answer arrays.',
    'host_boundary': 'Host code already implements ray propagation, collision handling, geometric transforms and tile assembly. Authored values control source-color gates, directions, stop policy, paint choice, tile counts and tile orientations; this is not learning these algorithms.',
    'demonstrations_only': True}
N.write_once(ROOT / 'AUTHORING_PACKET_CANDIDATE.json', packet)


def tensor_schema(shape):
    if not shape:
        return {'type': 'number', 'minimum': -9, 'maximum': 9}
    return {'type': 'array', 'minItems': shape[0], 'maxItems': shape[0], 'items': tensor_schema(shape[1:])}


params = {name: {'type': 'object', 'additionalProperties': False,
    'properties': {key: tensor_schema(shape) for key, shape in S.TEMPLATES[name]['shapes'].items()},
    'required': list(S.TEMPLATES[name]['shapes'])} for name in packet['composition']}
schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'array', 'minItems': 1,
    'maxItems': 1, 'items': {'type': 'object', 'additionalProperties': False,
        'properties': {'task_id': {'const': 'f5b8619d'}, 'params': {'type': 'object',
            'additionalProperties': False, 'properties': params, 'required': packet['composition']},
            'rationale': {'type': 'string', 'maxLength': 4096}}, 'required': ['task_id','params','rationale']}}
N.write_once(ROOT / 'RESPONSE_SCHEMA_CANDIDATE.json', schema)
N.write_once(ROOT / 'PACKET_MANIFEST.json', N.signed({'status': 'prepared candidate; no dispatch',
    'selection_hash': N.read_signed(ROOT / 'SELECTION.json')['hash'],
    'packet_sha256': N.digest_bytes((ROOT / 'AUTHORING_PACKET_CANDIDATE.json').read_bytes()),
    'schema_sha256': N.digest_bytes((ROOT / 'RESPONSE_SCHEMA_CANDIDATE.json').read_bytes()),
    'credit_usage': None, 'automatic_dispatch': False,
    'importer_support': 'not yet implemented; current native importer only accepts assoc_induct'}))
standard = json.loads((FIRST / 'STANDARD.json').read_bytes())
standard['limitations'] = ['Prepared search-only structure plan; no LLM proposal or query evaluation',
    'Hand-coded demonstrations probe feasibility only, not learned or transferred capability',
    'Native credit usage unknown; no model call authorized by this artifact', 'Not a qualified ONE_STANDARD experiment']
N.write_once(ROOT / 'STANDARD.json', standard)
print(json.dumps({'status': 'candidate_plan_frozen', 'root': str(ROOT), 'positions': full_authored_positions(chain),
    'hand_coded_training': [r['correct'] for r in proof['hand_coded_chain_results']],
    'zero_training': [r['correct'] for r in proof['zero_chain_results']],
    'original_task_zero_training': [r['correct'] for r in proof['zero_snap_to_line_results']],
    'runtime_errors': sum(row['runtime_error'] is not None for key in
        ('hand_coded_chain_results','zero_chain_results','zero_snap_to_line_results') for row in proof[key]),
    'new_model_calls': 0, 'query_predictions': 0}, sort_keys=True))
