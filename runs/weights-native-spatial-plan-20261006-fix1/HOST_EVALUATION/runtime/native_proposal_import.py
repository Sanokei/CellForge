"""One separately accounted native Codex artifact; never invokes a model service.

Prepare before consuming the response. Import consumes one immutable attempt,
including invalid/refused artifacts. Verify recomputes with saved host code and
hash-bound public-training data. Credits are unknown, not zero. No automatic
population admission or report evaluation; the CLI maker guard is untouched.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import sys

import cell_contracts as C
import fresh_inputs as F
import skillcore as S

SOURCE_FILES = ('native_proposal_import.py', 'fresh_inputs.py', 'skillcore.py', 'cell_contracts.py')
MAX_RESPONSE = 65536
RANDOM_SEED = 2026100602
SPATIAL_TEMPLATES = ('ray_cast', 'tile_orient')
SPATIAL_KIND = 'ray_cast_then_tile_orient/1'
SPATIAL_INSTRUCTION = ('Propose all literal tensor values for both cells. The host will compose ray_cast then '
    'tile_orient. Return exactly one response object matching the accompanying schema. Numeric values must be '
    'finite and in [-9,9]. No code or task-answer arrays.')
SPATIAL_BOUNDARY = ('Host code already implements ray propagation, collision handling, geometric transforms '
    'and tile assembly. Authored values control source-color gates, directions, stop policy, paint choice, '
    'tile counts and tile orientations; this is not learning these algorithms.')


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON field')
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def write_once(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def signed(body):
    return dict(body, hash=C.digest(body))


def read_signed(path):
    value = decode(path.read_bytes())
    if value.get('hash') != C.digest({k: v for k, v in value.items() if k != 'hash'}):
        raise ValueError('signed host record changed: ' + path.name)
    return value


def stop(root):
    if any(os.path.lexists(path) for path in (root / 'STOP', root / 'HOST_EVALUATION' / 'STOP')):
        raise RuntimeError('STOP blocks native import/evaluation')


def map_numbers(value, draw):
    if isinstance(value, dict):
        return {key: map_numbers(item, draw) for key, item in value.items()}
    if isinstance(value, list):
        return [map_numbers(item, draw) for item in value]
    return draw()


def response_entry(raw, task_id):
    """Exact data-only response wrapper, shared by the two bounded interfaces."""
    if len(raw) > MAX_RESPONSE:
        raise ValueError('response exceeds 65536 bytes')
    value = decode(raw)
    if type(value) is not list or len(value) != 1:
        raise ValueError('exactly one candidate required; refusal/invalid attempts remain charged')
    entry = value[0]
    if type(entry) is not dict or set(entry) != {'task_id', 'params', 'rationale'} or entry['task_id'] != task_id:
        raise ValueError('exact candidate fields and bound search task ID required')
    if type(entry['rationale']) is not str or len(entry['rationale'].encode('utf-8')) > 4096:
        raise ValueError('rationale exceeds 4096 UTF-8 bytes')
    return entry


def validate_response(raw, task_id, reference):
    """Strict wrapper plus existing literal-type/shape and registered-range APIs."""
    entry = response_entry(raw, task_id)
    params = C.validate_tensor_override(entry['params'], reference)
    if not S.valid_params('assoc_induct', params):
        raise ValueError('registered assoc_induct tensors in [-9,9] required')
    return copy.deepcopy(entry)


def spatial_cell(params):
    if type(params) is not dict or set(params) != set(SPATIAL_TEMPLATES):
        raise ValueError('exactly ray_cast and tile_orient tensor maps required')
    leaves = []
    for name in SPATIAL_TEMPLATES:
        # Existing strict literal checks reject bools/strings and malformed arrays.
        C.validate_tensor_override(params[name], params[name])
        if not S.valid_params(name, params[name]):
            raise ValueError('spatial leaf tensors differ from registered shape/range')
        leaves.append(S.make_cell(name, params[name]))
    if sum(S.positions(leaf) for leaf in leaves) != 167:
        raise ValueError('all 167 newly authored leaf positions must be counted')
    return S.typed_chain_cell(*leaves)


def validate_spatial_response(raw, task_id, reference):
    entry = response_entry(raw, task_id)
    if type(entry['params']) is not dict or set(entry['params']) != set(SPATIAL_TEMPLATES):
        raise ValueError('exactly ray_cast and tile_orient tensor maps required')
    for name in SPATIAL_TEMPLATES:
        C.validate_tensor_override(entry['params'][name], reference[name])
    spatial_cell(entry['params'])
    return copy.deepcopy(entry)


def expected_spatial_packet(demonstrations):
    return {'schema': 'cellforge-native-spatial-candidate-packet/1', 'status': 'candidate plan; no author dispatched',
        'task_id': 'f5b8619d', 'partition': 'search_fitness', 'demonstrations': demonstrations,
        'composition': list(SPATIAL_TEMPLATES), 'literal_positions': 167,
        'templates': {name: {'shapes': {k: list(v) for k, v in S.TEMPLATES[name]['shapes'].items()},
                             'semantics': S.TEMPLATES[name]['doc']} for name in SPATIAL_TEMPLATES},
        'instruction': SPATIAL_INSTRUCTION, 'host_boundary': SPATIAL_BOUNDARY, 'demonstrations_only': True}


def expected_spatial_schema():
    def tensor(shape):
        return ({'type': 'array', 'minItems': shape[0], 'maxItems': shape[0], 'items': tensor(shape[1:])}
                if shape else {'type': 'number', 'minimum': -9, 'maximum': 9})
    params = {name: {'type': 'object', 'additionalProperties': False,
        'properties': {key: tensor(shape) for key, shape in S.TEMPLATES[name]['shapes'].items()},
        'required': list(S.TEMPLATES[name]['shapes'])} for name in SPATIAL_TEMPLATES}
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'array', 'minItems': 1,
        'maxItems': 1, 'items': {'type': 'object', 'additionalProperties': False,
            'properties': {'task_id': {'const': 'f5b8619d'}, 'params': {'type': 'object',
                'additionalProperties': False, 'properties': params, 'required': list(SPATIAL_TEMPLATES)},
                'rationale': {'type': 'string', 'maxLength': 4096}}, 'required': ['task_id', 'params', 'rationale']}}


def validate_spatial_packet(packet, schema, demonstrations):
    if packet != expected_spatial_packet(demonstrations) or schema != expected_spatial_schema():
        raise ValueError('author context must be exact search demonstrations and generic forward semantics')


def charged_positions(protocol):
    expected = 167 if protocol.get('response_kind') == SPATIAL_KIND else 102
    if protocol.get('positions_per_arm', expected) != expected:
        raise ValueError('protocol position charge differs from full authored interface')
    return {'native': expected, 'random': expected}


def prepare(root, fresh_config):
    root = Path(root).resolve()
    stop(root)
    request = decode((root / 'REQUEST_MANIFEST.json').read_bytes())
    packet_bytes = (root / 'AUTHORING_PACKET.json').read_bytes()
    schema_bytes = (root / 'RESPONSE_SCHEMA.json').read_bytes()
    packet = decode(packet_bytes)
    if (request['maximum_submissions'] != 1 or request['template'] != 'assoc_induct'
            or request['positions_per_proposal'] != 102 or request['partition'] != 'search_fitness'
            or digest_bytes(packet_bytes) != request['packet_sha256']
            or digest_bytes(schema_bytes) != request['schema_sha256']):
        raise ValueError('one immutable 102-position search request required')
    inputs = F.load(fresh_config)
    if digest_bytes(inputs['source_bytes']) != request['source_manifest_sha256']:
        raise ValueError('dataset source differs from authoring request')
    by_id = {row['id']: row for row in inputs['rows']['search_fitness']}
    row = by_id[request['task_id']]
    if (packet['task_id'] != row['id'] or packet['template'] != 'assoc_induct'
            or packet['demonstrations'] != row['task']['train']
            or set(packet) != {'schema', 'partition', 'task_id', 'template', 'tensor_shapes',
                               'literal_positions', 'template_semantics', 'instruction', 'demonstrations'}):
        raise ValueError('packet differs from permitted search demonstrations')
    reference = next(cell for cell in inputs['baseline']['cells'] if cell['template'] == 'assoc_induct')
    sources = {name: digest_bytes(Path(__file__).with_name(name).read_bytes()) for name in SOURCE_FILES}
    if sources['skillcore.py'] != request['skillcore_sha256']:
        raise ValueError('evaluator changed since authoring request')
    body = {'schema': 'cellforge-native-import-protocol/1', 'maximum_submissions': 1,
        'task_id': request['task_id'], 'template': 'assoc_induct', 'positions_per_arm': 102,
        'request_hashes': {name: digest_bytes((root / name).read_bytes()) for name in
                           ('REQUEST_MANIFEST.json', 'AUTHORING_PACKET.json', 'RESPONSE_SCHEMA.json')},
        'fresh_inputs': fresh_config, 'source_hashes': sources, 'reference_cell': reference,
        'random_seed': RANDOM_SEED, 'random_distribution': 'uniform[-9,9] at all 102 positions',
        'partitions': {name: [r['id'] for r in inputs['rows'][name]] for name in
                       ('search_fitness', 'development_selection')},
        'reserved_report_ids': inputs['report_ids'], 'report_accessed': False,
        'scope_hash': C.digest(inputs['rows']), 'automatic_admission': False,
        'credit_usage': None, 'authoring_route': 'native Codex agent file artifact; not maker.ask'}
    return freeze_protocol(root, body)


def freeze_protocol(root, body):
    host = root / 'HOST_EVALUATION'
    host.mkdir()  # existing/partial work is preserved, never silently retried
    (host / 'runtime').mkdir()
    for name in SOURCE_FILES:
        (host / 'runtime' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_once(host / 'PROTOCOL.json', signed(body))
    return {'status': 'prepared', 'protocol_hash': C.digest(body), 'positions_per_arm': body['positions_per_arm']}


def prepare_v2(root):
    """Bind the single authorized feedback revision; do not read its response."""
    root = Path(root).resolve()
    stop(root)
    request = read_signed(root / 'REQUEST_MANIFEST.json')
    if (request['schema'] != 'cellforge-native-request-manifest/2'
            or request['maximum_submissions'] != 1 or request['positions_per_proposal'] != 102
            or request['matched_random_positions_required'] != 102):
        raise ValueError('one complete V2 replacement and matched control required')
    original = Path(request['original_run']).resolve()
    memory = root / 'HOST_MEMORY'
    # Reverify the existing journal through its frozen trusted host adapter.
    # Snapshot execution avoids rebinding old provenance to this new importer.
    manifest = decode((memory / 'EVOLUTION_MANIFEST.json').read_bytes())
    if C.digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}) != manifest['manifest_hash']:
        raise ValueError('feedback manifest changed')
    for key, digest in manifest['source_hashes'].items():
        if key.startswith('runtime:'):
            name = key.removeprefix('runtime:')
            if Path(name).name != name or digest_bytes((memory / 'runtime' / name).read_bytes()) != digest:
                raise ValueError('frozen feedback runtime changed')
    replay = subprocess.run([sys.executable, str(memory / 'runtime' / 'native_failure_feedback.py'),
        'verify', str(original), str(root)], capture_output=True, text=True, timeout=30)
    if replay.returncode or decode(replay.stdout).get('status') != 'verified':
        raise ValueError('trusted search feedback replay failed')
    previous = read_signed(original / 'HOST_EVALUATION' / 'PROTOCOL.json')
    prior_attempt = read_signed(original / 'HOST_EVALUATION' / 'ATTEMPT.json')
    prior_result = read_signed(original / 'HOST_EVALUATION' / 'RESULT.json')
    packet = decode((root / 'AUTHORING_PACKET_V2.json').read_bytes())
    parent = prior_result['cells']['native']
    if (request['parent_hashes'] != [parent['id']]
            or packet['previous_candidate'] != {'id': parent['id'], 'params': parent['params']}
            or S.make_cell('assoc_induct', parent['params']) != parent
            or prior_attempt['charged_positions'] != {'native': 102, 'random': 102}
            or prior_result['attempt_hash'] != prior_attempt['hash']
            or prior_attempt['protocol_hash'] != previous['hash']):
        raise ValueError('V2 parent or previous charged attempt changed')
    inputs = F.load(previous['fresh_inputs'])
    if C.digest(inputs['rows']) != previous['scope_hash']:
        raise ValueError('V2 must use exactly the original frozen panels')
    sources = {name: digest_bytes(Path(__file__).with_name(name).read_bytes()) for name in SOURCE_FILES}
    if sources['skillcore.py'] != previous['source_hashes']['skillcore.py']:
        raise ValueError('V2 evaluator changed')
    evidence = [original / 'HOST_EVALUATION' / name for name in
                ('PROTOCOL.json', 'ATTEMPT.json', 'RESULT.json', 'RESPONSE_CAPTURE.bin')]
    evidence += [memory / name for name in ('EVOLUTION_MANIFEST.json', 'EVOLUTION_EVENTS.jsonl',
                 'SEARCH_RECEIPT.json', 'FAILURES.jsonl', 'VERIFIED_FEEDBACK.json')]
    body = {k: copy.deepcopy(v) for k, v in previous.items() if k not in ('hash', 'request_hashes', 'source_hashes')}
    body.update(request_hashes={name: digest_bytes((root / name).read_bytes()) for name in
                ('REQUEST_MANIFEST.json', 'AUTHORING_PACKET_V2.json', 'RESPONSE_SCHEMA.json')},
        source_hashes=sources, reference_cell=parent, random_seed=2026100603,
        parent_hashes=[parent['id']], operator='complete_tensor_replacement', native_sequence_number=2,
        prior_attempt_hash=prior_attempt['hash'], cumulative_positions_after_attempt={'native': 204, 'random': 204},
        bound_evidence={str(path): digest_bytes(path.read_bytes()) for path in evidence})
    return freeze_protocol(root, body)


def verify_bound_evidence(protocol):
    for filename, digest in protocol.get('bound_evidence', {}).items():
        if digest_bytes(Path(filename).read_bytes()) != digest:
            raise ValueError('V2 parent, feedback or previous evidence changed')


def prepare_spatial(root, prior_root):
    """Freeze the reviewed two-leaf founder interface without consuming a response."""
    root, prior_root = Path(root).resolve(), Path(prior_root).resolve()
    stop(root)
    manifest = read_signed(root / 'PACKET_MANIFEST.json')
    selection = read_signed(root / 'SELECTION.json')
    proof = read_signed(root / 'HOST_FEASIBILITY.json')
    packet_bytes = (root / 'AUTHORING_PACKET_CANDIDATE.json').read_bytes()
    schema_bytes = (root / 'RESPONSE_SCHEMA_CANDIDATE.json').read_bytes()
    if (manifest['selection_hash'] != selection['hash']
            or digest_bytes(packet_bytes) != manifest['packet_sha256']
            or digest_bytes(schema_bytes) != manifest['schema_sha256']):
        raise ValueError('frozen spatial selection, packet or schema changed')
    previous = read_signed(prior_root / 'HOST_EVALUATION' / 'PROTOCOL.json')
    prior_attempt = read_signed(prior_root / 'HOST_EVALUATION' / 'ATTEMPT.json')
    prior_result = read_signed(prior_root / 'HOST_EVALUATION' / 'RESULT.json')
    if (previous.get('native_sequence_number') != 2 or prior_result['attempt_hash'] != prior_attempt['hash']
            or prior_attempt['protocol_hash'] != previous['hash']
            or prior_attempt['charged_positions'] != {'native': 102, 'random': 102}
            or prior_result['cumulative_charged_positions'] != {'native': 204, 'random': 204}):
        raise ValueError('completed separately charged native sequence required')
    verify_bound_evidence(previous)
    inputs = F.load(previous['fresh_inputs'])
    if C.digest(inputs['rows']) != previous['scope_hash']:
        raise ValueError('spatial evaluation must retain the frozen panels')
    rows = {row['id']: row for row in inputs['rows']['search_fitness']}
    if (selection['selected_task_id'] != 'f5b8619d' or selection['templates'] != list(SPATIAL_TEMPLATES)
            or selection['inspected_demonstration_hashes'] != {key: C.digest(row['task']['train']) for key, row in rows.items()}
            or digest_bytes(inputs['source_bytes']) != selection['source_manifest_sha256']):
        raise ValueError('search-only selection/source binding changed')
    validate_spatial_packet(decode(packet_bytes), decode(schema_bytes), rows['f5b8619d']['task']['train'])
    reference = proof['hand_coded_chain']
    params = {part['template']: part['params'] for part in reference['parts']}
    if (spatial_cell(params) != reference or proof['llm_authored'] is not False
            or proof['weight_positions'] != 167):
        raise ValueError('separate disclosed hand-coded baseline required')
    sources = {name: digest_bytes(Path(__file__).with_name(name).read_bytes()) for name in SOURCE_FILES}
    if sources['skillcore.py'] != selection['skillcore_sha256']:
        raise ValueError('spatial evaluator changed after feasibility review')
    evidence = [prior_root / 'HOST_EVALUATION' / name for name in ('PROTOCOL.json', 'ATTEMPT.json', 'RESULT.json')]
    body = {k: copy.deepcopy(previous[k]) for k in ('schema', 'fresh_inputs', 'partitions', 'reserved_report_ids',
        'scope_hash', 'report_accessed', 'automatic_admission', 'credit_usage', 'authoring_route')}
    body.update(maximum_submissions=1, task_id='f5b8619d', response_kind=SPATIAL_KIND,
        template='typed_chain', composition=list(SPATIAL_TEMPLATES), positions_per_arm=167,
        reference_cell=reference, reference_arm='hand_coded', source_hashes=sources,
        request_hashes={name: digest_bytes((root / name).read_bytes()) for name in
            ('PACKET_MANIFEST.json', 'SELECTION.json', 'HOST_FEASIBILITY.json',
             'AUTHORING_PACKET_CANDIDATE.json', 'RESPONSE_SCHEMA_CANDIDATE.json')},
        random_seed=2026100604, random_distribution='uniform[-9,9] at all 167 leaf positions including inactive slots',
        parent_hashes=[], operator='founder', native_sequence_number=3, prior_attempt_hash=prior_attempt['hash'],
        cumulative_positions_after_attempt={'native': 371, 'random': 371},
        accounting_scope='native ARC grid proposal sequence; other experiments accounted separately',
        bound_evidence={str(path): digest_bytes(path.read_bytes()) for path in evidence},
        selection_basis='search demonstration inspection and disclosed hand-coded feasibility; no independent generalization claim',
        host_algorithm_boundary=SPATIAL_BOUNDARY, new_algorithms_invented_by_weights=False)
    return freeze_protocol(root, body)


def context(root):
    root = Path(root).resolve()
    stop(root)
    host = root / 'HOST_EVALUATION'
    protocol = read_signed(host / 'PROTOCOL.json')
    verify_bound_evidence(protocol)
    for name, digest in protocol['request_hashes'].items():
        if digest_bytes((root / name).read_bytes()) != digest:
            raise ValueError('frozen author request changed')
    for name, digest in protocol['source_hashes'].items():
        if (digest_bytes((host / 'runtime' / name).read_bytes()) != digest
                or digest_bytes(Path(__file__).with_name(name).read_bytes()) != digest):
            raise ValueError('use the exact saved host runtime; code changed')
    inputs = F.load(protocol['fresh_inputs'])
    if C.digest(inputs['rows']) != protocol['scope_hash']:
        raise ValueError('development task scope changed')
    return root, host, protocol, inputs


def score(cell, rows, root):
    result = {}
    for row in rows:
        stop(root)
        task, predictions, errors = row['task'], [], []
        # Evaluator receives only input grids and demonstration pairs, never a
        # task object containing query outputs. Exceptions stay null, not false.
        for query in task['test']:
            try:
                predictions.append(S.run(cell, copy.deepcopy(query['input']), copy.deepcopy(task['train'])))
            except Exception as error:
                predictions.append(None)
                errors.append({'type': type(error).__name__, 'message': str(error)[:300]})
        result[row['id']] = {'correct': None if errors else predictions == [q['output'] for q in task['test']],
            'prediction_sha256': C.digest(predictions), 'errors': errors}
    return result


def compute(root, protocol, inputs, raw):
    reference = protocol['reference_cell']
    rng = random.Random(protocol['random_seed'])
    spatial = protocol.get('response_kind') == SPATIAL_KIND
    params = {part['template']: part['params'] for part in reference['parts']} if spatial else reference['params']
    make_cell = spatial_cell if spatial else lambda p: S.make_cell('assoc_induct', p)
    validator = validate_spatial_response if spatial else validate_response
    reference_arm = 'hand_coded' if spatial else 'reference'
    arms = {reference_arm: reference,
        'zero': make_cell(map_numbers(params, lambda: 0.0)),
        'random': make_cell(map_numbers(params, lambda: rng.uniform(-9, 9)))}
    try:
        entry = validator(raw, protocol['task_id'], params)
        arms['native'] = make_cell(entry['params'])
        validation = {'status': 'valid', 'error': None}
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as error:
        validation = {'status': 'invalid_or_refused', 'error': type(error).__name__ + ': ' + str(error)[:300]}
    evaluations = {arm: {panel: score(cell, inputs['rows'][panel], root) for panel in
                         ('search_fitness', 'development_selection')} for arm, cell in arms.items()}
    result = {'validation': validation, 'cells': arms, 'evaluations': evaluations,
        'charged_positions': charged_positions(protocol),
        'attempts': {'native': 1, 'random': 1}, 'fixed_diagnostic_arms': ['zero', reference_arm],
        'runtime_errors': {arm: sum(len(v['errors']) for rows in values.values() for v in rows.values())
                           for arm, values in evaluations.items()},
        'counts': {arm: {panel: sum(v['correct'] is True for v in rows.values()) for panel, rows in values.items()}
                    for arm, values in evaluations.items()},
        'query_outputs_supplied_to_predictor': False, 'report_accessed': False,
        'scientific_acceptance': False, 'retained_into_existing_population': False}
    if 'native_sequence_number' in protocol:
        result.update(native_sequence_number=protocol['native_sequence_number'],
            parent_hashes=protocol['parent_hashes'], operator=protocol['operator'],
            cumulative_charged_positions=protocol['cumulative_positions_after_attempt'])
    return result


def consume(root, author_thread, author_turn, author_model, author_reasoning):
    root, host, protocol, inputs = context(root)
    if any(not re.fullmatch('[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', identity)
           for identity in (author_thread, author_turn)):
        raise ValueError('actual author thread and turn UUIDs required')
    if not isinstance(author_model, str) or not 1 <= len(author_model) <= 100:
        raise ValueError('actual supervising-platform model label required')
    if author_reasoning not in ('none', 'low', 'medium', 'high', 'unknown'):
        raise ValueError('actual author reasoning label required')
    proposal_path = root / 'NATIVE_PROPOSAL.json'
    info = proposal_path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('native response must be an ordinary file, not a link')
    with proposal_path.open('rb') as stream:
        raw = stream.read(MAX_RESPONSE + 1)
    # Consume before parsing/scoring. Even invalid/refused/oversized content
    # spends the one submission; a partial import requires manual reconciliation.
    attempt = signed({'schema': 'cellforge-native-attempt/1', 'protocol_hash': protocol['hash'],
        'attempt': 1, 'charged_positions': charged_positions(protocol),
        'author_thread': author_thread, 'author_turn': author_turn,
        'author_model': author_model, 'author_reasoning': author_reasoning,
        'authoring_route': protocol['authoring_route'], 'authoring_credit_usage': None,
        'cli_meter_receipt': None, 'captured_response_sha256': digest_bytes(raw),
        'capture_complete': len(raw) <= MAX_RESPONSE, 'captured_bytes': len(raw)})
    write_once(host / 'ATTEMPT.json', attempt)
    with (host / 'RESPONSE_CAPTURE.bin').open('xb') as stream:
        stream.write(raw)
    result = signed(dict(compute(root, protocol, inputs, raw), attempt_hash=attempt['hash'],
                         protocol_hash=protocol['hash'], schema='cellforge-native-result/1'))
    write_once(host / 'RESULT.json', result)
    return {'status': 'evaluated', 'result_hash': result['hash'], 'validation': result['validation'],
            'counts': result['counts'], 'runtime_errors': result['runtime_errors'], 'credit_usage': None}


def verify(root):
    root, host, protocol, inputs = context(root)
    attempt, result = read_signed(host / 'ATTEMPT.json'), read_signed(host / 'RESULT.json')
    raw = (host / 'RESPONSE_CAPTURE.bin').read_bytes()
    if digest_bytes(raw) != attempt['captured_response_sha256'] or attempt['protocol_hash'] != protocol['hash']:
        raise ValueError('response/attempt provenance mismatch')
    if (attempt['attempt'] != 1 or attempt['charged_positions'] != charged_positions(protocol)
            or attempt['authoring_credit_usage'] is not None or attempt['cli_meter_receipt'] is not None):
        raise ValueError('native accounting or route differs from protocol')
    # Complete captures also remain bound to the original agent output file.
    if attempt['capture_complete'] and (root / 'NATIVE_PROPOSAL.json').read_bytes() != raw:
        raise ValueError('original native response changed after import')
    expected = signed(dict(compute(root, protocol, inputs, raw), attempt_hash=attempt['hash'],
                           protocol_hash=protocol['hash'], schema='cellforge-native-result/1'))
    if expected != result:
        raise ValueError('independent result replay differs')
    return {'status': 'verified', 'result_hash': result['hash'], 'counts': result['counts'],
            'runtime_errors': result['runtime_errors'], 'credit_usage': None, 'report_accessed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'prepare-v2', 'prepare-spatial', 'import', 'verify'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--plan', type=Path, help='existing fresh-input plan, prepare only')
    parser.add_argument('--author-thread')
    parser.add_argument('--author-turn')
    parser.add_argument('--author-model')
    parser.add_argument('--author-reasoning')
    parser.add_argument('--prior-native-run', type=Path, help='completed native V2 run, prepare-spatial only')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.plan is None:
            parser.error('--plan is required')
        result = prepare(args.root, decode(args.plan.read_bytes())['fresh_inputs'])
    elif args.command == 'prepare-v2':
        result = prepare_v2(args.root)
    elif args.command == 'prepare-spatial':
        if args.prior_native_run is None:
            parser.error('--prior-native-run is required')
        result = prepare_spatial(args.root, args.prior_native_run)
    elif args.command == 'import':
        if any(value is None for value in (args.author_thread, args.author_turn, args.author_model, args.author_reasoning)):
            parser.error('actual author thread, turn, model and reasoning are required')
        result = consume(args.root, args.author_thread, args.author_turn, args.author_model, args.author_reasoning)
    else:
        result = verify(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
