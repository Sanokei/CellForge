"""Exactly four supervised spatial replacements; no model service or dispatcher.

Frozen search-only author payloads, one charged submission per slot, matched
random controls, and read-only replay. Existing templates and old runs stay
unchanged. The new author/control vocabulary disables transpose-only tile
entries so every allowed orientation preserves rectangular dimensions.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random

import native_proposal_import as N

SLOTS = ('author01', 'author02', 'author03', 'author04')
FILES = (*N.SOURCE_FILES, 'native_parallel_batch.py')
SEEDS = dict(zip(SLOTS, (2026100611, 2026100612, 2026100613, 2026100614)))
POLICY = 'rectangular-safe-spatial-literals/1'
INSTRUCTION = ('Use only this payload. Do not use tools, filesystem access, external sources or another '
    'author. Return exactly one JSON array element matching the schema, without markdown. Write every '
    'literal value for both cells, including inactive entries. This is one complete replacement of the '
    'historical parent, not a request for code. No second submission or free schema repair is included.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_bytes(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def compact(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def leaf_params(cell):
    params = {part['template']: part['params'] for part in cell['parts']}
    require(N.spatial_cell(params) == cell, 'exact typed ray_cast then tile_orient cell required')
    return copy.deepcopy(params)


def response_schema():
    schema = N.expected_spatial_schema()
    vector = schema['items']['properties']['params']['properties']['tile_orient']['properties']['orient']['items']['items']
    number = vector.pop('items')
    vector['prefixItems'] = [copy.deepcopy(number) for _ in range(4)] + [{'type': 'number', 'const': -9} for _ in range(4)]
    vector['items'] = False
    return schema


def validate(raw, reference):
    entry = N.validate_spatial_response(raw, 'f5b8619d', reference)
    for row in entry['params']['tile_orient']['orient']:
        for values in row:
            require(values[4:] == [-9] * 4, 'all transpose/quarter-turn entries must be literal -9')
    return entry


def random_params(reference, seed):
    rng = random.Random(seed)
    params = {'ray_cast': N.map_numbers(reference['ray_cast'], lambda: rng.uniform(-9, 9)),
              'tile_orient': {key: [rng.uniform(-9, 9) for _ in range(4)] for key in ('per', 'pec')}}
    params['tile_orient']['orient'] = [[[rng.uniform(-9, 9) for _ in range(4)] + [-9.0] * 4
                                      for _ in range(4)] for _ in range(4)]
    return params


def author_packet(slot, demonstrations, parent, feedback):
    return {'schema': 'cellforge-four-author-spatial-packet/1', 'author_slot': slot,
        'task_id': 'f5b8619d', 'partition': 'search_fitness', 'composition': list(N.SPATIAL_TEMPLATES),
        'literal_positions': 167, 'demonstrations': demonstrations,
        'templates': N.expected_spatial_packet(demonstrations)['templates'],
        'parent': {'id': parent['id'], 'params': leaf_params(parent)}, 'search_feedback': feedback,
        'constraints': {'policy': POLICY, 'all_numbers': 'finite numbers in [-9,9]',
            'orient': 'In EVERY orient[i][j] vector, entries 4,5,6,7 must equal -9; entries 0,1,2,3 remain in [-9,9]. Lowest-index argmax breaks ties. All 167 values must be supplied.',
            'reason': 'Only shape-preserving identity, horizontal/vertical flips and 180-degree rotation are allowed in this new batch. The historical parent predates this constraint; it must not be silently clipped by the host.'},
        'instruction': INSTRUCTION, 'host_algorithm_boundary': N.SPATIAL_BOUNDARY,
        'qualification': 'Host-defined spatial primitive composition; no invented-algorithm or generalization claim.'}


def feedback_from(parent, demonstrations):
    index = min(range(len(demonstrations)), key=lambda i: (len(demonstrations[i]['input']) * len(demonstrations[i]['input'][0]), i))
    pair = demonstrations[index]
    context = copy.deepcopy(demonstrations[:index] + demonstrations[index + 1:])
    actual = N.S.run(parent, copy.deepcopy(pair['input']), context)
    params = leaf_params(parent)
    tile, ray = params['tile_orient'], params['ray_cast']
    argmax = lambda values: max(range(len(values)), key=lambda i: values[i])
    pr, pc = argmax(tile['per']) + 1, argmax(tile['pec']) + 1
    return {'training_example_index': index, 'input': pair['input'], 'expected': pair['output'],
        'actual': actual, 'matches': actual == pair['output'], 'runtime_error': None,
        'method': 'Host replay of historical parent on one search training example; that example excluded from context. No query or development/report feedback.',
        'observed_forward_choices': {'stop_policy_index': argmax(ray['stop']), 'tile_counts': [pr, pc],
            'active_orientation_indices': [[argmax(tile['orient'][i][j]) for j in range(pc)] for i in range(pr)],
            'paint_mode': 'source color' if max(ray['paint']) <= 0 else 'fixed color'},
        'cause_hypotheses': 'No established diagnosis is supplied; infer from the demonstrations and actual values.'}


def prepare(root, previous):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    require(not root.exists(), 'new batch root required; prior evidence is never overwritten')
    previous_protocol = N.read_signed(previous / 'HOST_EVALUATION/PROTOCOL.json')
    base_root = Path(previous_protocol['original_spatial_root'])
    _, _, base, inputs = N.context(base_root)
    batch = N.read_signed(previous / 'HOST_EVALUATION/BATCH_RESULT.json')
    result = N.read_signed(previous / 'HOST_EVALUATION/RESULT2.json')
    attempt = N.read_signed(previous / 'HOST_EVALUATION/ATTEMPT2.json')
    raw = (previous / 'TEXT_ROUTE_RECEIPTS/ATTEMPT2_RESPONSE.json').read_bytes()
    require(batch['cumulative_native_arc_positions'] == {'native': 538, 'random': 538}
            and batch['result_hashes'][1] == result['hash'] and result['attempt_hash'] == attempt['hash']
            and N.digest_bytes(raw) == attempt['response_sha256'], 'prior charged native lineage differs')
    parent = N.spatial_cell(N.validate_spatial_response(raw, 'f5b8619d', leaf_params(base['reference_cell']))['params'])
    require(parent == result['cells']['native'], 'parent identity does not match saved response')
    rows = inputs['rows']['search_fitness']
    demos = next(row['task']['train'] for row in rows if row['id'] == 'f5b8619d')
    feedback = feedback_from(parent, demos)
    root.mkdir()
    host = root / 'HOST_EVALUATION'
    host.mkdir()
    (host / 'runtime').mkdir()
    (root / 'AUTHORS').mkdir()
    N.write_once(root / 'SEARCH_FEEDBACK.json', N.signed({'schema': 'cellforge-four-author-search-feedback/1',
        'task_id': 'f5b8619d', 'parent_hash': parent['id'], 'previous_attempt_hash': attempt['hash'],
        'search_content_hash': N.C.digest(rows), 'demonstration_hash': N.C.digest(demos), 'feedback': feedback}))
    author_hashes = {}
    for slot in SLOTS:
        folder = root / 'AUTHORS' / slot
        folder.mkdir()
        packet, schema = compact(author_packet(slot, demos, parent, feedback)), compact(response_schema())
        payload = INSTRUCTION.encode() + b'\n\n<packet>\n' + packet + b'</packet>\n<response_schema>\n' + schema + b'</response_schema>\n'
        for name, data in [('PACKET.json', packet), ('RESPONSE_SCHEMA.json', schema), ('AUTHOR_INPUT.txt', payload)]:
            write_bytes(folder / name, data)
        author_hashes[slot] = {name: {'sha256': N.digest_bytes((folder / name).read_bytes()), 'bytes': (folder / name).stat().st_size}
                              for name in ('PACKET.json', 'RESPONSE_SCHEMA.json', 'AUTHOR_INPUT.txt')}
    sources = {}
    for name in FILES:
        data = Path(__file__).with_name(name).read_bytes()
        write_bytes(host / 'runtime' / name, data)
        sources[name] = N.digest_bytes(data)
    paths = [previous / 'HOST_EVALUATION' / name for name in ('PROTOCOL.json', 'BATCH_RESULT.json', 'RESULT2.json', 'ATTEMPT2.json')]
    paths += [previous / 'TEXT_ROUTE_RECEIPTS' / name for name in ('SOURCE.json', 'ATTEMPT2_RESPONSE.json')]
    body = {'schema': 'cellforge-four-author-spatial-protocol/1', 'base_root': str(base_root), 'base_protocol': base,
        'source_hashes': sources, 'bound_evidence': {str(path): N.digest_bytes(path.read_bytes()) for path in paths},
        'author_inputs': author_hashes, 'slots': list(SLOTS), 'maximum_submissions_per_slot': 1,
        'parent_cell': parent, 'feedback_sha256': N.digest_bytes((root / 'SEARCH_FEEDBACK.json').read_bytes()),
        'policy': POLICY, 'positions_per_arm_per_attempt': 167, 'free_positions': 103, 'fixed_positions': 64,
        'maximum_positions_per_arm': 668, 'previous_native_arc_positions': 538,
        'cumulative_positions_if_all_four_consumed': 1206, 'random_seeds': SEEDS,
        'random_distribution': '103 free literals independently uniform[-9,9]; 64 orientation tail literals fixed to -9, identical restrictions for native and random arms; all 167 charged',
        'diagnostic_arms': ['parent', 'hand_coded', 'zero'], 'diagnostic_constraint_exception': 'Unmodified historical/zero diagnostic tensors need not satisfy the new proposal vocabulary.',
        'selection_basis': 'Spatial family and task chosen using search demonstrations and prior search feedback; not independent generalization.',
        'ranking': 'descriptive best-of-four per arm: most development task successes, then search successes, then ascending cell ID; no automatic admission',
        'authoring_mode': 'parent-coordinated isolated native authors; no local model dispatcher',
        'credit_ceiling_owner_stated': 62000, 'credit_usage': None, 'maximum_author_calls': 4,
        'automatic_dispatch': False, 'automatic_retry': False, 'automatic_admission': False, 'report_accessed': False}
    N.write_once(host / 'PROTOCOL.json', N.signed(body))
    N.write_once(root / 'STANDARD.json', {'arms': {'llm': {}, 'random': {}}, 'budget_per_arm': 4,
        'seeds': list(SEEDS.values()), 'seeds_declared_before_run': True, 'niches': ['spatial_composition'],
        'founders': [], 'genomes': [], 'proposals': [], 'scientific_acceptance': False,
        'limitations': ['Prepared only; four author submissions pending', 'Host-defined spatial primitives and search-selected task',
                        'Single task/family; no crossover, router, broad milestone or independent generalization claim', 'Credit use unknown, not zero']})
    return preflight(root)


def context(root):
    root = Path(root).resolve()
    N.stop(root)
    protocol = N.read_signed(root / 'HOST_EVALUATION/PROTOCOL.json')
    N.verify_bound_evidence(protocol)
    for name, digest in protocol['source_hashes'].items():
        for path in (Path(__file__).with_name(name), root / 'HOST_EVALUATION/runtime' / name):
            require(N.digest_bytes(path.read_bytes()) == digest, 'use exact frozen batch source')
    for slot, entries in protocol['author_inputs'].items():
        for name, entry in entries.items():
            raw = (root / 'AUTHORS' / slot / name).read_bytes()
            require(N.digest_bytes(raw) == entry['sha256'] and len(raw) == entry['bytes'], 'frozen author payload changed')
    require(N.digest_bytes((root / 'SEARCH_FEEDBACK.json').read_bytes()) == protocol['feedback_sha256'], 'search feedback changed')
    _, _, base, inputs = N.context(protocol['base_root'])
    require(base == protocol['base_protocol'], 'frozen evaluator/data scope changed')
    return root, protocol, inputs


def preflight(root):
    root, protocol, inputs = context(root)
    demos = next(row['task']['train'] for row in inputs['rows']['search_fitness'] if row['id'] == 'f5b8619d')
    feedback = feedback_from(protocol['parent_cell'], demos)
    require(N.read_signed(root / 'SEARCH_FEEDBACK.json')['feedback'] == feedback, 'search-only feedback failed replay')
    for slot in SLOTS:
        folder = root / 'AUTHORS' / slot
        require((folder / 'PACKET.json').read_bytes() == compact(author_packet(slot, demos, protocol['parent_cell'], feedback)), 'nonallowlisted author context')
        require((folder / 'RESPONSE_SCHEMA.json').read_bytes() == compact(response_schema()), 'unexpected response vocabulary')
    consumed = sum((root / 'HOST_EVALUATION' / slot / 'ATTEMPT.json').exists() for slot in SLOTS)
    return {'status': 'preflight_verified', 'protocol_hash': protocol['hash'], 'author_inputs': protocol['author_inputs'],
        'consumed_attempts': consumed, 'positions_per_arm_consumed': consumed * 167, 'new_model_calls_by_host': 0,
        'credit_usage': None, 'report_accessed': False}


def parse_source(raw, expected_context_hash):
    origin = N.decode(raw)
    require(set(origin) == {'author_task_name', 'author_model', 'author_reasoning', 'route', 'context_sha256',
                           'context_provenance', 'response_provenance', 'credit_usage'}, 'exact supervisor source fields required')
    require(all(type(origin[k]) is str and 1 <= len(origin[k]) <= 500 for k in
                ('author_task_name', 'author_model', 'author_reasoning', 'route', 'context_provenance', 'response_provenance')),
            'parent-reported native author identity required')
    require(origin['credit_usage'] is None, 'credit usage must remain unknown without a meter')
    if origin['context_provenance'] == 'verbatim_frozen_payload_parent_attested':
        require(origin['context_sha256'] == expected_context_hash, 'actual payload differs from frozen context')
    else:
        require(origin['context_provenance'] == 'parent_transcription_exact_bytes_unknown'
                and origin['context_sha256'] is None, 'do not claim a reference hash as actual author context')
    return origin


def compute(root, protocol, inputs, slot, raw):
    parent = protocol['parent_cell']
    reference = leaf_params(parent)
    arms = {'parent': parent, 'hand_coded': protocol['base_protocol']['reference_cell'],
        'zero': N.spatial_cell(N.map_numbers(reference, lambda: 0.0)),
        'random': N.spatial_cell(random_params(reference, protocol['random_seeds'][slot]))}
    try:
        entry = validate(raw, reference)
        arms['native'] = N.spatial_cell(entry['params'])
        validation = {'status': 'valid', 'error': None}
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as error:
        validation = {'status': 'invalid_or_refused', 'error': type(error).__name__ + ': ' + str(error)[:300]}
    scores = {arm: {panel: N.score(cell, inputs['rows'][panel], root) for panel in
                   ('search_fitness', 'development_selection')} for arm, cell in arms.items()}
    return {'validation': validation, 'cells': arms, 'evaluations': scores,
        'counts': {arm: {panel: sum(row['correct'] is True for row in rows.values()) for panel, rows in panels.items()}
                   for arm, panels in scores.items()},
        'runtime_errors': {arm: sum(len(row['errors']) for rows in panels.values() for row in rows.values()) for arm, panels in scores.items()},
        'charged_positions': {'native': 167, 'random': 167}, 'parent_hashes': [parent['id']],
        'operation': 'complete_literal_replacement', 'report_accessed': False, 'scientific_acceptance': False,
        'automatic_admission': False, 'credit_usage': None}


def consume(root, slot, response_path, source_path):
    require(slot in SLOTS, 'one of four frozen author slots required')
    root, protocol, inputs = context(root)
    source_raw = Path(source_path).read_bytes()
    require(len(source_raw) <= 8192, 'bounded supervisor receipt required')
    origin = parse_source(source_raw, protocol['author_inputs'][slot]['AUTHOR_INPUT.txt']['sha256'])
    with Path(response_path).open('rb') as stream:
        raw = stream.read(N.MAX_RESPONSE + 1)
    folder = root / 'HOST_EVALUATION' / slot
    folder.mkdir()  # Partial/completed slots cannot be retried or overwritten.
    attempt = N.signed({'schema': 'cellforge-parallel-native-attempt/1', 'protocol_hash': protocol['hash'], 'slot': slot,
        'source': origin, 'source_sha256': N.digest_bytes(source_raw), 'response_sha256': N.digest_bytes(raw),
        'capture_complete': len(raw) <= N.MAX_RESPONSE, 'captured_bytes': len(raw),
        'charged_positions': {'native': 167, 'random': 167}, 'credit_usage': None,
        'random_seed': protocol['random_seeds'][slot], 'author_input_sha256': protocol['author_inputs'][slot]['AUTHOR_INPUT.txt']['sha256']})
    N.write_once(folder / 'ATTEMPT.json', attempt)
    write_bytes(folder / 'SOURCE.json', source_raw)
    write_bytes(folder / 'RESPONSE.bin', raw)
    result = N.signed(dict(compute(root, protocol, inputs, slot, raw), schema='cellforge-parallel-native-result/1',
                          protocol_hash=protocol['hash'], attempt_hash=attempt['hash'], slot=slot))
    N.write_once(folder / 'RESULT.json', result)
    return {'status': 'evaluated', 'slot': slot, 'result_hash': result['hash'], 'validation': result['validation'],
            'counts': result['counts'], 'runtime_errors': result['runtime_errors'], 'charged_positions': result['charged_positions']}


def verify(root, slot):
    require(slot in SLOTS, 'one of four frozen author slots required')
    root, protocol, inputs = context(root)
    folder = root / 'HOST_EVALUATION' / slot
    attempt, result = N.read_signed(folder / 'ATTEMPT.json'), N.read_signed(folder / 'RESULT.json')
    raw, source_raw = (folder / 'RESPONSE.bin').read_bytes(), (folder / 'SOURCE.json').read_bytes()
    require(attempt['slot'] == slot and attempt['protocol_hash'] == protocol['hash']
            and attempt['response_sha256'] == N.digest_bytes(raw) and attempt['source_sha256'] == N.digest_bytes(source_raw)
            and attempt['charged_positions'] == {'native': 167, 'random': 167}
            and attempt['random_seed'] == protocol['random_seeds'][slot], 'attempt capture/accounting changed')
    require(parse_source(source_raw, protocol['author_inputs'][slot]['AUTHOR_INPUT.txt']['sha256']) == attempt['source'], 'source identity changed')
    expected = N.signed(dict(compute(root, protocol, inputs, slot, raw), schema='cellforge-parallel-native-result/1',
                            protocol_hash=protocol['hash'], attempt_hash=attempt['hash'], slot=slot))
    require(expected == result, 'independent saved-tensor/control replay differs')
    return {'status': 'verified', 'slot': slot, 'result_hash': result['hash'], 'counts': result['counts'], 'runtime_errors': result['runtime_errors']}


def finalize(root):
    root, protocol, _ = context(root)
    results = []
    for slot in SLOTS:
        verify(root, slot)
        results.append(N.read_signed(root / 'HOST_EVALUATION' / slot / 'RESULT.json'))
    best = {}
    for arm in ('native', 'random'):
        eligible = [r for r in results if arm in r['cells'] and r['runtime_errors'][arm] == 0]
        ranked = sorted(eligible, key=lambda r: (-r['counts'][arm]['development_selection'], -r['counts'][arm]['search_fitness'], r['cells'][arm]['id']))
        best[arm] = {'slot': ranked[0]['slot'], 'cell_id': ranked[0]['cells'][arm]['id'], 'counts': ranked[0]['counts'][arm]} if ranked else None
    ids = [r['cells']['native']['id'] for r in results if 'native' in r['cells']]
    batch = N.signed({'schema': 'cellforge-four-author-result/1', 'protocol_hash': protocol['hash'],
        'result_hashes': [r['hash'] for r in results], 'best_per_arm': best, 'valid_candidate_ids': ids,
        'distinct_candidate_count': len(set(ids)), 'charged_positions': {'native': 668, 'random': 668},
        'cumulative_native_arc_positions': {'native': 1206, 'random': 1206}, 'credit_usage': None,
        'automatic_admission': False, 'scientific_acceptance': False, 'report_accessed': False})
    N.write_once(root / 'HOST_EVALUATION/BATCH_RESULT.json', batch)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight', 'import', 'verify', 'finalize'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--slot', choices=SLOTS)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.previous is None: parser.error('--previous is required')
        value = prepare(args.root, args.previous)
    elif args.command in ('import', 'verify'):
        if args.slot is None: parser.error('--slot is required')
        if args.command == 'import':
            if args.response is None or args.source is None: parser.error('--response and --source are required')
            value = consume(args.root, args.slot, args.response, args.source)
        else: value = verify(args.root, args.slot)
    else:
        value = (preflight if args.command == 'preflight' else finalize)(args.root)
    print(json.dumps(value, sort_keys=True))
