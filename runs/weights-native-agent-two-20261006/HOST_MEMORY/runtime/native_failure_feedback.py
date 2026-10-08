"""Bounded host adapter for one completed native proposal; never calls a model.

Only frozen search task files are opened. The original result is projected to
search once during preparation; callbacks bind its bytes without inspecting
development results. New evidence and prompts live in a separate directory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import evolution_manifest as E
import failure_journal as J
import native_proposal_import as N
import skillcore as S

FILES = tuple(dict.fromkeys((*N.SOURCE_FILES, 'failure_journal.py',
                            'evolution_manifest.py', 'native_failure_feedback.py')))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    raw = Path(path).read_bytes()
    require(N.digest_bytes(raw) == expected, 'immutable source bytes changed')
    return N.decode(raw)


def load_search(original):
    N.stop(original)
    host = original / 'HOST_EVALUATION'
    protocol = N.read_signed(host / 'PROTOCOL.json')
    for name, digest in protocol['request_hashes'].items():
        checked(original / name, digest)
    for name, digest in protocol['source_hashes'].items():
        for path in (host / 'runtime' / name, Path(__file__).with_name(name)):
            require(N.digest_bytes(path.read_bytes()) == digest, 'evaluator source changed')
    attempt = N.read_signed(host / 'ATTEMPT.json')
    require(attempt['protocol_hash'] == protocol['hash'] and attempt['attempt'] == 1
            and attempt['charged_positions'] == {'native': 102, 'random': 102}
            and attempt['capture_complete'], 'complete charged native attempt required')
    raw = (host / 'RESPONSE_CAPTURE.bin').read_bytes()
    require(N.digest_bytes(raw) == attempt['captured_response_sha256']
            and raw == (original / 'NATIVE_PROPOSAL.json').read_bytes(), 'proposal changed')
    entry = N.validate_response(raw, protocol['task_id'], protocol['reference_cell']['params'])
    candidate = S.make_cell('assoc_induct', entry['params'])
    fresh = checked(protocol['fresh_inputs']['manifest'], protocol['fresh_inputs']['sha256'])
    source = checked(fresh['source'], fresh['source_sha256'])
    rows = []
    require([item['id'] for item in source['panels']['search_fitness']]
            == protocol['partitions']['search_fitness'], 'original search order changed')
    for item in source['panels']['search_fitness']:
        path = Path(fresh['source']).parent / 'training' / (item['id'] + '.json')
        task = checked(path, item['sha256'])
        payload = path.read_bytes()
        require(hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
                == item['git_blob'], 'pinned upstream task changed')
        rows.append({'id': item['id'], 'task': task})
    packet = N.decode((original / 'AUTHORING_PACKET.json').read_bytes())
    task = next(row['task'] for row in rows if row['id'] == protocol['task_id'])
    require(packet['demonstrations'] == task['train'], 'original search demonstrations changed')
    return protocol, attempt, candidate, rows, packet


def observation(original, candidate, rows, task_id):
    scores = N.score(candidate, rows, original)
    task = next(row['task'] for row in rows if row['id'] == task_id)
    score = scores[task_id]
    category, error_type = 'correct', None
    if score['errors']:
        category = 'runtime_error'
        name = score['errors'][0]['type']
        error_type = name if name in J.RUNTIME_ERRORS else 'other'
    elif not score['correct']:
        category = 'mismatch_output_values'
        for pair in task['test']:
            actual = S.run(candidate, copy.deepcopy(pair['input']), copy.deepcopy(task['train']))
            if len(actual) != len(pair['output']) or any(len(a) != len(b) for a, b in zip(actual, pair['output'])):
                category = 'mismatch_output_shape'
                break
    # A deterministic first failing leave-one-out TRAINING example. Query grids
    # and labels never enter feedback. Demo failures are separate observations.
    demo = None
    for index, pair in enumerate(task['train']):
        context = copy.deepcopy(task['train'][:index] + task['train'][index + 1:])
        try:
            actual = S.run(candidate, copy.deepcopy(pair['input']), context)
            demo = J.demonstration(index, pair['input'], pair['output'], actual)
        except Exception:
            demo = J.demonstration(index, pair['input'], pair['output'], actual_status='runtime_error')
        if demo['actual'] != demo['expected']:
            break
    return scores, {'category': category, 'outcome': score['correct'], 'parent_outcome': None,
        'tensor_change': J.tensor_change(None, candidate['params']),
        'demonstration': demo, 'runtime_error_type': error_type}


def connect(original, destination):
    memory = destination / 'HOST_MEMORY'
    receipt = N.read_signed(memory / 'SEARCH_RECEIPT.json')
    protocol, attempt, candidate, rows, packet = load_search(original)
    status = E.read_manifest_status(memory)
    require(not status['sealed'], 'report-sealed memory cannot supply search feedback')
    manifest = status['manifest']
    for name in FILES:
        digest = manifest['source_hashes']['runtime:' + name]
        for path in (Path(__file__).with_name(name), memory / 'runtime' / name):
            require(N.digest_bytes(path.read_bytes()) == digest, 'memory runtime changed')
    reference = S.cell_reference(candidate)
    contract = {k: v for k, v in reference.items() if k not in ('weight_id', 'reference_hash')}
    template_hash = J.digest({'contract': contract, 'implementation': protocol['source_hashes']['skillcore.py']})
    scope = J.make_scope(manifest, rows, [template_hash])
    journal_attempt = J.make_attempt(scope, protocol['task_id'], template_hash=template_hash,
        candidate_hash=candidate['id'], parent_hashes=[], proposal_hash=attempt['captured_response_sha256'],
        attempt_hash=attempt['hash'], operator='founder', hypotheses=['representation_limit', 'wrong_mapping'])
    expected_request = {'scope': scope, 'attempt': journal_attempt}

    def verify(request):
        require(request == expected_request, 'candidate, ordered parents, attempt or scope differs')
        current_protocol, current_attempt, current_candidate, current_rows, _ = load_search(original)
        require((current_protocol, current_attempt, current_candidate, current_rows)
                == (protocol, attempt, candidate, rows), 'frozen host identities changed')
        current_status = E.read_manifest_status(memory)
        require(not current_status['sealed'] and current_status['manifest'] == manifest,
                'memory scope changed or sealed')
        require(N.read_signed(memory / 'SEARCH_RECEIPT.json') == receipt, 'search receipt changed')
        # Full terminal-result bytes are bound; callback never parses its other panels.
        require(N.digest_bytes((original / 'HOST_EVALUATION' / 'RESULT.json').read_bytes())
                == receipt['terminal_result_sha256'], 'terminal result changed')
        require(receipt['protocol_hash'] == protocol['hash'] and receipt['attempt_hash'] == attempt['hash']
                and receipt['candidate_hash'] == candidate['id'] and receipt['parent_hashes'] == []
                and receipt['search_content_hash'] == J.digest(rows), 'receipt identities differ')
        scores, observed = observation(original, candidate, rows, protocol['task_id'])
        require(scores == receipt['search_scores'] and observed == receipt['observation'],
                'search evidence differs on independent recomputation')
        return dict(observed, binding_hash=J.digest(request), evidence_hash=receipt['hash'], partition='search_fitness')

    return scope, journal_attempt, verify, candidate, packet


def retrieve(original, destination):
    scope, attempt, verify, candidate, packet = connect(original, destination)
    with J.FailureJournal(destination / 'HOST_MEMORY' / 'FAILURES.jsonl', scope) as journal:
        feedback = J.retrieve(scope, journal.records(), task_id=attempt['task_id'],
            template_hash=attempt['template_hash'], parent_hashes=[candidate['id']],
            verify=verify, max_records=1, max_prompt_bytes=8192)
        head = journal.head_hash
    require(len(feedback['record_hashes']) == 1 and not feedback['quarantined'], 'verified feedback unavailable')
    packet = copy.deepcopy(packet)
    packet['schema'] = 'cellforge-native-authoring-packet/2'
    packet['previous_candidate'] = {'id': candidate['id'], 'params': candidate['params']}
    packet['search_feedback'] = feedback['prompt']
    packet['feedback_method'] = 'First failing training demonstration predicted using all OTHER training demonstrations; query outputs omitted.'
    packet['hypothesis_caution'] = 'Representation limit and wrong mapping are competing unverified hypotheses. This observation does not establish whether different weights can solve the task.'
    packet['instruction'] += ' Propose one complete replacement of all 102 positions. Previous tensors and verified search feedback are supplied. Hypotheses are not established causes.'
    return packet, feedback, head


def prepare(original, destination):
    require(not destination.exists(), 'new immutable destination required')
    protocol, attempt, candidate, rows, _ = load_search(original)
    terminal = N.read_signed(original / 'HOST_EVALUATION' / 'RESULT.json')
    require(terminal['protocol_hash'] == protocol['hash'] and terminal['attempt_hash'] == attempt['hash']
            and terminal['validation']['status'] == 'valid' and terminal['cells']['native'] == candidate
            and terminal['report_accessed'] is False, 'completed native result required')
    scores, observed = observation(original, candidate, rows, protocol['task_id'])
    require(scores == terminal['evaluations']['native']['search_fitness'], 'saved search score fails replay')
    destination.mkdir()
    memory = destination / 'HOST_MEMORY'
    memory.mkdir()
    (memory / 'runtime').mkdir()
    sources = {'search-fitness-task-content': J.digest(rows), 'native-attempt': attempt['hash'],
               'native-protocol': protocol['hash'], 'native-result': terminal['hash']}
    for name in FILES:
        payload = Path(__file__).with_name(name).read_bytes()
        (memory / 'runtime' / name).write_bytes(payload)
        sources['runtime:' + name] = N.digest_bytes(payload)
    E.freeze_manifest(memory, {'purpose': 'post-terminal search-only feedback; no model calls',
        'candidate_id': candidate['id'], 'parent_ids': [], 'model_calls': 0}, sources,
        dict(protocol['partitions'], report=protocol['reserved_report_ids']))
    N.write_once(memory / 'SEARCH_RECEIPT.json', N.signed({'schema': 'cellforge-native-search-receipt/1',
        'protocol_hash': protocol['hash'], 'attempt_hash': attempt['hash'], 'candidate_hash': candidate['id'],
        'parent_hashes': [], 'search_content_hash': J.digest(rows), 'search_scores': scores,
        'terminal_result_sha256': N.digest_bytes((original / 'HOST_EVALUATION' / 'RESULT.json').read_bytes()),
        'observation': observed}))
    scope, journal_attempt, verify, _, _ = connect(original, destination)
    record = J.make_record(scope, journal_attempt, verify=verify)
    with J.FailureJournal(memory / 'FAILURES.jsonl', scope, create=True) as journal:
        journal.append(record, verify=verify)
    packet, feedback, head = retrieve(original, destination)
    N.write_once(destination / 'AUTHORING_PACKET_V2.json', packet)
    (destination / 'RESPONSE_SCHEMA.json').write_bytes((original / 'RESPONSE_SCHEMA.json').read_bytes())
    N.write_once(memory / 'VERIFIED_FEEDBACK.json', dict(feedback, journal_head=head))
    request = {'schema': 'cellforge-native-request-manifest/2', 'status': 'prepared; no author dispatched',
        'task_id': protocol['task_id'], 'partition': 'search_fitness', 'maximum_submissions': 1,
        'positions_per_proposal': 102, 'matched_random_positions_required': 102,
        'parent_hashes': [candidate['id']], 'credit_usage': None, 'cli_meter_receipt': None,
        'packet_sha256': N.digest_bytes((destination / 'AUTHORING_PACKET_V2.json').read_bytes()),
        'schema_sha256': N.digest_bytes((destination / 'RESPONSE_SCHEMA.json').read_bytes()),
        'original_run': str(original), 'journal_head': head, 'record_hashes': feedback['record_hashes'],
        'no_selection_or_report_feedback': True, 'automatic_dispatch': False, 'automatic_admission': False}
    N.write_once(destination / 'REQUEST_MANIFEST.json', N.signed(request))
    standard = N.decode((original / 'STANDARD.json').read_bytes())
    standard['limitations'][0] = 'Prepared one second native-agent request only; no second proposal or evaluation yet'
    N.write_once(destination / 'STANDARD.json', standard)
    return check(original, destination)


def check(original, destination):
    N.stop(destination)
    packet, feedback, head = retrieve(original, destination)
    require(packet == N.decode((destination / 'AUTHORING_PACKET_V2.json').read_bytes()), 'frozen packet changed')
    request = N.read_signed(destination / 'REQUEST_MANIFEST.json')
    checked(destination / 'AUTHORING_PACKET_V2.json', request['packet_sha256'])
    checked(destination / 'RESPONSE_SCHEMA.json', request['schema_sha256'])
    require((destination / 'RESPONSE_SCHEMA.json').read_bytes() == (original / 'RESPONSE_SCHEMA.json').read_bytes()
            and request['journal_head'] == head, 'response schema or journal head changed')
    require(N.decode((destination / 'HOST_MEMORY' / 'VERIFIED_FEEDBACK.json').read_bytes())
            == dict(feedback, journal_head=head), 'retrieved feedback changed')
    scope, attempt, callback, _, _ = connect(original, destination)
    rejected = []
    for field, value in [('candidate_hash', '0' * 64), ('parent_hashes', ['0' * 64]),
                         ('attempt_hash', '0' * 64), ('proposal_hash', '0' * 64)]:
        changed = copy.deepcopy(attempt)
        changed[field] = value
        try:
            callback({'scope': scope, 'attempt': changed})
        except ValueError:
            rejected.append(field)
    require(len(rejected) == 4, 'changed provenance was accepted')
    original_protocol = N.read_signed(original / 'HOST_EVALUATION' / 'PROTOCOL.json')
    serialized = json.dumps(packet)
    forbidden = original_protocol['partitions']['development_selection'] + original_protocol['reserved_report_ids']
    require(not any(task_id in serialized for task_id in forbidden), 'non-search task ID in author packet')
    return {'status': 'verified', 'packet': str(destination / 'AUTHORING_PACKET_V2.json'),
        'packet_sha256': request['packet_sha256'], 'journal_head': head, 'rejected_tampering': rejected,
        'feedback_bytes': len(feedback['prompt'].encode()), 'records': 1, 'new_model_calls': 0,
        'query_outputs_exported': False, 'selection_report_feedback': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify'))
    parser.add_argument('original', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = (prepare if args.command == 'prepare' else check)(args.original.resolve(), args.destination.resolve())
    print(json.dumps(result, sort_keys=True))
