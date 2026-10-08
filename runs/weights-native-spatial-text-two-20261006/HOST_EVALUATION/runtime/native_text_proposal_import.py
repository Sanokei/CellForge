"""Two already-authored parent-transcribed responses; no model or author executor.

Uses the frozen spatial evaluator while keeping an explicit text-route ledger.
The actual compacted author prompt/feedback bytes are unavailable: their hashes
remain null, and reference packet hashes never masquerade as actual context.
"""
import argparse
import copy
from pathlib import Path
import json

import native_proposal_import as N

FILES = (*N.SOURCE_FILES, 'native_text_proposal_import.py')
SEEDS = (2026100605, 2026100606)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(root):
    receipts = root / 'TEXT_ROUTE_RECEIPTS'
    value = N.decode((receipts / 'SOURCE.json').read_bytes())
    require(value['author_task_name'] == '/root/author_spatial_weights_text'
            and value['author_cloud_uuid'] is None and value['author_turn_uuid'] is None
            and value['actual_author_context_sha256'] is None
            and value['actual_author_context_bytes'] is None
            and value['reference_context_is_actual_author_context'] is False
            and value['authoring_credit_usage'] is None,
            'parent-reported native task identity and unknown actual context/credits required')
    require(len(value['responses']) == 2, 'exactly two parent-supplied responses required')
    raw = []
    for ordinal, item in enumerate(value['responses'], 1):
        require(item['ordinal'] == ordinal and item['file'] == f'ATTEMPT{ordinal}_RESPONSE.json',
                'fixed ordered response filenames required')
        data = (receipts / item['file']).read_bytes()
        require(len(data) == item['bytes'] and N.digest_bytes(data) == item['sha256'],
                'exact parent-supplied canonical response changed')
        raw.append(data)
    return value, raw


def prepare(root, original):
    root, original = Path(root).resolve(), Path(original).resolve()
    N.stop(root)
    _, _, base, _ = N.context(original)
    require(base['response_kind'] == N.SPATIAL_KIND and N.charged_positions(base) == {'native': 167, 'random': 167},
            'frozen two-leaf spatial protocol required')
    origin, _ = source(root)
    receipts = root / 'TEXT_ROUTE_RECEIPTS'
    for name in ('AUTHORING_PACKET_CANDIDATE.json', 'RESPONSE_SCHEMA_CANDIDATE.json'):
        require((receipts / ('REFERENCE_' + name)).read_bytes() == (original / name).read_bytes(),
                'reference input differs from frozen spatial protocol')
    sources = {}
    host = root / 'HOST_EVALUATION'
    host.mkdir()
    (host / 'runtime').mkdir()
    for name in FILES:
        path = Path(__file__).with_name(name)
        payload = path.read_bytes()
        if name in base['source_hashes']:
            require(N.digest_bytes(payload) == base['source_hashes'][name], 'use the exact frozen spatial source')
        sources[name] = N.digest_bytes(payload)
        (host / 'runtime' / name).write_bytes(payload)
    receipt_files = ['SOURCE.json', 'ATTEMPT1_RESPONSE.json', 'ATTEMPT2_RESPONSE.json',
        'REFERENCE_AUTHORING_PACKET_CANDIDATE.json', 'REFERENCE_RESPONSE_SCHEMA_CANDIDATE.json',
        'REFERENCE_TEXT_AUTHOR_CONTEXT.json']
    body = {'schema': 'cellforge-native-two-text-attempts/1', 'maximum_submissions': 2,
        'original_spatial_root': str(original), 'base_protocol': base,
        'source_hashes': sources, 'receipt_hashes': {name: N.digest_bytes((receipts / name).read_bytes()) for name in receipt_files},
        'authoring_route': origin['route'], 'author_task_name': origin['author_task_name'],
        'actual_author_context_sha256': None, 'revision_feedback_sha256': None,
        'random_seeds': list(SEEDS), 'positions_per_arm_per_attempt': 167,
        'previous_native_arc_positions': {'native': 204, 'random': 204},
        'text_batch_positions_after_both_attempts': {'native': 334, 'random': 334},
        'cumulative_native_arc_positions_after_both_attempts': {'native': 538, 'random': 538},
        'credits': None, 'source_provenance': 'parent canonical transcription; exact platform bytes unavailable',
        'author_feedback': 'static schema/semantic feedback only, according to parent; no evaluated outcomes',
        'scientific_acceptance': False, 'automatic_admission': False, 'new_model_calls': 0}
    N.write_once(host / 'PROTOCOL.json', N.signed(body))
    return {'status': 'prepared', 'protocol_hash': N.C.digest(body), 'attempts': 2, 'positions_per_arm_per_attempt': 167}


def context(root):
    root = Path(root).resolve()
    N.stop(root)
    host = root / 'HOST_EVALUATION'
    protocol = N.read_signed(host / 'PROTOCOL.json')
    for name, digest in protocol['source_hashes'].items():
        for path in (host / 'runtime' / name, Path(__file__).with_name(name)):
            require(N.digest_bytes(path.read_bytes()) == digest, 'frozen text-route source changed')
    for name, digest in protocol['receipt_hashes'].items():
        require(N.digest_bytes((root / 'TEXT_ROUTE_RECEIPTS' / name).read_bytes()) == digest, 'text-route source receipt changed')
    origin, raw = source(root)
    _, _, base, inputs = N.context(protocol['original_spatial_root'])
    require(base == protocol['base_protocol'], 'original spatial protocol changed')
    return root, host, protocol, origin, raw, inputs


def effective_protocol(protocol, ordinal):
    require(ordinal in (1, 2), 'bounded ordinal required')
    base = copy.deepcopy(protocol['base_protocol'])
    base.update(random_seed=protocol['random_seeds'][ordinal - 1], native_sequence_number=ordinal + 2,
        cumulative_positions_after_attempt={arm: 204 + 167 * ordinal for arm in ('native', 'random')})
    return base


def attempt_receipt(protocol, origin, raw, ordinal, previous_hash):
    return N.signed({'schema': 'cellforge-native-text-attempt/1', 'protocol_hash': protocol['hash'],
        'attempt': ordinal, 'previous_attempt_hash': previous_hash,
        'author_task_name': origin['author_task_name'], 'author_cloud_uuid': None, 'author_turn_uuid': None,
        'author_model': origin['author_model'], 'author_reasoning': origin['author_reasoning'],
        'authoring_route': origin['route'], 'identity_evidence': 'parent report',
        'response_sha256': N.digest_bytes(raw), 'response_bytes': len(raw),
        'source_receipt_sha256': protocol['receipt_hashes']['SOURCE.json'],
        'actual_author_context_sha256': None, 'revision_feedback_sha256': None,
        'charged_positions': {'native': 167, 'random': 167}, 'authoring_credit_usage': None,
        'random_seed': protocol['random_seeds'][ordinal - 1], 'cli_meter_receipt': None})


def result_for(root, protocol, inputs, raw, attempt):
    value = N.compute(root, effective_protocol(protocol, attempt['attempt']), inputs, raw)
    return N.signed(dict(value, schema='cellforge-native-text-result/1', attempt_hash=attempt['hash'],
                         protocol_hash=protocol['hash'], text_attempt=attempt['attempt']))


def summary(protocol, attempts, results):
    return N.signed({'schema': 'cellforge-native-text-batch-result/1', 'protocol_hash': protocol['hash'],
        'attempt_hashes': [item['hash'] for item in attempts], 'result_hashes': [item['hash'] for item in results],
        'validation': [item['validation'] for item in results], 'counts': [item['counts'] for item in results],
        'runtime_errors': [item['runtime_errors'] for item in results],
        'charged_positions': {'native': 334, 'random': 334},
        'cumulative_native_arc_positions': {'native': 538, 'random': 538},
        'credits': None, 'actual_author_context_sha256': None, 'new_model_calls': 0,
        'report_accessed': False, 'scientific_acceptance': False, 'automatic_admission': False})


def consume(root):
    root, host, protocol, origin, responses, inputs = context(root)
    if any((host / name).exists() for name in ('ATTEMPT1.json', 'ATTEMPT2.json', 'RESULT1.json', 'RESULT2.json', 'BATCH_RESULT.json')):
        raise FileExistsError('consumed or partial text batch must not be retried/overwritten')
    attempts, results, previous = [], [], '0' * 64
    for ordinal, raw in enumerate(responses, 1):
        N.stop(root)
        attempt = attempt_receipt(protocol, origin, raw, ordinal, previous)
        N.write_once(host / f'ATTEMPT{ordinal}.json', attempt)  # Consume even invalid response before parsing.
        result = result_for(root, protocol, inputs, raw, attempt)
        N.write_once(host / f'RESULT{ordinal}.json', result)
        attempts.append(attempt)
        results.append(result)
        previous = attempt['hash']
    value = summary(protocol, attempts, results)
    N.write_once(host / 'BATCH_RESULT.json', value)
    return dict(value, status='evaluated')


def verify(root):
    root, host, protocol, origin, responses, inputs = context(root)
    attempts, results, previous = [], [], '0' * 64
    for ordinal, raw in enumerate(responses, 1):
        attempt = attempt_receipt(protocol, origin, raw, ordinal, previous)
        require(attempt == N.read_signed(host / f'ATTEMPT{ordinal}.json'), 'attempt/source/accounting binding differs')
        result = result_for(root, protocol, inputs, raw, attempt)
        require(result == N.read_signed(host / f'RESULT{ordinal}.json'), 'independent tensor/control replay differs')
        attempts.append(attempt)
        results.append(result)
        previous = attempt['hash']
    expected = summary(protocol, attempts, results)
    require(expected == N.read_signed(host / 'BATCH_RESULT.json'), 'two-attempt summary differs')
    return dict(expected, status='verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'import', 'verify'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--spatial-source', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.spatial_source is None:
            parser.error('--spatial-source is required')
        result = prepare(args.root, args.spatial_source)
    else:
        result = (consume if args.command == 'import' else verify)(args.root)
    print(json.dumps(result, sort_keys=True))
