"""Read-only search-packet diagnostic; no authoring, generation, or evaluation access.

The packet contains already-exposed search success labels. This is a feasibility
check, never evidence of learned routing or independent generalization.
"""
import argparse
import copy
import importlib.util
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def analyze(packet):
    ids = packet['expert_ids']
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError('exactly two distinct experts required')
    rows = packet['search_examples']
    if not rows or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('nonempty, uniquely identified search rows required')
    groups, counts, missed, margins = defaultdict(list), [0, 0], [], []
    oracle = identity = 0
    for row in rows:
        f, successes = row['features'], row['successful_expert_ids']
        if len(f) != 2 or any(type(x) not in (float, int) or not math.isfinite(x) for x in f):
            raise ValueError('two finite numeric features required; execution errors are not features')
        if len(set(successes)) != len(successes) or not set(successes) <= set(ids):
            raise ValueError('success labels must be a unique subset of roster')
        target = set(successes)
        pick = 0 if f[0] >= f[1] else 1  # Historical first-argmax tie rule.
        oracle += bool(target)
        identity += ids[pick] in target
        for i, expert in enumerate(ids):
            counts[i] += expert in target
        if target and ids[pick] not in target:
            missed.append(row['task_id'])
        if len(target) == 1:
            i = ids.index(next(iter(target)))
            margins.append(f[i] - f[1-i])
        groups[tuple(f)].append((row['task_id'], target))
    conflicts, repeated, ceiling = [], [], 0
    for features, group in groups.items():
        support = [sum(expert in targets for _, targets in group) for expert in ids]
        ceiling += max(support)
        if len(group) > 1:
            repeated.append([name for name, _ in group])
        solvable = [(name, targets) for name, targets in group if targets]
        if solvable and not set.intersection(*(targets for _, targets in solvable)):
            conflicts.append({'features': list(features), 'task_ids': [name for name, _ in solvable]})
    return {'tasks': len(rows), 'expert_ids': ids, 'single_expert_correct': counts,
            'oracle_correct': oracle, 'identity_correct': identity,
            'oracle_minus_identity': oracle - identity, 'identity_misses_solved_by_expert': missed,
            'exact_feature_groups': len(groups), 'repeated_feature_task_groups': repeated,
            'contradictory_feature_groups': conflicts,
            'feature_deterministic_ceiling': ceiling,
            'minimum_unique_success_margin': min(margins) if margins else None,
            'scope': 'Previously exposed search feedback only; no learned-router or qualification claim.'}


def verify_search_replay(root, runtime):
    """Verify only saved search artifacts and replay frozen typed tensor experts.

    No final/report/sealed files, generators, oracle functions, or run entrypoints
    are opened or invoked. Runtime exceptions abort rather than becoming losses.
    """
    def signed(path):
        value = json.loads(path.read_bytes())
        body = {k: v for k, v in value.items() if k != 'hash'}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
        if value.get('hash') != digest:
            raise ValueError('record hash mismatch: ' + path.name)
        return value

    protocol = signed(root / 'HOST_EVALUATION/PROTOCOL.json')
    roster = signed(root / 'HOST_EVALUATION/ROSTER.json')
    receipt = signed(root / 'AUTHORING/router_founder.json')
    raw = (root / 'AUTHORING/router_founder.txt').read_bytes()
    search_raw = (root / 'MIXED_SEARCH.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() != receipt['payload_sha256'] or
            hashlib.sha256(search_raw).hexdigest() != protocol['file_hashes']['MIXED_SEARCH.json'] or
            receipt['protocol_hash'] != protocol['hash'] or
            roster['protocol_hash'] != protocol['hash'] or
            receipt['roster_hash'] != roster['hash'] or roster['development_used'] is not False):
        raise ValueError('search artifact binding mismatch')
    if hashlib.sha256(runtime.read_bytes()).hexdigest() != protocol['source_hashes']['skillcore.py']:
        raise ValueError('frozen runtime hash mismatch')
    packet = json.loads(raw.split(b'\n', 1)[1])
    cells = [protocol['column'], roster['native_specialist']['genome']]
    if [c['id'] for c in cells] != packet['expert_ids'] or roster['expert_ids'] != packet['expert_ids']:
        raise ValueError('expert roster mismatch')
    for cell in cells:
        if (cell['kind'] != 'chain' or len(cell['parts']) != 2 or
                [p.get('template') for p in cell['parts']] != ['ray_cast', 'tile_orient']):
            raise ValueError('only frozen typed spatial pair is supported')
    spec = importlib.util.spec_from_file_location('_headroom_frozen_skillcore', runtime)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    examples = []
    for row in json.loads(search_raw):
        task = row['task']
        demos = task['train']
        if not demos or not task['test']:
            raise ValueError('nonempty demonstrations and queries required')
        features, successes = [], []
        for cell in cells:
            total = 0.0
            for i, pair in enumerate(demos):
                actual = core.np.asarray(core.run(cell, copy.deepcopy(pair['input']),
                                                 copy.deepcopy(demos[:i] + demos[i+1:])))
                expected = core.np.asarray(pair['output'])
                total += 1.0 if actual.shape != expected.shape else float(core.np.mean(actual != expected))
            features.append(-total / len(demos))
            # Query labels are used only after prediction, never passed to run.
            predictions = [core.run(cell, copy.deepcopy(q['input']), copy.deepcopy(demos))
                           for q in task['test']]
            if predictions == [q['output'] for q in task['test']]:
                successes.append(cell['id'])
        examples.append({'task_id': row['id'], 'features': features,
                         'successful_expert_ids': successes})
    if examples != packet['search_examples']:
        raise ValueError('frozen replay differs from exposed author packet')
    return {'search_replay_matches_packet': True, 'runtime_errors': 0,
            'protocol_hash': protocol['hash'], 'roster_hash': roster['hash'],
            'search_sha256': hashlib.sha256(search_raw).hexdigest(),
            'runtime_sha256': hashlib.sha256(runtime.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path, help='existing AUTHORING/router_founder.txt')
    parser.add_argument('--verify-search-root', type=Path)
    parser.add_argument('--frozen-runtime', type=Path)
    args = parser.parse_args()
    if bool(args.verify_search_root) != bool(args.frozen_runtime):
        parser.error('verification requires both search root and frozen runtime')
    raw = args.packet.read_bytes()
    # Never execute the packet instruction; consume its JSON as data only.
    packet = json.loads(raw.split(b'\n', 1)[1])
    result = analyze(packet)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    if args.verify_search_root:
        if args.packet.resolve() != (args.verify_search_root / 'AUTHORING/router_founder.txt').resolve():
            parser.error('packet must belong to verification root')
        result.update(verify_search_replay(args.verify_search_root, args.frozen_runtime))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
