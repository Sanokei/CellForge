"""One finite two-specialist typed-bank experiment, without service admission.

No model dispatcher; no specialist_runtime change. Every forward exception is
explicit. Search selects both arms before the one-time sealed final opening.
"""
import argparse
import copy
import json
from pathlib import Path
import random

import native_transfer_curriculum as T
import router_evolution as R

N, P, S = T.N, T.P, T.N.S
H_TASK = 'synthetic-horizontal-mirror-v1'
SLOTS = ('specialist_founder', 'specialist_repair', 'router_founder', 'router_repair')
SEEDS = dict(zip(SLOTS, range(20261006411, 20261006415)))
FILES = (*T.FILES, 'router_evolution.py', 'native_composition_batch.py')
SEALED = 'HOST_EVALUATION/MIXED_DEVELOPMENT_SEALED.json'
COLUMN_ID = '778d0678b19cd5d5723db1e988d22dca35182719e210c2b5a0afb4e45064aa34'
IDENTITY = {'route.W': [[1, 0], [0, 1]], 'route.b': [0, 0]}
ZERO = {'route.W': [[0, 0], [0, 0]], 'route.b': [0, 0]}
ALWAYS_H = {'route.W': [[0, 0], [0, 0]], 'route.b': [0, 1]}


def h_reference():
    p = T.reference_params()
    p['ray_cast']['dirs'] = [-9, -9, 9, -9, -9, -9, 9, -9]
    for i, j, k in ((0, 0, 0), (0, 1, 0), (1, 0, 2), (1, 1, 2)):
        p['tile_orient']['orient'][i][j] = [9 if n == k else -9 for n in range(8)]
    return p


def oracle(family, grid):
    if family == 'C':
        return T.oracle(grid)
    P.require(family == 'H', 'unknown synthetic family')
    w = len(grid[0])
    filled = []
    for row in grid:
        values = [v for v in row if v]
        P.require(len(values) <= 1, 'at most one marker per row')
        filled.append([values[0] if values else 0] * w)
    return [row + row for row in filled] + [row + row for row in reversed(filled)]


def fixture_check(column):
    grid = [[0, 2, 0], [4, 0, 0]]  # Dimensions absent from every new panel.
    P.require(oracle('H', grid) == [[2] * 6, [4] * 6, [4] * 6, [2] * 6], 'manual H oracle fixture')
    cells = [column, N.spatial_cell(h_reference())]
    for family, cell in zip(('C', 'H'), cells):
        P.require(S.run(cell, grid, []) == oracle(family, grid), 'separate reference fixture mismatch')
    return {'manual_horizontal_oracle': True, 'both_reference_fixtures': True, 'new_panel_predictions': 0}


def draw(rng, size, colors, used):
    h, w = size
    n = rng.choice((2, 3))
    for _ in range(3000):
        grid = [[0] * w for _ in range(h)]
        for r, c, color in zip(rng.sample(range(h), n), rng.sample(range(w), n), rng.sample(list(colors), n)):
            grid[r][c] = color
        key = T.geometry(grid)
        if key not in used:
            used.add(key)
            return grid
    raise ValueError('bounded structural generation exhausted; do not tune to scores')


def generate(excluded):
    used = set(excluded)
    small = ((4, 5), (5, 4), (5, 6), (6, 5), (4, 6), (6, 4))
    large = ((7, 8), (8, 7), (8, 9), (9, 8), (9, 10), (10, 9))
    panels = {}
    for name, seed, count in (('h_demos', 20261006401, 6), ('h_search', 20261006402, 12)):
        rng = random.Random(seed)
        panels[name] = [{'id': f'{name}-{i}', 'family': 'H', 'input': draw(rng, small[i % 6], (2, 4, 5), used)} for i in range(count)]
    def task(rng, tid, family, sizes, palette, index, stratum):
        grids = [draw(rng, sizes[(index + k) % len(sizes)], palette, used) for k in range(5)]
        return {'id': tid, 'family': family, 'stratum': stratum, 'train_inputs': grids[:3], 'query_inputs': grids[3:]}
    rng = random.Random(20261006403)
    panels['mixed_search'] = [task(rng, f'mixed-search-{i}', 'CH'[i % 2], small, (2, 4, 5), i, 'search') for i in range(12)]
    rng, dev = random.Random(20261006404), []
    for i in range(8):
        dev.append(task(rng, f'color-{i}', 'CH'[i % 2], small, (1, 3, 6, 7, 8, 9), i, 'unseen_color'))
    for i in range(4):
        base = task(rng, f'perm-{i}-base', 'CH'[i % 2], small, (2, 4, 5), i, 'permutation')
        permutation = [0] + [1 + (c + i) % 9 for c in range(1, 10)]
        mapped = copy.deepcopy(base)
        mapped.update(id=f'perm-{i}-mapped', permutation=permutation)
        for key in ('train_inputs', 'query_inputs'):
            mapped[key] = [[[permutation[v] for v in row] for row in grid] for grid in base[key]]
        dev += [base, mapped]
    for i in range(8):
        dev.append(task(rng, f'size-{i}', 'CH'[i % 2], large, (2, 4, 5), i, 'larger_rectangle'))
    panels['development'] = dev
    return panels


def label_task(row):
    return {'id': row['id'], 'family': row['family'], 'stratum': row['stratum'],
            'task': {'train': [{'input': g, 'output': oracle(row['family'], g)} for g in row['train_inputs']],
                     'test': [{'input': g, 'output': oracle(row['family'], g)} for g in row['query_inputs']]}}


def h_schema():
    value = T.schema()
    value['items']['properties']['task_id']['const'] = H_TASK
    return value


def h_text(demos, parent=None, failure=None):
    text = T.author_text(demos, parent, failure)
    return text.replace(T.TASK.encode(), H_TASK.encode())


def prepare(root, previous):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    P.require(not root.exists(), 'new batch root required')
    result = N.read_signed(previous / 'HOST_EVALUATION/founder_a/RESULT.json')
    terminal = N.read_signed(previous / 'HOST_EVALUATION/TERMINAL_STATUS.json')
    column = result['cells']['native']
    P.require(column['id'] == COLUMN_ID and terminal['transfer_criterion_passed'], 'approved column specialist differs')
    checks = fixture_check(column)
    # Old inputs only exclude duplicates. No prior development labels or scores select new candidates.
    paths = [previous / 'SEARCH_DEMONSTRATIONS.json', previous / 'SEARCH_FITNESS.json', previous / T.SEALED]
    excluded = {T.geometry(row['input']) for path in paths for row in N.decode(path.read_bytes())}
    panels = generate(excluded)
    demos = [{'input': row['input'], 'output': oracle('H', row['input'])} for row in panels['h_demos']]
    search = [dict(row, output=oracle('H', row['input'])) for row in panels['h_search']]
    mixed = [label_task(row) for row in panels['mixed_search']]
    root.mkdir()
    (root / 'AUTHORING').mkdir()
    (root / 'HOST_EVALUATION/runtime').mkdir(parents=True)
    files = {'AUTHORING/HORIZONTAL_FOUNDER.txt': h_text(demos), 'AUTHORING/HORIZONTAL_SCHEMA.json': P.compact(h_schema()),
             'H_DEMONSTRATIONS.json': P.compact(demos), 'H_SEARCH.json': P.compact(search),
             'MIXED_SEARCH.json': P.compact(mixed), SEALED: P.compact(panels['development'])}
    hashes = {}
    for name, raw in files.items():
        P.write_bytes(root / name, raw)
        hashes[name] = N.digest_bytes(raw)
    source_hashes = {}
    for name in FILES:
        raw = Path(__file__).with_name(name).read_bytes()
        P.write_bytes(root / 'HOST_EVALUATION/runtime' / name, raw)
        source_hashes[name] = N.digest_bytes(raw)
    bound = paths + [previous / 'HOST_EVALUATION/founder_a/RESULT.json', previous / 'HOST_EVALUATION/TERMINAL_STATUS.json']
    protocol = N.signed({'schema': 'cellforge-native-composition/1', 'column': column, 'file_hashes': hashes,
        'source_hashes': source_hashes, 'bound_evidence': {str(p): N.digest_bytes(p.read_bytes()) for p in bound},
        'fixture_checks': checks, 'random_seeds': SEEDS, 'panel_seeds': [20261006401, 20261006402, 20261006403, 20261006404],
        'counts': {'h_demonstrations': 6, 'h_search': 12, 'mixed_search': 12, 'mixed_development': 24,
                   'demonstrations_per_mixed_task': 3, 'queries_per_mixed_task': 2},
        'max_completions': 4, 'charges': {'specialist_attempt': 167, 'router_attempt': 6, 'maximum_per_arm': 346},
        'specialist_constraints': '103 free values uniform[-9,9] in random control, 64 fixed orientation tails -9; all167 charged.',
        'router_constraints': 'All6 W[2,2]/b[2] literals finite in[-9,9]; matched random all6 uniform[-9,9].',
        'repair_rule': 'At most one full replacement of a valid search-failing founder in each stage. Invalid and perfect founders receive none.',
        'selection': 'Both native and random champions: greatest search exact task count, then ascending genome hash. Never development.',
        'specialist_gate': 'Selected horizontal specialist12/12 H search with zero errors; otherwise no router authoring.',
        'routing_features': 'Only negative mean leave-one-out demonstration error, aligned to immutable [column,horizontal] roster. All forward exceptions are explicit errors, never numerical feature values.',
        'baselines': ['single_C', 'single_H', 'zero_router', 'always_H', 'fixed_identity', 'reference_bank', 'zero_H_identity', 'each_random_H_identity', 'each_matched_random_router'],
        'acceptance': {'native_search_tasks': 12, 'native_final_tasks': 24, 'minimum_margin_over_best_single': 6,
                       'minimum_margin_over_every_matched_random_router': 6, 'runtime_errors': 0, 'reference_final_tasks': 24,
                       'permutation_pairs': 4, 'fixed_identity': 'report separately; matching it is not learned-router superiority'},
        'development_policy': 'Freeze all task inputs and family/oracle source now; materialize final labels once after all search-only choices. No final feedback or retry.',
        'service_boundary': 'Typed-bank experiment only. specialist_runtime ray_cast rejection/resource guard unchanged.',
        'automatic_dispatch': False, 'automatic_admission': False, 'credit_usage': None})
    N.write_once(root / 'HOST_EVALUATION/PROTOCOL.json', protocol)
    N.write_once(root / 'STANDARD.json', {'arms': {'llm': {}, 'random': {}}, 'budget_per_arm': 4,
        'seeds': list(SEEDS.values()), 'seeds_declared_before_run': True, 'niches': ['synthetic_composition'],
        'founders': [], 'genomes': [], 'proposals': [], 'scientific_acceptance': False,
        'limitations': ['Prospective bounded experiment; not a completed ONE_STANDARD ledger. Operational charges are explicit attempt receipts.']})
    return preflight(root)


def context(root):
    root = Path(root).resolve()
    N.stop(root)
    protocol = N.read_signed(root / 'HOST_EVALUATION/PROTOCOL.json')
    for name, digest in protocol['source_hashes'].items():
        for path in (Path(__file__).with_name(name), root / 'HOST_EVALUATION/runtime' / name):
            P.require(N.digest_bytes(path.read_bytes()) == digest, 'frozen source changed')
    for name, digest in protocol['file_hashes'].items():
        if name != SEALED:
            P.require(N.digest_bytes((root / name).read_bytes()) == digest, 'frozen search artifact changed')
    N.verify_bound_evidence(protocol)
    load = lambda name: N.decode((root / name).read_bytes())
    return root, protocol, load('H_DEMONSTRATIONS.json'), load('H_SEARCH.json'), load('MIXED_SEARCH.json')


def preflight(root):
    root, p, demos, _, _ = context(root)
    raw = (root / 'AUTHORING/HORIZONTAL_FOUNDER.txt').read_bytes()
    P.require(raw == h_text(demos), 'founder packet is not allowlisted')
    return {'status': 'ready', 'protocol_hash': p['hash'], 'horizontal_payload_sha256': N.digest_bytes(raw),
            'horizontal_payload_bytes': len(raw), 'development_opened': (root / 'HOST_EVALUATION/FINAL_OPEN.json').exists()}


def open_authoring(root):
    P.require(not any((root / 'HOST_EVALUATION' / n).exists() for n in ('SEARCH_SELECTION.json', 'FINAL_OPEN.json', 'TERMINAL_REJECTION.json')), 'authoring closed')


def strict_features(cells, task):
    errors = []
    demos = task['train']
    P.require(bool(demos), 'routing needs visible demonstrations')
    for cell in cells:
        total = 0.0
        for i, pair in enumerate(demos):
            # Exceptions propagate to the explicit row error; never convert them to 1.
            actual = S.np.asarray(S.run(cell, copy.deepcopy(pair['input']), copy.deepcopy(demos[:i] + demos[i + 1:])))
            expected = S.np.asarray(pair['output'])
            total += 1.0 if actual.shape != expected.shape else float(S.np.mean(actual != expected))
        errors.append(-total / len(demos))
    return errors


def bank_score(cells, params, rows, root):
    bank, outcomes = S.export_bank(cells, params), []
    for row in rows:
        N.stop(root)
        clean = {'train': copy.deepcopy(row['task']['train']),
                 'test': [{'input': copy.deepcopy(q['input'])} for q in row['task']['test']]}
        failures, original_run = [], S.run
        def watched_run(*args, **kwargs):
            try:
                return original_run(*args, **kwargs)
            except Exception as exc:
                failures.append(type(exc).__name__ + ': ' + str(exc)[:200])
                raise
        # Serial, bounded host only. Capture even exceptions the legacy demo
        # scorer catches internally; restore the unchanged core function always.
        S.run = watched_run
        try:
            f = strict_features(cells, clean)
            index = int(S.np.argmax(S.np.asarray(params['route.W']) @ S.np.asarray(f) + S.np.asarray(params['route.b'])))
            actual = S.predict_bank(bank, clean)
            P.require(not failures, 'expert runtime failure: ' + '; '.join(failures[:3]))
            P.require(all(type(guesses) is list and len(guesses) == 1 for guesses in actual), 'one prediction per query required')
            predictions = [guesses[0] for guesses in actual]
            error = None
            correct = predictions == [q['output'] for q in row['task']['test']]
        except Exception as exc:
            f, index, predictions, correct = None, None, None, None
            error = type(exc).__name__ + ': ' + str(exc)[:300]
        finally:
            S.run = original_run
        outcomes.append({'id': row['id'], 'correct': correct, 'features': f, 'pick': index, 'predictions': predictions, 'error': error})
    return {'correct': sum(r['correct'] is True for r in outcomes), 'errors': sum(r['error'] is not None for r in outcomes), 'rows': outcomes}


def is_router(slot):
    return slot.startswith('router_')


def needs_repair(result):
    return result['validation']['status'] == 'valid' and result['scores']['native']['correct'] < 12


def stage_results(root, stage):
    founder = verify(root, stage + '_founder')
    results = [founder]
    repair = root / 'HOST_EVALUATION' / (stage + '_repair')
    P.require(needs_repair(founder) == repair.exists(), 'eligible replacement must finish; ineligible replacement forbidden')
    if repair.exists(): results.append(verify(root, stage + '_repair'))
    return results


def best(results, arm):
    ranked = sorted((r for r in results if arm in r['genomes'] and r['scores'][arm]['errors'] == 0),
                    key=lambda r: (-r['scores'][arm]['correct'], r['genomes'][arm].get('id', r['genomes'][arm].get('hash'))))
    if not ranked: return None
    r = ranked[0]
    return {'slot': r['slot'], 'genome': r['genomes'][arm], 'search_correct': r['scores'][arm]['correct'], 'result_hash': r['hash']}


def roster(root):
    saved = N.read_signed(root / 'HOST_EVALUATION/ROSTER.json')
    results = stage_results(root, 'specialist')
    chosen = best(results, 'native')
    P.require(chosen and chosen['search_correct'] == 12 and saved['native_specialist'] == chosen, 'horizontal roster binding changed')
    _, p, _, _, _ = context(root)
    P.require(saved['protocol_hash'] == p['hash'] and saved['expert_ids'] == [p['column']['id'], chosen['genome']['id']], 'expert order changed')
    return saved, [p['column'], chosen['genome']]


def router_text(root, cells, mixed, parent=None, failure=None):
    examples = []
    for row in mixed:
        features = strict_features(cells, row['task'])
        successes = []
        for cell in cells:
            predictions = [S.run(cell, copy.deepcopy(q['input']), copy.deepcopy(row['task']['train'])) for q in row['task']['test']]
            if predictions == [q['output'] for q in row['task']['test']]: successes.append(cell['id'])
        examples.append({'task_id': row['id'], 'features': features, 'successful_expert_ids': successes})
    ids = [c['id'] for c in cells]
    body = {'expert_ids': ids, 'features': 'negative mean leave-one-out demonstration pixel error; shape mismatch=1; runtime failures abort authoring',
            'search_examples': examples, 'shapes': {'route.W': [2, 2], 'route.b': [2]},
            'instruction': 'Scores=W@features+b; first argmax chooses expert. Author one shared literal router; all6 values finite in[-9,9]. Never change experts/order/features. Return exactly [{"task_id":"router","params":{"route.W":[[number,number],[number,number]],"route.b":[number,number]},"rationale":"text"}]. No tools/code/extra keys. Search successful IDs are author feedback only, never inference inputs.'}
    if parent is not None: body.update(parent=parent, first_search_failure=failure)
    return b'Use only this payload. No inherited context, tools or external sources.\n' + P.compact(body)


def packet(root, slot):
    if slot == 'specialist_founder': return (root / 'AUTHORING/HORIZONTAL_FOUNDER.txt').read_bytes()
    receipt = N.read_signed(root / 'AUTHORING' / (slot + '.json'))
    raw = (root / 'AUTHORING' / (slot + '.txt')).read_bytes()
    P.require(receipt['payload_sha256'] == N.digest_bytes(raw), 'staged payload changed')
    _, p, demos, search, mixed = context(root)
    if slot == 'router_founder':
        _, cells = roster(root)
        expected = router_text(root, cells, mixed)
    else:
        parent = verify(root, slot.replace('_repair', '_founder'))
        P.require(needs_repair(parent) and receipt['parent_result_hash'] == parent['hash'], 'repair no longer eligible')
        failed = next(r for r in parent['scores']['native']['rows'] if r['correct'] is not True)
        if is_router(slot):
            _, cells = roster(root)
            expected = router_text(root, cells, mixed, parent['genomes']['native'], failed)
        else:
            case = next(r for r in search if r['id'] == failed['id'])
            expected = h_text(demos, parent['genomes']['native'], {'partition': 'search', 'input': case['input'],
                'expected': case['output'], 'actual': failed['actual'], 'error': failed['error']})
    P.require(receipt['protocol_hash'] == p['hash'] and raw == expected, 'staged payload not allowlisted')
    return raw


def compute(root, p, demos, search, mixed, slot, raw):
    parent_ids = []
    if slot.endswith('_repair'):
        parent = verify(root, slot.replace('_repair', '_founder'))['genomes']['native']
        parent_ids = [parent.get('id', parent.get('hash'))]
    if is_router(slot):
        _, cells = roster(root)
        ids = [c['id'] for c in cells]
        genomes = {'random': R.make_genome(ids, R.random_params(ids, p['random_seeds'][slot]))}
    else:
        genomes = {'random': N.spatial_cell(P.random_params(h_reference(), p['random_seeds'][slot]))}
    try:
        if is_router(slot):
            entry = N.response_entry(raw, 'router')
            genomes['native'] = R.make_genome(ids, entry['params'], parents=parent_ids)
        else:
            entry = N.validate_spatial_response(raw, H_TASK, h_reference())
            for row in entry['params']['tile_orient']['orient']:
                for v in row: P.require(v[4:] == [-9] * 4, 'fixed orientation tail differs')
            genomes['native'] = N.spatial_cell(entry['params'])
        validation = {'status': 'valid', 'error': None}
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
        validation = {'status': 'invalid_or_refused', 'error': type(exc).__name__ + ': ' + str(exc)[:300]}
    scores = {arm: bank_score(cells, g['params'], mixed, root) if is_router(slot) else T.evaluate(g, search, demos, root) for arm, g in genomes.items()}
    count = 6 if is_router(slot) else 167
    return {'genomes': genomes, 'validation': validation, 'scores': scores,
            'parent_genome_ids': parent_ids, 'operation': 'complete_literal_replacement' if parent_ids else 'founder',
            'charged_positions': {'native': count, 'random': count}, 'development_used': False}


def consume(root, slot, response, source):
    P.require(slot in SLOTS, 'unknown slot')
    root, p, demos, search, mixed = context(root)
    open_authoring(root)
    payload = packet(root, slot)
    source_raw = Path(source).read_bytes()
    P.require(len(source_raw) <= 8192, 'bounded supervisor source receipt required')
    origin = P.parse_source(source_raw, N.digest_bytes(payload))
    with Path(response).open('rb') as stream: raw = stream.read(N.MAX_RESPONSE + 1)
    folder = root / 'HOST_EVALUATION' / slot
    folder.mkdir()
    count = 6 if is_router(slot) else 167
    attempt = N.signed({'slot': slot, 'protocol_hash': p['hash'], 'payload_sha256': N.digest_bytes(payload),
        'source_sha256': N.digest_bytes(source_raw), 'source': origin, 'response_sha256': N.digest_bytes(raw),
        'captured_bytes': len(raw), 'capture_complete': len(raw) <= N.MAX_RESPONSE,
        'random_seed': p['random_seeds'][slot], 'charged_positions': {'native': count, 'random': count}, 'credit_usage': None})
    N.write_once(folder / 'ATTEMPT.json', attempt)
    P.write_bytes(folder / 'SOURCE.json', source_raw)
    P.write_bytes(folder / 'RESPONSE.bin', raw)
    result = N.signed(dict(compute(root, p, demos, search, mixed, slot, raw), slot=slot, protocol_hash=p['hash'], attempt_hash=attempt['hash']))
    N.write_once(folder / 'RESULT.json', result)
    return summary(result)


def summary(r):
    return {'slot': r['slot'], 'result_hash': r['hash'], 'validation': r['validation'], 'charged_positions': r['charged_positions'],
            'search': {arm: {'correct': s['correct'], 'errors': s['errors']} for arm, s in r['scores'].items()}}


def verify(root, slot):
    root, p, demos, search, mixed = context(root)
    folder = root / 'HOST_EVALUATION' / slot
    attempt, result = N.read_signed(folder / 'ATTEMPT.json'), N.read_signed(folder / 'RESULT.json')
    raw, source = (folder / 'RESPONSE.bin').read_bytes(), (folder / 'SOURCE.json').read_bytes()
    count, payload = (6 if is_router(slot) else 167), packet(root, slot)
    P.require(attempt['slot'] == slot and attempt['protocol_hash'] == p['hash']
        and attempt['response_sha256'] == N.digest_bytes(raw) and attempt['source_sha256'] == N.digest_bytes(source)
        and attempt['payload_sha256'] == N.digest_bytes(payload) and attempt['random_seed'] == p['random_seeds'][slot]
        and attempt['charged_positions'] == {'native': count, 'random': count}
        and P.parse_source(source, N.digest_bytes(payload)) == attempt['source'], 'attempt evidence changed')
    expected = N.signed(dict(compute(root, p, demos, search, mixed, slot, raw), slot=slot, protocol_hash=p['hash'], attempt_hash=attempt['hash']))
    P.require(expected == result, 'saved search replay differs')
    return result


def prepare_repair(root, slot):
    P.require(slot in ('specialist_repair', 'router_repair'), 'repair slot required')
    root, p, demos, search, mixed = context(root)
    open_authoring(root)
    parent = verify(root, slot.replace('_repair', '_founder'))
    P.require(needs_repair(parent), 'no repair for invalid or search-perfect founder')
    failed = next(r for r in parent['scores']['native']['rows'] if r['correct'] is not True)
    if is_router(slot):
        _, cells = roster(root)
        raw = router_text(root, cells, mixed, parent['genomes']['native'], failed)
    else:
        case = next(r for r in search if r['id'] == failed['id'])
        raw = h_text(demos, parent['genomes']['native'], {'partition': 'search', 'input': case['input'],
            'expected': case['output'], 'actual': failed['actual'], 'error': failed['error']})
    receipt = N.signed({'protocol_hash': p['hash'], 'parent_result_hash': parent['hash'], 'payload_sha256': N.digest_bytes(raw)})
    P.write_bytes(root / 'AUTHORING' / (slot + '.txt'), raw)
    N.write_once(root / 'AUTHORING' / (slot + '.json'), receipt)
    return receipt


def freeze_roster(root):
    root, p, _, _, mixed = context(root)
    open_authoring(root)
    results = stage_results(root, 'specialist')
    chosen, control = best(results, 'native'), best(results, 'random')
    if not chosen or chosen['search_correct'] != 12:
        rejection = N.signed({'reason': 'horizontal specialist gate failed', 'results': [r['hash'] for r in results],
            'charged_positions': {'native': 167 * len(results), 'random': 167 * len(results)}, 'development_opened': False})
        N.write_once(root / 'HOST_EVALUATION/TERMINAL_REJECTION.json', rejection)
        return rejection
    cells = [p['column'], chosen['genome']]
    raw = router_text(root, cells, mixed)  # Refuses runtime failures before author dispatch.
    record = N.signed({'protocol_hash': p['hash'], 'native_specialist': chosen, 'random_specialist': control,
                      'expert_ids': [c['id'] for c in cells], 'development_used': False})
    N.write_once(root / 'HOST_EVALUATION/ROSTER.json', record)
    P.write_bytes(root / 'AUTHORING/router_founder.txt', raw)
    receipt = N.signed({'protocol_hash': p['hash'], 'payload_sha256': N.digest_bytes(raw), 'roster_hash': record['hash']})
    N.write_once(root / 'AUTHORING/router_founder.json', receipt)
    return receipt


def final_compute(root, p, mixed, specialists, routers, selected):
    raw = (root / SEALED).read_bytes()
    P.require(N.digest_bytes(raw) == p['file_hashes'][SEALED], 'sealed panel changed')
    panel = N.decode(raw)
    rows = [label_task(r) for r in panel]
    _, cells = roster(root)
    scores = {r['slot'] + ':' + arm: bank_score(cells, g['params'], rows, root) for r in routers for arm, g in r['genomes'].items()}
    for name, params in (('single_C', ZERO), ('single_H', ALWAYS_H), ('zero_router', ZERO), ('fixed_identity', IDENTITY)):
        scores[name] = bank_score(cells, params, rows, root)
    reference_cells = [p['column'], N.spatial_cell(h_reference())]
    scores['reference_bank'] = bank_score(reference_cells, IDENTITY, rows, root)
    scores['zero_H_identity'] = bank_score([p['column'], N.spatial_cell(N.map_numbers(h_reference(), lambda: 0.0))], IDENTITY, rows, root)
    for r in specialists:
        scores[r['slot'] + ':random_H_identity'] = bank_score([p['column'], r['genomes']['random']], IDENTITY, rows, root)
    winner = selected['native']
    native_key = winner['slot'] + ':native' if winner else None
    pairs = []
    if native_key:
        outcomes = {r['id']: r['predictions'] for r in scores[native_key]['rows']}
        for i in range(4):
            before, after = outcomes[f'perm-{i}-base'], outcomes[f'perm-{i}-mapped']
            mapping = next(r['permutation'] for r in panel if r['id'] == f'perm-{i}-mapped')
            pairs.append(before is not None and after == [[[mapping[v] for v in row] for row in grid] for grid in before])
    random_max = max(scores[r['slot'] + ':random']['correct'] for r in routers)
    single_max = max(scores[k]['correct'] for k in ('single_C', 'single_H'))
    errors = sum(s['errors'] for s in scores.values()) + sum(s['errors'] for r in specialists + routers for s in r['scores'].values())
    valid = errors == 0 and scores['reference_bank']['correct'] == 24
    accepted = bool(valid and winner and winner['search_correct'] == 12 and scores[native_key]['correct'] == 24
                    and all(pairs) and len(pairs) == 4 and 24 - random_max >= 6 and 24 - single_max >= 6)
    charge = 167 * len(specialists) + 6 * len(routers)
    return N.signed({'schema': 'cellforge-native-composition-result/1', 'protocol_hash': p['hash'],
        'selected_on_search_only': selected, 'scores': scores, 'permutation_pairs': pairs,
        'experiment_valid': valid, 'comparative_acceptance': accepted,
        'identity_margin': scores[native_key]['correct'] - scores['fixed_identity']['correct'] if native_key else None,
        'learned_router_superiority': bool(native_key and valid and scores[native_key]['correct'] > scores['fixed_identity']['correct']),
        'random_max_for_acceptance_only': random_max, 'best_single_score': single_max,
        'charged_positions': {'native': charge, 'random': charge}, 'credit_usage': None,
        'labels_sha256': N.C.digest(rows), 'service_admitted': False, 'arc_qualification': False})


def finalize(root, replay=False):
    root, p, _, _, mixed = context(root)
    specialists, routers = stage_results(root, 'specialist'), stage_results(root, 'router')
    selected = {arm: best(routers, arm) for arm in ('native', 'random')}
    selection = N.signed({'protocol_hash': p['hash'], 'selected': selected,
        'result_hashes': [r['hash'] for r in specialists + routers], 'development_used': False})
    host = root / 'HOST_EVALUATION'
    if replay:
        P.require(N.read_signed(host / 'SEARCH_SELECTION.json') == selection, 'search selection changed')
        P.require(N.read_signed(host / 'FINAL_OPEN.json')['selection_hash'] == selection['hash'], 'final opening changed')
    else:
        N.write_once(host / 'SEARCH_SELECTION.json', selection)
        N.write_once(host / 'FINAL_OPEN.json', N.signed({'protocol_hash': p['hash'], 'selection_hash': selection['hash']}))
    result = final_compute(root, p, mixed, specialists, routers, selected)
    if replay: P.require(N.read_signed(host / 'FINAL_RESULT.json') == result, 'independent final replay differs')
    else: N.write_once(host / 'FINAL_RESULT.json', result)
    return {'status': 'verified' if replay else 'finalized', 'result_hash': result['hash'], 'experiment_valid': result['experiment_valid'],
            'comparative_acceptance': result['comparative_acceptance'], 'identity_margin': result['identity_margin'],
            'charged_positions': result['charged_positions'], 'counts': {k: s['correct'] for k, s in result['scores'].items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight', 'import', 'verify', 'prepare-repair', 'freeze-roster', 'finalize', 'replay-final'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--slot', choices=SLOTS)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--source', type=Path)
    a = parser.parse_args()
    if a.command == 'prepare':
        if not a.previous: parser.error('--previous required')
        value = prepare(a.root, a.previous)
    elif a.command == 'import':
        if not all((a.slot, a.response, a.source)): parser.error('--slot, --response and --source required')
        value = consume(a.root, a.slot, a.response, a.source)
    elif a.command in ('verify', 'prepare-repair'):
        if not a.slot: parser.error('--slot required')
        value = summary(verify(a.root, a.slot)) if a.command == 'verify' else prepare_repair(a.root, a.slot)
    elif a.command in ('finalize', 'replay-final'): value = finalize(a.root, replay=a.command == 'replay-final')
    else: value = (freeze_roster if a.command == 'freeze-roster' else preflight)(a.root)
    print(json.dumps(value, sort_keys=True))
