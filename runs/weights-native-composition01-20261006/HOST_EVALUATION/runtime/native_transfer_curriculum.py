"""Finite synthetic transfer experiment; no model dispatch or ARC qualification.

Development inputs are frozen but their oracle labels are not materialized until
the one-time final opening. Native AND random winners are selected on search.
"""
import argparse
import copy
import json
from pathlib import Path
import random

import native_parallel_batch as P

N = P.N
TASK = 'synthetic-column-mirror-v1'
SLOTS = ('founder_a', 'founder_b', 'repair_a', 'repair_b')
PARENTS = {'repair_a': 'founder_a', 'repair_b': 'founder_b'}
SEEDS = dict(zip(SLOTS, range(20261006211, 20261006215)))
FILES = (*P.FILES, 'native_transfer_curriculum.py')
SEALED = 'HOST_EVALUATION/DEVELOPMENT_INPUTS_SEALED.json'
INSTRUCTION = ('Use only this payload. No tools, filesystem, external sources or other authors. '
    'Infer one fixed rule shared by all examples. Return only one JSON array containing one object '
    'with exactly task_id, params, rationale, matching response_schema. Supply all 167 literal numbers '
    'including inactive slots; no code, ellipses, duplicate keys or answer grids. No free schema repair. '
    'Both primitives ignore examples at runtime. Host code supplies their algorithms; your weights '
    'select behavior. This is synthetic transfer, not ARC or chatbot qualification.')


def oracle(grid):
    """Independent declarative labels: column broadcast, then mirrored blocks."""
    h, w = len(grid), len(grid[0])
    P.require(h > 0 and w > 0 and all(len(row) == w for row in grid), 'rectangular nonempty grid')
    colors = []
    for c in range(w):
        values = [row[c] for row in grid if row[c] != 0]
        P.require(len(values) <= 1, 'at most one marker per column')
        colors.append(values[0] if values else 0)
    filled = [colors[:] for _ in range(h)]
    top = [row + list(reversed(row)) for row in filled]
    bottom = [row + list(reversed(row)) for row in reversed(filled)]
    return top + bottom


def reference_params():
    orient = [[[9, -9, -9, -9, -9, -9, -9, -9] for _ in range(4)] for _ in range(4)]
    for i, j, k in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)):
        orient[i][j] = [9 if n == k else -9 for n in range(8)]
    return {'ray_cast': {'src': [-9] + [9] * 9, 'dirs': [9, -9, -9, -9, 9, -9, -9, -9],
                        'stop': [-9, -9, 9], 'paint': [-9] * 10},
            'tile_orient': {'per': [-9, 9, -9, -9], 'pec': [-9, 9, -9, -9], 'orient': orient}}


def schema():
    value = P.response_schema()
    value['items']['properties']['task_id']['const'] = TASK
    return value


def validate(raw):
    entry = N.validate_spatial_response(raw, TASK, reference_params())
    for row in entry['params']['tile_orient']['orient']:
        for vector in row:
            P.require(vector[4:] == [-9] * 4, 'orientation entries 4..7 must equal -9')
    return entry


def geometry(grid):
    """Deduplicate layout up to palette relabeling and shape-preserving flips."""
    def normalized(rows):
        mapping = {0: 0}
        return [[mapping.setdefault(v, len(mapping)) for v in row] for row in rows]
    variants = (grid, [list(reversed(row)) for row in grid], list(reversed(grid)),
                [list(reversed(row)) for row in reversed(grid)])
    return min(json.dumps(normalized(rows), separators=(',', ':')) for rows in variants)


def draw(rng, h, w, count, palette, used):
    for _ in range(1000):
        grid = [[0] * w for _ in range(h)]
        colors = rng.sample(list(palette), count)
        for c, color in zip(rng.sample(range(w), count), colors):
            grid[rng.randrange(h)][c] = color
        # Require a meaningful left/right distinction in the filled pattern.
        columns = [next((row[c] for row in grid if row[c]), 0) for c in range(w)]
        key = geometry(grid)
        if columns != list(reversed(columns)) and key not in used:
            used.add(key)
            return grid
    raise ValueError('bounded input generation exhausted; do not tune the seed')


def inputs():
    used, panels = set(), {}
    dimensions = ((3, 4), (4, 3), (4, 4), (3, 5), (5, 3), (5, 4))
    for name, seed, count in (('demonstrations', 20261006201, 6), ('search', 20261006202, 12)):
        rng, rows = random.Random(seed), []
        for i in range(count):
            h, w = dimensions[i % len(dimensions)]
            n = min(1 + i % 3, w - 1)
            rows.append({'id': f'{name}-{i:02}', 'input': draw(rng, h, w, n, (2, 4, 5), used)})
        panels[name] = rows
    rng, rows = random.Random(20261006203), []
    for i in range(8):
        h, w = dimensions[i % len(dimensions)]
        rows.append({'id': f'color-{i:02}', 'stratum': 'unseen_color',
                     'input': draw(rng, h, w, min(2 + i % 2, w - 1), (1, 3, 6, 7, 8, 9), used)})
    for i in range(4):
        h, w = dimensions[i]
        grid = draw(rng, h, w, min(2 + i % 2, w - 1), (2, 4, 5), used)
        # Full nonbackground bijection; every label changes, background stays 0.
        shift = 1 + i
        permutation = [0] + [1 + (c - 1 + shift) % 9 for c in range(1, 10)]
        transformed = [[permutation[v] for v in row] for row in grid]
        rows += [{'id': f'perm-{i}-base', 'stratum': 'permutation', 'pair': i, 'role': 'base', 'input': grid},
                 {'id': f'perm-{i}-mapped', 'stratum': 'permutation', 'pair': i, 'role': 'mapped',
                  'permutation': permutation, 'input': transformed}]
    large = ((6, 7), (7, 6), (6, 8), (8, 6), (7, 9), (9, 7), (8, 9), (9, 8))
    for i, (h, w) in enumerate(large):
        rows.append({'id': f'size-{i:02}', 'stratum': 'larger_rectangle',
                     'input': draw(rng, h, w, 1 + i % 3, (2, 4, 5), used)})
    panels['development'] = rows
    return panels


def fixture_check():
    fixtures = ([[0, 2, 0], [4, 0, 0]], [[1, 0, 0, 9, 0], [0, 0, 6, 0, 0]],
                [[0, 0], [0, 0], [7, 0], [0, 0], [0, 3], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    P.require(oracle(fixtures[0]) == [[4, 2, 0, 0, 2, 4]] * 4, 'independent manual fixture differs')
    ref = N.spatial_cell(reference_params())
    for grid in fixtures:
        P.require(N.S.run(ref, grid, []) == oracle(grid), 'reference/oracle preparation fixture mismatch')
    return {'fixtures': 3, 'manual_oracle_fixture': True, 'reference_exact': True,
            'prospective_development_scored': False, 'fixture_input_hash': N.C.digest(fixtures)}


def author_text(demos, parent=None, failure=None):
    packet = {'task_id': TASK, 'demonstrations': demos, 'composition': ['ray_cast', 'tile_orient'],
              'semantics': {name: N.S.TEMPLATES[name]['doc'] for name in N.SPATIAL_TEMPLATES},
              'constraints': 'All numbers finite in [-9,9]. EVERY orient[i][j][4:8] must be [-9,-9,-9,-9]. '
                             'Argmax ties choose the lowest index. All 167 positions charged.',
              'response_schema': schema()}
    if parent is not None:
        packet['parent'] = {'cell_id': parent['id'], 'params': P.leaf_params(parent)}
        packet['search_failure'] = failure
    return INSTRUCTION.encode() + b'\n' + P.compact(packet)


def prepare(root):
    root = Path(root).resolve()
    P.require(not root.exists(), 'new experiment root required')
    checks = fixture_check()  # Separate fixtures only, before prospective panels exist.
    panels = inputs()
    demos = [{'input': row['input'], 'output': oracle(row['input'])} for row in panels['demonstrations']]
    search = [dict(row, output=oracle(row['input'])) for row in panels['search']]
    root.mkdir()
    (root / 'HOST_EVALUATION/runtime').mkdir(parents=True)
    (root / 'AUTHORING').mkdir()
    contents = {'AUTHORING/FOUNDER_INPUT.txt': author_text(demos),
                'AUTHORING/RESPONSE_SCHEMA.json': P.compact(schema()),
                'SEARCH_DEMONSTRATIONS.json': P.compact(demos), 'SEARCH_FITNESS.json': P.compact(search),
                SEALED: P.compact(panels['development'])}
    hashes = {}
    for name, raw in contents.items():
        P.write_bytes(root / name, raw)
        hashes[name] = N.digest_bytes(raw)
    sources = {}
    for name in FILES:
        raw = Path(__file__).with_name(name).read_bytes()
        P.write_bytes(root / 'HOST_EVALUATION/runtime' / name, raw)
        sources[name] = N.digest_bytes(raw)
    protocol = N.signed({'schema': 'cellforge-synthetic-transfer/1', 'task_id': TASK,
        'source_hashes': sources, 'file_hashes': hashes, 'fixture_checks': checks, 'random_seeds': SEEDS,
        'panel_seeds': {'demonstrations': 20261006201, 'search': 20261006202, 'development': 20261006203},
        'counts': {'demonstrations': 6, 'search': 12, 'development': 24}, 'slots': list(SLOTS),
        'parents': PARENTS, 'max_completions': 4, 'positions_per_attempt_per_arm': 167,
        'free_positions': 103, 'fixed_positions': 64, 'max_positions_per_arm': 668,
        'eligibility': 'Two founders; one replacement only for each valid founder with search errors. No repair for invalid or search-perfect founders.',
        'selection': 'For BOTH arms: greatest search exact count, then ascending cell ID; all valid zero-error consumed candidates eligible. Freeze before opening development.',
        'acceptance': {'native_search_exact': 12, 'native_development_exact': 24, 'permutation_pairs': 4,
                       'minimum_margin_over_all_random_development_scores': 6,
                       'all_scored_arms_runtime_errors': 0, 'reference_development_exact': 24},
        'development_policy': 'Inputs frozen; labels unmaterialized until one-time final opening after search-only selection. Never feed development outcomes to an author.',
        'generator_rule': 'One marker per occupied column; fill that column with its color; tile identity/fliplr/flipud/rot180 in a 2x2 arrangement.',
        'limitations': ['Host implements the rule family; not ARC qualification or algorithm discovery.',
                       'Column filling makes vertical flips observationally redundant.',
                       'Synthetic seeds do not imply independent founder/model seeds.'],
        'credit_usage': None, 'automatic_dispatch': False, 'automatic_admission': False})
    N.write_once(root / 'HOST_EVALUATION/PROTOCOL.json', protocol)
    N.write_once(root / 'STANDARD.json', {'arms': {'llm': {}, 'random': {}}, 'budget_per_arm': 4,
        'seeds': list(SEEDS.values()), 'seeds_declared_before_run': True, 'niches': ['synthetic_spatial_transfer'],
        'founders': [], 'genomes': [], 'proposals': [], 'scientific_acceptance': False,
        'limitations': ['Prospective bounded study, not a qualified ONE_STANDARD ledger.',
                        'Operational attempted-position ledger is in HOST_EVALUATION; no fabricated founder records.']})
    return preflight(root)


def context(root):
    root = Path(root).resolve()
    N.stop(root)
    protocol = N.read_signed(root / 'HOST_EVALUATION/PROTOCOL.json')
    for name, digest in protocol['source_hashes'].items():
        for path in (Path(__file__).with_name(name), root / 'HOST_EVALUATION/runtime' / name):
            P.require(N.digest_bytes(path.read_bytes()) == digest, 'frozen source changed')
    for name, digest in protocol['file_hashes'].items():
        if name != SEALED:  # Not even read before final opening.
            P.require(N.digest_bytes((root / name).read_bytes()) == digest, 'frozen search/author artifact changed')
    return root, protocol, N.decode((root / 'SEARCH_DEMONSTRATIONS.json').read_bytes()), N.decode((root / 'SEARCH_FITNESS.json').read_bytes())


def preflight(root):
    root, protocol, demos, rows = context(root)
    P.require((root / 'AUTHORING/FOUNDER_INPUT.txt').read_bytes() == author_text(demos), 'unexpected founder context')
    P.require(len(demos) == 6 and len(rows) == 12, 'frozen search sizes differ')
    return {'status': 'ready', 'protocol_hash': protocol['hash'], 'founder_input_sha256': protocol['file_hashes']['AUTHORING/FOUNDER_INPUT.txt'],
            'founder_input_bytes': (root / 'AUTHORING/FOUNDER_INPUT.txt').stat().st_size,
            'development_opened': (root / 'HOST_EVALUATION/FINAL_OPEN.json').exists(), 'credit_usage': None}


def evaluate(cell, rows, demos, root):
    output = []
    for row in rows:
        N.stop(root)
        try:
            actual = N.S.run(cell, copy.deepcopy(row['input']), copy.deepcopy(demos))
            error = None
        except Exception as exc:
            actual, error = None, type(exc).__name__ + ': ' + str(exc)[:300]
        output.append({'id': row['id'], 'correct': None if error else actual == row['output'],
                       'actual': actual, 'error': error})
    return {'correct': sum(row['correct'] is True for row in output),
            'errors': sum(row['error'] is not None for row in output), 'rows': output}


def needed_repair(result):
    return result['validation']['status'] == 'valid' and result['scores']['native']['correct'] < 12


def packet_for(root, slot):
    if slot not in PARENTS:
        return (root / 'AUTHORING/FOUNDER_INPUT.txt').read_bytes()
    receipt = N.read_signed(root / 'AUTHORING' / (slot + '.json'))
    raw = (root / 'AUTHORING' / (slot + '.txt')).read_bytes()
    parent = verify(root, PARENTS[slot])
    P.require(needed_repair(parent) and receipt['parent_result_hash'] == parent['hash']
              and receipt['payload_sha256'] == N.digest_bytes(raw), 'repair eligibility/packet changed')
    _, protocol, demos, search = context(root)
    P.require(receipt['protocol_hash'] == protocol['hash'] and raw == repair_text(parent, demos, search),
              'repair payload contains nonallowlisted context')
    return raw


def compute(root, protocol, demos, search, slot, raw):
    cells = {'random': N.spatial_cell(P.random_params(reference_params(), protocol['random_seeds'][slot]))}
    try:
        entry = validate(raw)
        cells['native'] = N.spatial_cell(entry['params'])
        validation = {'status': 'valid', 'error': None}
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as error:
        validation = {'status': 'invalid_or_refused', 'error': type(error).__name__ + ': ' + str(error)[:300]}
    return {'validation': validation, 'cells': cells,
            'scores': {arm: evaluate(cell, search, demos, root) for arm, cell in cells.items()},
            'charged_positions': {'native': 167, 'random': 167}, 'development_opened': False}


def consume(root, slot, response, source):
    P.require(slot in SLOTS, 'unknown slot')
    root, protocol, demos, search = context(root)
    P.require(not any((root / 'HOST_EVALUATION' / name).exists() for name in ('SEARCH_SELECTION.json', 'FINAL_OPEN.json')), 'authoring closed')
    payload = packet_for(root, slot)
    source_raw = Path(source).read_bytes()
    P.require(len(source_raw) <= 8192, 'bounded supervisor receipt required')
    origin = P.parse_source(source_raw, N.digest_bytes(payload))
    with Path(response).open('rb') as stream:
        raw = stream.read(N.MAX_RESPONSE + 1)
    folder = root / 'HOST_EVALUATION' / slot
    folder.mkdir()
    attempt = N.signed({'schema': 'cellforge-transfer-attempt/1', 'protocol_hash': protocol['hash'], 'slot': slot,
        'payload_sha256': N.digest_bytes(payload), 'response_sha256': N.digest_bytes(raw),
        'source_sha256': N.digest_bytes(source_raw), 'source': origin, 'captured_bytes': len(raw),
        'capture_complete': len(raw) <= N.MAX_RESPONSE, 'credit_usage': None,
        'charged_positions': {'native': 167, 'random': 167}, 'random_seed': protocol['random_seeds'][slot]})
    N.write_once(folder / 'ATTEMPT.json', attempt)
    P.write_bytes(folder / 'SOURCE.json', source_raw)
    P.write_bytes(folder / 'RESPONSE.bin', raw)
    result = N.signed(dict(compute(root, protocol, demos, search, slot, raw), slot=slot,
                          protocol_hash=protocol['hash'], attempt_hash=attempt['hash']))
    N.write_once(folder / 'RESULT.json', result)
    return summary(result)


def summary(result):
    return {'slot': result['slot'], 'result_hash': result['hash'], 'validation': result['validation'],
            'search': {arm: {'correct': score['correct'], 'errors': score['errors']} for arm, score in result['scores'].items()},
            'charged_positions': result['charged_positions']}


def verify(root, slot):
    root, protocol, demos, search = context(root)
    folder = root / 'HOST_EVALUATION' / slot
    attempt, result = N.read_signed(folder / 'ATTEMPT.json'), N.read_signed(folder / 'RESULT.json')
    raw, source = (folder / 'RESPONSE.bin').read_bytes(), (folder / 'SOURCE.json').read_bytes()
    payload = packet_for(root, slot)
    P.require(attempt['response_sha256'] == N.digest_bytes(raw) and attempt['source_sha256'] == N.digest_bytes(source)
              and attempt['payload_sha256'] == N.digest_bytes(payload) and attempt['slot'] == slot
              and attempt['protocol_hash'] == protocol['hash'] and attempt['random_seed'] == protocol['random_seeds'][slot]
              and attempt['charged_positions'] == {'native': 167, 'random': 167}
              and P.parse_source(source, N.digest_bytes(payload)) == attempt['source'], 'attempt changed')
    expected = N.signed(dict(compute(root, protocol, demos, search, slot, raw), slot=slot,
                            protocol_hash=protocol['hash'], attempt_hash=attempt['hash']))
    P.require(expected == result, 'saved search replay differs')
    return result


def repair_text(parent, demos, search):
    failed = next(row for row in parent['scores']['native']['rows'] if row['correct'] is not True)
    case = next(row for row in search if row['id'] == failed['id'])
    feedback = {'id': case['id'], 'input': case['input'], 'expected': case['output'], 'actual': failed['actual'],
                'runtime_error': failed['error'], 'partition': 'search', 'selection': 'first failure in frozen search order'}
    return author_text(demos, parent['cells']['native'], feedback)


def prepare_repair(root, slot):
    P.require(slot in PARENTS, 'repair slot required')
    root, protocol, demos, search = context(root)
    P.require(not any((root / 'HOST_EVALUATION' / name).exists() for name in ('SEARCH_SELECTION.json', 'FINAL_OPEN.json')), 'authoring closed')
    parents = {name: verify(root, name) for name in ('founder_a', 'founder_b')}
    parent = parents[PARENTS[slot]]
    P.require(needed_repair(parent), 'no repair for invalid or search-perfect founder')
    raw = repair_text(parent, demos, search)
    P.write_bytes(root / 'AUTHORING' / (slot + '.txt'), raw)
    receipt = N.signed({'slot': slot, 'parent_result_hash': parent['hash'], 'protocol_hash': protocol['hash'],
                        'payload_sha256': N.digest_bytes(raw), 'development_feedback': False})
    N.write_once(root / 'AUTHORING' / (slot + '.json'), receipt)
    return receipt


def selection(root):
    results = [verify(root, slot) for slot in ('founder_a', 'founder_b')]
    for slot, parent in PARENTS.items():
        required = needed_repair(next(r for r in results if r['slot'] == parent))
        present = (root / 'HOST_EVALUATION' / slot).exists()
        P.require(required == present, 'eligible repairs must finish; ineligible repairs are forbidden')
        if present:
            results.append(verify(root, slot))
    best = {}
    for arm in ('native', 'random'):
        eligible = [r for r in results if arm in r['cells'] and r['scores'][arm]['errors'] == 0]
        ranked = sorted(eligible, key=lambda r: (-r['scores'][arm]['correct'], r['cells'][arm]['id']))
        best[arm] = {'slot': ranked[0]['slot'], 'cell_id': ranked[0]['cells'][arm]['id'],
                     'search_correct': ranked[0]['scores'][arm]['correct']} if ranked else None
    return results, N.signed({'schema': 'cellforge-search-only-selection/1',
        'results': {r['slot']: r['hash'] for r in results}, 'selected': best,
        'charged_positions': {'native': len(results) * 167, 'random': len(results) * 167}, 'development_used': False})


def final_compute(root, protocol, demos, search, results, selected):
    raw = (root / SEALED).read_bytes()
    P.require(N.digest_bytes(raw) == protocol['file_hashes'][SEALED], 'sealed development input changed')
    panel = N.decode(raw)
    rows = [dict(row, output=oracle(row['input'])) for row in panel]  # FIRST label materialization.
    cells = {f'{r["slot"]}:{arm}': cell for r in results for arm, cell in r['cells'].items()}
    cells.update(reference=N.spatial_cell(reference_params()), zero=N.spatial_cell(N.map_numbers(reference_params(), lambda: 0.0)))
    scores = {key: evaluate(cell, rows, demos, root) for key, cell in cells.items()}
    diagnostics_search = {key: evaluate(cells[key], search, demos, root) for key in ('reference', 'zero')}
    native = selected['selected']['native']
    key = native['slot'] + ':native' if native else None
    permutation_matches = []
    if key:
        predictions = {row['id']: row['actual'] for row in scores[key]['rows']}
        for i in range(4):
            mapping = next(row['permutation'] for row in panel if row['id'] == f'perm-{i}-mapped')
            before, after = predictions[f'perm-{i}-base'], predictions[f'perm-{i}-mapped']
            permutation_matches.append(before is not None and after == [[mapping[v] for v in row] for row in before])
    best_random_count = max(scores[r['slot'] + ':random']['correct'] for r in results)
    all_errors = sum(score['errors'] for score in scores.values()) + sum(score['errors'] for score in diagnostics_search.values())
    all_errors += sum(score['errors'] for r in results for score in r['scores'].values())
    valid = all_errors == 0 and scores['reference']['correct'] == 24 and diagnostics_search['reference']['correct'] == 12
    accepted = bool(valid and key and native['search_correct'] == 12 and scores[key]['correct'] == 24
                    and all(permutation_matches) and len(permutation_matches) == 4
                    and scores[key]['correct'] - best_random_count >= 6)
    return N.signed({'schema': 'cellforge-synthetic-transfer-final/1', 'protocol_hash': protocol['hash'],
        'selection_hash': selected['hash'], 'selected_on_search_only': selected['selected'],
        'development_scores': scores, 'diagnostic_search': diagnostics_search,
        'development_labels_sha256': N.C.digest(rows), 'permutation_pair_matches': permutation_matches,
        'max_random_development_score_for_acceptance_only': best_random_count,
        'experiment_valid': valid, 'transfer_criterion_passed': accepted,
        'charged_positions': selected['charged_positions'], 'credit_usage': None,
        'automatic_admission': False, 'arc_qualification': False})


def finalize(root, replay=False):
    root, protocol, demos, search = context(root)
    results, selected = selection(root)  # Never reads development to choose either winner.
    host = root / 'HOST_EVALUATION'
    if replay:
        saved = N.read_signed(host / 'SEARCH_SELECTION.json')
        opened = N.read_signed(host / 'FINAL_OPEN.json')
        P.require(saved == selected and opened['selection_hash'] == selected['hash']
                  and opened['protocol_hash'] == protocol['hash'], 'saved selection changed')
    else:
        N.write_once(host / 'SEARCH_SELECTION.json', selected)
        N.write_once(host / 'FINAL_OPEN.json', N.signed({'protocol_hash': protocol['hash'],
                     'selection_hash': selected['hash'], 'one_time_final_open': True}))
    result = final_compute(root, protocol, demos, search, results, selected)
    if replay:
        P.require(result == N.read_signed(host / 'FINAL_RESULT.json'), 'final saved-tensor replay differs')
    else:
        N.write_once(host / 'FINAL_RESULT.json', result)
    return {'status': 'verified' if replay else 'finalized', 'result_hash': result['hash'],
            'selected_on_search_only': result['selected_on_search_only'], 'experiment_valid': result['experiment_valid'],
            'transfer_criterion_passed': result['transfer_criterion_passed'], 'charged_positions': result['charged_positions'],
            'development_counts': {key: score['correct'] for key, score in result['development_scores'].items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight', 'import', 'verify', 'prepare-repair', 'finalize', 'replay-final'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--slot', choices=SLOTS)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.command in ('import', 'verify', 'prepare-repair') and not args.slot:
        parser.error('--slot required')
    if args.command == 'import':
        if not args.response or not args.source:
            parser.error('--response and --source required')
        result = consume(args.root, args.slot, args.response, args.source)
    elif args.command == 'verify': result = summary(verify(args.root, args.slot))
    elif args.command == 'prepare-repair': result = prepare_repair(args.root, args.slot)
    elif args.command == 'finalize': result = finalize(args.root)
    elif args.command == 'replay-final': result = finalize(args.root, replay=True)
    else: result = (prepare if args.command == 'prepare' else preflight)(args.root)
    print(json.dumps(result, sort_keys=True))
