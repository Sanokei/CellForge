"""Fail-closed full architecture gate, independently replayed by trusted executors.

JSON receipts alone cannot demonstrate execution. ``replay`` is an explicit map
of trusted cell/attachment/ensemble executors, each (genome, request)->prediction.
This gate establishes architecture coverage, never scientific or ARC acceptance.
"""
import copy
import json
import random
import math
import time
from .cell_population import CellPopulation, validate_literal, _inheritance
from .joint_weight_population import digest
from .red_queen_history import OpponentHistory

FORMAT = 'cellforge-two-level-execution/1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _raw(provenance):
    raw = provenance.get('raw_response')
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as error:
            raise ValueError('raw author response must contain exact JSON literals') from error
    return raw


def _measurement(receipt, genome, executor):
    """Replay one frozen manifest, excluding targets and case IDs from the executor."""
    _require(isinstance(receipt, dict), 'executed measurement required')
    _require(receipt.get('partition') == 'development', 'reporting cannot select')
    _require(receipt.get('genome_hash') == digest(genome), 'execution bound to wrong literal genome')
    cases = receipt.get('cases')
    _require(isinstance(cases, list) and cases, 'nonempty frozen execution cases required')
    _require(receipt.get('manifest_hash') == digest(cases), 'execution manifest differs from frozen cases')
    ids = []
    for case in cases:
        _require(isinstance(case, dict) and set(case) == {'id', 'request', 'target'}, 'exact target-separated cases required')
        _require(isinstance(case['id'], str) and case['id'], 'task identity required')
        _require(isinstance(case['request'], dict) and not any(k in case['request'] for k in
                    ('target', 'expected', 'output', 'answer', 'task_id', 'niche')), 'target-blind execution request required')
        ids.append(case['id'])
    _require(len(set(ids)) == len(ids), 'distinct frozen task identities required')
    predictions = {case['id']: executor(copy.deepcopy(genome), copy.deepcopy(case['request'])) for case in cases}
    outcomes = {case['id']: predictions[case['id']] == case['target'] for case in cases}
    _require(receipt.get('predictions') == predictions and receipt.get('outcomes') == outcomes,
             'recorded predictions/outcomes do not reproduce from literal tensors')
    _require(all(type(v) is bool for v in receipt['outcomes'].values()), 'strict boolean outcomes required')
    return predictions, outcomes


def _zero(value):
    return [_zero(v) for v in value] if isinstance(value, list) else 0


def _control_genome(genome, name, seed=None):
    result = copy.deepcopy(genome)
    if name == 'parent':
        return None
    if set(result) == {'cells', 'router', 'topology'}:
        if name == 'random':
            _require(type(seed) is int, 'explicit reproducible random control seed required')
            rng = random.Random(seed)
            def draw(value):
                return [draw(v) for v in value] if isinstance(value, list) else rng.uniform(-9, 9)
            result['cells'] = {cid: {k: draw(params[k]) for k in sorted(params)} for cid, params in sorted(result['cells'].items())}
            result['router'] = {k: draw(result['router'][k]) for k in sorted(result['router'])}
        elif name == 'zero':
            result['cells'] = {cid: {k: _zero(v) for k, v in params.items()} for cid, params in result['cells'].items()}
            result['router'] = {k: _zero(v) for k, v in result['router'].items()}
        elif name == 'uniform-router':
            result['router'] = {k: _zero(v) for k, v in result['router'].items()}
        elif name.startswith('ablate-cell:'):
            cid = name.split(':', 1)[1]
            _require(cid in result['cells'], 'unknown whole-MoE cell ablation')
            result['cells'][cid] = {k: _zero(v) for k, v in result['cells'][cid].items()}
        else:
            raise ValueError('unsupported whole-MoE numerical control')
        return result
    _require(set(result) == {'spec', 'founders'}, 'typed graph control genome required')
    if name == 'random':
        _require(type(seed) is int, 'explicit reproducible random control seed required')
        rng = random.Random(seed)
        def draw(value):
            return [draw(v) for v in value] if isinstance(value, list) else rng.uniform(-9, 9)
        for node in result['spec']['nodes']:
            if node['kind'] == 'glue':
                node['params'] = {k: draw(node['params'][k]) for k in sorted(node['params'])}
    elif name == 'zero':
        for cell in result['founders'].values():
            for layer in cell['layers']:
                for key in layer:
                    layer[key] = _zero(layer[key])
        for node in result['spec']['nodes']:
            if node['kind'] == 'glue':
                node['params'] = {k: _zero(v) for k, v in node['params'].items()}
    elif name.startswith('ablate-cell:'):
        contributor = name.split(':', 1)[1]
        _require(contributor in result['founders'], 'unknown contributor ablation')
        for layer in result['founders'][contributor]['layers']:
            for key in layer:
                layer[key] = _zero(layer[key])
    elif name.startswith('ablate-glue:'):
        ident = name.split(':', 1)[1]
        nodes = [n for n in result['spec']['nodes'] if n['id'] == ident and n['kind'] == 'glue']
        _require(len(nodes) == 1, 'unknown authored glue ablation')
        nodes[0]['params'] = {k: _zero(v) for k, v in nodes[0]['params'].items()}
    else:
        raise ValueError('unsupported numerical control')
    return result

def _controls(bundle, genome, executor, required):
    base, _ = _measurement(bundle['measurement'], genome, executor)
    controls = bundle.get('controls', {})
    _require(set(controls) >= set(required), 'executed parent/random/zero and contributor/glue controls required')
    for name in required:
        control = controls[name]
        expected = _control_genome(genome, name, control.get('seed'))
        _require(expected is None or control['genome'] == expected, 'control numerics differ from actual zero/random/contributor/glue intervention')
        _require(control['measurement']['cases'] == bundle['measurement']['cases'], 'controls require identical task budget')
        _require(digest(control['genome']) != digest(genome), 'control repeats candidate literal genome')
        predicted, _ = _measurement(control['measurement'], control['genome'], executor)
        if name.startswith('ablate-') or name == 'uniform-router':
            _require(predicted != base, 'registered contributor or authored glue has no measured causal effect')
    return base


def _cell_genome(cell):
    return {'structure': cell['structure'], 'params': cell['params']}


def _author(provenance, raw_key, applied):
    _require(isinstance(provenance, dict) and provenance.get('model') == 'gpt-6.1-sol'
             and provenance.get('reasoning') == 'low' and provenance.get('prompt'),
             'requested gpt-6.1-sol low-reasoning author and actual prompt required')
    raw = _raw(provenance)
    _require(isinstance(raw, dict) and raw.get(raw_key) == applied,
             'raw author response differs from applied literal genome')


def _lineage(population, attachment_child=None):
    cells = population.cells
    sexual = []
    for cid, cell in cells.items():
        validate_literal(cell['params'])
        _require(cid == digest(_cell_genome(cell)) and cell['id'] == cid, 'registry identity differs from literal tensors')
        _require(all(cell.get(k) for k in ('family', 'niche', 'niche_manifest')), 'persistent family/niche commitment required')
        parents = cell.get('parents', [])
        if not parents:
            _author(cell['provenance'], 'tensors', cell['params'])
            continue
        _require(len(parents) in (1, 2) and len(set(parents)) == len(parents)
                 and all(p in cells for p in parents), 'distinct actual registry parents required')
        charges = [c for c in population.charges if c.get('child') == cid and c.get('parents') == parents and c.get('charged') is True]
        _require(charges, 'every registered child attempt must be charged')
        provenance = cell['provenance']
        if cid == attachment_child:
            # Registered typed graph birth is checked against exact parent layers and glue below.
            continue
        if len(parents) == 2:
            inherited = _inheritance(cells[parents[0]]['params'], cells[parents[1]]['params'], provenance.get('inheritance'))
            if any(c.get('arm', '').startswith('llm') for c in charges):
                _author(provenance, 'tensors', cell['params'])
                _require(cell['params'] != inherited and provenance['prompt'].get('mutation_parent_hash') == digest(inherited),
                         'post-crossover author must actually mutate the inherited tensors shown in its prompt')
                _require(provenance.get('mutation_parent_hash') == digest(inherited)
                         and provenance.get('newly_authored') and provenance.get('typed_io'),
                         'post-crossover literal mutation must bind inherited numerical child')
            else:
                _require(cell['params'] == inherited, 'crossover tensors differ from replayed inheritance')
            sexual.append(cid)
        elif any(c.get('arm', '').startswith('llm') for c in charges):
            _author(provenance, 'tensors', cell['params'])
    _require(sexual, 'actual two-parent numerical inheritance required')
    return sexual


def _cell_execution(population, execution, executor):
    evidence = execution.get('cells', {})
    _require(set(evidence) == set(population.cells), 'every registered cell must have independent execution evidence')
    families, niches = set(), set()
    for cid, cell in population.cells.items():
        sections = evidence[cid]
        _require(set(sections) == {'fresh', 'archive', 'anchor'}, 'fresh/archive/anchor execution required')
        for label, measurement in sections.items():
            _, outcomes = _measurement(measurement, _cell_genome(cell), executor)
            score = cell['scores'][label]
            _require(score['outcomes'] == outcomes and score['manifest'] == measurement['manifest_hash'],
                     'cell admission scores differ from replayed manifest/outcomes')
        families.add(cell['family']); niches.add(cell['niche'])
    _require(len(families) >= 2 and len(niches) >= 2, 'parallel structural/skill families and independent niches required')
    events = population.events
    contemporary = set()
    for event in events:
        kind, cid = event.get('event'), event.get('cell')
        if kind == 'seed':
            _require(cid in population.cells and not population.cells[cid]['parents'], 'seed event must bind actual founder')
            contemporary.add(cid)
        elif kind == 'birth':
            _require(cid in population.cells and event.get('parents') == population.cells[cid]['parents']
                     and all(p in contemporary for p in event['parents']), 'birth must use contemporary distinct live parents')
        elif kind == 'admitted':
            _require(cid in population.cells and population.cells[cid]['parents'], 'admission must bind registered birth')
            contemporary.add(cid)
            contemporary.discard(population.cells[cid]['parents'][0])
        elif kind == 'retired':
            contemporary.discard(cid)
    _require(contemporary == {c for c, record in population.cells.items() if record['state'] == 'live'},
             'selection event replay differs from final live population')
    _require(any(e.get('event') == 'birth' for e in events), 'registered cell births required')
    for cid, cell in population.cells.items():
        if not cell['parents']:
            continue
        recipient = population.cells[cell['parents'][0]]
        selection = execution.get('selection', {}).get(cid, {})
        caps = selection.get('caps', {})
        wall_limit, size_limit = caps.get('max_wall_seconds'), caps.get('max_numeric_positions')
        _require(type(wall_limit) in (int, float) and math.isfinite(wall_limit) and wall_limit > 0
                 and type(size_limit) is int and size_limit > 0, 'predeclared finite resource caps required')
        started = time.perf_counter()
        _, fixture_outcomes = _measurement(selection.get('fixtures'), _cell_genome(cell), executor)
        elapsed = time.perf_counter() - started
        def positions(value):
            return sum(positions(v) for v in value) if isinstance(value, list) else 1
        caps_passed = elapsed <= wall_limit and sum(positions(v) for v in cell['params'].values()) <= size_limit
        fixtures_passed = all(fixture_outcomes.values())
        history = population.histories[recipient.get('history_key', recipient['structure'])].records
        earlier = []
        for opponent in history:
            if opponent['graph_hash'] == cid:
                break
            earlier.append(opponent)
        decision = OpponentHistory(earlier).admission(recipient['scores'], dict(cell['scores'],
                    caps_passed=caps_passed, fixtures_passed=fixtures_passed))
        # A rejected child need not improve. An admitted child must reproduce selection.
        admitted = any(e.get('event') == 'admitted' and e.get('cell') == cid for e in events)
        _require(type(selection.get('accepted')) is bool and selection['accepted'] == admitted
                 and (not admitted or decision['accepted']), 'admitted cell has no replayed paired development gain or resource/fixture pass')
        _require(any(e.get('event') == ('admitted' if admitted else 'rejected') and e.get('cell') == cid for e in events),
                 'every cell birth requires explicit selection result')
    for records in population.histories.values():
        for record in records.records:
            cid = record['graph_hash']
            _require(cid in population.cells and population.cells[cid]['state'] != 'rejected' and record['scores'] == population.cells[cid]['scores']
                     and record['provenance'] == population.cells[cid]['provenance'], 'historical opponents must bind executed registry cells')


def _attachment(population, bundle, executor):
    from .adhesion import validate_adhesion
    spec, catalogue = bundle['spec'], bundle['catalogue']
    contract = validate_adhesion(spec, catalogue)
    parents = bundle['parent_cells']
    _require(set(parents) == set(spec['parents']) and len(set(parents.values())) >= 2,
             'typed attachment must bind distinct actual population parents')
    families = set()
    for name, cid in parents.items():
        _require(cid in population.cells, 'attachment contributor absent from registry')
        cell = population.cells[cid]
        # Named layer tensors bind the catalogue's executable typed definition.
        tensors = {str(i)+':'+k: v for i, layer in enumerate(catalogue[name]['layers']) for k, v in layer.items()}
        _require(tensors == cell['params'], 'typed contributor numerics differ from actual population parent')
        families.add(cell['family'])
    _require(len(families) >= 2 and contract['new_numeric_positions'] > 0,
             'cross-family numerical attachment and authored IO glue required')
    child = population.cells.get(bundle.get('child_id'))
    _require(child is not None and set(child['parents']) == set(parents.values()), 'attachment must be a registered population birth')
    glue = {n['id']+':'+k: v for n in spec['nodes'] if n['kind'] == 'glue' for k, v in n['params'].items()}
    _require(child['params'] == glue and child['provenance'].get('graph_hash') == contract['candidate_hash'],
             'registered attachment child must bind exact authored glue and typed graph')
    _author(bundle['provenance'], 'spec', spec)
    _require(bundle['provenance']['prompt'].get('parent_hashes') ==
             {name: digest(population.cells[cid]['params']) for name, cid in parents.items()},
             'attachment author prompt must bind actual contributing parent tensors')
    _require(child['provenance'].get('raw_response') == bundle['provenance'].get('raw_response')
             and child['provenance'].get('model') == bundle['provenance'].get('model'),
             'registered attachment author provenance differs from executed graph')
    genome = {'spec': spec, 'founders': catalogue}
    required = ['parent', 'random', 'zero'] + ['ablate-cell:'+p for p in spec['parents']] + [
               'ablate-glue:'+n['id'] for n in spec['nodes'] if n['kind'] == 'glue']
    _controls(bundle, genome, executor, required)


def _ensemble(run, execution, executor, population):
    archive = run.get('ensemble_population', {})
    records = archive.get('records', {})
    history = archive.get('historical_opponents', [])
    _require(records and history and len(set(history)) == len(history) and all(h in records and records[h].get('state') != 'rejected' for h in history), 'separate actual MoE registry and opponent lineage required')
    live = archive.get('live')
    _require(live in records and records[live].get('state') in ('admitted', 'live', 'bootstrap-control'), 'valid independent MoE live identity required')
    evidence = execution.get('ensemble', {})
    _require(set(evidence) == set(records), 'every whole-MoE genome requires independent execution')
    proposed = False
    for cid, record in records.items():
        genome = record.get('genome', {})
        _require(set(genome) == {'cells', 'router', 'topology'} and genome['topology'],
                 'whole-MoE requires literal cell groups, authored router and executable topology')
        _require(len(genome['cells']) >= 2 and all(cell in population.cells and params == population.cells[cell]['params']
                 for cell, params in genome['cells'].items()), 'whole-MoE must consume actual selected registry weight cells')
        _require(len({population.cells[cell]['family'] for cell in genome['cells']}) >= 2, 'whole-MoE must combine complementary live families')
        router = genome['router']
        validate_literal(router)
        n = len(genome['cells'])
        _require(set(router) == {'route.W', 'route.b'} and len(router['route.W']) == n
                 and all(len(row) == n for row in router['route.W']) and len(router['route.b']) == n,
                 'whole-MoE literal router must align actual selected cells')
        _require(cid == digest(genome), 'whole-MoE identity differs from executable literal genome')
        _author(record.get('provenance', {}), 'genome', genome)
        _require(record['provenance']['prompt'].get('cell_hashes') ==
                 {cell: digest(params) for cell, params in genome['cells'].items()},
                 'whole-MoE author prompt must bind selected actual cell groups')
        if record.get('parents'):
            proposed = True
            parent = record['parents'].get('moe')
            _require(parent in records and any(c.get('charged') is True and c.get('child') == cid
                     and c.get('parent') == parent for c in archive.get('charges', [])), 'charged separate MoE parent required')
            _require(record['provenance'].get('parent_hash') == parent
                     and record['provenance']['prompt'].get('parent_hash') == parent,
                     'whole-MoE author must bind actual archived parent')
            controls = ['parent', 'random', 'zero', 'uniform-router'] + ['ablate-cell:'+cell for cell in genome['cells']]
            _require(evidence[cid].get('controls', {}).get('parent', {}).get('genome') == records[parent]['genome'],
                     'whole-MoE parent control bound to wrong archive member')
            _controls(evidence[cid], genome, executor, controls)
            events = [e for e in archive.get('events', []) if e.get('event') == 'proposal' and e.get('id') == cid and e.get('parent') == parent]
            _require(len(events) == 1 and type(events[0].get('accepted')) is bool, 'whole-MoE attempt requires independent selection event')
            if events[0]['accepted']:
                outcomes = evidence[cid]['measurement']['outcomes']
                _require(all(sum(outcomes.values()) > sum(evidence[cid]['controls'][label]['measurement']['outcomes'].values())
                             for label in controls), 'admitted MoE requires executed gain over parent and controls')
                for hid in history:
                    if hid == cid:
                        break
                    _require(not any(v and not outcomes.get(task, False) for task, v in records[hid]['outcomes'].items()),
                             'admitted MoE regresses on executed historical opponents')
                _require(record['state'] in ('admitted', 'live', 'retired') and cid in history, 'MoE admission and archive membership disagree')
            else:
                _require(record['state'] == 'rejected' and cid not in history and cid != live, 'rejected MoE cannot enter live/history membership')
        else:
            _measurement(evidence[cid]['measurement'], genome, executor)
        _require(evidence[cid]['measurement']['outcomes'] == record['outcomes'], 'whole-MoE archive scores differ from execution')
    _require(proposed, 'whole MoE must evolve independently of cell population')

def _broad(population, bundle, executor):
    families = {c['family'] for c in population.cells.values() if c['state'] == 'live'}
    reps = bundle.get('representatives', {})
    _require(set(reps) == families, 'broad lane requires every eligible live family')
    for family, cid in reps.items():
        _require(cid in population.cells and population.cells[cid]['state'] == 'live'
                 and population.cells[cid]['family'] == family, 'broad representative must be live and belong to its family')
    environments = bundle.get('environments', [])
    _require(environments, 'actual broad development environments required')
    executed = set()
    for environment in environments:
        _require(environment.get('partition') == 'development', 'broad environment must be development only')
        rows = environment.get('receipts', {})
        _require(set(rows) == set(reps.values()), 'every broad family requires an applicability receipt')
        for cid, row in rows.items():
            cell = population.cells[cid]
            compatible = environment.get('input_type') == cell['provenance'].get('input_type')
            _require(environment.get('input_type') and cell['provenance'].get('input_type'), 'explicit typed applicability required')
            if not compatible:
                _require(row.get('status') == 'inapplicable' and row.get('correct') is None
                         and row.get('scored_tasks') == 0 and row.get('reason'), 'incompatible input must remain unscored')
            else:
                _require(row.get('status') == 'executed', 'compatible broad environment must actually execute')
                _measurement(row['measurement'], _cell_genome(cell), executor)
                executed.add(cell['family'])
    _require(executed == families, 'every eligible family must reach an applicable broad execution')


def require_two_level_run(run, *, replay=None):
    """Require complete executed architecture; intermediate runs fail this gate.

    ``architecture_execution`` format cellforge-two-level-execution/1 contains:
    cells[cid][fresh|archive|anchor] measurements; attachment bundle; ensemble[cid]
    measurement/control bundles; broad representative/environment applicability
    bundles. Measurements bind genome_hash, development partition, manifest_hash
    of exact {id,request,target} cases, predictions and boolean outcomes. Controls
    carry their actual genome and a measurement on the identical cases.
    """
    try:
        _require(run.get('selection_partition') == 'development', 'reporting cannot select')
        population = CellPopulation.from_snapshot(run['population'])
        execution = run.get('architecture_execution', {})
        _lineage(population, execution.get('attachment', {}).get('child_id'))
        _require(execution.get('format') == FORMAT, 'complete replayable architecture execution bundle required')
        _require(isinstance(replay, dict) and all(callable(replay.get(k)) for k in ('cell', 'attachment', 'ensemble')),
                 'trusted target-blind cell, attachment and whole-ensemble executors required')
        assembly = run.get('assembly', {})
        _require(assembly.get('cell_ids') and all(c in population.cells and population.cells[c]['state'] == 'live'
                 for c in assembly['cell_ids']), 'assembly requires actual selected live population cells')
        population.assemble(assembly['router'], assembly['cell_ids'])
        _cell_execution(population, execution, replay['cell'])
        _attachment(population, execution['attachment'], replay['attachment'])
        _ensemble(run, execution, replay['ensemble'], population)
        _broad(population, execution['broad'], replay['cell'])
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError('missing or malformed executed architecture evidence: '+str(error)) from error
    return True








