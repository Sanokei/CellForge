"""Serial owner-authorized maker run, isolated from the head-center driver."""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import bfcl_skill as B
import head_arc_cells as H
import skillcore as S


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    global B
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--oracle', action='store_true')
    p.add_argument('--start-tier', type=int, choices=(0, 1), default=1)
    p.add_argument('--module', default='bfcl_skill')
    a = p.parse_args()
    B = importlib.import_module(a.module)
    template, niche = B.TEMPLATE, B.NICHENAME
    shapes = S.TEMPLATES[template]['shapes']
    positions = sum(int(np.prod(shape)) for shape in shapes.values())
    a.root.mkdir(parents=True, exist_ok=True)
    (a.root / 'SERVICE.pid').write_text(str(os.getpid()))
    seeds = [1701, 1702, 1703]
    manifest = a.root / 'STANDARD.json'
    if manifest.exists():
        standard = json.loads(manifest.read_text())
    else:
        standard = {'arms': {'llm': {}, 'random': {}}, 'seeds': seeds, 'seeds_declared_before_run': True,
                    'budget_per_arm': 0, 'charged_positions_per_arm': 0, 'niches': [niche], 'founders': [], 'genomes': [], 'proposals': [],
                    'scientific_acceptance': False, 'oracle': a.oracle,
                    'limitations': ['BFCL-derived generated requests, not an official BFCL score',
                                    'No ARC transfer or joint router/crossover result', 'Three maker seed entries share a prompt; not independent model draws'],
                    'module': a.module, 'template': template, 'positions': positions}
        write(manifest, standard)
    rows = B.schema_rows(a.source)
    rng = random.Random(91877)
    rng.shuffle(rows)
    # Source ids are disjoint across generated development and clean reporting.
    devrows, reportrows = rows[:48], rows[48:96]
    if len(devrows) < 24 or len(reportrows) < 24:
        raise RuntimeError('not enough disjoint schema rows')
    write(a.root / 'SPLIT.json', {'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
          'dev_ids': [r['id'] for r in devrows], 'report_ids': [r['id'] for r in reportrows],
          'seeds': seeds, 'report_used_for_selection': False})
    write(a.root / 'TASKS.json', {'development': {niche: [r['id'] for r in devrows]}})
    tier = a.start_tier
    parents = {s: None for s in seeds}
    random_parents = {s: None for s in seeds}
    start = 1
    previous = sorted(a.root.glob('GENERATION-*.json'), key=lambda x: int(x.stem.split('-')[-1]))
    if previous:
        last = json.loads(previous[-1].read_text())
        start, tier = last['generation'] + 1, last['next_tier']
        parents = {int(k): v for k, v in last['parents'].items()}
        random_parents = {int(k): v for k, v in last.get('random_parents', {}).items()} or random_parents
        if not last.get('random_parents'):
            standard['limitations'].append('Generations before lineage correction scored random proposals without random-parent admission; not balanced evolution evidence')
        if tier==1:
            current=[B.generated(row,33000+i,tier) for i,row in enumerate(devrows)]
            if all(parents[s] and B.score(parents[s],current)[0]>=.95*len(current) for s in seeds):
                print('Solved tier already has its generation; no further maker calls',flush=True)
                return
    for gen in range(start, a.rounds + 1):
        if (a.root / 'STOP').exists():
            write(a.root / 'STATUS.json', {'status': 'owner_stop', 'utc': now()})
            return
        write(a.root / 'STATUS.json', {'status': 'authoring', 'utc': now(), 'generation': gen, 'tier': tier})
        dev = [B.generated(row, 33000 + i, tier) for i, row in enumerate(devrows)]
        write(a.root / ('DEV-%d.json' % gen), dev)
        items = []
        for seed in seeds:
            item = {'task_id': str(seed), 'examples': dev[:4]}
            if previous:
                last_generation = json.loads(previous[-1].read_text())
                prior = [d for d in last_generation['decisions'] if d['arm'] == 'llm' and d['seed'] == seed and d.get('candidate')]
                if prior:
                    candidate_id = prior[-1]['candidate']
                    genome = next(g for g in reversed(standard['genomes']) if g['hash'] == candidate_id)
                    rejected = S.make_cell(template, json.loads((a.root / genome['weights_file']).read_text()))
                    _, outcomes = B.score(rejected, dev)
                    feedback = []
                    for pair, ok in zip(dev, outcomes):
                        if ok:
                            continue
                        try:
                            prediction = S.run(rejected, pair['input'], pair.get('context', []))
                        except Exception as error:
                            prediction = type(error).__name__ + ': ' + str(error)
                        feedback.append(dict(pair, predicted=prediction))
                        if len(feedback) == 6:
                            break
                    item.update(last_attempt_params=rejected['params'], last_attempt_failures=feedback)
            if parents[seed]:
                _, outcomes = B.score(parents[seed], dev)
                failures = []
                for pair, ok in zip(dev, outcomes):
                    if ok:
                        continue
                    try:
                        pred = S.run(parents[seed], pair['input'], pair.get('context', []))
                    except Exception as e:
                        pred = type(e).__name__
                    failures.append(dict(pair, predicted=pred))
                    if len(failures) == 4:
                        break
                item.update(previous_params=parents[seed]['params'], counterexamples=failures)
            items.append(item)
        prompt = ('Do not use tools, run commands, inspect files, search, or write code. Reason from the examples and answer literal tensors only. '
                  'A tool invocation invalidates the whole submission. Write literal tensors only for each seeded cell. ' + S.TEMPLATES[template]['doc'] +
                  ' All entries finite in [-9,9]. Return JSON array [{task_id,params,rationale}], one per seed. '
                  'Shapes: ' + json.dumps({k:list(s) for k,s in shapes.items()}) + '. Infer from generated examples, revise from failures. '
                  'A fresh submission must differ from all your previous submissions.\n' + json.dumps(items))
        if not (a.root / ('PROMPTS-%d.json' % gen)).exists():
            write(a.root / ('PROMPTS-%d.json' % gen), {'utc': now(), 'prompt': prompt})
        work = a.root / ('AUTHOR-%d' % gen)
        if a.oracle:
            params = {'route': [1, 0, 0, 0], 'bind': [9, 0, 0, 1], 'type': (np.eye(4)*3).tolist()}
            raw = [{'task_id': str(s), 'params': params, 'rationale': 'ORACLE plumbing only'} for s in seeds]
            work.mkdir(exist_ok=True)
            write(work / 'response.json', raw)
        else:
            try:
                if (work / 'EXIT.json').exists() and not (a.root / ('GENERATION-%d.json' % gen)).exists():
                    events = (work / 'events.jsonl').read_text()
                    if 'command_execution' in events or 'mcp_tool_call' in events:
                        raise RuntimeError('recovered historical tool-using maker attempt; no release')
                    raise RuntimeError('recovered incomplete historical maker attempt; no release')
                raw = H.ask_maker(prompt, work, False, None, [])
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                write(work / 'INVALID.json', {'utc': now(), 'reason': str(error), 'charged_declared_positions': positions * len(seeds)})
                raw = [{'task_id': str(s), 'params': None, 'rationale': 'invalid maker attempt: ' + str(error)} for s in seeds]
        decisions, charges = [], []
        all_saturated = True
        for seed in seeds:
            submitted = [x for x in raw if str(x.get('task_id')) == str(seed)] if isinstance(raw, list) else []
            # Every returned entry is charged, even invalid or repeated tensors.
            for entry in submitted or [None]:
                cell = S.make_cell(template, entry['params']) if entry and S.valid_params(template, entry.get('params')) else None
                r = random.Random(seed * 1000 + gen * 31 + len(charges))
                control = S.make_cell(template, {k: np.array([r.uniform(-9, 9) for _ in range(int(np.prod(shape)))]).reshape(shape).tolist()
                                 for k, shape in shapes.items()})
                baseline = B.score(parents[seed], dev)[0] if parents[seed] else 0
                all_saturated &= baseline >= .95 * len(dev)
                for arm, proposed in [('llm', cell), ('random', control)]:
                    score = B.score(proposed, dev)[0] if proposed else 0
                    repeated = proposed and any(x['arm'] == arm and x['seed'] == seed and x['hash_or_null'] == proposed['id'] for x in standard['proposals'])
                    arm_parent = parents[seed] if arm == 'llm' else random_parents[seed]
                    arm_baseline, old_outcomes = B.score(arm_parent, dev) if arm_parent else (0, [False] * len(dev))
                    _, new_outcomes = B.score(proposed, dev) if proposed else (0, [False] * len(dev))
                    fresh = [B.generated(row, 55000 + i, tier) for i, row in enumerate(devrows)]
                    fresh_gain = (B.score(proposed, fresh)[0] if proposed else 0) > (B.score(arm_parent, fresh)[0] if arm_parent else 0)
                    no_regression = all(not before or after for before, after in zip(old_outcomes, new_outcomes))
                    accepted = bool(proposed and not repeated and score > arm_baseline and fresh_gain and no_regression)
                    outcome = 'invalid' if proposed is None else 'neutral' if repeated else 'accepted' if accepted else 'rejected'
                    if accepted:
                        (parents if arm == 'llm' else random_parents)[seed] = proposed
                    filename = '%s-%d-%d-%d.json' % (arm, seed, gen, len(decisions))
                    if proposed:
                        write(a.root / filename, proposed['params'])
                        standard['genomes'].append({'hash': proposed['id'], 'niche': niche, 'origin':
                            ('llm_founder' if gen == 1 else 'llm_mutation') if arm == 'llm' else 'random_mutation',
                            'parents': [arm_parent['id']] if arm_parent else [], 'weights_file': filename, 'accuracy': score / len(dev), 'generation': gen, 'arm': arm})
                        if arm == 'llm' and gen == 1:
                            standard['founders'].append({'hash': proposed['id'], 'niche': niche, 'author': 'llm',
                                'actor': 'gpt-6.1-sol low' if not a.oracle else 'ORACLE', 'rationale': entry.get('rationale', ''), 'weights_file': filename})
                    standard['proposals'].append({'arm': arm, 'seed': seed, 'hash_or_null': proposed['id'] if proposed else None, 'charged': True, 'outcome': outcome})
                    charges.append({'arm': arm, 'seed': seed, 'positions': positions, 'charged_positions': positions, 'charged': positions})
                    decisions.append({'seed': seed, 'niche': niche, 'arm': arm, 'candidate': proposed['id'] if proposed else None,
                        'solved': score, 'accepted': accepted, 'reasons': [outcome], 'total': len(dev),
                        'baseline': arm_baseline, 'no_regression': no_regression, 'fresh_gain': fresh_gain})
        write(a.root / ('CHARGES-%d.json' % gen), charges)
        (a.root / ('CHARGES-%d.jsonl' % gen)).write_text(''.join(json.dumps(x) + '\n' for x in charges))
        standard['charged_positions_per_arm'] = sum(
            x.get('positions', 0) for path in a.root.glob('CHARGES-*.jsonl') for x in
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if x['arm'] == 'llm')
        counts = [sum(x['arm'] == arm and x['seed'] == seed for x in standard['proposals'])
                  for arm in ('llm', 'random') for seed in seeds]
        standard['budget_per_arm'] = max(counts)
        standard['proposal_counts_equal'] = len(set(counts)) == 1
        # Once the easy tier saturates, change tasks; never rewrite that solved tier.
        solved = [B.score(parents[s], dev)[0] if parents[s] else 0 for s in seeds]
        next_tier = 1 if tier == 0 and min(solved) >= .95 * len(dev) else tier
        write(a.root / ('GENERATION-%d.json' % gen), {'generation': gen, 'utc': now(), 'total': len(dev), 'tier': tier,
              'next_tier': next_tier, 'decisions': decisions, 'parents': parents, 'random_parents': random_parents})
        write(manifest, standard)
        previous = sorted(a.root.glob('GENERATION-*.json'), key=lambda x: int(x.stem.split('-')[-1]))
        print(json.dumps({'generation': gen, 'tier': tier, 'solved': solved, 'total': len(dev)}), flush=True)
        saturated_hard = tier == 1 and min(solved) >= .95 * len(dev)
        tier = next_tier
        if saturated_hard:
            break
    # Reporting is computed once after all selection, and cannot alter parents.
    report = [B.generated(row, 87000 + i, tier) for i, row in enumerate(reportrows)]
    write(a.root / 'REPORT_TASKS.json', report)
    write(a.root / 'REPORT.json', {'utc': now(), 'tier': tier, 'total': len(report), 'oracle': a.oracle,
          'scores': {s: B.score(parents[s], report)[0] if parents[s] else 0 for s in seeds},
          'random_scores': {s: B.score(random_parents[s], report)[0] if random_parents[s] else 0 for s in seeds},
          'used_for_selection': False})
    write(a.root / 'STATUS.json', {'status': 'rounds_finished_not_goal_complete', 'utc': now(), 'generations': gen})


if __name__ == '__main__':
    main()
