# SPDX-License-Identifier: MIT-0
"""Discrete context-key recipe pilot, matched physical evaluations, no gradients.

Synthetic development only. No ARC files, downloads, APIs or final split reads.
Every candidate executes the compiled CellForge graph, including demo adaptation.
"""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
from build_grid_context_memory_gene import build_context_memory_gene, generate_context_memory_family
from cellforge.runtime.grid import GridPredictor
from cellforge.governance.ledger import Ledger


FOUNDERS = (511, 186, 56, 146)  # full, orthogonal cross, center+horizontal, center+vertical


def masked_graph(mask, parents=(), operator='authored-seed'):
    if type(mask) is not int or not 0 <= mask < 512 or not mask & 16:
        raise ValueError('nine-bit mask must retain center')
    g = build_context_memory_gene()
    g.params['key.W'] *= np.array([bool(mask & (1 << i)) for i in range(9)])[None]
    g.meta.update(context_mask=mask, operator=operator, parent_hashes=list(parents),
                  construction='discrete context projection recipe/1',
                  gradients=False, query_adaptation=False)
    g.validate()
    return g


def offspring(a, b, rng, crossover=True):
    mask = a
    if crossover:
        slots = int(rng.integers(512))
        mask = (a & slots) | (b & (511 ^ slots))
    index = int(rng.choice([0, 1, 2, 3, 5, 6, 7, 8]))
    return (mask ^ (1 << index)) | 16


def evaluate(g, tasks):
    predictor = GridPredictor(g, backend='compiled')
    solved = 0; pixels = 0.; batches = [0., 0.]
    start = time.perf_counter(); cpu = time.process_time()
    for index, task in enumerate(tasks):
        pair = task['test'][0]
        outputs = predictor(copy.deepcopy(task['train']), copy.deepcopy(pair['input']))
        expected = np.array(pair['output'])
        accuracy = max(float(np.mean(np.array(x) == expected))
                       if np.array(x).shape == expected.shape else 0. for x in outputs)
        solved += int(accuracy == 1.); pixels += accuracy
        batches[index % 2] += accuracy
    return dict(tasks=len(tasks), solved=solved, pixel_accuracy=pixels/len(tasks),
                batch_accuracy=[x/(len(tasks)//2) for x in batches],
                wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu)


def main():
    import resource  # Linux measurement host; genome helpers also import on Windows.
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--proposals', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.proposals <= 64: raise ValueError('bounded proposals1..64')
    out = Path(args.out)
    if out.exists(): raise ValueError('fresh output required')
    out.mkdir(parents=True); ledger = Ledger(out/'ledger.jsonl')
    # Independent fresh seeds, fixed before any candidate score.
    selection = generate_context_memory_family(24, 2477000)['tasks']
    report = generate_context_memory_family(24, 2487000)['tasks']
    def digest(tasks):
        return hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    fingerprints = lambda tasks: {json.dumps({'train':t['train'],'test':t['test']}, sort_keys=True) for t in tasks}
    if fingerprints(selection) & fingerprints(report): raise ValueError('I/O overlap')
    policy = dict(protocol='context-mask-discrete-pilot/1', proposals_per_arm=args.proposals,
                  founders=list(FOUNDERS), selection_tasks=24, report_tasks=24,
                  selection_hash=digest(selection), report_hash=digest(report),
                  arms=['crossover','mutation','random'], physical_evaluations_per_arm=4+args.proposals,
                  fitness='pixel accuracy; offspring accepted only if both disjoint batches improve or tie and total improves',
                  archive='all candidates, including rejected and duplicates; no evaluation cache',
                  maker_input='prior aggregate fine-memory failure evidence only; founder masks authored before pilot',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gradients=False, benchmark_files_loaded=False, final_evaluations=0,
                  limitations='small directional-copy synthetic pilot; no native ARC or sealed claim')
    (out/'policy.json').write_text(json.dumps(policy, indent=2)+'\n')
    ledger.append('policy_frozen', policy, 'operator')
    winners = []
    for arm in policy['arms']:
        rng = np.random.default_rng(2497000)
        archive = []; accepted = 0
        def score(mask, parents, operator):
            g = masked_graph(mask, parents, operator); h = g.save(out/arm/'graphs')
            row = dict(mask=mask, graph_hash=h, parents=parents, operator=operator,
                       fitness=evaluate(g, selection))
            archive.append(row); ledger.append('candidate_scored', dict(arm=arm, **row), 'operator')
            return row
        population = [score(mask, [], 'authored-seed') for mask in FOUNDERS]
        for _ in range(args.proposals):
            ranked = sorted(population, key=lambda x:(x['fitness']['pixel_accuracy'],-x['mask']), reverse=True)
            a, b = ranked[:2]
            mask = int(rng.integers(512)) | 16 if arm=='random' else offspring(a['mask'], b['mask'], rng, arm=='crossover')
            child = score(mask, [] if arm=='random' else [a['graph_hash'],b['graph_hash']] if arm=='crossover' else [a['graph_hash']], arm)
            old = a['fitness']; new = child['fitness']
            improvement = all(x >= y for x,y in zip(new['batch_accuracy'],old['batch_accuracy'])) and new['pixel_accuracy'] > old['pixel_accuracy']
            accepted += improvement
            if improvement: population = ranked[:3]+[child]
        winner = max(archive, key=lambda x:(x['fitness']['pixel_accuracy'],-x['mask'])) if arm=='random' else max(population,key=lambda x:(x['fitness']['pixel_accuracy'],-x['mask']))
        winners.append(dict(arm=arm, winner=winner, accepted=accepted, evaluations=len(archive)))
    # Freeze every winner before report scoring; report cannot influence search.
    ledger.append('winners_frozen', {'winners':winners}, 'operator')
    for row in winners:
        g = masked_graph(row['winner']['mask'], row['winner']['parents'], row['winner']['operator'])
        row['report'] = evaluate(g, report)
        row['parameter_slots'] = g.stats()['unique_param_storage']
        ledger.append('winner_reported', row, 'operator')
    result = dict(policy=policy, winners=winners,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  ledger_verified=ledger.verify())
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
