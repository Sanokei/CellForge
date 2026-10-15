"""Bounded discrete parameter GA versus founder random search; no gradients."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from calibrate_language_evolution import content_fitness


def parameter_scope(graph, roots):
    seen, params = set(), set()
    def visit(name):
        if name in seen: return
        seen.add(name)
        params.update(graph.params_of(name))
        for child in graph.children(name): visit(child)
    for name in roots: visit(name)
    return sorted(p for p in params if np.any(graph.params[p]))


def mutate(parent, names, seed):
    rng = np.random.default_rng(seed)
    child = parent.copy()
    name = names[int(rng.integers(len(names)))]
    array = child.params[name]
    active = np.flatnonzero(array.ravel())
    indices = rng.choice(active, min(8, len(active)), replace=False)
    # Same fixed relative-noise operator in both arms; discrete selection only.
    sigma = 0.05 * max(1., float(np.median(np.abs(array.ravel()[active]))))
    before = array.ravel()[indices].copy()
    array.ravel()[indices] += rng.normal(0., sigma, len(indices))
    proposal = {"parent_hash": parent.hash(), "seed": seed, "parameter": name,
                "indices": indices.tolist(), "before": before.tolist(),
                "after": array.ravel()[indices].tolist(), "sigma": sigma}
    child.meta["evolution_parameter_proposal"] = proposal
    child.validate()
    return child, proposal


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    private, out = Path(args.private).resolve(), Path(args.out)
    if private.parent != Path('/srv/cellforge-private') or out.exists():
        raise SystemExit('operator-private existing split and fresh output required')
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    out.mkdir()
    founder = Graph.load(args.artifact)
    assert founder.hash() == 'f0b43dbe9c5f0582750e80589b7fa9c6cf55362bb3660b4a7a2db2f3cf8d7beb'
    policy = json.loads((private/'partition-policy.json').read_text())
    raw = (private/'selection.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == policy['selection_sha256']
    selection = json.loads(raw)
    scopes = {'router': parameter_scope(founder, ['router']),
              'experts': parameter_scope(founder, [n for n,d in founder.defs.items() if d.get('role') == 'expert'])}
    assert all(scopes.values())
    config = {'protocol': 'discrete-byte-parameter-minipilot/1', 'founder_hash': founder.hash(),
              'generations': 3, 'islands': ['router', 'experts'], 'proposals_per_island_generation': 1,
              'mutation': '8 uniformly sampled nonzero coordinates of one uniformly sampled active parameter; Gaussian sigma .05*max(1,median abs active)',
              'seed_rule': '20260930 + generation*100 + island index', 'selection': 'strict lower development content bpb, otherwise retain parent',
              'random_control': 'same 6 seeds/scopes/evaluator, every proposal starts from founder; retain best for reporting only',
              'scope': 'parameter-only calibration; module crossover, zero activation, migration, curriculum selection are subsequent stages',
              'partition_policy': policy, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': {'address_space_bytes': 2*1024**3, 'threads': 1, 'nice': 15, 'external_timeout_seconds': 600},
              'report_rule': 'open report-only split only after all selection and winner hashes frozen; never feed its scores into selection',
              'claims': 'tiny public-source development pilot; no independent final, QA, AGI or frontier evidence'}
    write(out/'config.json', config)
    ledger = Ledger(out/'ledger.jsonl')
    ledger.append('configuration_frozen', config, 'operator')
    started = time.monotonic()
    fitness = content_fitness(founder, selection)
    assert fitness['eligible_for_selection']
    baseline = fitness['fitness_bits_per_byte']
    founder.save(out/'graphs'/founder.hash())
    parents = {k: founder for k in scopes}
    losses = {k: baseline for k in scopes}
    random_best, random_loss = founder, baseline
    frozen = [dict(generation=0, ga_hash=founder.hash(), random_hash=founder.hash(), ga_selection_bpb=baseline, random_selection_bpb=baseline)]
    for generation in range(1, 4):
        for index, (island, names) in enumerate(scopes.items()):
            seed = 20260930 + generation*100 + index
            for arm, parent in [('ga', parents[island]), ('random', founder)]:
                child, proposal = mutate(parent, names, seed)
                digest = child.save(out/'graphs'/child.hash())
                tick = time.monotonic()
                result = content_fitness(child, selection)
                score = result['fitness_bits_per_byte']
                accepted = result['eligible_for_selection'] and score < (losses[island] if arm == 'ga' else random_loss)
                if accepted:
                    if arm == 'ga': parents[island], losses[island] = child, score
                    else: random_best, random_loss = child, score
                ledger.append('candidate_evaluation', dict(generation=generation, island=island, arm=arm,
                    child_hash=digest, proposal=proposal, fitness=result, seconds=time.monotonic()-tick,
                    accepted=bool(accepted), donor_retained=True), 'operator')
        best_island = min(losses, key=lambda k: (losses[k], k))
        frozen.append(dict(generation=generation, ga_hash=parents[best_island].hash(), random_hash=random_best.hash(),
                           ga_selection_bpb=losses[best_island], random_selection_bpb=random_loss))
        write(out/'selection-progress.json', frozen)
    write(out/'frozen-winners.json', frozen)
    ledger.append('all_winners_frozen', {'winners': frozen}, 'operator')
    report_raw = (private/'report-only.json').read_bytes()
    assert hashlib.sha256(report_raw).hexdigest() == policy['report_only_sha256']
    reporting = json.loads(report_raw)
    assert {x['family'] for x in selection}.isdisjoint({x['family'] for x in reporting})
    ledger.append('report_split_opened_after_freeze', {'selection_complete': True}, 'operator')
    measured = {}
    for row in frozen:
        for arm in ('ga', 'random'):
            digest = row[arm+'_hash']
            if digest not in measured:
                measured[digest] = content_fitness(Graph.load(out/'graphs'/digest), reporting)
            row[arm+'_report_bpb'] = measured[digest]['fitness_bits_per_byte']
            row[arm+'_report_failed'] = measured[digest]['next_byte']['failed_documents']
    result = {'config': config, 'curve': frozen, 'total_seconds': time.monotonic()-started,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'candidate_evaluations': 12, 'report_unique_graph_evaluations': len(measured),
              'gradient_steps': 0, 'gradient_estimators': 0, 'final_evaluations': 0}
    ledger.append('report_complete', result, 'operator')
    result['ledger_entries_verified'] = ledger.verify()
    write(out/'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__': main()
