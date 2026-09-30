"""Freeze aggregate handoff evidence without copying task data or active logs."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[3]

def read(path):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    complete = b''.join(line for line in lines if line.endswith(b'\n'))
    rows = [json.loads(line.removeprefix(b'RESULT ')) for line in complete.splitlines() if line.strip()]
    return rows, dict(rows=len(rows), sha256=hashlib.sha256(complete).hexdigest(), bytes=len(complete))

def save(directory, evidence):
    (directory / 'HANDOFF_VERIFIED.json').write_text(json.dumps(evidence, indent=2) + '\n')

def main():
    arc = ROOT / 'research/claude_arc'
    dsl = arc / 'dsl_port'
    B, bh = read(dsl / 'cache/res_base.jsonl')
    D, dh = read(dsl / 'cache/res_dsl.jsonl')
    assert len(B) == len(D) == 1000
    assert len({r['task'] for r in B}) == len({r['task'] for r in D}) == 1000
    assert {r['task'] for r in B} == {r['task'] for r in D}
    for r in B + D:
        assert r['split'] == ('val' if zlib.crc32(r['task'].encode()) % 5 == 0 else 'dev')
    assert all(r['n_gen'] == 0 for r in D if r['split'] == 'val')
    # Execute only definitions: analyze.py otherwise runs main on import.
    ns = {}
    exec((dsl / 'analyze.py').read_text().rsplit('\nmain()', 1)[0], ns)
    stats = {}
    bm = {r['task']: r for r in B}; dm = {r['task']: r for r in D}
    for split in ('dev', 'val'):
        tasks = [r['task'] for r in B if r['split'] == split]
        for mode in ('strict', 'top2'):
            res, _ = ns['score'](tasks, bm, dm, mode, False, split)
            stats[split + '_' + mode] = dict(n=len(tasks), **res)
    save(dsl, dict(status='complete baseline and DSL; coverage job separate', sources=dict(baseline=bh, dsl=dh), split='crc32(task ID without .json) % 5', statistics=stats, tests=9,
                   caveats=['Validation differs from the 228-task filename-hash split.', 'Combined analyzer deduplicates by object identity, not prediction hash; counts reflect that policy.', 'No generator filtering on validation.']))
    grok = arc / 'grok'; rows, meta = read(grok / 'results.jsonl')
    keys = [(r['ti'], r['seed'], r['arm'], r['lam']) for r in rows]
    assert len(set(keys)) == len(keys)
    counts = collections.defaultdict(lambda: dict(runs=0, fit_ever=0, fit_final=0, delayed_generalization=0))
    for r in rows:
        assert r['snaps'][-1]['check'] == 100000
        c = counts[f"{r['arm']} lam={r['lam']} control={r['ctrl']}"]
        c['runs'] += 1; c['fit_ever'] += r['efit'] is not None
        final = r['snaps'][-1]; c['fit_final'] += final['demo_exact']
        f = r['fitsnap']
        c['delayed_generalization'] += bool(f and r['efit'] <= 30000 and f['extra20'] < .5 and final['demo_exact'] and final['extra20'] >= .5 and final['extra20'] >= f['extra20'] + .3)
    save(grok, dict(status='complete historical sweep' if len(rows) == 180 else 'partial sweep', expected_runs=180, source=meta, statistics=dict(counts), caveats=['Historical evolved neural-weight experiment; excluded from pure authored/fitted runtime.', 'No additional weight search launched.', 'C stop-at-fit is derived, not an independent arm.']))
    mv = arc / 'multiview'; rows, meta = read(mv / 'results/val.jsonl')
    assert len({r['task'] for r in rows}) == len(rows)
    assert all(r['bucket'] == zlib.crc32(r['task'].encode()) % 5 == 0 for r in rows)
    spec = importlib.util.spec_from_file_location('mv_analysis', mv / 'analyze.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    dev, devmeta = read(mv / 'results/dev.jsonl') if (mv / 'results/dev.jsonl').exists() else ([], None)
    assert all(r['bucket'] == zlib.crc32(r['task'].encode()) % 5 != 0 for r in dev)
    assert len({r['task'] for r in dev}) == len(dev)
    da = mod.arms(dev) if dev else {}
    eligible = {k: v for k, v in da.items() if k[0] in 'BDE'}
    selected = max(eligible, key=lambda k: (eligible[k]['solved'], -eligible[k]['fp'])) if eligible else None
    va = mod.arms(rows)
    save(mv, dict(status='complete validation' if len(rows) == 228 else 'partial validation', development_status='complete' if len(dev) == 772 else 'incomplete', development_source=devmeta, development_statistics=da, development_selected_arm=selected, selected_validation=va.get(selected), expected_validation_tasks=228, source=meta, split='crc32(filename including .json) % 5', statistics=va, tests=8, caveats=['Arm selection uses development outcomes only.', 'Runtime estimates reuse cached timings rather than separately timed arms.']))
    a3 = ROOT / 'research/claude_arc3'; stats = {}
    for path in sorted((a3 / 'results').glob('*.jsonl')):
        rows, meta = read(path)
        assert len({(r['game'], r['seed']) for r in rows}) == len(rows)
        stats[path.name] = dict(source=meta, complete=len(rows)==25, levels=sum(r['levels'] for r in rows), available_levels=sum(r['n_levels'] for r in rows), mean_score=sum(r['score'] for r in rows)/len(rows), seconds=sum(r['secs'] for r in rows), peak_mb=max(r['peak_mb'] for r in rows))
    save(a3, dict(status='full v5 seeds 0,1,2 completed; some flag ablations partial', statistics=stats, caveats=['Compare complete 25-game runs only.', 'v4 has only seed 0; no paired multi-seed advantage established.', 'No hidden games measured.', 'Full v5 clears fewer levels than v4 on seed 0.']))

if __name__ == '__main__':
    main()
