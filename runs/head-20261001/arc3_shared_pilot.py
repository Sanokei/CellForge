"""One shared founder on three public development games; run/verify without novelty claims."""
import argparse
import hashlib
import json
from pathlib import Path
import arc3_env as E
import arc3cell as C
import maker as M
from lab_manager import CONTRACT
GAMES = ['ar25', 'lp85', 'sc25']
STEPS = 1500

def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def evaluate(entry, checkpoint=None):
    rows = []
    source = '\n'.join(entry['source_lines'])
    for game in GAMES:
        result = E.play(source, entry['params'], game, STEPS, seed=0, keep_trace=True)
        controls = E.controls(source, entry['params'], game, STEPS, seed=0)
        best = max([controls['zero']['levels'], controls['blind']['levels']] + [r['levels'] for r in controls['random']])
        all_controls = [controls['zero'], controls['blind']] + controls['random']
        controls_valid = all(not r.get('error') for r in all_controls)
        credit = result['levels'] if controls_valid and not result.get('error') and result['levels'] > best else 0
        rows.append(dict(game=game, result=result, controls=controls, controls_valid=controls_valid, credited=credit))
        if checkpoint:
            save(checkpoint, dict(games=rows, complete=len(rows) == len(GAMES)))
    return dict(games=rows, credited_total=sum(r['credited'] for r in rows), raw_total=sum(r['result']['levels'] for r in rows))

def metrics(result):
    def run_metrics(r):
        return (r['levels'], r['used'], r.get('error'), r['level_steps'])
    return [(r['game'], run_metrics(r['result']), r['credited'], r['controls_valid'], run_metrics(r['controls']['zero']), run_metrics(r['controls']['blind']), [run_metrics(x) for x in r['controls']['random']]) for r in result['games']]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['run', 'verify'])
    ap.add_argument('root', type=Path)
    ap.add_argument('--repair-from', type=Path, help='One new charged proposal from a saved runtime failure; not a continuation loop')
    a = ap.parse_args()
    if a.command == 'verify':
        entry = json.loads((a.root / 'FOUNDER.json').read_text(encoding='utf-8'))
        recorded = json.loads((a.root / 'RESULT.json').read_text(encoding='utf-8'))
        actual = evaluate(entry)
        report = dict(matches=metrics(actual) == metrics(recorded), recomputed=actual)
        save(a.root / 'VERIFY.json', report)
        print(json.dumps(dict(matches=report['matches'], credited_total=actual['credited_total'])))
        return 0 if report['matches'] else 1
    used, reset = M.last_used()
    if used is None or used >= M.cap():
        raise SystemExit('Current shared budget reading below cap required.')
    a.root.mkdir(parents=True, exist_ok=False)
    manifest = dict(kind='shared-policy-development-probe/1', created=M.now(), games=GAMES, steps=STEPS, seed=0, writer_calls_max=1, budget_start=used, budget_reset=reset, held_out=False, scientific_acceptance=False, trace_limit=400, stop_rule='One maker call regardless of outcome')
    standard = dict(arms=dict(llm=dict(kind='founder_probe'), random=dict(kind='tensor_ablation_only')), seeds=[0], seeds_declared_before_run=True, budget_per_arm=1, niches=['arc3_shared_development'], founders=[], genomes=[], proposals=[], scientific_acceptance=False, limitations=['One seed and founder; no mutations/crossover', 'Public development games, not held-out', 'Random ablations are not an equally charged random-evolution arm', 'No matched explorer or joint ARC-1/2/3 model', 'Expected standard contract/scientific failures; no waiver'])
    save(a.root / 'PROBE.json', manifest)
    save(a.root / 'STANDARD.json', standard)
    prompt = CONTRACT + '\nOVERRIDE the single-game objective. Write ONE SHARED generic policy; set game=shared. No game facts, identities, screenshots or solutions are provided. Infer object roles and action effects online using frames and persistent mem. Literal tensors guide exploration and short planning. The SAME source and weights run unchanged with fresh memory for each environment. No game fingerprints or stored solutions. Public development pilot, not held-out.'
    if a.repair_from:
        parent = json.loads((a.repair_from / 'FOUNDER.json').read_text(encoding='utf-8'))
        previous = json.loads((a.repair_from / 'RESULT.json').read_text(encoding='utf-8'))
        failures = sorted({row['result']['error'] for row in previous['games'] if row['result'].get('error')})
        if not failures:
            raise SystemExit('Repair requires a recorded runtime failure; no tuning from game scores.')
        manifest['repair_parent'] = str(a.repair_from)
        manifest['repair_parent_sha256'] = hashlib.sha256((a.repair_from / 'FOUNDER.json').read_bytes()).hexdigest()
        prompt += '\nRUNTIME REPAIR ONLY. Previous shared proposal never executed a valid action. Fix the concrete runtime failures, retaining the generic online mechanism. Return complete source and literal tensors. Allowed builtins are: ' + ', '.join(sorted(C.restricted_globals()['__builtins__'])) + '\nFailures: ' + json.dumps(failures) + '\nPrior proposal: ' + json.dumps(parent)
    save(a.root / 'PROMPT.json', dict(prompt=prompt))
    (a.root / 'PROMPT.txt').write_text(prompt, encoding='utf-8')
    proposal = dict(arm='llm', seed=0, hash_or_null=None, charged=True, outcome='invalid')
    try:
        raw = M.ask(prompt, a.root / 'AUTHOR', role='writer', track='arc3-shared-probe')
        save(a.root / 'RESPONSE.json', raw)
        entry = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], dict) else {}
        source = '\n'.join(entry.get('source_lines', []))
        ok, why = C.validate(source)
        if ok:
            ok, why = C.check_shapes(entry.get('shapes', {}), entry.get('params', {}))
        if not ok:
            manifest['invalid'] = why
        else:
            save(a.root / 'FOUNDER.json', entry)
            save(a.root / 'WEIGHTS.json', dict(tensors=entry['params']))
            digest = hashlib.sha256(json.dumps(entry['params'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            standard['founders'].append(dict(hash=digest, niche='arc3_shared_development', author='llm', actor='metered_maker', rationale=entry.get('rationale', ''), weights_file='WEIGHTS.json'))
            proposal.update(hash_or_null=digest, outcome='neutral')
            result = evaluate(entry, a.root / 'PARTIAL_RESULT.json')
            save(a.root / 'RESULT.json', result)
            manifest['credited_total'] = result['credited_total']
            print(json.dumps(dict(raw_total=result['raw_total'], credited_total=result['credited_total'])), flush=True)
    except M.BudgetExhausted as error:
        manifest['budget_refusal'] = str(error)
        proposal['charged'] = False
    except Exception as error:
        manifest['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        standard['proposals'].append(proposal)
        manifest['budget_end'], manifest['budget_reset_end'] = M.last_used()
        save(a.root / 'STANDARD.json', standard)
        save(a.root / 'PROBE.json', manifest)
    return 0 if 'credited_total' in manifest else 1

if __name__ == '__main__':
    raise SystemExit(main())
