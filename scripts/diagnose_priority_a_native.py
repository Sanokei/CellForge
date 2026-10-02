# SPDX-License-Identifier: MIT-0
"""Train-only fired-cell evidence from the existing exposed DEV200 anchor.

Run after the current scoring batch terminates, sequentially on the device.
This is diagnostic inference, not a candidate proposal or final evaluation.
"""
import argparse,hashlib,json,time,resource
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.construction.maker_diagnostics import train_counterexamples

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();completed=json.loads((a.batch/'result.json').read_text())
    if len(completed['rows'])!=9 or completed['report_loaded']!=0:raise ValueError('complete unreported batch required')
    if a.out.exists():raise ValueError('fresh diagnostic destination required')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    paths=[x for h,x in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in a.training.glob('*.json')) if int(h,16)%5!=0][:200]
    if len(paths)!=200:raise ValueError('complete DEV200 required')
    manifest=hashlib.sha256();tasks=[]
    for path in paths:
        raw=path.read_bytes();manifest.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest())
        tasks.append(dict(task_id=path.stem,family='public-training',train=json.loads(raw)['train']))
    if manifest.hexdigest()!='4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4':
        raise ValueError('DEV200 manifest mismatch')
    # Source query dictionaries do not enter the diagnostic task objects.
    a.out.mkdir();start=time.perf_counter();counts=[]
    for row in completed['rows']:
        stem=f"{row['seed']}-{row['arm']}";g=Graph.load(a.batch/(stem+'-selected'))
        if g.hash()!=row['selected_hash']:raise ValueError('exact selected graph required')
        diag=train_counterexamples(g,tasks[:100],limit=5,per_task_limit=1)
        diag['native_manifest']=manifest.hexdigest()
        diag['diagnostic_population']=dict(source='fixed first100 tasks of existing DEV200 order',
            task_ids=[t['task_id'] for t in tasks[:100]])
        (a.out/(stem+'.json')).write_text(json.dumps(diag,indent=2))
        count=dict(seed=row['seed'],arm=row['arm'],graph_hash=g.hash(),
            failures=len(diag['counterexamples']),calls=diag['diagnostic_predictor_calls'])
        counts.append(count);print(json.dumps(count),flush=True)
    (a.out/'result.json').write_text(json.dumps(dict(rows=counts,wall_seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='bounded train-only diagnosis on exposed DEV200; no candidate mutations or report scoring'),indent=2))

if __name__=='__main__':main()
