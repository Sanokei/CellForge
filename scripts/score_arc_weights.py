"""Score frozen iteration checkpoints and matched weight ablations; never select or author."""
import argparse, hashlib, importlib.util, json, os, pathlib, random, time

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)

def transform(value, rng=None):
    if isinstance(value, dict): return {k: transform(v, rng) for k, v in value.items()}
    if isinstance(value, list): return [transform(v, rng) for v in value]
    return 0.0 if rng is None else rng.uniform(-9, 9)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run', type=pathlib.Path)
    parser.add_argument('--arc-root', type=pathlib.Path, required=True)
    args=parser.parse_args(); root=args.run.resolve()
    os.environ['ARC_ROOT']=str(args.arc_root.resolve())
    plan=json.loads((root/'ARC_DECLARATION.json').read_text())
    lock=root/'ARC_EVALUATION.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        started=time.perf_counter()
        spec=importlib.util.spec_from_file_location('frozen_arc_eval',root/'arc_eval.py')
        ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
        paths=[p for h,p in sorted((hashlib.sha256(p.stem.encode()).hexdigest(),p)
               for p in (args.arc_root/'training').glob('*.json')) if int(h,16)%5!=0][:200]
        if len(paths)!=200:raise ValueError('existing DEV200 corpus missing')
        commitment=hashlib.sha256()
        tasks=[]
        for path in paths:
            raw=path.read_bytes();sha=hashlib.sha256(raw).digest()
            commitment.update(path.stem.encode()+b':'+sha)
            raw_task=json.loads(raw)
            # Targets remain exclusively in the scoring process.
            view={'train':[{'input':p['input'],'output':p['output']} for p in raw_task['train']],
                  'test':[{'input':p['input']} for p in raw_task['test']]}
            tasks.append((path.stem,view,[p['output'] for p in raw_task['test']]))
        if commitment.hexdigest()!=plan['task_cohort_sha256']:raise ValueError('DEV200 content changed')
        save(root/'ARC_TASK_MANIFEST.json',{'split':'training','ids':[t[0] for t in tasks],
             'sha256':commitment.hexdigest(),'selection_use':False})
        def measure(weights,template,label):
            per={};errors=[];tick=time.perf_counter()
            for index,(tid,view,expected) in enumerate(tasks):
                try:
                    attempts=ev.predict_with_template(weights,view,template)
                    per[tid]=len(attempts)==len(expected) and all(any(a==target for a in result[:2])
                              for result,target in zip(attempts,expected))
                except Exception as error:
                    per[tid]=False;errors.append({'task_id':tid,'error':type(error).__name__+': '+str(error)})
                if (index+1)%20==0:
                    print(json.dumps({'stage':label,'tasks':index+1,'solved':sum(per.values())}),flush=True)
                    save(root/'ARC_PROGRESS.json',{'stage':label,'tasks':index+1,'solved':sum(per.values())})
            return {'solved':sum(per.values()),'total':len(tasks),'per_task':per,
                    'errors':errors,'seconds':time.perf_counter()-tick}
        history=[];measurements=[]
        for checkpoint in plan['checkpoints']:
            path=root/checkpoint['weights_file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=checkpoint['file_sha256']:
                raise ValueError('checkpoint weights changed')
            weights=json.loads(path.read_text())['tensors']
            row=measure(weights,checkpoint['template'],f"iteration-{checkpoint['iteration']}")
            record=dict(checkpoint,**row);measurements.append(record)
            history.append({'iteration':checkpoint['iteration'],'solved':row['solved'],
                            'weights_hash':checkpoint['weights_hash'],
                            'weights_file':checkpoint['weights_file'],'source_run':checkpoint['source_run'],
                            'measured_retrospectively':True})
            save(root/'ARC_HISTORY.json',history);save(root/'ARC_ITERATION_SCORES.json',measurements)
        final=plan['checkpoints'][-1];weights=json.loads((root/final['weights_file']).read_text())['tensors']
        zero=measure(transform(weights),final['template'],'zero-weights')
        save(root/'ARC_ZERO_CONTROL.json',zero)
        controls=[]
        for seed in plan['random_seeds']:
            control=transform(weights,random.Random(seed))
            save(root/f'control-random-{seed}.json',{'tensors':control})
            row=measure(control,final['template'],f'random-{seed}')
            controls.append(dict(seed=seed,**row));save(root/'ARC_RANDOM_CONTROLS.json',controls)
        claimed={k:measurements[-1][k] for k in ('solved','total','per_task')}
        score={'split':'training','task_ids':[t[0] for t in tasks],
               'weights_file':final['weights_file'],'evaluator':'arc_eval.py','claimed':claimed,
               'controls':{'zero_weights':{'solved':zero['solved']},
                           'random_weights':{'solved':max(r['solved'] for r in controls),
                             'seeds':plan['random_seeds'],'aggregation':'maximum of three physical evaluations'}},
               'history':history,'arc1_arc2_merged_by_task_id':True,
               'scope':'Existing ARC-AGI-2 public-training DEV200, containing ARC-1 tasks by shared ID. Retrospective accuracy audit, no new generation or independent transfer claim.',
               'arc_root':str(args.arc_root.resolve()),'task_manifest_sha256':commitment.hexdigest(),
               'seconds':time.perf_counter()-started,'scientific_acceptance':False}
        save(root/'ARC_SCORE.json',score)
        print(json.dumps({'finished':True,'solved':claimed['solved'],'zero':zero['solved'],
                         'random':[r['solved'] for r in controls],'history':[h['solved'] for h in history]}),flush=True)
    finally:lock.unlink()

if __name__=='__main__':main()
