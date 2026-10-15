"""Matched frozen-model long-prefix numerics/performance; no task labels/fitting."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


def replay(graph,engine,text,mode):
    tokens=[1]+[byte+3 for byte in text.encode('utf-8')]
    width=len(graph.meta['vocab'])
    ctx=np.zeros((1,graph.meta['ctx_len'],width))
    state=np.zeros((1,graph.meta.get('state_dim',0)))
    digest=hashlib.sha256()
    started=time.perf_counter()
    for t,token in enumerate(tokens):
        ctx=np.roll(ctx,-1,axis=1)
        ctx[:,-1]=0
        ctx[:,-1,token]=1
        inputs={'ctx':ctx,'prev':ctx[:,-1],'state':state,'step':np.array([[float(t)]])}
        if mode=='compiled-state-prefill' and t+1<len(tokens):
            output=engine.run_bundle_outputs(graph.entries['step'],inputs,['state'],'/gen')
        else:
            output=engine.run_bundle(graph.entries['step'],inputs,'/gen')
        digest.update(output['state'].tobytes())
        state=np.where(np.isfinite(output['state']),output['state'],0)
    final_logits_hash=hashlib.sha256(output['logits'].tobytes()).hexdigest()
    generated=[]
    for t in range(8):
        logits=output['logits'].reshape(1,-1).copy()
        logits[:,[0,1]]=-np.inf
        token=int(logits.argmax(-1)[0])
        generated.append(token)
        if token==2:
            break
        ctx=np.roll(ctx,-1,axis=1)
        ctx[:,-1]=0
        ctx[:,-1,token]=1
        output=engine.run_bundle(graph.entries['step'],{'ctx':ctx,'prev':ctx[:,-1],'state':state,
                 'step':np.array([[float(len(tokens)+t)]])},'/gen')
        digest.update(output['state'].tobytes())
        state=np.where(np.isfinite(output['state']),output['state'],0)
    return {'mode':mode,'elapsed_seconds':time.perf_counter()-started,'state_sequence_sha256':digest.hexdigest(),
            'final_prompt_logits_sha256':final_logits_hash,'generated_token_ids':generated,'exec':engine.stats.as_dict(),
            'counter_scope':'executed demanded closure semantic ops (cached constants still counted)' if mode=='compiled-state-prefill' else 'full eager semantic ops (cached constants still counted)'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prefix-bytes',type=int,default=1024)
    args=parser.parse_args()
    if not 1<=args.prefix_bytes<=8192:
        parser.error('prefix bound1..8192')
    graph=Graph.load(args.artifact)
    text=('Public question: summarize observations. OCR observation: abc 42, uncertainty.\n'*
          (args.prefix_bytes//70+2))[:args.prefix_bytes]
    results=[]
    compilation=[]
    for mode in ['reference','compiled-full','compiled-state-prefill']:
        started=time.perf_counter()
        engine=Interpreter(graph) if mode=='reference' else CompiledInterpreter(graph)
        compilation.append({'mode':mode,'seconds':time.perf_counter()-started})
        results.append(replay(graph,engine,text,mode))
    reference=results[0]
    exact_keys=['state_sequence_sha256','final_prompt_logits_sha256','generated_token_ids']
    equivalent=all(all(result[key]==reference[key] for key in exact_keys) for result in results)
    full_stats_equal=results[0]['exec']==results[1]['exec']
    report={'protocol':'compiled-prefix-differential/1','model_hash':graph.hash(),'prefix_bytes':len(text.encode()),
            'prefix_sha256':hashlib.sha256(text.encode()).hexdigest(),'exact_numerical_and_token_equivalence':equivalent,
            'full_compiled_stats_exact':full_stats_equal,'compilation':compilation,'results':results,
            'python':platform.python_version(),'numpy':np.__version__,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'backend_sha256':hashlib.sha256(Path(__import__('cellforge.runtime.compiled',fromlist=['x']).__file__).read_bytes()).hexdigest(),
            'limitations':'Single authored model/prefix performance; no benchmark ability gain; state-only demand drops unused prompt logits only.'}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'exact':equivalent,'stats_exact':full_stats_equal,'seconds':[r['elapsed_seconds'] for r in results],
                      'state_prefill_speedup':results[0]['elapsed_seconds']/results[2]['elapsed_seconds']}))
    if not equivalent or not full_stats_equal:
        raise SystemExit(1)


if __name__=='__main__':
    main()
