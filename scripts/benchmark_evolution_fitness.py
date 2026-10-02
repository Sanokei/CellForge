"""One bounded local throughput qualification, immutable source snapshot caller."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.lm import score_texts
from cellforge.reference.evolution_fitness import EvolutionFitness,compiled_content_fitness


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--items',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('preserve prior receipt')
    g=Graph.load(args.model);items=[x for x in json.loads(Path(args.items).read_text())['items'] if x['kind']=='next_byte']
    sources=[Path(__file__),Path(__import__('cellforge.reference.evolution_fitness',fromlist=['x']).__file__),
             Path(__import__('cellforge.runtime.compiled',fromlist=['x']).__file__),
             Path(__import__('cellforge.runtime.lm',fromlist=['x']).__file__),
             Path(__import__('cellforge.runtime.ops',fromlist=['x']).__file__)]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    receipt={'protocol':'language-throughput-qualification/1','scope':'local author-visible next-byte development, no selection search',
             'founder_hash':g.hash(),'manifest_sha256':hashlib.sha256(Path(args.items).read_bytes()).hexdigest(),
             'source_hashes':hashes,'window_bytes':64,'cases':[]}
    tick=time.perf_counter();e=EvolutionFitness(g,items);receipt['prefix_cache_startup_seconds']=time.perf_counter()-tick
    variants=[('founder',g)]
    for name in ('cl_router.sp.W1','cl_bigram.W10'):
        child=g.copy();child.params[name].flat[0]+=.125;variants.append((name,child))
    state=g.copy();state.params['cl_name_write.e0'].flat[0]+=.125;variants.append(('state_mutation_fallback',state))
    for label,child in variants:
        tick=time.perf_counter();scores,info=score_texts(child,[i['text'] for i in items]);reference_seconds=time.perf_counter()-tick
        expected=[float(info['per_pos_bits'][j,:len(i['text'].encode())].sum()) for j,i in enumerate(items)]
        tick=time.perf_counter();cold=e.evaluate(child);cold_wall=time.perf_counter()-tick
        tick=time.perf_counter();warm=e.evaluate(child);warm_wall=time.perf_counter()-tick
        tick=time.perf_counter();full=compiled_content_fitness(child,items);full_wall=time.perf_counter()-tick
        deltas={key:max(abs(a-b) for a,b in zip(result['next_byte']['per_document_content_bits'],expected))
                for key,result in [('cold',cold),('warm',warm),('full',full)]}
        assert max(deltas.values())<=1e-10 and not scores['failed'].any()
        receipt['cases'].append({'label':label,'reference_seconds':reference_seconds,'reference_content_bits':expected,
           'cold_cached':cold,'cold_total_seconds':cold_wall,'cold_end_to_end_evals_per_second':1/cold_wall,
           'warm_cached':warm,'warm_total_seconds':warm_wall,'warm_end_to_end_evals_per_second':1/warm_wall,
           'full_compiled':full,'full_compiled_total_seconds':full_wall,'max_content_bit_deltas':deltas})
    receipt['source_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    receipt['claim']='full prefix/context/state retained; numerical parity within 1e-10; no loss pruning or general QA evidence'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('cases','source_hashes')}))
    print(json.dumps([{'case':x['label'],'reference_s':x['reference_seconds'],'cold_total_s':x['cold_total_seconds'],
                      'warm_total_s':x['warm_total_seconds'],'full_compiled_total_s':x['full_compiled_total_seconds'],
                      'deltas':x['max_content_bit_deltas']} for x in receipt['cases']]))


if __name__=='__main__':main()
