"""Operator-only WB6 fast-fitness parity/cost; no evolution or report split."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.reference.evolution_fitness import EvolutionFitness,compiled_content_fitness
from cellforge.reference.language_curriculum import validate_partition_manifests
from evolve_language_islands import selection_items,reject_adaptive_mixer


def main():
    p=argparse.ArgumentParser();p.add_argument('--fitted-artifact',required=True)
    for name in ('batch-a','batch-b','context-a','context-b'):p.add_argument('--'+name,required=True)
    p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve previous qualification receipt')
    source_paths=[Path(__file__),Path(__import__('cellforge.reference.evolution_fitness',fromlist=['x']).__file__),
                  Path(__import__('cellforge.runtime.compiled',fromlist=['x']).__file__),Path(__import__('cellforge.runtime.ops',fromlist=['x']).__file__),
                  Path(__import__('evolve_language_islands').__file__)]
    source_before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    start=time.perf_counter();g=Graph.load(args.fitted_artifact)
    reject_adaptive_mixer(g)
    if not g.meta.get('fitted_provenance'):raise ValueError('fitted provenance required')
    paths=[Path(getattr(args,n)) for n in ('batch_a','batch_b','context_a','context_b')]
    raw=[path.read_bytes() for path in paths];items=[selection_items(json.loads(r)) for r in raw]
    if any(not x.get('score_byte_ranges') for group in items[2:] for x in group):raise ValueError('masked context items required')
    validate_partition_manifests({'partition':'selection','items':items[0]+items[2]},
                                 {'partition':'report','items':items[1]+items[3]})
    # Deterministic structural mutation, no measured-value sweep or selection.
    child=g.copy();root=child.defs[child.entries['step']]
    gate=next((node for node in root['nodes'] if node['id']=='authored_residual' and node['use']=='op:scale'),None)
    if gate is None:raise ValueError('declared WB authored_residual gate required')
    before=gate['args']['c'];gate['args']['c']=.01 if before!=.01 else .02
    child.meta['qualification_mutation']={'gate':'authored_residual','before':before,'after':gate['args']['c'],'selection':False}
    cases=[];qualified=True
    for label,group in zip(('corpus_a','corpus_b','context_a','context_b'),items):
        tick=time.perf_counter();scorer=EvolutionFitness(g,group);startup=time.perf_counter()-tick
        for name,candidate in (('seed',g),('residual_mutation',child)):
            tick=time.perf_counter();fast=scorer.evaluate(candidate);cold=time.perf_counter()-tick
            tick=time.perf_counter();full=compiled_content_fitness(candidate,group);full_seconds=time.perf_counter()-tick
            deltas=[abs(a-b) for a,b in zip(fast['next_byte']['per_document_content_bits'],full['next_byte']['per_document_content_bits'])]
            passed=(len(deltas)==len(group) and max(deltas)<=1e-10 and fast['eligible_for_selection'] and full['eligible_for_selection'] and
                    fast['next_byte']['failed_documents']==full['next_byte']['failed_documents'])
            qualified &= passed
            cases.append({'batch':label,'candidate':name,'hash':candidate.hash(),'prefix_startup_seconds':startup,
                          'fast':fast,'full_compiled':full,'cold_total_seconds':cold,'cold_evals_per_second':1/cold,
                          'full_compiled_seconds':full_seconds,'max_document_bit_delta':max(deltas),'parity_passed':passed})
    source_after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    qualified &= source_before==source_after
    receipt={'protocol':'wb6-fast-fitness-qualification/1','seed_hash':g.hash(),'mutation_hash':child.hash(),
             'manifest_sha256':[hashlib.sha256(r).hexdigest() for r in raw],
             'qualified':qualified,'cases':cases,'total_seconds':time.perf_counter()-start,
             'source_hashes':source_before,'source_hashes_after':source_after,'sources_unchanged':source_before==source_after,
             'selection_proposals':0,'report_evaluations':0,'scope':'seed + fixed residual mutation on both existing DEV corpus/context batches; no evolution, report access or scale qualification'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'qualified':qualified,'cases':len(cases),'max_bit_delta':max(x['max_document_bit_delta'] for x in cases),'seconds':receipt['total_seconds']}))
    if not qualified:raise SystemExit('WB6 fast-fitness parity failed')


if __name__=='__main__':main()
