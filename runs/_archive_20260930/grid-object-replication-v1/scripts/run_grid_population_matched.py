# SPDX-License-Identifier: MIT-0
"""Canonical living-genotype pilot with equal physical selection evaluation.

No gradient methods. Query labels are reporting-only after winner freeze.
The earlier pilot and its immutable receipts are intentionally untouched.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.construction.living_population import LivingPopulation
from cellforge.reference.grid_curriculum import FAMILIES,generate_task
from cellforge.reference.grid_false_positives import score_with_consistency
from build_grid_two_slot_genome import GENES,SLOTS,instantiate_genome,crossover,mutate,assert_compatible
from build_grid_adjacency_recolor_gene import generate_adjacency_family
from build_grid_object_association_child import family as object_family

EXPECTED_NATIVE='4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def canonical_genotype(base,graph):
    """Validate declared function; metadata lineage cannot rejuvenate it."""
    assert_compatible(base,graph)
    for key in ('grid_interface','grid_attempts','state_dim','profile'):
        if base.meta.get(key)!=graph.meta.get(key): raise ValueError('runtime contract mismatch: '+key)
    alleles=graph.meta.get('two_slot_genome',{}).get('alleles')
    if not isinstance(alleles,list) or len(alleles)!=2 or any(g not in base.meta['two_slot_genome']['available_genes'] for g in alleles):
        raise ValueError('two validated declared alleles required')
    for slot,allele in zip(SLOTS,alleles):
        if graph.defs[slot]!=base.defs[allele]: raise ValueError('slot disagrees with declared bank allele')
    graph.validate();check_profile(graph,'ladder-micro/1')
    return digest({'shared_base_hash':base.hash(),'alleles':alleles})


def evaluate_selection(graph,selection,*,predictor_factory=None):
    """Physically attempt EVERY demo, including those after initial failure."""
    if any(set(task)!={'train'} for task in selection): raise ValueError('selection view must contain train pairs only')
    start=time.perf_counter();cpu=time.process_time();calls=0;solves=[];errors=[]
    factory=predictor_factory or (lambda g:GridPredictor(g,backend='compiled'))
    try: predictor=factory(graph)
    except Exception as exc:
        return {'solves':[False]*len(selection),'solved':0,'tasks':len(selection),'failed':True,
                'errors':[{'phase':'predictor_init','type':type(exc).__name__}],'predictor_calls':0,
                'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}
    for task in selection:
        success=True
        for demo in task['train']:
            calls+=1
            try:
                values=predictor(task['train'],demo['input'])
                matched=demo['output'] in values
            except Exception as exc:
                errors.append({'phase':'demo_call','type':type(exc).__name__}); matched=False
            success=bool(matched) and success
        solves.append(bool(success))
    failed=bool(errors)
    # Execution failures exclude a genotype globally from selection/living.
    if failed: solves=[False]*len(selection)
    return {'solves':solves,'solved':sum(solves),'tasks':len(solves),'failed':failed,'errors':errors,
            'predictor_calls':calls,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}


def tick_with_evaluation(population,evaluation):
    """Existing neutral genotypes retain age and cannot revive at death tick."""
    existing=evaluation['genotype_hash'] in {m.genotype_hash for m in population.members}
    if existing: population.admit([evaluation]);population.advance_age()
    else: population.advance_age();population.admit([evaluation])


def frozen_corpora(private,*,per_family=20,include_object=False):
    if type(per_family) is not int or not 1<=per_family<=20:raise ValueError('bounded per-family count1..20 required')
    selection_groups=[generate_task(f,1741000+j*10000+i) for j,f in enumerate(FAMILIES) for i in range(per_family)]
    selection_groups+=generate_adjacency_family(per_family,1741000)['tasks']
    validation=[generate_task(f,1941000+j*10000+i) for j,f in enumerate(FAMILIES) for i in range(per_family)]
    validation+=generate_adjacency_family(per_family,1941000)['tasks']
    if include_object:
        for target,seed in ((selection_groups,2837000),(validation,2847000)):
            for task in object_family(seed)['tasks'][:per_family]:
                task['family']='object_recolor';target.append(task)
    fingerprints=lambda tasks:[digest({'train':t['train'],'test':t['test']}) for t in tasks]
    a,b=fingerprints(selection_groups),fingerprints(validation)
    if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)&set(b): raise ValueError('synthetic exact I/O duplicates/overlap')
    selection=[{'train':t['train']} for t in selection_groups]
    for task in selection_groups:
        # Remove original generated query dictionaries, not only a view.
        task.pop('test',None);task.pop('metadata',None)
    del selection_groups
    directory=Path(private)/'arcagi2/data/training'
    paths=[p for k,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in directory.glob('*.json')) if int(k,16)%5!=0][:200]
    if len(paths)!=200: raise ValueError('nativeDEV200 missing')
    native_hash=hashlib.sha256()
    for path in paths: native_hash.update(path.stem.encode()+b':'+hashlib.sha256(path.read_bytes()).digest())
    if native_hash.hexdigest()!=EXPECTED_NATIVE: raise ValueError('nativeDEV content manifest mismatch')
    commitments={'selection_manifest':digest(selection),'selection_full_io_manifest':digest(a),
                 'validation_full_io_manifest':digest(b),'exact_io_duplicate_audit':'zero within/cross-split duplicates before policy freeze',
                 'native_content_manifest':native_hash.hexdigest(),'selection_tasks':len(selection),
                 'selection_demo_pairs':sum(len(t['train']) for t in selection),'validation_tasks':len(validation)}
    return selection,validation,paths,commitments


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inventory',required=True);parser.add_argument('--private',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--object-bank',action='store_true');parser.add_argument('--per-family',type=int,default=20)
    parser.add_argument('--seed-base',type=int,default=772031);parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args();out=Path(args.out)
    if out.exists(): raise ValueError('fresh output required')
    inventory_path=Path(args.inventory);inventory=json.loads(inventory_path.read_bytes());graphs={}
    for row in inventory['artifacts']:
        graph=Graph.load(inventory_path.parent/row['name'])
        if graph.hash()!=row['hash']: raise ValueError('frozen graph changed')
        graphs[row['name']]=graph
    base=graphs['base'];founders=[graphs['parent-object-identity' if args.object_bank else 'parent-fill-identity'],graphs['parent-identity-adjacency']]
    for graph in graphs.values(): canonical_genotype(base,graph)
    selection,validation,native_paths,commitments=frozen_corpora(args.private,per_family=args.per_family,include_object=args.object_bank)
    policy={'protocol':'living-two-slot-matched/2','arms':['crossover','mutation_only','random_search'],
            'proposals_per_arm':4,'founder_evaluations_per_arm':2,'physical_evaluations_per_arm':6,
            'capacity':4,'max_age':3,'seed_base':args.seed_base,'selection_seed_base':1741000,'validation_seed_base':1941000,
            'native_query_scoring':not args.synthetic_only,
            'object_bank':args.object_bank,'per_family':args.per_family,
            'object_selection_seed':2837000 if args.object_bank else None,'object_validation_seed':2847000 if args.object_bank else None,
            'fitness':'all demonstration pairs reconstructed by either ranked output; not fixed-rank consistency or query success',
            'genotype_identity':'validated shared base hash plus two functional alleles; full artifact hash separate lineage',
            'mortality':'neutral contemporary genotype preserves age, including death boundary; no elite immortality',
            'selection_cache':'none; independently execute two founders and every proposal in each arm',
            'winner':'eligible archive maximum absolute demo solve count, canonical genotype then artifact hash tie; archive never parent pool',
            'mate_rule':'same-hash draw replaced by lowest canonical-hash distinct contemporary when possible',
            'random_control':'uniform two alleles from same bank; independent of parent fitness; same physical call budget',
            'mortality_sharing':'same mechanics in crossover/mutation; no individual-mechanism efficacy isolation',
            'report_cache':'identical canonical winner may reuse report only, explicitly not independent replicates',
            'extinction':'remaining proposal budget executes failed baseline placeholders; no archive-parent fallback, no eligible birth/winner',
            'audit_evaluations':0,'final_evaluations':0,'gradient_steps':0,'gradient_estimators':0,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shared_base_hash':base.hash(),
            'source_graphs':{k:g.hash() for k,g in graphs.items()},**commitments,
            'limitations':'four proposals per arm, narrow synthetic curriculum and exposed nativeDEV; no efficacy/frontier claim'}
    out.mkdir();ledger=Ledger(out/'ledger.jsonl')
    def write(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
    write('policy.json',policy);ledger.append('policy_frozen',policy,'operator')
    winners={};attempts={};costs={};physical=0
    for arm in policy['arms']:
        tick=time.perf_counter();cpu=time.process_time();records=[];representatives={};attempts[arm]=[]
        pop=LivingPopulation(capacity=4,max_age=3,task_manifest=commitments['selection_manifest'],n_tasks=len(selection))
        def evaluated(graph,kind,forced_failure=None):
            nonlocal physical
            canonical=canonical_genotype(base,graph);fitness=evaluate_selection(graph,selection);physical+=1
            if forced_failure:
                fitness.update({'failed':True,'solves':[False]*len(selection),'solved':0})
                fitness['errors'].append({'phase':'proposal','type':forced_failure})
            record={'arm':arm,'kind':kind,'canonical_genotype':canonical,'artifact_hash':graph.hash(),
                    'genome':graph.meta['two_slot_genome'],'fitness':fitness}
            records.append(record);representatives[canonical]=graph
            ledger.append('physical_selection_evaluation',record,'operator')
            admission={'genotype_hash':canonical,'task_manifest':commitments['selection_manifest'],'solves':fitness['solves'],'failed':fitness['failed']}
            return record,admission
        founder_admissions=[]
        for founder_index,graph in enumerate(founders):
            record,admission=evaluated(graph,'founder');founder_admissions.append(admission)
            graph.save(out/(arm+'-founder-'+str(founder_index+1)))
        pop.admit(founder_admissions)
        for index in range(4):
            seed=policy['seed_base']+index
            if pop.members:
                sampling=pop.sample_parents(seed=seed,count=2);chosen=[representatives[h] for h in sampling['parent_hashes']]
            else: sampling={'parent_hashes':[],'probabilities':{},'seed':seed};chosen=[]
            if arm=='crossover' and chosen:
                if canonical_genotype(base,chosen[0])==canonical_genotype(base,chosen[1]) and len(pop.members)>1:
                    chosen[1]=representatives[next(m.genotype_hash for m in pop.members if m.genotype_hash!=canonical_genotype(base,chosen[0]))]
                child=crossover(chosen[0],chosen[1],seed)
            elif arm=='mutation_only' and chosen: child=mutate(chosen[0],seed)
            elif arm=='random_search':
                available=base.meta['two_slot_genome']['available_genes']
                rng=np.random.default_rng(seed);child=instantiate_genome(base,[available[int(x)] for x in rng.integers(len(available),size=2)],operator_seed=seed)
            else:
                # A failed birth still consumes physical evaluation budget;
                # this baseline placeholder is never a parent or eligible winner.
                child=base.copy();child.meta['failed_birth']={'reason':'extinct','seed':seed}
            forced_failure='ExtinctPopulation' if not chosen and arm!='random_search' else None
            record,admission=evaluated(child,'proposal',forced_failure);record.update({'attempt':index+1,'sampling':sampling,'actual_parent_artifact_hashes':[g.hash() for g in chosen] if arm!='random_search' else []})
            attempts[arm].append(record);child.save(out/(arm+'-attempt-'+str(index+1)))
            tick_with_evaluation(pop,admission);ledger.append('population_tick',{'arm':arm,'metrics':pop.metrics(),'events':pop.events},'operator')
        eligible=[r for r in records if not r['fitness']['failed']]
        if not eligible: raise RuntimeError('no eligible archive winner')
        best=min(eligible,key=lambda r:(-r['fitness']['solved'],r['canonical_genotype'],r['artifact_hash']))
        # Load exact archived artifact, not merely the latest canonical representative.
        if best['kind']=='founder': winner=next(g for g in founders if g.hash()==best['artifact_hash'])
        else: winner=Graph.load(out/(arm+'-attempt-'+str(best['attempt'])))
        winners[arm]=winner;winner.save(out/(arm+'-winner'))
        costs[arm]={'physical_evaluations':len(records),'predictor_calls':sum(r['fitness']['predictor_calls'] for r in records),
                    'evaluation_cpu_seconds':sum(r['fitness']['cpu_seconds'] for r in records),'evaluation_wall_seconds':sum(r['fitness']['wall_seconds'] for r in records),
                    'arm_cpu_seconds':time.process_time()-cpu,'arm_wall_seconds':time.perf_counter()-tick,
                    'failed_evaluations':sum(r['fitness']['failed'] for r in records)}
        ledger.append('arm_complete',{'arm':arm,'costs':costs[arm],'winner':best},'operator');write('selection-progress.json',{'attempts':attempts,'costs':costs})
    write('winners-frozen.json',{arm:{'artifact_hash':g.hash(),'canonical_genotype':canonical_genotype(base,g)} for arm,g in winners.items()})
    ledger.append('winners_frozen',{arm:g.hash() for arm,g in winners.items()},'operator')
    # Decode native query labels only AFTER selected hashes are frozen.
    native=[];recheck=hashlib.sha256()
    for path in native_paths:
        raw=path.read_bytes();recheck.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest())
        if not args.synthetic_only:native.append((path.stem,json.loads(raw)))
    if recheck.hexdigest()!=commitments['native_content_manifest']: raise ValueError('native content changed after policy freeze')
    validation_counts={}
    for task in validation:
        family=task.get('family',task.get('metadata',{}).get('family'))
        validation_counts[family]=validation_counts.get(family,0)+1
    if len(validation_counts)!=(10 if args.object_bank else 9) or set(validation_counts.values())!={args.per_family}: raise ValueError('balanced validation family manifest changed')
    reports={};rows=[]
    for arm,graph in winners.items():
        canonical=canonical_genotype(base,graph);reused=canonical in reports
        if not reused:
            predictor=GridPredictor(graph,backend='compiled')
            reports[canonical]=[(dict(score_with_consistency([(str(i),t) for i,t in enumerate(validation)],predictor,record_tasks=True),split='fresh-synthetic-validation'))]
            if not args.synthetic_only:reports[canonical].append(dict(score_with_consistency(native,predictor,record_tasks=True),split='native-development'))
            reports[canonical][0].update({'macro_family_exact_accuracy':reports[canonical][0]['tasks_solved']/len(validation),
                                         'macro_scope':f'{len(validation_counts)} equal-size {args.per_family}-task families; macro equals micro, not per-family measurements',
                                         'family_task_counts':validation_counts})
        for report in reports[canonical]: rows.append(dict(report,arm=arm,artifact_hash=graph.hash(),canonical_genotype=canonical,report_reused=reused,independent_replicate=False))
    import resource
    expected_calls=6*commitments['selection_demo_pairs']
    result={'policy':policy,'attempts':attempts,'costs':costs,'physical_selection_evaluations':physical,'expected_predictor_calls_per_arm':expected_calls,
            'equal_physical_calls':all(c['predictor_calls']==expected_calls for c in costs.values()),'rows':rows,
            'winners':{a:g.hash() for a,g in winners.items()},'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'limitations':policy['limitations']}
    ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify();write('result.json',result)
    print(json.dumps({'rows':[{k:v for k,v in row.items() if k!='task_records'} for row in rows],'physical_selection_evaluations':physical,'costs':costs}))
if __name__=='__main__':main()
