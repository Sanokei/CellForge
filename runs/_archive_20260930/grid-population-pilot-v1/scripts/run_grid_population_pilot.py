# SPDX-License-Identifier: MIT-0
"""Equal-budget discrete graph population pilot; selection only on demo pairs."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.construction.living_population import LivingPopulation
from cellforge.reference.grid_curriculum import FAMILIES,generate_task
from cellforge.reference.grid_false_positives import score_with_consistency
from build_grid_two_slot_genome import GENES,instantiate_genome,crossover,mutate
from build_grid_adjacency_recolor_gene import generate_adjacency_family

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--private',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
 if out.exists():raise ValueError('fresh output required')
 invpath=Path(a.inventory);inv=json.loads(invpath.read_bytes());graphs={}
 for row in inv['artifacts']:
  g=Graph.load(invpath.parent/row['name'])
  if g.hash()!=row['hash']:raise ValueError('frozen graph changed')
  graphs[row['name']]=g
 base=graphs['base'];parents=[graphs['parent-fill-identity'],graphs['parent-identity-adjacency']]
 groups=[generate_task(f,1741000+j*10000+i) for j,f in enumerate(FAMILIES) for i in range(20)]
 groups+=generate_adjacency_family(20,1741000)['tasks']
 # Selection corpus physically omits query labels/inputs.
 selection=[{'train':task['train']} for task in groups];manifest=digest(selection)
 policy={'protocol':'living-two-slot-pilot/1','proposals_per_arm':4,'arms':['crossover','mutation_only','random_search'],'capacity':4,'max_age':3,'selection_tasks':len(selection),'selection_manifest':manifest,'selection_seed_base':1741000,'seed_base':772031,'fitness':'all demonstrations reconstructed by either ranked output, task Boolean; no queries','winner':'highest absolute solve count among evaluated archive, hash tie','random_control':'uniform two alleles from shared four-allele bank, no fitness-guided proposal','mate_rule':'if sampled same hash twice and another contemporary exists, second mate is lowest-hash distinct contemporary','cache_rule':'all proposals counted; same two alleles reuse deterministic fitness, reported as cache hits','mortality_sharing':'same living helper on crossover/mutation; controls do not isolate these mechanisms','validation_seed_base':1941000,'validation_per_family':20,'public_evaluation_loaded':False,'audit_evaluations':0,'final_evaluations':0,'gradients':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_graphs':{k:g.hash() for k,g in graphs.items()},'limitations':'tiny four-proposal pilot, narrow synthetic curriculum, exposed nativeDEV; no frontier inference'}
 out.mkdir();ledger=Ledger(out/'ledger.jsonl')
 def write(name,x):(out/name).write_text(json.dumps(x,indent=2)+'\n')
 write('policy.json',policy);ledger.append('policy_frozen',policy,'operator');cache={};allgraphs={};attempts={};winners={}
 def evaluate(g):
  # Every proposal counted; identical functional alleles share an evaluation cache.
  key=tuple(g.meta['two_slot_genome']['alleles']);allgraphs[g.hash()]=g
  if key in cache:return cache[key],True
  predictor=GridPredictor(g,backend='compiled');start=time.perf_counter();cpu=time.process_time();solves=[]
  for task in selection:
   success=True
   for demo in task['train']:
    success=success and demo['output'] in predictor(task['train'],demo['input'])
   solves.append(bool(success))
  row={'solves':solves,'solved':sum(solves),'tasks':len(solves),'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start};cache[key]=row
  return row,False
 initial=[]
 for g in parents:
  row,hit=evaluate(g);initial.append((g,row));ledger.append('founder_scored',{'hash':g.hash(),'fitness':row,'cache_hit':hit},'operator')
 for arm in policy['arms']:
  pop=LivingPopulation(capacity=4,max_age=3,task_manifest=manifest,n_tasks=len(selection));archive={g.hash():row for g,row in initial};pop.admit([{'genotype_hash':g.hash(),'task_manifest':manifest,'solves':row['solves']} for g,row in initial]);attempts[arm]=[]
  for i in range(4):
   seed=policy['seed_base']+i;sampling=pop.sample_parents(seed=seed,count=2);chosen=[allgraphs[h] for h in sampling['parent_hashes']]
   # Distinct contemporary mates whenever possible, avoiding accidental selfing.
   if arm=='crossover' and chosen[0].hash()==chosen[1].hash() and len(pop.members)>1:
    chosen[1]=allgraphs[next(m.genotype_hash for m in pop.members if m.genotype_hash!=chosen[0].hash())]
   if arm=='crossover':child=crossover(chosen[0],chosen[1],seed)
   elif arm=='mutation_only':child=mutate(chosen[0],seed)
   else:
    rng=np.random.default_rng(seed);child=instantiate_genome(base,[GENES[int(x)] for x in rng.integers(4,size=2)],operator_seed=seed)
   row,hit=evaluate(child);archive[child.hash()]=row
   receipt={'arm':arm,'attempt':i+1,'hash':child.hash(),'genome':child.meta['two_slot_genome'],'sampling':sampling,'actual_parents':[g.hash() for g in chosen] if arm!='random_search' else [],'fitness':row,'cache_hit':hit}
   attempts[arm].append(receipt);ledger.append('proposal_scored',receipt,'operator');child.save(out/(arm+'-attempt-'+str(i+1)))
   pop.advance_age();pop.admit([{'genotype_hash':child.hash(),'task_manifest':manifest,'solves':row['solves']}]);ledger.append('population_tick',{'arm':arm,'metrics':pop.metrics(),'events':pop.events},'operator')
  winner=min(archive,key=lambda h:(-archive[h]['solved'],h));winners[arm]=allgraphs[winner];winners[arm].save(out/(arm+'-winner'));write('selection-progress.json',attempts)
 write('winners-frozen.json',{arm:g.hash() for arm,g in winners.items()});ledger.append('winners_frozen',{arm:g.hash() for arm,g in winners.items()},'operator')
 validation=[generate_task(f,1941000+j*10000+i) for j,f in enumerate(FAMILIES) for i in range(20)]+generate_adjacency_family(20,1941000)['tasks']
 directory=Path(a.private)/'arcagi2/data/training';paths=[p for k,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in directory.glob('*.json')) if int(k,16)%5!=0][:200]
 if len(paths)!=200:raise ValueError('nativeDEV200 missing')
 native=[(p.stem,json.loads(p.read_bytes())) for p in paths];rows=[];reportcache={}
 for arm,g in winners.items():
  key=tuple(g.meta['two_slot_genome']['alleles'])
  if key not in reportcache:
   predictor=GridPredictor(g,backend='compiled');reportcache[key]=[dict(score_with_consistency([(str(i),t) for i,t in enumerate(validation)],predictor),split='fresh-synthetic-validation'),dict(score_with_consistency(native,predictor),split='native-development')]
  for row in reportcache[key]:rows.append(dict(row,arm=arm,hash=g.hash(),alleles=list(key)))
 result={'policy':policy,'attempts':attempts,'winners':{arm:g.hash() for arm,g in winners.items()},'rows':rows,'unique_functional_evaluations':len(cache),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'limitations':policy['limitations']}
 ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify();write('result.json',result);print(json.dumps({'rows':rows,'unique_functional_evaluations':len(cache)}))
if __name__=='__main__':main()
