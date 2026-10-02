# SPDX-License-Identifier: MIT-0
"""Operator attribution from committed selection records, not archive admission."""
import json
from pathlib import Path
from cellforge.governance.ledger import Ledger

def main():
 totals={arm:dict(proposals=0,failed=0,duplicate_genotypes=0,strict_record_gains=0,physical_calls=0,cpu_seconds=0.) for arm in ('crossover','mutation_only','random_search')};runs=[]
 for seed in (772211,772221,772231):
  root=Path(f'runs/grid-expanded-replication-{seed}');l=Ledger(root/'ledger.jsonl');assert l.verify()==108
  entries=l.entries();r=json.loads((root/'result.json').read_text());assert l.find('completed')[0]['payload']=={k:v for k,v in r.items() if k!='ledger_verified'}
  run=dict(seed=seed,arms={})
  for arm in totals:
   founders=[e['payload'] for e in entries if e['type']=='physical_selection_evaluation' and e['payload']['arm']==arm and e['payload']['kind']=='founder']
   assert len(founders)==2;best=max(x['fitness']['solved'] for x in founders);seen={x['canonical_genotype'] for x in founders};events=[]
   for record in r['attempts'][arm]:
    t=totals[arm];fitness=record['fitness'];t['proposals']+=1;t['physical_calls']+=fitness['predictor_calls'];t['cpu_seconds']+=fitness['cpu_seconds']
    failed=bool(fitness['failed']);duplicate=record['canonical_genotype'] in seen
    t['failed']+=failed;t['duplicate_genotypes']+=duplicate;seen.add(record['canonical_genotype'])
    gain=not failed and fitness['solved']>best
    if gain:t['strict_record_gains']+=1
    events.append(dict(attempt=record['attempt'],artifact_hash=record['artifact_hash'],operator=record['genome'].get('operator'),solved=fitness['solved'],previous_best=best,strict_gain=gain,duplicate=duplicate,failed=failed))
    if not failed:best=max(best,fitness['solved'])
   run['arms'][arm]=dict(founder_best=max(x['fitness']['solved'] for x in founders),final_selection_best=best,events=events)
  runs.append(run)
 for t in totals.values():t['strict_gain_per_proposal']=t['strict_record_gains']/t['proposals']
 result=dict(protocol='expanded-operator-attribution/1',runs=runs,totals=totals,scope='strict running-best demonstration selection gains; not living admission, fresh validation or isolated operator causality',data_access='existing development receipts only')
 Path('runs/grid-expanded-operator-attribution-v1.json').write_text(json.dumps(result,indent=2));print(json.dumps(totals))
if __name__=='__main__':main()
