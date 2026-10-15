"""Fixed typed two-slot discrete crossover; no task data or host inference.

Every genome retains the full authored bank, cells and state capacity. Only
flat geometry aliases8/9 differ. This builder does not establish efficacy.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile
from cellforge.construction.transactions import import_def

GENES=('geometry_0','bbox.geometry','bank_local_fill.gene','bank_adjacency_recolor.gene')
SLOTS=('slot8_geometry','slot9_geometry')


def build_base(parent=None,recolor=None):
    repo=Path(__file__).resolve().parents[1]
    parent=parent or Graph.load(repo/'runs/arc-grid-genes/curriculum-portfolio-v1/parent')
    recolor=recolor or Graph.load(repo/'runs/arc-grid-genes/adjacency-recolor-v1')
    graph=parent.copy()
    if graph.meta['state_dim']!=1010:raise ValueError('fixed ten-slot state1010 required')
    imported=import_def(graph,recolor,'adjacency_recolor.gene',prefix='bank_')
    graph.defs['slot8_geometry']=copy.deepcopy(graph.defs['bbox.geometry'])
    for name in (graph.entries['grid_step'],'expert_8'):
        for node in graph.defs[name]['nodes']:
            if node['use']=='bbox.geometry':node['use']='slot8_geometry'
    for gene in GENES:
        definition=graph.defs[gene]
        if set(definition['inputs'])!={'input','input_height','input_width'}:raise ValueError('typed geometry ports required')
        if set(definition['outputs'])!={'colors','height','width'}:raise ValueError('typed geometry outputs required')
        if any(node['use'] in graph.defs and graph.defs[node['use']]['kind']!='cell' for node in definition['nodes']):raise ValueError('flat gene required')
    graph.meta['two_slot_genome']={'protocol':'fixed-two-slot-genome/1','slots':list(SLOTS),
        'alleles':['bbox.geometry','geometry_0'],'available_genes':list(GENES),'founder_hash':parent.hash(),
        'recolor_donor_hash':recolor.hash(),'import_def_lineage':imported,
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'fixed shared authored bank; homologous-slot recombination only; no efficacy/LLM advantage claim'}
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1');return graph


def instantiate_genome(base,genome,parent_hashes=None,operator_seed=None):
    genome=tuple(genome)
    available=tuple(base.meta['two_slot_genome']['available_genes'])
    if len(genome)!=2 or any(gene not in available for gene in genome):raise ValueError('two declared alleles required')
    for gene in available:
        definition=base.defs[gene]
        if set(definition.get('inputs',[]))!={'input','input_height','input_width'} or set(definition.get('outputs',{}))!={'colors','height','width'}:raise ValueError('typed bank geometry required')
    graph=base.copy()
    for slot,gene in zip(SLOTS,genome):graph.defs[slot]=copy.deepcopy(graph.defs[gene])
    record=graph.meta['two_slot_genome']
    record.update({'alleles':list(genome),'parent_hashes':list(parent_hashes or [base.hash()]),
                   'operator_seed':operator_seed,'operator':'instantiate declared alleles'})
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1');return graph


def assert_compatible(left,right):
    if left.meta['two_slot_genome']['available_genes']!=right.meta['two_slot_genome']['available_genes']:raise ValueError('declared gene banks differ')
    if left.params.keys()!=right.params.keys() or any(not np.array_equal(left.params[k],right.params[k]) for k in left.params):raise ValueError('immutable shared parameters differ')
    if left.defs.keys()!=right.defs.keys() or any(left.defs[k]!=right.defs[k] for k in left.defs if k not in SLOTS):raise ValueError('shared bank/wiring differs')
    if left.entries!=right.entries or left.meta['state_dim']!=right.meta['state_dim']:raise ValueError('state/entry mismatch')


def crossover(left,right,seed):
    assert_compatible(left,right);rng=np.random.default_rng(seed)
    alleles=[left.meta['two_slot_genome']['alleles'],right.meta['two_slot_genome']['alleles']]
    donors=list(map(int,rng.integers(2,size=2)))
    child=instantiate_genome(left,[alleles[donors[i]][i] for i in range(2)],[left.hash(),right.hash()],seed)
    child.meta['two_slot_genome'].update({'operator':'homologous uniform two-slot crossover','donor_per_slot':donors,
                                        'no_change_vs_left':child.meta['two_slot_genome']['alleles']==alleles[0]})
    return child


def mutate(parent,seed):
    available=parent.meta['two_slot_genome']['available_genes']
    rng=np.random.default_rng(seed);slot=int(rng.integers(2));choice=available[int(rng.integers(len(available)))]
    alleles=parent.meta['two_slot_genome']['alleles'].copy();before=alleles[slot];alleles[slot]=choice
    child=instantiate_genome(parent,alleles,[parent.hash()],seed)
    child.meta['two_slot_genome'].update({'operator':f'one-slot uniform {len(available)}-allele mutation','mutated_slot':slot,
                                        'old_allele':before,'new_allele':choice,'no_change':before==choice})
    return child


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve prior genome artifacts')
    base=build_base();left=instantiate_genome(base,(GENES[2],GENES[0]));right=instantiate_genome(base,(GENES[0],GENES[3]))
    child=instantiate_genome(base,(GENES[2],GENES[3]),[left.hash(),right.hash()])
    child.meta['two_slot_genome'].update({'operator':'explicit complementary homologous crossover','donor_per_slot':[0,1]})
    rows=[]
    for name,graph in [('base',base),('parent-fill-identity',left),('parent-identity-adjacency',right),('child-fill-adjacency',child)]:
        digest=graph.save(out/name);rows.append({'name':name,'hash':digest,'genome':graph.meta['two_slot_genome'],'stats':graph.stats(),'profile':check_profile(graph,'ladder-micro/1')})
    receipt={'protocol':'two-slot-genome-inventory/1','artifacts':rows,'benchmark_tasks_read':0,'efficacy_measured':False}
    (out/'INVENTORY.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
