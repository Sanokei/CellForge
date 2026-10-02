# SPDX-License-Identifier: MIT-0
"""Seven to nine fixed typed slots; graph-owned demonstration adaptation."""
import argparse,copy,json
from pathlib import Path
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile
from build_grid_associative import Nodes
from build_grid_gene_pool import ranked_two


def build_expanded_genome(donor,alleles=None,*,pre_rank_limitations=None):
    bank=donor.meta.get('two_slot_genome',donor.meta.get('expanded_genome'))['available_genes']
    alleles=list(bank if alleles is None else alleles)
    if not 7<=len(alleles)<=9 or any(x not in bank for x in alleles):raise ValueError('seven to nine declared typed alleles required')
    count=8+len(alleles)
    g=donor.copy();template=copy.deepcopy(g.defs['expert_0']);mapping=copy.deepcopy(g.defs['mapping_0'])
    names=[f'geometry_{i}' for i in range(8)]
    for i,gene in enumerate(alleles,8):
        name=f'expanded_slot_{i}';definition=g.defs[gene]
        if set(definition['inputs'])!={'input','input_height','input_width'} or set(definition['outputs'])!={'colors','height','width'}:raise ValueError('typed flat geometry required')
        g.defs[name]=copy.deepcopy(definition);names.append(name)
        g.defs[f'mapping_{i}']=copy.deepcopy(mapping)
        g.defs[f'mapping_{i}']['nodes'][0]['args'].update(start=i*101,stop=i*101+100)
        expert=copy.deepcopy(template)
        for node in expert['nodes']:
            if node['use']=='geometry_0':node['use']=name
            if node['use']=='mapping_0':node['use']=f'mapping_{i}'
        g.defs[f'expert_{i}']=expert
    n=Nodes();scores=[]
    for i in range(count):
        state=n.slice('$state',i*101,(i+1)*101);counts=n.reshape(n.slice(state,0,100),[10,10])
        total=n.sum(n.add('op:sum',{'x':counts},axis=1));majority=n.sum(n.add('op:max',{'x':counts},axis=1))
        conflict=n.op('sub',a=total,b=majority)
        error=n.add('op:scale',{'x':n.slice(state,100,101)},c=9001.)
        scores.append(n.op('neg',x=n.op('add',a=conflict,b=error)))
    g.defs['router']=n.bundle('router',['state'],{'scores':n.add('op:concat',{f'x{i}':x for i,x in enumerate(scores)},axis=0)})
    n=Nodes();updates=[]
    for i,name in enumerate(names):
        geometry=n.add(name,{p:'$'+p for p in ('input','input_height','input_width')})
        updates.append(n.add('association_update',dict(colors=geometry+'.colors',height=geometry+'.height',width=geometry+'.width',
                   output='$output',output_height='$output_height',output_width='$output_width',phase='$phase',state=n.slice('$state',i*101,(i+1)*101))))
    state=n.add('op:concat',{f'x{i}':x for i,x in enumerate(updates)},axis=0)
    router=n.add('router',{'state':state})
    dispatch=n.add('dispatch',dict(scores=router+'.scores',state=state,input='$input',input_height='$input_height',input_width='$input_width'),branches=[f'expert_{i}' for i in range(count)])
    g.defs['root']=n.bundle('root',['input','input_height','input_width','output','output_height','output_width','phase','state'],
                                dict(state=state,**{p:dispatch+'.'+p for p in ('height_logits','width_logits','color_logits')}))
    g.meta.pop('two_slot_genome',None);g.meta.pop('portfolio',None)
    g.meta.update(state_dim=count*101,grid_attempts=1,gradients=False,
                  expanded_genome=dict(slots=list(range(8,count)),alleles=alleles,available_genes=bank,donor_hash=donor.hash()),
                  limitations='seven mutable flat slots; demo-association routing proxy; no evolutionary efficacy claim')
    if len(alleles)!=7:g.meta['limitations']='nine or eight fixed typed slots; demo-association routing proxy; no evolutionary efficacy claim'
    # Ranking records its immediate parent hash, including construction metadata.
    # Explicit historical metadata permits exact replay without falsifying that hash.
    if pre_rank_limitations is not None:
        if not isinstance(pre_rank_limitations,str):raise ValueError('text construction metadata required')
        g.meta['limitations']=pre_rank_limitations
    g=ranked_two(g);g.validate(Limits(strict_microcells=True));check_profile(g,'ladder-micro/1');return g


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    donor=Graph.load(a.donor)
    control=['scaling_bank2_repeat.gene','scaling_bank3_repeat.gene']+['geometry_0']*5
    rows=[]
    for name,alleles in [('expanded',None),('two-active-control',control)]:
        g=build_expanded_genome(donor,alleles)
        rows.append(dict(name=name,hash=g.save(a.out/name),stats=g.stats(),genome=g.meta['expanded_genome']))
    (a.out/'INVENTORY.json').write_text(json.dumps(dict(artifacts=rows,benchmark_tasks_read=0,efficacy_measured=False),indent=2)+'\n')
    print(json.dumps(rows))
