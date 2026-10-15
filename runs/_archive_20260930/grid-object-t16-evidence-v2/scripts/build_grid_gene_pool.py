"""Graph-owned grid gene pools and lossless ranked two-attempt decoding.

No benchmark tasks are read during construction. Demonstration reconstruction,
color association, recurrent updates and both ranks execute in CellForge IR.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.construction.transactions import import_def


def ranked_two(parent):
    graph=parent.copy();root=graph.defs[graph.entries['grid_step']]
    dispatch=next(node for node in root['nodes'] if node['use']=='dispatch')
    score_ref=dispatch['in']['scores'];router_id=score_ref.split('.')[0]
    router_node=next(n for n in root['nodes'] if n['id']==router_id)
    router=graph.defs[router_node['use']]
    n=len(dispatch['args']['branches'])
    # Integral demo errors dominate this fixed small lexicographic tie offset.
    graph.params['rank_tie_offsets']=-np.arange(n,dtype=float)*1e-8
    router['nodes'] += [{'id':'rank_tie','use':'param:rank_tie_offsets'},
                        {'id':'rank_scores','use':'op:add','in':{'a':router['outputs']['scores'],'b':'rank_tie'}}]
    router['outputs']['scores']='rank_scores'
    graph.defs['rank_second']={'kind':'bundle','role':'router','inputs':['scores'],'nodes':[
        {'id':'winner','use':'op:onehot_argmax','in':{'x':'$scores'},'args':{'axis':0}},
        {'id':'lo','use':'op:min','in':{'x':'$scores'},'args':{'axis':0,'keepdims':True}},
        {'id':'hi','use':'op:max','in':{'x':'$scores'},'args':{'axis':0,'keepdims':True}},
        {'id':'span','use':'op:sub','in':{'a':'hi','b':'lo'}},
        {'id':'gap','use':'op:bias','in':{'x':'span'},'args':{'c':1}},
        {'id':'penalty','use':'op:mul','in':{'a':'winner','b':'gap'}},
        {'id':'scores','use':'op:sub','in':{'a':'$scores','b':'penalty'}}], 'outputs':{'scores':'scores'}}
    second=copy.deepcopy(dispatch);second['id']='rank_dispatch_second';second['in']['scores']='rank_router_second.scores'
    root['nodes'] += [{'id':'rank_router_second','use':'rank_second','in':{'scores':score_ref}},second]
    for port in ('height_logits','width_logits','color_logits'):
        name='rank_pair_'+port
        root['nodes'].append({'id':name,'use':'op:concat','in':{'x0':root['outputs'][port], 'x1':second['id']+'.'+port},'args':{'axis':0}})
        root['outputs'][port]=name
    graph.meta.update({'grid_attempts':2,'construction':'graph gene pool ranked two/1; direct authored, no fitting',
                       'gene_pool':{'parent_hash':parent.hash(),'branches':dispatch['args']['branches'],
                                    'ranking':'demo reconstruction scores; fixed index tie; graph masks first winner before second dispatch',
                                    'neural_scope':'graph-owned authored transforms and associations, not Python program search'}})
    graph.validate();check_profile(graph,'ladder-micro/1');return graph


def fixed_single_gene(parent,index):
    """Coverage control: retain graph association updates, force one gene.

    Two cloned-score dispatch slots reference the same expert; this is a fixed
    single-gene control, not an adaptive pair or two independent capabilities.
    """
    graph=parent.copy();root=graph.defs[graph.entries['grid_step']]
    dispatch=next(n for n in root['nodes'] if n['use']=='dispatch');branches=dispatch['args']['branches']
    if not 0<=index<len(branches):raise ValueError('gene index out of range')
    root_router=next(n for n in root['nodes'] if n['id']==dispatch['in']['scores'].split('.')[0])
    router=graph.defs[root_router['use']]
    if root['outputs']['state']==dispatch['in']['scores']:
        root['outputs']['state']='$state'
    router['nodes']=[{'id':'zero','use':'op:scale','in':{'x':'$state'},'args':{'c':0}},
                     {'id':'two','use':'op:slice','in':{'x':'zero'},'args':{'axis':0,'start':0,'stop':2}}]
    router['outputs']={'scores':'two'}
    dispatch['args']['branches']=[branches[index],branches[index]]
    graph.meta.update({'construction':'fixed single graph gene coverage control','gene_pool':{'parent_hash':parent.hash(),'fixed_gene':branches[index],
                      'limitations':'two dispatch slots share one expert; no adaptive ranking claim'}})
    graph.validate();check_profile(graph,'ladder-micro/1');return graph


def breed_fixed_pair(left,right):
    """Exact component crossover: two fixed genes, one decoded attempt each.

    This measures union coverage. It does not claim adaptive parent ranking.
    Imported cells/association updates execute in the child's graph.
    """
    if any(p.meta.get('grid_attempts')!=1 for p in (left,right)):
        raise ValueError('pair donors must each emit one attempt')
    g=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':2,
                  'state_dim':sum(p.meta['state_dim'] for p in (left,right)),
                  'profile':'ladder-micro/1','construction':'fixed two-gene wiring crossover/1; no fitting',
                  'gene_pool':{'parent_hashes':[left.hash(),right.hash()],
                    'ranking':'fixed left/right union coverage; no adaptive pair ranking',
                    'donor_genes':[p.meta.get('gene_pool',{}).get('fixed_gene') for p in (left,right)]}})
    nodes=[];ports=[];offset=0;inputs=set()
    for side,parent in (('left_',left),('right_',right)):
        name=parent.entries['grid_step'];import_def(g,parent,name,prefix=side)
        root=g.defs.pop(side+name);inputs.update(root['inputs'])
        state=side+'state'
        nodes.append({'id':state,'use':'op:slice','in':{'x':'$state'},'args':{'axis':0,'start':offset,'stop':offset+parent.meta['state_dim']}})
        offset+=parent.meta['state_dim']
        def rewrite(ref):
            if ref=='$state':return state
            return ref if ref.startswith('$') else side+ref
        for original in root['nodes']:
            node=copy.deepcopy(original);node['id']=side+node['id'];node['in']={k:rewrite(v) for k,v in node.get('in',{}).items()};nodes.append(node)
        ports.append({k:rewrite(v) for k,v in root['outputs'].items()})
    outputs={}
    for port in ('state','height_logits','width_logits','color_logits'):
        name='pair_'+port;nodes.append({'id':name,'use':'op:concat','in':{'x0':ports[0][port],'x1':ports[1][port]},'args':{'axis':0}});outputs[port]=name
    g.defs['root']={'kind':'bundle','role':'root','inputs':sorted(inputs),'nodes':nodes,'outputs':outputs};g.entries={'grid_step':'root'}
    g.validate();check_profile(g,'ladder-micro/1');return g


def export_inventory(out,repo):
    if out.exists():raise ValueError('preserve prior gene inventory')
    out.mkdir(parents=True);inventory=[];families={}
    for family,path in (('associative8','runs/grid-associative/artifact'),('geometry20','runs/grid20/artifact')):
        parent=Graph.load(repo/path);families[family]=parent
        count=8 if family=='associative8' else 20
        for index in range(count):
            gene=fixed_single_gene(parent,index);relative=f'{family}-gene-{index:02d}';digest=gene.save(out/relative)
            inventory.append({'family':family,'index':index,'path':relative,'hash':digest,'stats':gene.stats()})
    # First matched breeding control: same identity geometry, color-associative
    # and fixed-color donors. New capability is not presumed from this union.
    pair=breed_fixed_pair(fixed_single_gene(families['associative8'],0),fixed_single_gene(families['geometry20'],0))
    digest=pair.save(out/'pair-assoc0-geometry0')
    record={'protocol':'grid-gene-inventory/1','genes':inventory,'pairs':[{'path':'pair-assoc0-geometry0','hash':digest,'ranking':'fixed donor union coverage'}],
            'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_tasks_read':0,'gradients':0}
    (out/'INVENTORY.json').write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--family',choices=['associative8','geometry20'],default='associative8');p.add_argument('--single',type=int);p.add_argument('--inventory',action='store_true')
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('preserve prior artifacts')
    repo=Path(__file__).resolve().parents[1]
    if args.inventory:
        record=export_inventory(out,repo);print(json.dumps({'genes':len(record['genes']),'pairs':record['pairs'],'inventory':str(out/'INVENTORY.json')}));return
    parent=Graph.load(repo/('runs/grid-associative/artifact' if args.family=='associative8' else 'runs/grid20/artifact'))
    graph=fixed_single_gene(parent,args.single) if args.single is not None else ranked_two(parent)
    digest=graph.save(out)
    card={'model_hash':digest,'parent_hash':parent.hash(),'family':args.family,'fixed_gene':args.single,'attempts':graph.meta['grid_attempts'],
          'stats':graph.stats(),'profile':check_profile(graph,'ladder-micro/1'),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'authored grid genes; no benchmark reads, host answer reconstruction, gradients or Python program search'}
    (out/'MODEL_CARD.json').write_text(json.dumps(card,indent=2)+'\n');print(json.dumps(card))


if __name__=='__main__':main()
