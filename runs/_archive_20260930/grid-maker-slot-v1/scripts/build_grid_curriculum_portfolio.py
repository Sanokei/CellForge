"""Fixed-capacity ten-slot typed grid portfolio and single-slot transactions.

All candidates contain the same authored gene bank. Slot9 starts as identity;
maker/control edits replace only its flat geometry definition. Gene authoring
is not evidence of an LLM advantage. No task/curriculum files are read here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile
from cellforge.construction.transactions import import_def,apply_ops
from build_grid_bbox_gene import append_bbox_gene,Nodes
from build_grid_gene_pool import ranked_two
from build_grid_local_fill_gene import build_local_fill_gene

GENES=tuple(['geometry_'+str(i) for i in range(1,8)]+['bbox.geometry','bank_local_fill.gene'])


def build_portfolio(parent,fill_donor=None):
    if parent.meta.get('grid_attempts')!=1:raise ValueError('unranked associative8 donor required')
    graph=append_bbox_gene(parent)
    fill=fill_donor or build_local_fill_gene()
    imported=import_def(graph,fill,'local_fill.gene',prefix='bank_')
    gene=graph.defs['bank_local_fill.gene']
    # Onehot input has sum1 on valid pixels and0 in padding. Compute its mask
    # inside the flat typed cluster, avoiding extra host ports/nested clusters.
    for node in gene['nodes']:
        node['in']={k:('valid_mask' if v=='$input_mask' else v) for k,v in node.get('in',{}).items()}
    gene['nodes'].insert(0,{'id':'valid_mask','use':'op:sum','in':{'x':'$input'},'args':{'axis':2,'keepdims':True}})
    gene['inputs']=['input','input_height','input_width']
    graph.defs['slot9_geometry']=copy.deepcopy(graph.defs['geometry_0'])
    graph.defs['mapping_9']=copy.deepcopy(graph.defs['mapping_0'])
    graph.defs['mapping_9']['nodes'][0]['args'].update(start=909,stop=1009)
    graph.defs['expert_9']=copy.deepcopy(graph.defs['expert_0'])
    for node in graph.defs['expert_9']['nodes']:
        if node['use']=='geometry_0':node['use']='slot9_geometry'
        if node['use']=='mapping_0':node['use']='mapping_9'
    n=Nodes();scores=[]
    for i in range(10):
        state=n.slice('$state',i*101,(i+1)*101);counts=n.reshape(n.slice(state,0,100),[10,10])
        total=n.sum(n.add('op:sum',{'x':counts},axis=1));majority=n.sum(n.add('op:max',{'x':counts},axis=1))
        conflict=n.op('sub',a=total,b=majority)
        error=n.add('op:scale',{'x':n.slice(state,100,101)},c=9001)
        scores.append(n.op('neg',x=n.op('add',a=conflict,b=error)))
    joined=n.add('op:concat',{f'x{i}':v for i,v in enumerate(scores)},axis=0)
    graph.defs['router']=n.bundle('router',['state'],{'scores':joined})
    n=Nodes();updates=[]
    for i in range(10):
        name='geometry_'+str(i) if i<8 else ('bbox.geometry' if i==8 else 'slot9_geometry')
        geometry=n.add(name,{'input':'$input','input_height':'$input_height','input_width':'$input_width'})
        updates.append(n.add('association_update',{'colors':geometry+'.colors','height':geometry+'.height','width':geometry+'.width',
            'output':'$output','output_height':'$output_height','output_width':'$output_width','phase':'$phase','state':n.slice('$state',i*101,(i+1)*101)}))
    state=n.add('op:concat',{f'x{i}':v for i,v in enumerate(updates)},axis=0)
    router=n.add('router',{'state':state})
    dispatch=n.add('dispatch',{'scores':router+'.scores','state':state,'input':'$input','input_height':'$input_height','input_width':'$input_width'},branches=['expert_'+str(i) for i in range(10)])
    graph.defs['root']=n.bundle('root',['input','input_height','input_width','output','output_height','output_width','phase','state'],
        {'state':state,**{p:dispatch+'.'+p for p in ('height_logits','width_logits','color_logits')}})
    graph.meta.update({'state_dim':1010,'construction':'fixed ten-slot authored portfolio/1',
        'portfolio':{'slots':10,'mutable_slot':9,'initial_gene':'geometry_0','available_genes':list(GENES),
        'parent_hash':parent.hash(),'fill_donor_hash':fill.hash(),'import_def_lineage':imported,
        'maker_scope':'slot selection from shared authored bank; authoring knowledge itself is not controlled LLM advantage'}})
    graph=ranked_two(graph)
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1')
    return graph


def slot_transaction(parent,gene,author='maker',task_id='slot9-rewire'):
    if gene not in GENES:raise ValueError('gene outside predeclared shared bank')
    definition=copy.deepcopy(parent.defs[gene])
    if definition['inputs']!=parent.defs['slot9_geometry']['inputs']:
        if set(definition['inputs'])!=set(parent.defs['slot9_geometry']['inputs']):raise ValueError('typed gene ports differ')
    return {'base':parent.hash(),'author':author,'task_id':task_id,
        'rationale':'rewire existing slot9 to '+gene+' from shared authored bank',
        'selected_gene':gene,'ops':[{'op':'put_def','name':'slot9_geometry','def':definition}]}


def random_control_transaction(parent,seed):
    gene=GENES[int(np.random.default_rng(seed).integers(len(GENES)))]
    transaction=slot_transaction(parent,gene,author='uniform-nonLLM-control')
    transaction['control_seed']=seed;transaction['sampling']='uniform nine available nonidentity genes; one slot rewire'
    return transaction


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve frozen portfolio')
    repo=Path(__file__).resolve().parents[1]
    graph=build_portfolio(Graph.load(repo/'runs/grid-associative/artifact'))
    digest=graph.save(out/'parent')
    proposal=slot_transaction(graph,'bank_local_fill.gene')
    (out/'MAKER_SLOT9_FILL_TRANSACTION.json').write_text(json.dumps(proposal,indent=2)+'\n')
    receipt={'protocol':'fixed-grid-portfolio/1','parent_hash':digest,'stats':graph.stats(),
        'profile':check_profile(graph,'ladder-micro/1'),'portfolio':graph.meta['portfolio'],
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_tasks_read':0,'curriculum_scored':False,
        'child_frozen':False,'transaction_note':'proposal template only; root MakerOperator must apply and ledger against actual train-pair diagnostics'}
    (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
