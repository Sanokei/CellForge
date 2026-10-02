"""Version-two graph pair ranking; version-one coverage artifacts untouched."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile

spec=importlib.util.spec_from_file_location('fixed_grid_pairs',Path(__file__).with_name('build_grid_gene_pool.py'))
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)


def ranked_pair(left,right):
    g=fixed.breed_fixed_pair(left,right);base_state_dim=g.meta['state_dim'];root=g.defs['root'];nodes=root['nodes']
    pair_nodes={n['id']:n for n in nodes}
    donor_ports=[{p:pair_nodes['pair_'+p]['in']['x'+str(i)] for p in ('height_logits','width_logits','color_logits')} for i in range(2)]
    def add(name,use,inputs,args=None):
        node={'id':name,'use':use,'in':inputs}
        if args is not None:node['args']=args
        nodes.append(node);return name
    def op(name,use,inputs,args=None):return add(name,'op:'+use,inputs,args)
    def reduce(name,ref):return op(name,'sum',{'x':ref},{'axis':0,'keepdims':True})
    op('ranking_target_mask','reshape',{'x':'$output_mask'},{'shape':[900]})
    reduce('ranking_cell_count','ranking_target_mask')
    op('ranking_zero','scale',{'x':'$phase'},{'c':0})
    op('ranking_one','bias',{'x':'ranking_zero'},{'c':1})
    op('ranking_denominator','maximum',{'a':'ranking_cell_count','b':'ranking_one'})
    op('ranking_query_neg','neg',{'x':'$phase'})
    op('ranking_demo_gate','bias',{'x':'ranking_query_neg'},{'c':1})
    scores=[]
    for i,ports in enumerate(donor_ports):
        tag='ranking_'+str(i)+'_'
        decoded=op(tag+'colors_hot','onehot_argmax',{'x':ports['color_logits']},{'axis':3})
        decoded=op(tag+'colors','reshape',{'x':decoded},{'shape':[30,30,10]})
        delta=op(tag+'delta','sub',{'a':decoded,'b':'$output'})
        squared=op(tag+'squared','square',{'x':delta})
        masked=op(tag+'masked','mul',{'a':squared,'b':'$output_mask'})
        flattened=op(tag+'flat','reshape',{'x':masked},{'shape':[9000]})
        error=reduce(tag+'sum',flattened)
        half=op(tag+'half','scale',{'x':error},{'c':.5})
        current=op(tag+'color_error','divide',{'a':half,'b':'ranking_denominator'})
        for port,target in (('height_logits','$output_height'),('width_logits','$output_width')):
            hot=op(tag+port+'_hot','onehot_argmax',{'x':ports[port]},{'axis':1})
            shape=op(tag+port+'_flat','reshape',{'x':hot},{'shape':[30]})
            diff=op(tag+port+'_diff','sub',{'a':shape,'b':target})
            square=op(tag+port+'_square','square',{'x':diff})
            total=reduce(tag+port+'_total',square)
            unit=op(tag+port+'_unit','scale',{'x':total},{'c':.5})
            current=op(tag+port+'_combined','add',{'a':current,'b':unit})
        gated=op(tag+'gated','mul',{'a':current,'b':'ranking_demo_gate'})
        scores.append(op(tag+'negative','neg',{'x':gated}))
    op('ranking_delta','concat',{'x0':scores[0],'x1':scores[1]},{'axis':0})
    op('ranking_previous','slice',{'x':'$state'},{'axis':0,'start':base_state_dim,'stop':base_state_dim+2})
    op('ranking_accumulated','add',{'a':'ranking_previous','b':'ranking_delta'})
    op('ranking_next_state','concat',{'x0':'pair_state','x1':'ranking_accumulated'},{'axis':0})
    g.params['pair_rank_tie']=np.array([0.,-1e-8])
    g.defs['pair_router']={'kind':'bundle','role':'router','inputs':['scores'],'nodes':[
        {'id':'tie','use':'param:pair_rank_tie'},
        {'id':'scores','use':'op:add','in':{'a':'$scores','b':'tie'}}], 'outputs':{'scores':'scores'}}
    # Exact masking mechanism is imported from the independently tested
    # ranked-two graph, but no neural donor definitions are replaced.
    temporary=fixed.ranked_two(left)
    g.defs['pair_second_router']=copy.deepcopy(temporary.defs['rank_second'])
    expert_inputs=[f'donor{i}_{port}' for i in range(2) for port in ('height_logits','width_logits','color_logits')]
    for i in range(2):
        g.defs['pair_output_'+str(i)]={'kind':'bundle','role':'expert','inputs':expert_inputs,'nodes':[
            {'id':port,'use':'op:scale','in':{'x':'$donor'+str(i)+'_'+port},'args':{'c':1}}
            for port in ('height_logits','width_logits','color_logits')],
            'outputs':{p:p for p in ('height_logits','width_logits','color_logits')}}
    ins={f'donor{i}_{p}':ports[p] for i,ports in enumerate(donor_ports) for p in ('height_logits','width_logits','color_logits')}
    add('ranking_first_router','pair_router',{'scores':'ranking_accumulated'})
    add('ranking_first_dispatch','dispatch',dict(ins,scores='ranking_first_router.scores'),{'branches':['pair_output_0','pair_output_1']})
    add('ranking_second_router','pair_second_router',{'scores':'ranking_first_router.scores'})
    add('ranking_second_dispatch','dispatch',dict(ins,scores='ranking_second_router.scores'),{'branches':['pair_output_0','pair_output_1']})
    for port in ('height_logits','width_logits','color_logits'):
        root['outputs'][port]=op('ranked_pair_'+port,'concat',{'x0':'ranking_first_dispatch.'+port,'x1':'ranking_second_dispatch.'+port},{'axis':0})
    root['outputs']['state']='ranking_next_state';root['inputs']=sorted(set(root['inputs'])|{'output_mask'})
    g.meta['state_dim']=base_state_dim+2
    g.meta['construction']='ranked two-gene graph crossover/2; graph-owned online normalized demo error'
    g.meta['gene_pool']['ranking']='graph cumulative online demo error: color mismatch / target cell count + wrong height + wrong width; graph first/second dispatch'
    g.meta['gene_pool']['ranking_limit']='association donors reconstruct after each online update; no retroactive final-state all-demo check'
    g.validate();check_profile(g,'ladder-micro/1');return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise SystemExit('preserve previous artifacts')
    g=ranked_pair(Graph.load(args.left),Graph.load(args.right));digest=g.save(out)
    print(json.dumps({'hash':digest,'stats':g.stats(),'metadata':g.meta['gene_pool']}))


if __name__=='__main__':main()
