"""Matched graph-owned demonstration selector versus fixed score control.

All original cells, association updates, state dimensions and score computation
remain present in the control. This is an output routing ablation, not a speed
optimization or a learned amortized selector. No benchmark files are accessed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from build_grid_gene_pool import ranked_two


def fixed_score_control(ranked):
    graph=ranked.copy()
    root=graph.defs[graph.entries['grid_step']]
    dispatch=next(n for n in root['nodes'] if n['use']=='dispatch')
    router_node=next(n for n in root['nodes'] if n['id']==dispatch['in']['scores'].split('.')[0])
    router=graph.defs[router_node['use']]
    original=router['outputs']['scores']
    router['nodes'] += [
        {'id':'ablation_zero','use':'op:scale','in':{'x':original},'args':{'c':0}},
        {'id':'ablation_fixed','use':'op:scale','in':{'x':'rank_tie'},'args':{'c':1e8}},
        {'id':'ablation_scores','use':'op:add','in':{'a':'ablation_zero','b':'ablation_fixed'}}]
    router['outputs']['scores']='ablation_scores'
    graph.meta['selector_ablation']={'parent_hash':ranked.hash(),
        'ranking':'graph fixed branch index scores; attempts select branches 0 and 1',
        'preserved':'all original cells/parameters, recurrent association updates and state capacity; original scores still evaluated',
        'scope':'constant selector control; no host sorting or CPU savings claim'}
    graph.validate();check_profile(graph,'ladder-micro/1')
    return graph


def export(out,repo,include_bbox=False):
    if out.exists():raise ValueError('preserve previous selector artifacts')
    parent=Graph.load(repo/'runs/grid-associative/artifact')
    families={'associative8':ranked_two(parent)}
    if include_bbox:
        from build_grid_bbox_gene import append_bbox_gene
        families['associative9-bbox']=ranked_two(append_bbox_gene(parent))
    out.mkdir(parents=True)
    records=[]
    for family,ranked in families.items():
        for mode,graph in [('demo-selector',ranked),('constant-selector',fixed_score_control(ranked))]:
            path=family+'-'+mode
            digest=graph.save(out/path)
            records.append({'family':family,'mode':mode,'path':path,'hash':digest,
                            'stats':graph.stats(),'profile':check_profile(graph,'ladder-micro/1'),
                            'state_dim':graph.meta['state_dim']})
    receipt={'protocol':'grid-selector-ablation/1','artifacts':records,'benchmark_tasks_read':0,
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'demo reconstruction branch ranking versus fixed 0/1 branch selection; association learning preserved; neither amortized learned selector nor compute reduction'}
    (out/'INVENTORY.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--include-bbox',action='store_true')
    args=p.parse_args();print(json.dumps(export(Path(args.out),Path(__file__).resolve().parents[1],args.include_bbox)))


if __name__=='__main__':main()
