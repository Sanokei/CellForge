"""Flat graph-owned serial composition with intermediate/final validity masks.

Typed genes must emit dimension onehots and canonical30x30x10 colors. Two
ordered stages are inlined; no runtime program search or host grid transform.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile

GENES=tuple(['geometry_'+str(i) for i in range(8)]+['bbox.geometry','bank_local_fill.gene','bank_adjacency_recolor.gene'])


def compose_flat(firstdef,seconddef):
    nodes=[]
    def inline(definition,prefix,inputs):
        if set(definition['inputs'])!={'input','input_height','input_width'} or set(definition['outputs'])!={'colors','height','width'}:
            raise ValueError('canonical flat geometry ports required')
        def rewrite(ref):return inputs[ref[1:]] if ref.startswith('$') else prefix+ref
        for original in definition['nodes']:
            if original['use']=='dispatch':raise ValueError('gene dispatch cannot be inlined as flat geometry')
            node=copy.deepcopy(original);node['id']=prefix+node['id']
            node['in']={key:rewrite(ref) for key,ref in node.get('in',{}).items()};nodes.append(node)
        return {key:rewrite(ref) for key,ref in definition['outputs'].items()}
    def validity_mask(outputs,prefix):
        def add(name,use,inputs=None,args=None):
            node={'id':prefix+name,'use':'op:'+use}
            if inputs:node['in']=inputs
            if args:node['args']=args
            nodes.append(node);return node['id']
        masks=[]
        # Reverse cumulative onehots give exactly1 for valid coordinates.
        for axis,dimension,shape in [('h','height',[30,1,1]),('w','width',[1,30,1])]:
            reverse=add(axis+'_reverse','slice',{'x':outputs[dimension]},{'axis':0,'step':-1})
            cumulative=add(axis+'_cumulative','cumsum',{'x':reverse},{'axis':0})
            valid=add(axis+'_valid','slice',{'x':cumulative},{'axis':0,'step':-1})
            masks.append(add(axis+'_reshape','reshape',{'x':valid},{'shape':shape}))
        mask=add('mask','mul',{'a':masks[0],'b':masks[1]})
        return add('colors','mul',{'a':outputs['colors'],'b':mask})
    first=inline(firstdef,'first__',{p:'$'+p for p in ('input','input_height','input_width')})
    intermediate=validity_mask(first,'mid__')
    second=inline(seconddef,'second__',{'input':intermediate,'input_height':first['height'],'input_width':first['width']})
    final=validity_mask(second,'final__')
    return {'kind':'bundle','role':'cluster','inputs':['input','input_height','input_width'],'nodes':nodes,
            'outputs':{'colors':final,'height':second['height'],'width':second['width']}}


def compose_into_slot9(base,firstgene,secondgene):
    if firstgene not in GENES or secondgene not in GENES:raise ValueError('genes outside shared bank')
    for name in (firstgene,secondgene):
        if any(node['use'] in base.defs and base.defs[node['use']]['kind']!='cell' for node in base.defs[name]['nodes']):raise ValueError('genes must be flat ops/cells')
    graph=base.copy();graph.defs['slot9_geometry']=compose_flat(base.defs[firstgene],base.defs[secondgene])
    graph.meta['serial_gene']={'protocol':'ordered-flat-serial-gene/1','slot':9,'alleles':[firstgene,secondgene],
        'parent_hash':base.hash(),'source_definition_sha256':[hashlib.sha256(json.dumps(base.defs[name],sort_keys=True).encode()).hexdigest() for name in (firstgene,secondgene)],
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mask_policy':'graph reverse-cumsum output dimension onehots; sanitize intermediate and final colors',
        'scope':'two ordered authored stages; no efficacy claim or host solver'}
    graph.meta['two_slot_genome']['serial_override_slot']=9
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1');return graph


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/arc-grid-genes/two-slot-genome-v1/base')
    p.add_argument('--first',choices=GENES,default='bank_adjacency_recolor.gene');p.add_argument('--second',choices=GENES,default='bank_local_fill.gene');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve prior serial artifact')
    base=Graph.load(args.base);graph=compose_into_slot9(base,args.first,args.second);digest=graph.save(out)
    def expanded_nodes(name):
        definition=graph.defs[name]
        if definition['kind']=='cell':return 1
        count=len(definition['nodes'])
        for node in definition['nodes']:
            if node['use']=='dispatch':count+=sum(expanded_nodes(branch) for branch in node['args']['branches'])
            elif node['use'] in graph.defs:count+=expanded_nodes(node['use'])
        return count
    receipt={'hash':digest,'stats':graph.stats(),'slot9_node_count':len(graph.defs['slot9_geometry']['nodes']),
        'expanded_ir_nodes_all_dispatch_branches':expanded_nodes(graph.entries['grid_step']),
        'base_parameter_storage':base.stats()['unique_param_storage'],'profile':check_profile(graph,'ladder-micro/1'),
        'serial_gene':graph.meta['serial_gene'],'benchmark_tasks_read':0,'efficacy_measured':False}
    (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
