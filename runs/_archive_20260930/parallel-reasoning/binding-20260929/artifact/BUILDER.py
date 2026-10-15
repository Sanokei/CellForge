"""Directly authored exact UTF-8 key retrieval, without fitted parameters."""
import argparse
import hashlib
import json
from pathlib import Path
from cellforge.ir.graph import Graph


def build_binding():
    nodes = []
    def add(op, inputs, **args):
        name = f'n{len(nodes)}'
        node = {'id': name, 'use': 'op:' + op, 'in': inputs}
        if args:
            node['args'] = args
        nodes.append(node)
        return name
    keys = add('reshape', {'x':'$keys'}, shape=[8,16*257])
    query = add('reshape', {'x':'$query'}, shape=[16*257,1])
    similarity = add('matmul', {'a':keys,'b':query})
    exact = add('step', {'x':add('bias', {'x':similarity}, c=-15.5)})
    active = add('reshape', {'x':'$active'}, shape=[8,1])
    match = add('mul', {'a':exact,'b':active})
    count = add('reshape', {'x':add('sum', {'x':match}, axis=0, keepdims=True)}, shape=[1])
    distance = add('abs', {'x':add('bias', {'x':count}, c=-1)})
    unique = add('relu', {'x':add('bias', {'x':add('neg', {'x':distance})}, c=1)})
    gate = add('reshape', {'x':unique}, shape=[1,1])
    weights = add('transpose', {'x':add('mul', {'a':match,'b':gate})}, perm=[1,0])
    values = add('reshape', {'x':'$values'}, shape=[8,32*257])
    output = add('reshape', {'x':add('matmul', {'a':weights,'b':values})}, shape=[32,257])
    return Graph(defs={'binding_predict':{'kind':'bundle','role':'cluster','inputs':['keys','values','query','active'],
                 'nodes':nodes,'outputs':{'value_logits':output,'matches':count}}},
                 entries={'predict':'binding_predict'}, meta={'task':'binding','binding_interface':'binding-graph/1',
                 'max_bindings':8,'key_bytes':16,'value_bytes':32,'construction':'authored exact equality and unique-match retrieval; no fitting',
                 'limitations':'typed retrieval only; no natural-language parsing, reasoning or learned tool selection'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit('output must be new')
    graph = build_binding()
    graph.validate()
    digest = graph.save(out)
    (out/'BUILDER.py').write_bytes(Path(__file__).read_bytes())
    (out/'MODEL_CARD.json').write_text(json.dumps({'model_hash':digest,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numeric_provenance':'fixed lengths and authored equality threshold, no dataset access or fitted weights','training':'none',
        'limitations':graph.meta['limitations']},indent=2))
    print(digest)


if __name__ == '__main__':
    main()
