"""Count-fitted Witten–Bell byte model expressed as explicit graph operations."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from ..ir.profiles import check_profile
from ..runtime.text_codec import BYTE_CODEC, BYTE_VOCAB


def context_key(context):
    key = 0
    for byte in context: key = key*257 + byte+1
    return key


def fit_counts(texts, order=6):
    counts = defaultdict(Counter)
    for text in texts:
        document = text.encode('utf8')
        for position, byte in enumerate(document):
            for n in range(min(order, position)+1):
                counts[document[position-n:position]][byte] += 1
    return counts


def reference_probabilities(counts, prefix, order=6):
    """Independent scalar reference; no IR/runtime dependency."""
    probabilities = np.full(256, 1/256.)
    for n in range(min(order, len(prefix))+1):
        row = counts.get(prefix[len(prefix)-n:] if n else b'')
        if not row: break
        total, unique = sum(row.values()), len(row)
        weight = total/(total+unique)
        probabilities *= 1-weight
        for byte, count in row.items(): probabilities[byte] += weight*count/total
    return probabilities


def wb_byte_seed(founder, partition, *, order=6, residual_weight=0.):
    if partition.get('partition') != 'dev_train_selection' or partition.get('source_split') != 'dev_train':
        raise ValueError('designated dev_train selection fit partition required')
    if founder.meta.get('text_codec') != BYTE_CODEC or founder.meta.get('vocab') != BYTE_VOCAB:
        raise ValueError('canonical byte founder required')
    if type(order) is not int or not 1 <= order <= 6 or not 0 <= residual_weight <= 1:
        raise ValueError('order1..6 and bounded residual weight required')
    items = partition['items']
    if not items or len({x['family'] for x in items}) != len(items):
        raise ValueError('nonempty unique fit families required')
    counts = fit_counts([x['text'] for x in items], order)
    ordered = sorted((context_key(context), row) for context, row in counts.items())
    keys, offsets, columns, values = [], [0], [], []
    for key, row in ordered:
        keys.append(key)
        total, unique = sum(row.values()), len(row)
        weight = total/(total+unique)
        for byte, count in sorted(row.items()):
            columns.append(byte); values.append(weight*count/total)
        columns.append(256); values.append(weight)
        offsets.append(len(values))
    graph = founder.copy()
    graph.params.update({'cfwb.keys': np.asarray(keys, dtype=np.float64),
                         'cfwb.offsets': np.asarray(offsets, dtype=np.float64),
                         'cfwb.columns': np.asarray(columns, dtype=np.float64),
                         'cfwb.values': np.asarray(values, dtype=np.float64),
                         'cfwb.uniform': np.full(256, 1/256.),
                         'cfwb.special_logits': np.full(3, -1000.)})
    nodes = [{'id': key, 'use': 'param:cfwb.'+key} for key in ('keys','offsets','columns','values','uniform','special_logits')]
    nodes.extend([
        {'id': 'one', 'use': 'op:sum', 'in': {'x': '$prev'}, 'args': {'axis': 0, 'keepdims': True}},
        {'id': 'base', 'use': 'op:mul', 'in': {'a': 'one', 'b': 'uniform'}}
    ])
    outputs = {}
    previous = 'base'
    K = int(graph.meta['ctx_len'])
    for n in range(order+1):
        tag = f'order_{n}'
        if n:
            weights = np.zeros((n, len(BYTE_VOCAB)))
            for j in range(n): weights[j, 3:] = np.arange(1,257)*257**(n-1-j)
            graph.params[f'cfwb.encode_{n}'] = weights.reshape(-1,1)
            nodes.extend([
                {'id': tag+'_weights', 'use': f'param:cfwb.encode_{n}'},
                {'id': tag+'_ctx', 'use': 'op:slice', 'in': {'x': '$ctx'}, 'args': {'axis': 0,'start': K-n,'stop': K}},
                {'id': tag+'_flat', 'use': 'op:reshape', 'in': {'x': tag+'_ctx'}, 'args': {'shape': [1,n*len(BYTE_VOCAB)]}},
                {'id': tag+'_key2', 'use': 'op:matmul', 'in': {'a': tag+'_flat','b': tag+'_weights'}},
                {'id': tag+'_key', 'use': 'op:reshape', 'in': {'x': tag+'_key2'}, 'args': {'shape': [1]}},
                {'id': tag+'_age', 'use': 'op:bias', 'in': {'x': '$step'}, 'args': {'c': -(n-1)}},
                {'id': tag+'_valid', 'use': 'op:step', 'in': {'x': tag+'_age'}}
            ])
        else:
            nodes.extend([
                {'id': tag+'_key', 'use': 'op:scale', 'in': {'x': '$step'}, 'args': {'c': 0}},
                {'id': tag+'_valid', 'use': 'op:scale', 'in': {'x': 'one'}, 'args': {'c': 1}}
            ])
        nodes.extend([
            {'id': tag+'_lookup', 'use': 'op:csr_lookup', 'in': {'query': tag+'_key', 'keys':'keys','offsets':'offsets','columns':'columns','values':'values'}, 'args': {'width':257}},
            {'id': tag+'_masked', 'use': 'op:mul', 'in': {'a': tag+'_lookup','b':tag+'_valid'}},
            {'id': tag+'_seen', 'use': 'op:slice', 'in': {'x': tag+'_masked'}, 'args': {'axis':0,'start':0,'stop':256}},
            {'id': tag+'_weight', 'use': 'op:slice', 'in': {'x': tag+'_masked'}, 'args': {'axis':0,'start':256,'stop':257}},
            {'id': tag+'_neg', 'use': 'op:neg', 'in': {'x':tag+'_weight'}},
            {'id': tag+'_escape', 'use': 'op:bias', 'in': {'x':tag+'_neg'}, 'args': {'c':1}},
            {'id': tag+'_backoff', 'use': 'op:mul', 'in': {'a':previous,'b':tag+'_escape'}},
            {'id': tag, 'use': 'op:add', 'in': {'a':tag+'_seen','b':tag+'_backoff'}}
        ])
        previous = tag
        if n >= 4: outputs[tag] = tag
    nodes.extend([
        {'id':'byte_logp','use':'op:log','in':{'x':previous}},
        {'id':'logits','use':'op:concat','in':{'x0':'special_logits','x1':'byte_logp'},'args':{'axis':0}}
    ])
    outputs.update({'probs':previous,'logits':'logits'})
    graph.defs['cfwb_trunk'] = {'kind':'bundle','role':'cluster','inputs':['ctx','prev','step'], 'nodes':nodes,'outputs':outputs}
    root = graph.defs[graph.entries['step']]
    root['nodes'].extend([
        {'id':'count_trunk','use':'cfwb_trunk','in':{'ctx':'$ctx','prev':'$prev','step':'$step'}},
        {'id':'authored_residual','use':'op:scale','in':{'x':root['outputs']['logits']},'args':{'c':residual_weight}},
        {'id':'count_moe_mix','use':'op:add','in':{'a':'count_trunk.logits','b':'authored_residual'}}
    ])
    root['outputs']['logits'] = 'count_moe_mix'
    canonical = json.dumps(partition,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    provenance = {'artifact_class':'fitted','method':'byte Witten-Bell interpolation, orders0..'+str(order),
                  'fit_partition':'dev_train_selection','fit_partition_sha256':hashlib.sha256(canonical).hexdigest(),
                  'corpus_sha256':partition['corpus_sha256'],'fit_documents':len(items),
                  'fit_content_bytes':sum(len(x['text'].encode()) for x in items),'fit_families':[x['family'] for x in items],
                  'eos_policy':'no EOS/BOS target counts; specials finite -1000 logits; reported content metric excludes EOS',
                  'count_contexts':len(keys),'csr_entries':len(values),'order':order,
                  'residual_weight':residual_weight,'founder_hash':founder.hash(),
                  'profile':'ladder/1, large immutable fitted CSR tables; pure founder remains ladder-micro/1',
                  'capacity_caveat':'not parameter-matched to pure founder; fitted-alone/evolution/random share the same seed',
                  'gradient_steps':0,'gradient_estimators':0,'report_data_used':False,'final_data_used':False}
    provenance['builder_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    graph.meta.update({'artifact_class':'fitted','fitted_provenance':provenance,'parent_hash':founder.hash(),
                       'training':'count-fitted without gradients','profile':'ladder/1'})
    graph.validate();check_profile(graph,'ladder/1')
    return graph, provenance, counts
