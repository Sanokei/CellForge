"""Explicitly fitted byte bigram trunk; count fitting only, no derivatives."""
import hashlib
import json
import numpy as np
from ..ir.profiles import check_profile
from ..runtime.lm import encode_text
from ..runtime.text_codec import BYTE_CODEC, BYTE_VOCAB


def fitted_byte_seed(founder, partition, *, smoothing=0.1, backoff_mass=10., residual_weight=0.05):
    if partition.get('partition') != 'dev_train_selection':
        raise ValueError('fit only an explicitly designated dev_train selection partition')
    if founder.meta.get('text_codec') != BYTE_CODEC or founder.meta.get('vocab') != BYTE_VOCAB:
        raise ValueError('canonical byte founder required')
    if smoothing <= 0 or backoff_mass <= 0 or not 0 <= residual_weight <= 1:
        raise ValueError('invalid predeclared smoothing/backoff/residual constants')
    items = partition.get('items', [])
    if not items or any(not isinstance(x.get('text'), str) or not x['text'] for x in items):
        raise ValueError('nonempty fit texts required')
    V = len(BYTE_VOCAB)
    counts = np.zeros((V, V), dtype=np.float64)
    unigram = np.full(V, smoothing)
    for item in items:
        tokens = encode_text(BYTE_VOCAB, item['text'], BYTE_CODEC)
        for prev, nxt in zip(tokens, tokens[1:]):
            counts[prev, nxt] += 1
            unigram[nxt] += 1
    unigram /= unigram.sum()
    logp = np.log((counts + backoff_mass*unigram[None, :]) /
                  (counts.sum(axis=1, keepdims=True) + backoff_mass))
    graph = founder.copy()
    flat = logp.ravel()
    nodes = []
    for i, start in enumerate(range(0, len(flat), 40)):
        key = f'fitted_ngram.tile_{i}'
        graph.params[key] = flat[start:start+40].copy()
        nodes.append({'id': f'tile_{i}', 'use': 'param:'+key})
    nodes.extend([
        {'id': 'flat', 'use': 'op:concat', 'in': {f'x{i}': f'tile_{i}' for i in range(len(nodes))}, 'args': {'axis': 0}},
        {'id': 'table', 'use': 'op:reshape', 'in': {'x': 'flat'}, 'args': {'shape': [V, V]}},
        {'id': 'prev_row', 'use': 'op:expand', 'in': {'x': '$prev'}, 'args': {'axis': 0}},
        {'id': 'lookup', 'use': 'op:matmul', 'in': {'a': 'prev_row', 'b': 'table'}},
        {'id': 'logits', 'use': 'op:reshape', 'in': {'x': 'lookup'}, 'args': {'shape': [V]}}
    ])
    graph.defs['fitted_ngram_trunk'] = {'kind': 'bundle', 'role': 'cluster', 'inputs': ['prev'],
                                        'nodes': nodes, 'outputs': {'logits': 'logits'}}
    root = graph.defs[graph.entries['step']]
    old_logits = root['outputs']['logits']
    root['nodes'].extend([
        {'id': 'fitted_trunk', 'use': 'fitted_ngram_trunk', 'in': {'prev': '$prev'}},
        {'id': 'authored_residual', 'use': 'op:scale', 'in': {'x': old_logits}, 'args': {'c': residual_weight}},
        {'id': 'fitted_combined', 'use': 'op:add', 'in': {'a': 'fitted_trunk.logits', 'b': 'authored_residual'}}
    ])
    root['outputs']['logits'] = 'fitted_combined'
    raw = json.dumps(partition, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    provenance = {'artifact_class': 'fitted', 'method': 'UTF8 byte bigram counts, add-alpha unigram backoff; BOS/EOS counted',
                  'fit_partition': 'dev_train_selection', 'fit_partition_sha256': hashlib.sha256(raw).hexdigest(),
                  'corpus_sha256': partition.get('corpus_sha256'), 'fit_families': [x['family'] for x in items],
                  'fit_documents': len(items), 'fit_content_bytes': sum(len(x['text'].encode()) for x in items),
                  'unigram_alpha': smoothing, 'backoff_mass': backoff_mass, 'authored_residual_weight': residual_weight,
                  'founder_hash': founder.hash(), 'gradient_steps': 0, 'gradient_estimators': 0,
                  'report_data_used': False, 'final_data_used': False,
                  'size_caveat': 'adds 259x259 fitted slots; founder comparison is not capacity-matched'}
    graph.meta.update({'artifact_class': 'fitted', 'fitted_provenance': provenance,
                       'parent_hash': founder.hash(), 'training': 'count-fitted; no gradient methods'})
    graph.validate()
    check_profile(graph, 'ladder-micro/1')
    return graph, provenance
