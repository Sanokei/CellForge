"""Small language-only curriculum; targets are scorer data, never host answers.

Selection initially uses next-byte content-only loss. Copy/recall/QA are
separate quality observations; no repair, specialist calls, or normalization.
"""
import hashlib
import json
import time
import numpy as np
from ..ir.graph import GraphError
from ..runtime.lm import score_texts,generate
from ..runtime.text_codec import BYTE_CODEC,BYTE_VOCAB

KINDS={'next_byte','copy','recall','qa'}


def validate_items(items):
    if not isinstance(items,list) or not items:raise ValueError('nonempty item list required')
    seen=set()
    for item in items:
        if not isinstance(item,dict) or not isinstance(item.get('id'),str) or not item['id'] or item['id'] in seen:
            raise ValueError('unique nonempty item IDs required')
        seen.add(item['id'])
        if item.get('kind') not in KINDS:raise ValueError('unknown curriculum kind')
        fields=('text',) if item['kind']=='next_byte' else ('prompt','expected')
        if any(not isinstance(item.get(k),str) or not item[k] for k in fields):
            raise ValueError('nonempty text/prompt/expected strings required')
        if item['kind']=='recall':
            distance=item.get('dependency_distance_bytes')
            if type(distance) is not int or not 0<=distance<=len(item['prompt'].encode('utf-8')):
                raise ValueError('recall requires bounded dependency_distance_bytes')
            if item['expected'] not in item['prompt']:raise ValueError('recall target must occur in supplied context')
    return items


def validate_partition_manifests(selection,report):
    """Called by an operator; does not expose report questions to selection."""
    if selection.get('partition')!='selection' or report.get('partition')!='report':
        raise ValueError('selection/report partition labels required')
    a,b=validate_items(selection['items']),validate_items(report['items'])
    ids=lambda items:{x['id'] for x in items}
    content=lambda items:{hashlib.sha256(x.get('text',x.get('prompt')).encode('utf-8')).hexdigest() for x in items}
    families=lambda items:{x['family'] for x in items if 'family' in x}
    if ids(a)&ids(b) or content(a)&content(b) or families(a)&families(b):
        raise ValueError('selection/report items or declared families overlap')
    return True


def evaluate_language_curriculum(graph,items,*,backend='compiled',state_only_prefill=True,max_new_tokens=24,include_raw=True):
    validate_items(items)
    if graph.meta.get('task')!='lm' or graph.meta.get('text_codec')!=BYTE_CODEC or graph.meta.get('vocab')!=BYTE_VOCAB:
        raise GraphError('curriculum requires canonical UTF8 byte language graph')
    if type(max_new_tokens) is not int or not 1<=max_new_tokens<=4096:raise ValueError('bounded generation required')
    start=time.perf_counter()
    raw=[];next_items=[x for x in items if x['kind']=='next_byte']
    total_bits=0.;total_bytes=0;failed=0
    penalty=max(8.,float(np.log2(len(BYTE_VOCAB))+1))
    if next_items:
        scores,info=score_texts(graph,[x['text'] for x in next_items])
        for j,item in enumerate(next_items):
            length=len(item['text'].encode('utf-8'))
            failure=bool(scores['failed'][j])
            bits=penalty*length if failure else float(info['per_pos_bits'][j,:length].sum())
            total_bits+=bits;total_bytes+=length;failed+=failure
            raw.append({'id':item['id'],'kind':'next_byte','content_bytes':length,'content_bits':bits,'failed':failure})
    generated=[x for x in items if x['kind']!='next_byte']
    quality={kind:{'n':0,'exact_text':0,'exact_terminated':0,'failures':0} for kind in ('copy','recall','qa')}
    long_context={'n':0,'exact_text':0,'exact_terminated':0}
    if generated:
        outputs,info=generate(graph,len(generated),prompts=[x['prompt'] for x in generated],max_len=max_new_tokens,
                              backend=backend,state_only_prefill=state_only_prefill)
        for j,(item,text) in enumerate(zip(generated,outputs)):
            failure=info['failure'][j]
            exact=text==item['expected']
            terminated=exact and failure is None
            group=quality[item['kind']]
            group['n']+=1;group['exact_text']+=int(exact);group['exact_terminated']+=int(terminated);group['failures']+=int(failure is not None)
            distance=item.get('dependency_distance_bytes')
            if item['kind']=='recall' and distance>8:
                long_context['n']+=1;long_context['exact_text']+=int(exact);long_context['exact_terminated']+=int(terminated)
            raw.append({'id':item['id'],'kind':item['kind'],'prompt':item['prompt'],'expected':item['expected'],
                'output':text,'token_ids':info['token_ids'][j],'failure':failure,'exact_text':exact,
                'exact_terminated':terminated,'dependency_distance_bytes':distance})
    report={'protocol':'language-curriculum/1','model_hash':graph.hash(),
       'items_sha256':hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest(),
       'fitness_bits_per_byte':total_bits/total_bytes if total_bytes else None,
       'eligible_for_selection':bool(total_bytes) and failed==0,
       'next_byte':{'content_bits':total_bits,'content_bytes':total_bytes,'failed_documents':failed,'eos_excluded':True,
                    'failure_bits_per_content_byte':penalty,'backend':'reference'},
       'generation':quality,'long_context_recall':long_context,
       'backend':backend,'state_only_prefill':state_only_prefill,'max_new_tokens':max_new_tokens,
       'seconds':time.perf_counter()-start,'selection_rule':'next-byte content loss only; all quality metrics separate initially',
       'limitations':'small authored curriculum, raw ordinaryLM output only; no independent frontier or AGI evidence'}
    if include_raw:report['items']=raw
    return report
