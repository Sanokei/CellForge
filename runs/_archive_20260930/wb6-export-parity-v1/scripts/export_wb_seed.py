"""Operator export/parity on current corpus; excludes prior report families."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.construction.wb_byte_seed import wb_byte_seed, reference_probabilities
from cellforge.reference.evolution_fitness import compiled_content_fitness
from cellforge.governance.ledger import Ledger
from prepare_fitted_evolution import context_items


def write(path, value): path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--eval',required=True)
    p.add_argument('--founder',required=True);p.add_argument('--exclude-private',required=True)
    p.add_argument('--private',required=True);p.add_argument('--public',required=True)
    p.add_argument('--parity-docs',type=int,default=24)
    args=p.parse_args()
    private,public=Path(args.private).resolve(),Path(args.public)
    if private.parent!=Path('/srv/cellforge-private') or private.exists() or public.exists():
        raise SystemExit('fresh operator/public destinations required')
    raw=Path(args.source).read_bytes();eval_raw=Path(args.eval).read_bytes()
    excluded={x['family'] for x in json.loads((Path(args.exclude_private)/'report-only.json').read_text())}
    train=json.loads(raw);evaluation=json.loads(eval_raw)
    if len(evaluation)<165:raise SystemExit('expected fixed165-document development corpus')
    excluded.update(hashlib.sha256(t.encode()).hexdigest() for t in evaluation[157:165])
    items=[]
    for text in train:
        family=hashlib.sha256(text.encode()).hexdigest()
        if family not in excluded: items.append({'family':family,'text':text})
    # Exact duplicates have identical count contribution in Claude's reference;
    # preserve multiplicity via occurrence IDs while reporting whole-text hash.
    seen={}
    for item in items:
        family=item['family'];seen[family]=seen.get(family,0)+1
        item['whole_text_sha256']=family;item['family']=family+'-'+str(seen[family])
    partition={'partition':'dev_train_selection','source_split':'dev_train',
               'corpus_sha256':hashlib.sha256(raw).hexdigest(),'items':items,
               'exclusion_policy':'prior pilot and newly reserved report-only whole-text families remain excluded',
               'excluded_prior_report_families':sorted(excluded)}
    private.mkdir(mode=0o700);public.mkdir()
    write(private/'fit-selection.json',partition)
    ledger=Ledger(public/'ledger.jsonl')
    started=time.perf_counter()
    seed,provenance,counts=wb_byte_seed(Graph.load(args.founder),partition)
    digest=seed.save(public/'fitted-seed')
    write(public/'fitted-seed'/'MODEL_CARD.json',dict(provenance,model_hash=digest,stats=seed.stats()))
    ledger.append('count_seed_export',dict(provenance,model_hash=digest), 'operator')
    for label,indices in [('a',range(0,8,2)),('b',range(1,8,2)),('report',range(157,165))]:
        texts=[evaluation[i] for i in indices]
        manifest={'partition':'report' if label=='report' else 'selection',
                  'corpus_sha256':hashlib.sha256(eval_raw).hexdigest(),
                  'items':[{'id':str(i),'family':hashlib.sha256(evaluation[i].encode()).hexdigest(),
                            'kind':'next_byte','text':evaluation[i][:64] if label!='report' else evaluation[i]} for i in indices],
                  'exposure':'public dev_eval previously scored by Claude; reporting-only development, never unexposed final',
                  'selection_window':'first64ASCIIbytes, state reset at document/window start; full-prefix long tasks separate'}
        assert all(x['text'].isascii() for x in manifest['items'])
        write(private/f'{label}.json',manifest)
    for label in ('a','b'): write(private/f'context-{label}.json',context_items(label))
    parity_items=[{'id':str(i),'kind':'next_byte','text':text} for i,text in enumerate(evaluation[:args.parity_docs])]
    reference_bits=[];reference_bytes=0
    tick=time.perf_counter()
    for item in parity_items:
        raw_text=item['text'].encode();bits=0.
        for i,byte in enumerate(raw_text): bits-=math.log2(reference_probabilities(counts,raw_text[max(0,i-6):i])[byte])
        reference_bits.append(bits);reference_bytes+=len(raw_text)
    reference_seconds=time.perf_counter()-tick
    graph_score=compiled_content_fitness(seed,parity_items)
    deltas=[abs(a-b) for a,b in zip(reference_bits,graph_score['next_byte']['per_document_content_bits'])]
    result={'protocol':'wb6-ir-parity/1','model_hash':digest,'provenance':provenance,
            'eval_source_sha256':hashlib.sha256(eval_raw).hexdigest(),'parity_documents':len(parity_items),
            'parity_content_bytes':reference_bytes,'reference_bits_per_byte':sum(reference_bits)/reference_bytes,
            'graph_bits_per_byte':graph_score['fitness_bits_per_byte'],'max_document_bits_delta':max(deltas),
            'parity_passed':max(deltas)<1e-8 and graph_score['eligible_for_selection'],
            'reference_seconds':reference_seconds,'graph_score':graph_score,'total_seconds':time.perf_counter()-started,
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations':'prior-report exclusion changes fit from Claude full2088-doc reference; compare IR/reference with identical fit and eval. Full165-doc 1.3816 and first24 are different scopes. No QA or frontier claim.'}
    write(public/'parity.json',result);ledger.append('ir_reference_parity',result,'operator')
    print(json.dumps({k:v for k,v in result.items() if k not in ('provenance','graph_score')}))
    if not result['parity_passed']:raise SystemExit('IR/reference mismatch; no evolution launch')


if __name__=='__main__':main()
