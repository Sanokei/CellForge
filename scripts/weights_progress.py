"""Current literal-weight reporting; historical allele charts remain separate."""

import argparse,collections,datetime,hashlib,json

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
try:
    from .report_verification import read_gate, component_metrics, population_metrics
except ImportError:
    from report_verification import read_gate, component_metrics, population_metrics

def collect(root):

    runs=[]

    for directory in sorted(Path(root).glob('weights-*')):

        if not directory.is_dir():continue

        records={};origins=collections.Counter();parents=[];changes=[];s={};inventory_errors=[]

        for file in (directory/'weights').glob('*.json'):

            if file.stem.endswith('-router'):continue

            payload=json.loads(file.read_text(encoding='utf-8-sig'))

            tensors=payload.get('tensors')

            if not tensors:continue

            h=hashlib.sha256(json.dumps(tensors,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

            if h!=file.stem:
                inventory_errors.append('unsupported_or_invalid_tensor_hash_format: '+str(file));continue

            records[h]={'parameters':sum(np.asarray(v).size for v in tensors.values()),'tensors':tensors}

        standard=directory/'STANDARD.json'

        if standard.exists():

            s=json.loads(standard.read_text());entries=s.get('genomes',[])

            for entry in entries:

                h=entry['hash']

                if h not in records:continue

                if records[h].get('origin') is not None:continue

                records[h]['origin']=entry.get('origin','unknown');origins[entry.get('origin','unknown')]+=1

                for parent in entry.get('parents',[]):parents.append([parent,h])

        for file in (directory/'provenance').glob('*.json'):

            payload=json.loads(file.read_text());response=payload.get('response',payload)

            # Authored responses are counted as receipts, not independent genomes.

            changes.append({'file':str(file),'model':response.get('model',payload.get('actor')),

                            'ops':len(response.get('ops',[]))})

        receipts=directory/'inheritance-receipts.json'

        scalar_changes=[]

        if receipts.exists():

            for x in json.loads(receipts.read_text()):

                scalar_changes.append({'child':x['child_hash'],'operator':x.get('operator'),

                       'changed_scalars':x['changed_scalars']})

        curve=directory/'learning_curve.json'

        gate=read_gate(directory)
        learning=json.loads(curve.read_text(encoding='utf-8-sig')) if curve.exists() else None
        plot_learning=learning if learning and all('generation' in row for row in learning.get('rows',[])) else None
        format_status='literal_tensor_inventory' if records and not inventory_errors else 'unsupported_or_missing_weight_format'
        if inventory_errors:
            gate['errors'].extend(inventory_errors);gate['scientific_acceptance']=False;gate['scientific_headline_allowed']=False
        runs.append({'run':directory.name,'distinct_genomes':len(records) if records and not inventory_errors else None,'origins':dict(origins),
                     'format_status':format_status,'inventory_errors':inventory_errors,'verification':gate,
                     'component_result':component_metrics(directory,gate),
                     'population_result':population_metrics(directory,gate),
                     'recorded_summary':json.loads((directory/'summary.json').read_text(encoding='utf-8-sig')) if (directory/'summary.json').exists() else None,
                     'recorded_learning':learning,

                     'parameter_counts':{h:r['parameters'] for h,r in records.items()},'parents':parents,

                     'maker_receipts':changes,'numeric_inheritance_changes':scalar_changes,

                     'learning':plot_learning,

                     'structural_families':s.get('structural_families',[]) if standard.exists() else [],

                     'skill_families':s.get('skill_families',[]) if standard.exists() else [],

                     'missing':'no supported real-task learning curve' if plot_learning is None else None})

    return runs

def stamp_gates(fig,data):
    blocked=[r['run'] for r in data if not r['verification']['scientific_headline_allowed']]
    if not data:
        fig.text(.5,.025,'SCIENTIFIC ACCEPTANCE BLOCKED; no reportable evidence.',ha='center',fontsize=8,color='darkred')
    elif blocked:
        fig.text(.5,.025,'SCIENTIFIC ACCEPTANCE BLOCKED; '+str(len(blocked))+' run(s) with failed/missing/unsupported gates. See report_verification.json.',ha='center',fontsize=8,color='darkred')
    else:
        fig.text(.5,.025,'Scientific gates independently verified for the reported scope.',ha='center',fontsize=8)


def generate(root,outdir):

    out=Path(outdir)

    if out.exists():raise ValueError('versioned report output must be fresh')

    data=collect(root);out.mkdir(parents=True)

    (out/'weights_population.json').write_text(json.dumps(data,indent=2))

    fig,ax=plt.subplots(figsize=(16,6));names=[x['run'] for x in data]

    ax.bar(names,[x['distinct_genomes'] if x['distinct_genomes'] is not None else np.nan for x in data]);ax.tick_params(axis='x',rotation=30)
    for i,run in enumerate(data):
        if run['distinct_genomes'] is None:ax.text(i,0,'unverified\nformat',ha='center',fontsize=7,va='bottom')
    ax.set_xlim(-.5,len(data)-.5)

    ax.set_ylabel('Distinct hash-verified literal tensor sets');ax.set_title('Weight population inventory; not evidence of learning')

    stamp_gates(fig,data);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(out/'weights_population.png',dpi=130);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5));any_curve=False

    for run in data:

        curve=run['learning']

        if not curve:continue

        for partition,style in [('development','--'),('reporting','-')]:

            for arm in ('llm','random','parent'):

                rows=[x for x in curve['rows'] if x['arm']==arm]

                if not rows:continue

                generations=sorted({x['generation'] for x in rows})

                means=[float(np.mean([x[partition]['solved']/x[partition]['tasks'] for x in rows if x['generation']==g])) for g in generations]

                ax.plot(generations,means,style,marker='o',label=run['run']+' '+arm+' '+partition+' [unverified mechanism]')

                # Seed-context means are not confidence intervals or maker replications.

                any_curve=True

    if any_curve:ax.legend(fontsize=7)

    else:ax.text(.5,.5,'No supported real-task learning curve available',ha='center',transform=ax.transAxes)

    ax.set_ylim(-.05,1.05);ax.set_xlabel('Charged revision generation');ax.set_ylabel('Exact task solutions / tasks')

    ax.set_title('Public-training task-held-out learning; no synthetic scores')

    stamp_gates(fig,data);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(out/'weights_learning.png',dpi=130);plt.close(fig)

    fig,ax=plt.subplots(figsize=(16,6))

    labels=[];values=[]

    for run in data:

        seen=set()

        for receipt in run['numeric_inheritance_changes']:

            if receipt['child'] in seen:continue

            seen.add(receipt['child']);labels.append(run['run']+'\n'+str(receipt['operator']));values.append(sum(receipt['changed_scalars'].values()))

    if labels:ax.bar(labels,values);ax.tick_params(axis='x',rotation=15)

    else:ax.text(.5,.5,'No numeric inheritance receipts',ha='center',transform=ax.transAxes)

    ax.set_title('Numerical inheritance edits; viability does not prove improvement');ax.set_ylabel('Actually changed scalar positions')

    stamp_gates(fig,data);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(out/'weights_inheritance.png',dpi=130);plt.close(fig)

    fig,ax=plt.subplots(figsize=(16,6))

    offsets={}

    for j,run in enumerate(data):

        hs=list(run['parameter_counts']);positions={}

        for i,h in enumerate(hs):

            positions[h]=(j,i/max(1,len(hs)-1))

        for parent,child in run['parents']:

            if parent in positions and child in positions:

                x,y=positions[parent];xx,yy=positions[child]

                ax.annotate('',xy=(xx+.25,yy),xytext=(x, y),arrowprops={'arrowstyle':'->','alpha':.2,'lw':.6})

        ax.scatter([j for _ in hs],[positions[h][1] for h in hs],s=12,label=run['run'])

    ax.set_xticks(range(len(data)),[r['run'] for r in data],rotation=20)

    ax.set_yticks([]);ax.set_title('Recorded literal-weight ancestry, including rejected candidates')

    stamp_gates(fig,data);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(out/'weights_lineage.png',dpi=130);plt.close(fig)

    fig,ax=plt.subplots(figsize=(16,6))

    fam_labels=[];fam_counts=[]

    for run in data:

        for kind,key in [('structure','structural_families'),('skill','skill_families')]:

            fam_labels.append(run['run']+'\n'+kind);fam_counts.append(len(set(run[key])))

    fig.set_size_inches(12,max(6,len(fam_labels)*.42))
    ax.barh(fam_labels,fam_counts);ax.tick_params(axis='y',labelsize=8);ax.invert_yaxis()

    ax.set_xlabel('Declared family labels');ax.set_title('Declared structures/skills; absent labels are not inferred diversity')

    stamp_gates(fig,data);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(out/'weights_families.png',dpi=130);plt.close(fig)

    gates={r['run']:r['verification'] for r in data}
    (out/'report_verification.json').write_text(json.dumps({'scientific_headline_allowed':all(g['scientific_headline_allowed'] for g in gates.values()) if gates else False,'runs':gates},indent=2),encoding='utf-8')
    components=[r for r in data if r['component_result'] is not None]
    (out/'bounded_component_results.json').write_text(json.dumps({r['run']:r['component_result'] for r in components},indent=2),encoding='utf-8')
    for run in components:
        component=run['component_result'];metrics=component['metrics']
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
        for ax,partition in zip(axes,('development','reporting')):
            labels=[m['arm']+((' seed '+str(m['seed'])) if m['seed'] is not None else '') for m in metrics]
            values=[m['scores'][partition]['state_correct']/m['scores'][partition]['total'] for m in metrics]
            bars=ax.bar(labels,values);ax.set_ylim(0,1.15);ax.tick_params(axis='x',rotation=25)
            for bar,m in zip(bars,metrics):
                score=m['scores'][partition];ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,str(score['state_correct'])+'/'+str(score['total']),ha='center')
            ax.set_title(partition);ax.set_ylabel('Fixture state checks correct / checks')
        fig.suptitle('Bounded lexical argument trial; provisional and unadmitted\nMetric status: '+component['metric_status'])
        stamp_gates(fig,[run]);fig.tight_layout(rect=(0,.09,1,.89));fig.savefig(out/'bounded_action_component.png',dpi=130);plt.close(fig)
    populations=[r for r in data if r['population_result'] is not None]
    (out/'population_component_results.json').write_text(json.dumps({r['run']:r['population_result'] for r in populations},indent=2),encoding='utf-8')
    for run in populations:
        diagnostic=run['population_result'];rows=diagnostic['raw_scores']
        fig,ax=plt.subplots(figsize=(12,5))
        labels=['seed '+str(row['seed']) for row in rows]
        ax.bar(labels,[row['raw_recorded_correct'] for row in rows],color='gray')
        for i,row in enumerate(rows):
            ax.text(i,row['raw_recorded_correct']+.05,'raw '+str(row['raw_recorded_correct'])+'/'+str(row['raw_recorded_case_count']),ha='center')
        ax.set_ylim(0,max([row['raw_recorded_case_count'] for row in rows]+[1])+1)
        invalid=diagnostic['valid_accuracy'] is None
        ax.set_title('INVALID OR UNVERIFIED FIXTURE: raw scores; no valid five-case accuracy' if invalid else 'Quality-verified bounded population component; scientific acceptance blocked')
        ax.set_ylabel('Recorded correct labels (not validated accuracy)' if invalid else 'Correct bounded fixture cases')
        if diagnostic.get('quality_errors'):fig.text(.5,.1,'; '.join(diagnostic['quality_errors']),ha='center',fontsize=8,color='darkred')
        stamp_gates(fig,[run]);fig.tight_layout(rect=(0,.15,1,1));fig.savefig(out/'population_fixture_diagnostics.png',dpi=130);plt.close(fig)
    notes=['Scientific acceptance is blocked unless supported independent replay and every receipt gate authorize it. Current action/population verifier never authorizes full scientific acceptance. Population accuracy additionally requires independent fixture quality and matching QUALITY_VERIFICATION.json. Report generation success is not experiment acceptance.',
           'Positive bounded component metrics are retained separately from ARC learning. Unsupported standard-weights/action/population formats are unverified or missing, never zero-genome validated experiments.',
           'Historical allele/genelist reports remain separate in scripts/genome_progress.py. Tensor hash consistency is not authorship or mechanism verification. Recorded learning is not independent mechanism validation. Missing topology/lineage/population verification remains blocking.',
           '', 'Per-run gate evidence:']
    for run in data:
        gate=run['verification'];notes.append(run['run']+': '+gate['status']+'; scientific_headline_allowed='+str(gate['scientific_headline_allowed']))
        notes.extend('  - '+error for error in gate['errors'])
        receipt=gate.get('receipt') or {};reason=receipt.get('scientific_standard',{}).get('reason')
        if reason:notes.append('  Recorded scientific gate reason: '+reason)
    (out/'README.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')

    return [str(x) for x in out.iterdir()]

if __name__=='__main__':

    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');p.add_argument('--outdir');a=p.parse_args()

    out=a.outdir or 'reports/weights-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

    print(json.dumps(generate(a.runs,out)))

