# SPDX-License-Identifier: MIT-0
"""Compare exposed development receipts; no model execution or answer routing."""


def native_records(result):
    rows=[r for r in result['rows'] if r['split']=='native-development']
    if len(rows)!=1:raise ValueError('exactly one native development row required')
    row=rows[0];records=row.get('task_records')
    if records is None:raise ValueError('per-task records missing; paired comparison unproven')
    indexed={}
    for item in records:
        identity=item['task_id']
        if identity in indexed:raise ValueError('duplicate task ID')
        successes=item['query_successes']
        if not successes or any(type(x) is not bool for x in successes):raise ValueError('nonempty boolean query outcomes required')
        if type(item['solved']) is not bool or item['solved']!=all(successes):raise ValueError('task solve verdict inconsistent')
        if type(item['demo_consistent']) is not bool:raise ValueError('boolean consistency required')
        if item['demo_consistent_but_wrong']!=(item['demo_consistent'] and not item['solved']):raise ValueError('false positive verdict inconsistent')
        indexed[identity]=item
    totals={'tasks':len(records),'tasks_solved':sum(x['solved'] for x in records),
            'test_inputs':sum(len(x['query_successes']) for x in records),
            'test_inputs_correct':sum(sum(x['query_successes']) for x in records),
            'demo_consistent_tasks':sum(x['demo_consistent'] for x in records),
            'demo_consistent_but_wrong_tasks':sum(x['demo_consistent_but_wrong'] for x in records)}
    if any(row[k]!=v for k,v in totals.items()):raise ValueError('aggregate/per-task disagreement')
    return indexed


def compare_native(candidate,control):
    manifest=candidate['policy']['native_manifest_sha256']
    if not manifest or manifest!=control['policy']['native_manifest_sha256']:raise ValueError('different native manifests')
    a,b=native_records(candidate),native_records(control)
    if set(a)!=set(b):raise ValueError('different task populations')
    if any(len(a[k]['query_successes'])!=len(b[k]['query_successes']) for k in a):raise ValueError('different query populations')
    gained=sorted(k for k in a if a[k]['solved'] and not b[k]['solved'])
    lost=sorted(k for k in a if b[k]['solved'] and not a[k]['solved'])
    return dict(protocol='native-development-paired-receipts/1',native_manifest_sha256=manifest,
                candidate_hash=candidate['policy']['graph_hash'],control_hash=control['policy']['graph_hash'],
                tasks=len(a),gained=gained,lost=lost,net_gain=len(gained)-len(lost),
                both_solved=sorted(k for k in a if a[k]['solved'] and b[k]['solved']),
                limitation='receipt consistency check, not independent score replication or sealed transfer')
