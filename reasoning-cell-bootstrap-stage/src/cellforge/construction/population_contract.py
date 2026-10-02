"""Executable architecture checks; historical ensemble runs cannot pass."""
def require_two_level_run(run):
    p=run.get('population',{})
    if not p.get('cells') or not p.get('live') or not p.get('histories'):raise ValueError('independent cells and histories required')
    events=p.get('events',[])
    if not any(e.get('event')=='birth' for e in events):raise ValueError('executed cell births required')
    if not run.get('ensemble_population',{}).get('histories'):raise ValueError('separate ensemble history required')
    for cell in p['cells'].values():
        if not cell.get('params') or any(k.startswith('slot') for k in cell['params']):raise ValueError('literal numerical cells required')
        if cell['parents'] and not cell.get('provenance'):raise ValueError('child provenance required')
    if run.get('selection_partition')!='development':raise ValueError('reporting cannot select')
    if not run.get('assembly',{}).get('cell_ids'):raise ValueError('evidence-earned cooperation required')
    return True

