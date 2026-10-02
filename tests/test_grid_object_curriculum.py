import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_grid_population_matched as m


def test_object_curriculum_is_disjoint_and_selection_excludes_query_fields(tmp_path,monkeypatch):
    directory=tmp_path/'arcagi2/data/training';directory.mkdir(parents=True)
    count=0;i=0
    while count<200:
        identity=f'task{i}';i+=1
        if int(hashlib.sha256(identity.encode()).hexdigest(),16)%5==0:continue
        (directory/(identity+'.json')).write_text(json.dumps({'train':[],'test':[]}));count+=1
    paths=[p for _,p in sorted((hashlib.sha256(p.stem.encode()).hexdigest(),p) for p in directory.glob('*.json'))]
    digest=hashlib.sha256()
    for path in paths:digest.update(path.stem.encode()+b':'+hashlib.sha256(path.read_bytes()).digest())
    monkeypatch.setattr(m,'EXPECTED_NATIVE',digest.hexdigest())
    selection,validation,_,commitments=m.frozen_corpora(tmp_path,per_family=2,include_object=True)
    assert len(selection)==len(validation)==20
    assert all(set(t)=={'train'} for t in selection)
    assert sum(t.get('family')=='object_recolor' for t in validation)==2
    assert commitments['selection_demo_pairs']==58
    selection,validation,_,commitments=m.frozen_corpora(tmp_path,per_family=2,include_object=True,include_scaling=True)
    assert len(selection)==len(validation)==24
    assert all(set(t)=={'train'} for t in selection)
    assert sum(t.get('family') in ('repeat2','repeat3') for t in validation)==4
    assert commitments['selection_demo_pairs']==70
