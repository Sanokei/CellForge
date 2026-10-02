import importlib.util,json,tarfile
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('bundler',Path(__file__).parents[1]/'scripts/build_research_source_bundle.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_closure_cycles_and_dynamic_sibling(tmp_path):
 d=tmp_path/'scripts';d.mkdir();(d/'main.py').write_text("from child import thing\nname='dynamic.py'\n")
 (d/'child.py').write_text('import main\n');(d/'dynamic.py').write_text('x=1\n')
 assert {p.name for p in m.script_closure(d/'main.py')}=={'main.py','child.py','dynamic.py'}
 (d/'dynamic.py').unlink()
 with pytest.raises(ValueError,match='missing script'):m.script_closure(d/'main.py')
def test_bundle_hashes_artifact_and_rejects_reuse(tmp_path):
 (tmp_path/'scripts').mkdir();(tmp_path/'scripts/main.py').write_text('import numpy\n')
 (tmp_path/'src').mkdir();(tmp_path/'src/a.py').write_text('x=1\n');(tmp_path/'artifact').mkdir();(tmp_path/'artifact/HASH').write_text('frozen')
 out=tmp_path/'out.tar.gz';r=m.build_bundle(tmp_path,'scripts/main.py',['artifact'],out)
 with tarfile.open(out) as t:assert set(t.getnames())=={'scripts/main.py','src/a.py','artifact/HASH'}
 assert all(len(x['sha256'])==64 for x in r['files']);assert r['preflight_executed'] is False
 with pytest.raises(ValueError,match='fresh'):m.build_bundle(tmp_path,'scripts/main.py',[],out)
def test_missing_artifact_rejected(tmp_path):
 (tmp_path/'main.py').write_text('x=1')
 with pytest.raises(ValueError,match='missing artifact'):m.build_bundle(tmp_path,'main.py',['absent'],tmp_path/'out.tar.gz')

def test_attached_builder_also_includes_its_transitive_siblings(tmp_path):
 d=tmp_path/'scripts';d.mkdir()
 (d/'main.py').write_text('x=1\n')
 (d/'builder.py').write_text('from donor import build\n')
 (d/'donor.py').write_text('import helper\n')
 (d/'helper.py').write_text('build=1\n')
 out=tmp_path/'out.tar.gz'
 m.build_bundle(tmp_path,'scripts/main.py',['scripts/builder.py'],out)
 with tarfile.open(out) as t:
  assert set(t.getnames())=={'scripts/main.py','scripts/builder.py','scripts/donor.py','scripts/helper.py'}
