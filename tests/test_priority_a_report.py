import json,subprocess,sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_priority_a import paired_interval

def test_paired_interval_uses_three_seed_variation():
    result=paired_interval([0.,.1,.2])
    assert result['mean']==pytest.approx(.1)
    assert result['lower']<0<result['upper']
    with pytest.raises(ValueError):paired_interval([.1])

def test_premature_report_does_not_open_corpus_or_create_output(tmp_path):
    plan=tmp_path/'PLAN.json';plan.write_text('{}')
    batch=tmp_path/'batch';batch.mkdir()
    (batch/'result.json').write_text(json.dumps(dict(proposals_per_arm_per_seed=2)))
    out=tmp_path/'report'
    script=Path(__file__).resolve().parents[1]/'scripts/report_priority_a.py'
    result=subprocess.run([sys.executable,str(script),'--plan',str(plan),'--final-batch',str(batch),'--out',str(out)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'all sixteen proposals required' in result.stderr
    assert not out.exists()
