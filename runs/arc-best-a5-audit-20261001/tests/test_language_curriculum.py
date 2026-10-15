import numpy as np
import pytest
from cellforge.ir.graph import Graph
from cellforge.runtime.text_codec import BYTE_CODEC,BYTE_VOCAB
from cellforge.reference.language_curriculum import evaluate_language_curriculum,validate_partition_manifests


def fixture(failed=False):
    nodes=[{'id':'logits','use':'op:scale','in':{'x':'$prev'},'args':{'c':1e308 if failed else 0}}]
    output='logits'
    if failed:
        nodes.append({'id':'overflow','use':'op:add','in':{'a':'logits','b':'logits'}});output='overflow'
    return Graph(defs={'step':{'kind':'bundle','inputs':['ctx','prev','step','state'],'nodes':nodes,
                              'outputs':{'logits':output,'state':'$state'}}},entries={'step':'step'},
       meta={'task':'lm','text_codec':BYTE_CODEC,'vocab':BYTE_VOCAB,'ctx_len':2,'state_dim':0})


def test_content_bytes_exclude_eos_and_failures_charge_every_byte():
    item=[{'id':'n','kind':'next_byte','text':'é'}]
    good=evaluate_language_curriculum(fixture(),item)
    assert good['next_byte']['content_bytes']==2
    assert good['fitness_bits_per_byte']==pytest.approx(np.log2(259))
    assert good['eligible_for_selection']
    with np.errstate(over='ignore'):
        bad=evaluate_language_curriculum(fixture(True),item)
    assert bad['next_byte']['content_bits']==pytest.approx(2*(np.log2(259)+1))
    assert not bad['eligible_for_selection']


def test_raw_generation_quality_separate_and_anchor_validated():
    item={'id':'q','kind':'recall','prompt':'red '+('x'*20),'expected':'red','dependency_distance_bytes':21}
    report=evaluate_language_curriculum(fixture(),[item],max_new_tokens=3)
    assert report['fitness_bits_per_byte'] is None
    assert not report['eligible_for_selection']
    assert report['long_context_recall']['n']==1
    assert report['generation']['recall']['exact_text']==0
    assert report['items'][0]['output']==''
    assert report['items'][0]['expected']=='red'
    with pytest.raises(ValueError):evaluate_language_curriculum(fixture(),[dict(item,dependency_distance_bytes=20)])


def test_report_whole_items_and_families_are_disjoint():
    a={'partition':'selection','items':[{'id':'a','kind':'next_byte','text':'hello','family':'one'}]}
    b={'partition':'report','items':[{'id':'b','kind':'next_byte','text':'other','family':'two'}]}
    assert validate_partition_manifests(a,b)
    for field,value in [('id','a'),('text','hello'),('family','one')]:
        item=dict(b['items'][0]);item[field]=value
        with pytest.raises(ValueError):validate_partition_manifests(a,{'partition':'report','items':[item]})
