import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from diagnose_context_mask_errors import scalar_key,explain


def test_padding_and_zero_color_are_distinct_in_masked_scalar_key():
    assert scalar_key([[0]],0,0,16)==11**4
    assert scalar_key([[0]],0,1,16)==0


def test_unknown_and_conflicting_counts_are_distinguished():
    task={'direction':[0,1],'train':[{'input':[[1]],'output':[[2]]},
                                  {'input':[[1]],'output':[[3]]}],
          'test':[{'input':[[1,4]],'output':[[2,5]]}]}
    prediction,counts,_=explain(task,16)
    assert prediction==[[1,4]]
    assert counts['wrong_conflicting']==1 and counts['wrong_unknown']==1
