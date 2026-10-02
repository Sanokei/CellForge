import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from expanded_grid_genetics import canonical

@pytest.mark.parametrize('backend',['reference','compiled'])
def test_modal_moe_crop_with_demonstration_recolor(backend):
 root=Path(__file__).parents[1]/'runs/modal-crop-moe-v1'
 base=Graph.load(root/'inactive-modal-control');g=Graph.load(root/'modal-crop-moe');canonical(base,g)
 def canvas(obj):
  x=[[9]*8 for _ in range(7)]
  for i in range(2):x[i+2][3:5]=obj[i]
  return x
 obj=[[1,2],[3,4]];query=[[4,3],[2,1]]
 demos=[{'input':canvas(obj),'output':[[2,3],[4,5]]}]
 p=GridPredictor(g,backend=backend)
 assert [[5,4],[3,2]] in p(demos,canvas(query))
 assert [[5,4],[3,2]] in p(demos,canvas(query))
