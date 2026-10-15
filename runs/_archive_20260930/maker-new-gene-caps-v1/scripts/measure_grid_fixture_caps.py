# SPDX-License-Identifier: MIT-0
"""Frozen graph fixture/cap receipt; Linux CPU, no benchmark labels."""
import argparse,json,time,hashlib,resource
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter
from cellforge.runtime.grid import pack_grid

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('fresh receipt required')
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
 g=Graph.load(a.graph);policy=dict(protocol='frozen-grid-fixture-caps/1',graph_hash=g.hash(),profile='ladder-micro/1',address_space_limit_bytes=3*1024**3,device_rss_limit_bytes=4*1024**3,stored_parameter_limit=1024,wall_limit_seconds=60,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='fixed boundary fixtures; not correctness on arbitrary tasks',gradients=False)
 check_profile(g,policy['profile']);assert g.meta.get('gradients') is False
 a.out.mkdir();(a.out/'policy.json').write_text(json.dumps(policy,indent=2));start=time.perf_counter();cpu=time.process_time()
 engines=[Interpreter(g),CompiledInterpreter(g)];states=[np.zeros((1,g.meta['state_dim'])) for _ in engines];cases=[]
 for size in (1,30):
  for phase in (0.,1.):
   grid=np.indices((size,size)).sum(0)%10;x,_,h,w=pack_grid(grid.tolist());values=[]
   for i,e in enumerate(engines):
    out=e.run_bundle(g.entries['grid_step'],dict(input=x,input_height=h,input_width=w,output=x if not phase else np.zeros_like(x),output_height=h if not phase else np.zeros_like(h),output_width=w if not phase else np.zeros_like(w),phase=np.array([[phase]]),state=states[i]))
    shapes=dict(state=(1,g.meta['state_dim']),color_logits=(1,g.meta['grid_attempts'],30,30,10),height_logits=(1,g.meta['grid_attempts'],30),width_logits=(1,g.meta['grid_attempts'],30))
    assert set(out)==set(shapes)
    for k,shape in shapes.items():assert out[k].shape==shape and np.isfinite(out[k]).all()
    states[i]=out['state'];values.append(out)
   for k in values[0]:np.testing.assert_array_equal(values[0][k],values[1][k])
   cases.append(dict(size=size,phase=phase,finite=True,parity=True))
 wall=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;stats=g.stats()
 caps=rss<=policy['device_rss_limit_bytes'] and stats['unique_param_storage']<=policy['stored_parameter_limit'] and wall<=policy['wall_limit_seconds']
 result=dict(policy=policy,cases=cases,fixtures_passed=True,caps_passed=caps,wall_seconds=wall,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=rss,stats=stats)
 (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(graph_hash=g.hash(),caps_passed=caps,fixtures_passed=True,wall_seconds=wall,peak_rss_bytes=rss)))
if __name__=='__main__':main()
