import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from build_grid_spatial_genes import (build_symmetry_gene, build_periodic_gene,
    build_drawing_gene, build_kron_gene, build_scale_gene, gene_specs, build_named_gene, DIRECTIONS)
from cellforge.runtime.grid import GridPredictor


def predict(g, x, backend):
    return GridPredictor(g, backend=backend)([{'input':[[1]],'output':[[1]]}], x)[0]


@pytest.mark.parametrize('backend', ['reference','compiled'])
def test_symmetry(backend):
    assert predict(build_symmetry_gene('mirror_h'), [[2,0,0],[0,0,3]],backend)==[[2,0,2],[3,0,3]]
    assert predict(build_symmetry_gene('mirror_v'), [[2,0],[0,0],[0,3]],backend)==[[2,3],[0,0],[2,3]]
    assert predict(build_symmetry_gene('diagonal'), [[0,4,0],[0,0,0],[0,0,0]],backend)==[[0,4,0],[4,0,0],[0,0,0]]
    x=[[0,4,0],[0,0,0]]
    assert predict(build_symmetry_gene('diagonal'),x,backend)==x
    assert predict(build_symmetry_gene('rotate180'),[[3,0,0],[0,0,0]],backend)==[[3,0,0],[0,0,3]]


@pytest.mark.parametrize('backend', ['reference','compiled'])
def test_period_inference_and_conflict(backend):
    x=[[2,3,0,0,0,0]]
    assert predict(build_periodic_gene('column'),x,backend)==[[2,3,2,3,2,3]]
    assert predict(build_periodic_gene('column',period=2),[[2,3,4,0]],backend)==[[2,3,4,0]]
    assert predict(build_periodic_gene('row'),[[2],[3],[0],[0]],backend)==[[2],[3],[2],[3]]
    assert predict(build_periodic_gene('both'),[[2,3,0,0],[4,5,0,0],[0,0,0,0],[0,0,0,0]],backend)==[[2,3,2,3],[4,5,4,5],[2,3,2,3],[4,5,4,5]]
    assert predict(build_periodic_gene(),[[0,0],[0,0]],backend)==[[0,0],[0,0]]
    x=[[2,3,4,0,0,0],[5,6,7,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    assert predict(build_periodic_gene('independent'),x,backend)==[[2,3,4,2,3,4],[5,6,7,5,6,7],[2,3,4,2,3,4],[5,6,7,5,6,7]]


@pytest.mark.parametrize('backend', ['reference','compiled'])
def test_drawing(backend):
    assert predict(build_drawing_gene('ray'),[[2,0,0,3,0]],backend)==[[2,2,2,3,3]]
    assert predict(build_drawing_gene('connect'),[[2,0,0,2,0]],backend)==[[2,2,2,2,0]]
    assert predict(build_drawing_gene('connect'),[[2,0,0,3,0]],backend)==[[2,0,0,3,0]]
    assert predict(build_drawing_gene('connect',('se',)),[[2,0,0],[0,0,0],[0,0,2]],backend)==[[2,0,0],[0,2,0],[0,0,2]]
    assert predict(build_drawing_gene('bounce',('se',),steps=4),[[0,2,0],[0,0,0],[0,0,0]],backend)==[[0,2,0],[2,0,2],[0,2,0]]
    assert predict(build_drawing_gene('bounce',('se',),steps=4),[[2,0,0]],backend)==[[2,2,2]]
    assert predict(build_drawing_gene('bounce',('se',),steps=4),[[2]],backend)==[[2]]


@pytest.mark.parametrize('backend', ['reference','compiled'])
def test_kron(backend):
    x=[[2,0],[3,4]]
    expected=np.kron(x,[[1,0],[0,1]]).tolist()
    assert predict(build_kron_gene(),x,backend)==expected
    assert predict(build_kron_gene('border',3),[[7]],backend)==[[7,7,7],[7,0,7],[7,7,7]]
    x=[[2]*16]
    assert predict(build_kron_gene(),x,backend)==x


def test_catalog_validates():
    assert len(gene_specs())==108
    for name in gene_specs():
        build_named_gene(name)


@pytest.mark.parametrize('backend', ['reference','compiled'])
def test_independent_scaling(backend):
    x=[[2,3],[4,5]]
    assert predict(build_scale_gene(2,3),x,backend)==np.repeat(np.repeat(x,2,axis=0),3,axis=1).tolist()
    enlarged=np.repeat(np.repeat(x,2,axis=0),3,axis=1).tolist()
    assert predict(build_scale_gene(2,3,'sample'),enlarged,backend)==x
    assert predict(build_scale_gene(2,3,'sample'),[[1,2,3]],backend)==[[1,2,3]]


@pytest.mark.parametrize('backend', ['reference','compiled'])
@pytest.mark.parametrize('mode', ['ray','connect','bounce'])
def test_drawing_against_scalar_paths(backend, mode):
    rng=np.random.default_rng(811)
    for direction,(dr,dc) in DIRECTIONS.items():
        g=build_drawing_gene(mode,(direction,),steps=5)
        for h,w in ((1,1),(1,6),(5,1),(5,6)):
            x=np.where(rng.random((h,w))<0.15,rng.integers(1,10,(h,w)),0)
            expected=x.copy()
            for r,c in zip(*np.nonzero(x)):
                color=x[r,c]; rr,cc=int(r),int(c); vr,vc=dr,dc
                if mode=='connect':
                    for k in range(1,30):
                        er,ec=r+dr*k,c+dc*k
                        if not (0<=er<h and 0<=ec<w): break
                        if x[er,ec]==color:
                            for j in range(1,k):
                                ar,ac=r+dr*j,c+dc*j
                                if x[ar,ac]==0: expected[ar,ac]=max(expected[ar,ac],color)
                else:
                    for _ in range(5):
                        if mode=='bounce':
                            if h==1: vr=0
                            elif not 0<=rr+vr<h: vr=-vr
                            if w==1: vc=0
                            elif not 0<=cc+vc<w: vc=-vc
                        rr+=vr; cc+=vc
                        if not (0<=rr<h and 0<=cc<w): break
                        if mode=='ray' and x[rr,cc]!=0: break
                        if x[rr,cc]==0: expected[rr,cc]=max(expected[rr,cc],color)
            assert predict(g,x.tolist(),backend)==expected.tolist()
