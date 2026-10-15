"""Typed existing-artifact boundaries and palette-aware numerical execution.

No implicit tensor padding, color clipping or cross-domain shape averaging.
"""
import copy
import time
import tracemalloc
import numpy as np
from cellforge.ir.graph import GraphError, array_hash
from cellforge.runtime.vision import OCRPredictor
from cellforge.runtime.deduction import HornDeductionPredictor
from cellforge.construction.joint_weight_population import digest, load_core


def measure_resources(call):
    """Local synchronous wall time and traced allocations, not process RSS."""
    if tracemalloc.is_tracing():
        raise ValueError('resource probe requires exclusive tracing ownership')
    tracemalloc.start()
    start=time.perf_counter()
    try:
        result=call()
        elapsed=time.perf_counter()-start
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result,{'wall_seconds':elapsed,'traced_peak_bytes':peak,
                   'scope':'local synchronous execution; traced allocation peak, not RSS'}


def calibrate_caps(samples):
    if len(samples)<3 or any(s['wall_seconds']<=0 or s['traced_peak_bytes']<0 for s in samples):
        raise ValueError('three valid baseline repetitions required')
    return {'wall_seconds':max(.25,4*max(s['wall_seconds'] for s in samples)),
            'traced_peak_bytes':max(16*1024*1024,4*max(s['traced_peak_bytes'] for s in samples)),
            'baseline':samples,'scope':'development-local 4x observed maximum with explicit floors; not ARC benchmark limits'}


def resource_pass(observed,caps):
    return all(observed[k]<=caps[k] for k in ('wall_seconds','traced_peak_bytes'))


def parity_row_stencil(tensors,grid):
    """Execute the maker's typed row0/parity stencil, without task IDs."""
    a=np.asarray(grid)
    shapes={'stencil.W':(2,3),'foreground.W':(1,10),'activation.b':(1,)}
    if a.ndim!=2 or not a.size or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>9):
        raise ValueError('ten-color rectangular grid required')
    if set(tensors)!=set(shapes) or any(np.asarray(tensors[n]).shape!=s
              or not np.isfinite(tensors[n]).all() for n,s in shapes.items()):
        raise ValueError('complete literal parity-stencil tensors required')
    if not np.any(a[0]) or np.any(a[1:]):return a.tolist()
    out=np.zeros_like(a)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            scores=[]
            for k in range(1,10):
                score=tensors['activation.b'][0]
                for i,offset in enumerate((-1,0,1)):
                    if 0<=c+offset<a.shape[1] and a[0,c+offset]==k:
                        score+=tensors['foreground.W'][0][k]*tensors['stencil.W'][r%2][i]
                scores.append(score)
            if max(scores)>0:out[r,c]=int(np.argmax(scores))+1
    return out.tolist()


class TypedArtifact:
    def __init__(self, graph):
        graph.validate()
        self.graph = graph.copy()
        if graph.meta.get('deduction_interface') == 'horn-closure-graph/1':
            self.family = 'positive-horn'
            self.predictor = HornDeductionPredictor(self.graph)
            self.keys = {'facts', 'rules', 'query'}
        elif 'pixel_contract' in graph.meta:
            self.family = 'ocr-raster'
            self.predictor = OCRPredictor(self.graph)
            self.keys = {'image'}
        else:
            raise GraphError('no verified typed artifact adapter')

    def predict(self, request):
        if set(request) != self.keys:
            raise GraphError('typed input only; targets and foreign modality inputs forbidden')
        if self.family == 'ocr-raster':
            return self.predictor(request['image'])
        return self.predictor(request['facts'], request['rules'], request['query'])

    def transfer_plan(self, donor):
        # Existing OCR glyph scores and Horn presence thresholds have no shared
        # feature/output semantics, regardless of numerical tensor shape.
        if self.family != donor.family:
            return {'accepted': False, 'recipient': self.graph.hash(),
                    'donor': donor.graph.hash(), 'reason': 'no aligned feature/output contract',
                    'repair_required': 'authored typed bridge with numeric parameters and independent semantic fixtures',
                    'copied_weights': []}
        return {'accepted': False, 'recipient': self.graph.hash(), 'donor': donor.graph.hash(),
                'reason': 'same domain alone does not establish cell feature alignment',
                'copied_weights': []}


class PaletteExecutor:
    def __init__(self, core_path, colors):
        if type(colors) is not int or not 1 <= colors <= 10:
            raise ValueError('explicit palette size 1..10 required')
        self.core = load_core(core_path)
        self.predictors = dict(self.core.PREDICTORS)
        self.core.K = colors
        self.colors = colors
        k = colors
        self.core.SHAPES = {
            'conv3x3_detect': {'conv.W': (9,k), 'conv.b': (1,), 'paint': (k,)},
            'colormap': {'map.W': (k,k)},
            'color_gate': {'gate': (k,), 'fill': (k,)},
            'flip_select': {'sel': (6,), 'post.W': (k,k)},
            'tile_select': {'per': (3,), 'pec': (3,)},
            'neighbor_count': {'self.W': (1,k), 'cnt.W': (1,k), 'cnt.b': (1,), 'paint': (k,)},
            'bbox_mask': {'pick': (k,), 'fill': (k,)},
        }

    def predict(self, template, tensors, grid):
        a = np.asarray(grid)
        if (a.ndim != 2 or not a.size or a.dtype.kind not in 'iu'
                or np.any(a < 0) or np.any(a >= self.colors)):
            raise ValueError('grid colors outside declared numerical palette or invalid grid')
        shapes = self.core.SHAPES[template]
        if set(tensors) != set(shapes) or any(np.asarray(tensors[n]).shape != s
                or not np.isfinite(tensors[n]).all() for n,s in shapes.items()):
            raise ValueError('complete finite literal tensors for declared palette required')
        pred = self.predictors[template](tensors,grid,False)[0]
        return pred


def repaired_count(tensors, grid):
    """Authored residuals extend count features with all eight directions."""
    a = np.asarray(grid)
    k = np.asarray(tensors['paint']).size
    shapes = {'self.W':(1,k), 'cnt.W':(1,k), 'cnt.b':(1,),
              'paint':(k,), 'residual.W':(8,k), 'residual.center':(1,k)}
    if set(tensors) != set(shapes) or any(np.asarray(tensors[n]).shape != s
            or not np.isfinite(tensors[n]).all() for n,s in shapes.items()):
        raise ValueError('finite complete repaired-count tensors required')
    if a.ndim != 2 or not a.size or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=k):
        raise ValueError('invalid repaired-count palette input')
    h,w=a.shape
    pad=np.full((h+2,w+2),-1);pad[1:-1,1:-1]=a
    f=np.asarray(tensors['self.W'])[0,a]+np.asarray(tensors['residual.center'])[0,a]+tensors['cnt.b'][0]
    offs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i,(dr,dc) in enumerate(offs):
        v=pad[1+dr:1+dr+h,1+dc:1+dc+w]
        weights=np.asarray(tensors['cnt.W'])[0]+np.asarray(tensors['residual.W'])[i]
        f=f+np.where(v>=0,weights[np.maximum(v,0)],0.)
    out=np.where(f>0,int(np.argmax(tensors['paint'])),a)
    return out.tolist(),f


def validate_repair(parent, donor, authored):
    for name in ('self.W','cnt.W','paint'):
        if authored[name] != parent[name]:
            raise ValueError('undeclared recipient material changed')
    if authored['cnt.b'] != donor['conv.b']:
        raise ValueError('inherited donor bias mismatch')
    offsets=[0,1,2,3,5,6,7,8]
    combined=np.asarray(authored['cnt.W'])+np.asarray(authored['residual.W'])
    center=np.asarray(authored['self.W'])+np.asarray(authored['residual.center'])
    if not np.array_equal(combined,np.asarray(donor['conv.W'])[offsets]) or not np.array_equal(center[0],donor['conv.W'][4]):
        raise ValueError('authored bridge fails exact directional feature reconstruction')
    return {'recipient_hash':digest(parent),'donor_hash':digest(donor),'child_hash':digest(authored),
            'inherited':{'recipient':['self.W','cnt.W','paint'],'donor':{'conv.b':'cnt.b'}},
            'newly_authored':['residual.W','residual.center'],
            'donor_information':'all eight directional rows and center reconstructed exactly',
            'limitation':'recipient paint retained; exact preactivation does not imply donor output equivalence'}
