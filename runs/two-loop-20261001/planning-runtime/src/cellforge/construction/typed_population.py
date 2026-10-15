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
from cellforge.runtime.binding import BindingPredictor
from cellforge.runtime.arithmetic import DecimalAdditionPredictor
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
        elif graph.meta.get('arithmetic_interface') == 'decimal-addition-graph/1':
            self.family = 'decimal-addition'
            self.predictor = DecimalAdditionPredictor(self.graph)
            self.keys = {'a', 'b'}
        elif graph.meta.get('binding_interface') == 'binding-graph/1':
            self.family = 'typed-binding'
            self.predictor = BindingPredictor(self.graph)
            self.keys = {'bindings', 'query'}
        else:
            raise GraphError('no verified typed artifact adapter')

    def predict(self, request):
        if not isinstance(request, dict) or set(request) != self.keys:
            raise GraphError('typed input only; targets and foreign modality inputs forbidden')
        if self.family == 'ocr-raster':
            return self.predictor(request['image'])
        if self.family == 'decimal-addition':
            return self.predictor(request['a'], request['b'])
        if self.family == 'typed-binding':
            return self.predictor(request['bindings'], request['query'])
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
    k = len(parent.get('paint', []))
    if not 1 <= k <= 10:
        raise ValueError('declared palette must contain 1..10 colors')
    layouts = (
        (parent, {'self.W':(1,k), 'cnt.W':(1,k), 'cnt.b':(1,), 'paint':(k,)}),
        (donor, dict({'conv.W':(9,k), 'conv.b':(1,)}, **({'paint':(k,)} if 'paint' in donor else {}))),
        (authored, {'self.W':(1,k), 'cnt.W':(1,k), 'cnt.b':(1,), 'paint':(k,),
                   'residual.W':(8,k), 'residual.center':(1,k)}))
    for tensors, shapes in layouts:
        if set(tensors) != set(shapes):
            raise ValueError('complete exact typed repair tensors required')
        for name, shape in shapes.items():
            a = np.asarray(tensors[name])
            if a.dtype.kind not in 'iuf' or a.shape != shape or not np.isfinite(a).all():
                raise ValueError('finite numeric repair tensors of declared shapes required')
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
            'scalar_accounting':{'newly_authored':9*k,'inherited_or_copied':3*k+1,'total':12*k+1},
            'donor_information':'all eight directional rows and center reconstructed exactly',
            'limitation':'recipient paint retained; exact preactivation does not imply donor output equivalence'}


def validate_threshold_net(cell, *, max_layers=8, max_width=64, max_parameters=4096):
    """Validate actual threshold-net-v1 tensors and ordered bit ports.

    This validates a typed boundary; the main worker supplies numerical forward.
    Tool matcher/router formats have separate contracts and cannot enter here.
    """
    if not isinstance(cell, dict) or set(cell) != {'inputs','outputs','layers'}:
        raise ValueError('threshold cell requires exactly inputs, outputs and layers')
    for key in ('inputs','outputs'):
        ports=cell[key]
        if (not isinstance(ports,list) or not 1 <= len(ports) <= max_width
                or any(not isinstance(p,str) or not p or len(p)>64 for p in ports)
                or len(set(ports))!=len(ports)):
            raise ValueError('nonempty distinct ordered bit ports required')
    layers=cell['layers']
    if not isinstance(layers,list) or not 1 <= len(layers) <= max_layers:
        raise ValueError('bounded explicit threshold layers required')
    width=len(cell['inputs']);positions=0
    for layer in layers:
        if not isinstance(layer,dict) or set(layer)!={'W','b'}:
            raise ValueError('literal W and b required for each threshold layer')
        try:
            W=np.asarray(layer['W']);b=np.asarray(layer['b'])
        except (ValueError,TypeError) as error:
            raise ValueError('rectangular numeric layer required') from error
        if (W.dtype.kind not in 'iuf' or b.dtype.kind not in 'iuf'
                or W.ndim!=2 or not 1 <= W.shape[0] <= max_width
                or W.shape[1]!=width or b.shape!=(W.shape[0],)
                or not np.isfinite(W).all() or not np.isfinite(b).all()):
            raise ValueError('finite aligned literal threshold tensors required')
        positions+=W.size+b.size;width=W.shape[0]
    if width!=len(cell['outputs']) or positions>max_parameters:
        raise ValueError('output ports or threshold capacity exceed declared contract')
    return {'format':'threshold-net-v1','inputs':list(cell['inputs']),
            'outputs':list(cell['outputs']),'input_width':len(cell['inputs']),
            'output_width':width,'numeric_positions':int(positions),
            'activation':'step: strictly greater than zero at every layer',
            'bit_order':'declared port order; integer bitfields little-endian',
            'cell_hash':digest(cell)}


class ThresholdNetArtifact:
    """Typed bridge around the main worker's numerical threshold runner."""
    def __init__(self, cell, forward):
        if not callable(forward):
            raise ValueError('explicit numerical forward runner required')
        self.contract=validate_threshold_net(cell)
        self.cell=copy.deepcopy(cell)
        self.forward=forward

    def predict(self, request):
        if not isinstance(request,dict) or set(request)!={'bits'}:
            raise ValueError('bits only; targets and foreign fields forbidden')
        bits=request['bits']
        if (not isinstance(bits,list) or len(bits)!=self.contract['input_width']
                or any(type(x) not in (int,bool) or x not in (0,1) for x in bits)):
            raise ValueError('exact declared width of input bits required')
        out=np.asarray(self.forward(copy.deepcopy(self.cell),list(bits)))
        if (out.dtype.kind not in 'iufb' or out.shape!=(self.contract['output_width'],)
                or not np.isfinite(out).all() or not np.isin(out,[0,1]).all()):
            raise ValueError('numerical runner violated output bit ports')
        return {'bits':out.astype(int).tolist(), 'output_ports':list(self.contract['outputs'])}


def project_bit_features(params, packet, *, output_ports):
    """Execute explicit authored affine bit glue; no implicit padding or fitting."""
    if (not isinstance(packet,dict) or set(packet)!={'bits','output_ports'}
            or not isinstance(packet['bits'],list) or not isinstance(packet['output_ports'],list)
            or len(packet['bits'])!=len(packet['output_ports']) or not packet['bits']
            or len(set(packet['output_ports']))!=len(packet['output_ports'])
            or any(not isinstance(p,str) or not p for p in packet['output_ports'])
            or any(type(x) not in (int,bool) or x not in (0,1) for x in packet['bits'])):
        raise ValueError('complete ordered bit packet required')
    if (not isinstance(output_ports,list) or not output_ports
            or len(set(output_ports))!=len(output_ports)
            or any(not isinstance(p,str) or not p for p in output_ports)):
        raise ValueError('explicit distinct destination ports required')
    if not isinstance(params,dict) or set(params)!={'W','b'}:
        raise ValueError('explicit literal affine bit glue required')
    W=np.asarray(params['W']);b=np.asarray(params['b'])
    if (W.dtype.kind not in 'iuf' or b.dtype.kind not in 'iuf'
            or W.shape!=(len(output_ports),len(packet['bits'])) or b.shape!=(len(output_ports),)
            or not np.isfinite(W).all() or not np.isfinite(b).all()):
        raise ValueError('finite glue tensors must align both ordered port sets')
    projected=W@np.asarray(packet['bits'])+b
    if not np.isin(projected,[0,1]).all():
        raise ValueError('authored affine glue did not produce valid bits')
    return {'bits':projected.astype(int).tolist(),'output_ports':list(output_ports)}
