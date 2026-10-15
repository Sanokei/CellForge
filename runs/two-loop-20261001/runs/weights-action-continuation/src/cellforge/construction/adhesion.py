"""Staged declarative bit adhesion. Main retains population and forward runtime.

Only named earlier-node/external bit wires and explicit affine glue execute.
No author code, task callback, label field, inferred binding or answer lookup.
"""
import copy
import numpy as np
from cellforge.construction.typed_population import validate_threshold_net, ThresholdNetArtifact, project_bit_features
from cellforge.construction.joint_weight_population import digest


def _ports(value):
    if not isinstance(value,list) or not value or len(value)>64 or any(not isinstance(x,str) or not x for x in value) or len(set(value))!=len(value):
        raise ValueError('bounded distinct ordered bit ports required')
    return value


def validate_adhesion(spec, catalogue, *, max_nodes=8, max_new_positions=16):
    if not isinstance(spec,dict) or set(spec)!={'format','inputs','nodes','outputs','parents','newly_authored'} or spec['format']!='typed-bit-adhesion/1':
        raise ValueError('exact declarative adhesion schema required')
    external=_ports(spec['inputs']);_ports(spec['parents'])
    nodes=spec['nodes']
    if not isinstance(nodes,list) or not 1<=len(nodes)<=max_nodes:
        raise ValueError('bounded ordered acyclic nodes required')
    seen={};dependencies={};used=set();new=[];positions=0;inherited={};contracts={}
    def ref(binding):
        if not isinstance(binding,dict):raise ValueError('explicit named bit binding required')
        if set(binding)=={'external'}:
            if binding['external'] not in external:raise ValueError('unknown external bit port')
            return None
        if set(binding)=={'node','port'}:
            source=binding['node']
            if source not in seen or binding['port'] not in seen[source]:
                raise ValueError('unknown, forward or cyclic output binding')
            return source
        raise ValueError('constants, labels, callbacks and implicit repairs forbidden')
    for node in nodes:
        if not isinstance(node,dict) or not isinstance(node.get('id'),str) or not node['id'] or node['id'] in seen:
            raise ValueError('distinct explicit node identifiers required')
        ident=node['id'];kind=node.get('kind')
        if kind=='cell':
            if set(node)!={'id','kind','cell','inputs'} or node['cell'] not in catalogue:
                raise ValueError('known literal founder and exact cell-node fields required')
            cell=catalogue[node['cell']];contract=validate_threshold_net(cell)
            inputs=contract['inputs'];outputs=contract['outputs'];used.add(node['cell'])
            inherited[node['cell']]={'cell_hash':contract['cell_hash'],'numeric_positions':contract['numeric_positions']}
            contracts[ident]=contract
        elif kind=='glue':
            if set(node)!={'id','kind','input_ports','output_ports','inputs','params'}:
                raise ValueError('explicit typed glue-node fields required')
            inputs=_ports(node['input_ports']);outputs=_ports(node['output_ports']);p=node['params']
            if not isinstance(p,dict) or set(p)!={'W','b'}:raise ValueError('literal affine glue W,b required')
            try:W=np.asarray(p['W']);b=np.asarray(p['b'])
            except (ValueError,TypeError) as error:raise ValueError('rectangular numerical glue required') from error
            if W.dtype.kind not in 'iuf' or b.dtype.kind not in 'iuf' or W.shape!=(len(outputs),len(inputs)) or b.shape!=(len(outputs),) or not np.isfinite(W).all() or not np.isfinite(b).all():
                raise ValueError('finite glue coefficients must match exact named ports')
            positions+=W.size+b.size;new.extend([ident+'.W',ident+'.b'])
        else:raise ValueError('only literal cell and affine glue nodes supported')
        if not isinstance(node['inputs'],dict) or set(node['inputs'])!=set(inputs):
            raise ValueError('every input has exactly one explicit binding')
        deps=set()
        for p in inputs:
            source=ref(node['inputs'][p])
            if source is not None:deps.add(source)
        dependencies[ident]=deps;seen[ident]=list(outputs)
    if set(spec['parents'])!=used or len(spec['parents'])!=len(used):
        raise ValueError('provenance must name exactly the executed founder definitions')
    if spec['newly_authored']!=new or positions>max_new_positions:
        raise ValueError('complete numeric authorship and bounded added capacity required')
    outputs=spec['outputs']
    if not isinstance(outputs,dict) or not outputs or len(outputs)>64 or any(not isinstance(k,str) or not k for k in outputs):
        raise ValueError('explicit ordered output bindings required')
    reachable=set()
    def visit(n):
        if n is None or n in reachable:return
        reachable.add(n)
        for d in dependencies[n]:visit(d)
    for binding in outputs.values():visit(ref(binding))
    if reachable!=set(seen):raise ValueError('unused donor nodes cannot inflate cooperation provenance')
    return {'format':spec['format'],'typed_io':{'inputs':external,'outputs':list(outputs),'domain':'ordered0/1bits'},
            'inherited':inherited,'newly_authored':new,'new_numeric_positions':int(positions),
            'inherited_unique_positions':sum(x['numeric_positions'] for x in inherited.values()),
            'executed_cell_instances':sum(n['kind']=='cell' for n in nodes),
            'expanded_inherited_positions':sum(contracts[n['id']]['numeric_positions'] for n in nodes if n['kind']=='cell'),
            'candidate_hash':digest({'spec':spec,'founders':{k:catalogue[k] for k in sorted(used)}}),
            'population_child_id':None,'admission':'main-owned; no birth, lineage or selection created'}


class AdhesionCandidate:
    def __init__(self,spec,catalogue,forward):
        if not callable(forward):raise ValueError('explicit trusted numeric cell forward required')
        self.receipt=validate_adhesion(spec,catalogue)
        self.spec=copy.deepcopy(spec);self.catalogue=copy.deepcopy(catalogue);self.forward=forward
    def execute(self,request):
        if not isinstance(request,dict) or set(request)!={'bits'} or not isinstance(request['bits'],list) or len(request['bits'])!=len(self.spec['inputs']) or any(type(x) not in (int,bool) or x not in (0,1) for x in request['bits']):
            raise ValueError('exact bit inputs only; targets and answer fields forbidden')
        external=dict(zip(self.spec['inputs'],request['bits']));packets={};trace=[]
        def resolve(binding):
            if 'external' in binding:return external[binding['external']]
            packet=packets[binding['node']]
            return packet['bits'][packet['output_ports'].index(binding['port'])]
        for node in self.spec['nodes']:
            ports=(self.catalogue[node['cell']]['inputs'] if node['kind']=='cell' else node['input_ports'])
            bits=[resolve(node['inputs'][p]) for p in ports]
            if node['kind']=='cell':
                packet=ThresholdNetArtifact(self.catalogue[node['cell']],self.forward).predict({'bits':bits})
            else:
                packet=project_bit_features(node['params'],{'bits':bits,'output_ports':ports},output_ports=node['output_ports'])
            packets[node['id']]=packet
            trace.append({'node':node['id'],'kind':node['kind'],'input_ports':list(ports),'inputs':bits,'output':packet})
        result={'bits':[resolve(b) for b in self.spec['outputs'].values()],'output_ports':list(self.spec['outputs'])}
        return {'packet':result,'trace':trace,'candidate_hash':self.receipt['candidate_hash'],'cell_calls':sum(n['kind']=='cell' for n in self.spec['nodes']),'glue_calls':sum(n['kind']=='glue' for n in self.spec['nodes'])}
