"""Immutable graph execution plans, differential against reference Interpreter.

Prebind references/ops/parameters and memoize input-independent primitive/cell
nodes only. No expert pruning, neural approximation, dtype change, or answer
logic. Counters retain reference *semantic* operations including cached work;
they do not claim physical compute executed after constant memoization.
"""
from __future__ import annotations
import math
import numpy as np
from ..ir.graph import GraphError, parse_ref, array_hash, canonical_json, sha256_bytes
from .interp import Interpreter
from .ops import ACTIVATIONS, OPS


class CompiledInterpreter(Interpreter):
    """Reference-compatible methods over an owned, frozen graph snapshot.

    Compilation cost is paid once. Returned cached constants are read-only;
    callers must not mutate graph outputs. Dispatch stays reference lazy top-k.
    """

    backend_version = 'cellforge-execution-plan/1'

    def __init__(self, graph, trace=False, trace_filter=None, *, reuse_from=None):
        try:
            graph.validate()
        except GraphError as error:
            if 'contains non-finite values' not in str(error):
                raise
            # The reference scorer intentionally observes nonfinite inference
            # and charges its failure penalty. Validate structure on a separate
            # temporary copy while retaining every original execution value.
            validation=graph.copy()
            for key,value in validation.params.items():
                if not np.isfinite(value).all():
                    validation.params[key]=np.where(np.isfinite(value),value,0.0)
            validation.validate()
        super().__init__(graph.copy(), trace, trace_filter)
        self.graph_hash = self.g.hash()
        for value in self.g.params.values():
            value.flags.writeable = False
        self._cell_plans = {}
        self._functions = {}
        self._nodes = {}
        self._constants = {}
        self.generated_sources = {}
        self._demand_keys = {}
        self._definition_signatures = {}
        self.reused_bundle_plans = 0
        self.compiled_bundle_plans = 0
        parameter_hashes = {k: array_hash(v) for k,v in self.g.params.items()}

        def signature(name):
            if name not in self._definition_signatures:
                payload = {'definition':self.g.defs[name],
                           'params':{p:parameter_hashes[p] for p in self.g.params_of(name)},
                           'children':{child:signature(child) for child in self.g.children(name)}}
                self._definition_signatures[name] = sha256_bytes(canonical_json(payload))
            return self._definition_signatures[name]

        for name in self.g.defs: signature(name)
        for name, definition in self.g.defs.items():
            if definition['kind']=='cell':
                p = {key:self.g.params[value] for key,value in definition['params'].items()}
                self._cell_plans[name] = (p,ACTIVATIONS[definition.get('act','identity')])
        for name, definition in self.g.defs.items():
            if definition['kind']=='bundle':
                if (reuse_from is not None and
                    self._definition_signatures[name] == reuse_from._definition_signatures.get(name)):
                    # Generated functions access self, so nested calls still
                    # resolve against the new candidate interpreter. Constants
                    # are shared only when the whole dependency closure matches.
                    self._functions[name] = reuse_from._functions[name]
                    self._nodes[name] = reuse_from._nodes[name]
                    self._constants[name] = reuse_from._constants[name]
                    self.generated_sources[name] = reuse_from.generated_sources[name]
                    self.reused_bundle_plans += 1
                else:
                    self._compile_bundle(name,definition)
                    self.compiled_bundle_plans += 1
        self.stats.node_execs = 0
        self.stats.macs = 0
        self.stats.peak_elements = 0
        self.stats.route_counts.clear()

    def run_cell(self, name, x):
        p, activation = self._cell_plans[name]
        W1,b1 = p['W1'],p['b1']
        if x.shape[-1] != W1.shape[1]:
            raise GraphError(f'cell {name}: input width {x.shape[-1]} != {W1.shape[1]}')
        lead = math.prod(x.shape[:-1])
        h = activation(x @ W1.T + b1)
        self.stats.macs += lead*W1.size
        if 'W2' in p:
            h = h @ p['W2'].T+p['b2']
            self.stats.macs += lead*p['W2'].size
        return h

    def _ports(self,node):
        use = node['use']
        if use.startswith(('op:','param:')):
            return ['out']
        if use=='dispatch':
            return list(self.g.defs[node['args']['branches'][0]]['outputs'])
        sub = self.g.defs[use]
        return ['y'] if sub['kind']=='cell' else list(sub['outputs'])

    def _compile_bundle(self,name,definition):
        nodes = definition['nodes']
        indexes = {node['id']:i for i,node in enumerate(nodes)}
        constants = {}
        node_plans = []

        def ref_expression(ref):
            kind,target,port = parse_ref(ref)
            if kind=='input':
                return f'inputs[{target!r}]'
            index = indexes[target]
            if port is None:
                ports = self._ports(nodes[index])
                if len(ports)!=1:
                    # Preserve reference error timing rather than reject compilation.
                    return f'self._default_output(v{index}, {name!r}, {target!r})'
                port = ports[0]
            if port not in self._ports(nodes[index]):
                return f'self._named_output(v{index}, {port!r}, {name!r}, {target!r})'
            return f'v{index}[{port!r}]'

        lines = ['def execute(self,inputs,path):']
        for key in definition.get('inputs',[]):
            lines += [f'    if {key!r} not in inputs:',f'        raise GraphError({("bundle "+name+": missing input "+key)!r})']
        lines += ['    stats = self.stats']
        for index,node in enumerate(nodes):
            use = node['use']
            args = node.get('args',{})
            fn = OPS[use[3:]][0] if use.startswith('op:') else None
            node_plans.append((fn,args,node))
            static_inputs = {}
            is_static = True
            for key,ref in node.get('in',{}).items():
                kind,target,port = parse_ref(ref)
                if kind=='input' or indexes[target] not in constants:
                    is_static = False
                    break
                previous = constants[indexes[target]][0]
                if port is None:
                    if len(previous)!=1:
                        is_static=False
                        break
                    static_inputs[key] = next(iter(previous.values()))
                else:
                    if port not in previous:
                        is_static=False
                        break
                    static_inputs[key] = previous[port]
            cached = None
            if is_static:
                try:
                    if use.startswith('param:'):
                        cached = ({'out':self.g.params[use[6:]][None,...]},0)
                    elif fn is not None:
                        value = fn(static_inputs,args)
                        macs = int(np.prod(value.shape))*static_inputs['a'].shape[-1] if use=='op:matmul' else 0
                        cached = ({'out':value},macs)
                    elif use!='dispatch' and self.g.defs[use]['kind']=='cell':
                        before = self.stats.macs
                        value = self.run_cell(use,static_inputs['x'])
                        cached = ({'y':value},self.stats.macs-before)
                except (ValueError,IndexError,KeyError,TypeError):
                    cached = None  # invalid constant graph fails at execution, as reference
            if cached is not None:
                for value in cached[0].values():
                    value.flags.writeable = False
                constants[index] = cached
            if cached is not None:
                lines += ['    stats.node_execs += 1']
                lines += [f'    v{index} = self._constants[{name!r}][{index}][0]']
                if cached[1]:
                    lines += [f'    stats.macs += {cached[1]}']
            else:
                inputs_expr = ','.join(f'{port!r}:{ref_expression(ref)}' for port,ref in node.get('in',{}).items())
                lines += [f'    ins = {{{inputs_expr}}}']
                lines += ['    stats.node_execs += 1']
                if fn is not None:
                    lines += ['    try:',f'        value = self._nodes[{name!r}][{index}][0](ins,self._nodes[{name!r}][{index}][1])',
                              '    except (ValueError,IndexError,KeyError,TypeError) as error:',
                              f'        raise GraphError({(name+"/"+node["id"]+" ("+use+"): ")!r}+str(error)) from error',
                              f'    v{index} = {{"out":value}}']
                    if use=='op:matmul':
                        lines += ['    stats.macs += math.prod(value.shape)*ins["a"].shape[-1]']
                elif use.startswith('param:'):
                    lines += [f'    v{index} = {{"out":self.g.params[{use[6:]!r}][None,...]}}']
                elif use=='dispatch':
                    lines += [f'    v{index} = self.run_dispatch({name!r}, self._nodes[{name!r}][{index}][2], ins, path+{("/"+node["id"])!r})']
                elif self.g.defs[use]['kind']=='cell':
                    lines += ['    if "x" not in ins:',f'        raise GraphError({(name+"/"+node["id"]+": cell instance needs input x")!r})',
                              f'    v{index} = {{"y":self.run_cell({use!r},ins["x"])}}']
                else:
                    lines += [f'    v{index} = self.run_bundle({use!r},ins,path+{("/"+node["id"])!r})']
            lines += [f'    for value in v{index}.values():',
                      '        if value.size > stats.peak_elements: stats.peak_elements = value.size',
                      '    if self.trace_enabled:',f'        npath = path+{("/"+node["id"])!r}',
                      '        if self.trace_filter is None or self.trace_filter(npath):',
                      f'            for key,value in v{index}.items(): self.trace[npath+"."+key] = value']
        output_expr = ','.join(f'{key!r}:{ref_expression(ref)}' for key,ref in definition['outputs'].items())
        lines += [f'    return {{{output_expr}}}']
        source = '\n'.join(lines)+'\n'
        namespace = {'GraphError':GraphError,'math':math}
        exec(compile(source,f'<cellforge-plan:{name}>','exec'),namespace)
        self._functions[name] = namespace['execute']
        self._nodes[name] = node_plans
        self._constants[name] = constants
        self.generated_sources[name] = source

    @staticmethod
    def _default_output(outs,name,target):
        if len(outs)!=1:
            raise GraphError(f'bundle {name}: {target} has several outputs; name a port')
        return next(iter(outs.values()))

    @staticmethod
    def _named_output(outs,port,name,target):
        if port not in outs:
            raise GraphError(f'bundle {name}: {target} has no output {port}')
        return outs[port]

    def run_bundle(self,name,inputs,path=''):
        return self._functions[name](self,inputs,path)

    def assert_frozen(self):
        """Check once at an inference boundary, not once per recurrent byte."""
        if self.g.hash()!=self.graph_hash:
            raise GraphError('compiled graph snapshot changed after plan creation')

    def cached_node_plan(self, name, cached_nodes):
        """Compile a root plan with caller-certified node values as inputs.

        This is an execution mechanism, not a cache validity decision. The
        evolution scorer verifies immutable dependency fingerprints, exact
        recurrent input trajectories and batch independence before calling it.
        Nested generated calls still use this interpreter's candidate snapshot.
        """
        definition = self.g.defs[name]
        cached_nodes = set(cached_nodes)
        nodes = {node['id']: node for node in definition['nodes']}
        if not cached_nodes.issubset(nodes):
            raise GraphError('unknown cached root node')
        bindings = {}
        for node_id in sorted(cached_nodes):
            for port in self._ports(nodes[node_id]):
                bindings[(node_id, port)] = f'cache_{len(bindings)}'

        def rewrite(ref):
            kind, target, port = parse_ref(ref)
            if kind == 'node' and target in cached_nodes:
                if port is None:
                    ports = self._ports(nodes[target])
                    if len(ports) != 1:
                        raise GraphError('cached multi-output reference requires port')
                    port = ports[0]
                return '$' + bindings[(target, port)]
            return ref

        reduced = {**definition,
                   'inputs': list(definition.get('inputs', [])) + list(bindings.values()),
                   'nodes': [{**node, 'in': {k: rewrite(v) for k, v in node.get('in', {}).items()}}
                             for node in definition['nodes'] if node['id'] not in cached_nodes],
                   'outputs': {k: rewrite(v) for k, v in definition['outputs'].items()}}
        key = f'@cached:{name}'
        self._compile_bundle(key, reduced)
        return self._functions[key], bindings

    def guard_batched_matmul(self):
        """Reject dynamic rank-two operands that may mix example rows.

        Registered per-example axis ops preserve batches, but NumPy matmul
        treats rank-two operands as matrices, potentially confusing a leading
        example axis with matrix rows. Constants (leading one) remain safe.
        Evolution caching falls back to sequential execution on this rejection.
        """
        def checked(inputs, args):
            for value in inputs.values():
                if value.ndim < 3 and value.shape[0] > 1:
                    raise GraphError('uncertified batched matrix axis')
            return OPS['matmul'][0](inputs, args)
        # Copy plans before replacing bindings: unchanged plans may be shared.
        for name, plans in list(self._nodes.items()):
            self._nodes[name] = [(checked if node['use']=='op:matmul' else fn,args,node)
                                 for fn,args,node in plans]

    def run_bundle_outputs(self,name,inputs,outputs,path=''):
        """Execute the pure dependency closure of explicitly demanded outputs.

        Only use when omitted outputs are discarded by the inference protocol
        (e.g. logits during prompt prefill). State dependencies are never skipped.
        Stats/traces cover executed closure, not discarded full-eager outputs.
        Nested bundles still execute eagerly, including reference lazy dispatch.
        Current registered ops are pure; adding effectful ops requires disabling
        this mode or extending its effect analysis before use.
        """
        outputs=tuple(outputs)
        definition=self.g.defs[name]
        if not outputs or len(set(outputs))!=len(outputs) or not set(outputs).issubset(definition['outputs']):
            raise GraphError('output demand must name existing distinct outputs')
        key=(name,outputs)
        if key not in self._demand_keys:
            nodes={node['id']:node for node in definition['nodes']}
            required=set()

            def need(ref):
                kind,target,_=parse_ref(ref)
                if kind=='input' or target in required:
                    return
                required.add(target)
                for dependency in nodes[target].get('in',{}).values():
                    need(dependency)

            for output in outputs:
                need(definition['outputs'][output])
            reduced={**definition,'nodes':[node for node in definition['nodes'] if node['id'] in required],
                     'outputs':{output:definition['outputs'][output] for output in outputs}}
            internal_key=f'@demand:{len(self._demand_keys)}:{name}'
            before=(self.stats.node_execs,self.stats.macs,self.stats.peak_elements)
            self._compile_bundle(internal_key,reduced)
            self.stats.node_execs,self.stats.macs,self.stats.peak_elements=before
            self._demand_keys[key]=internal_key
        return self._functions[self._demand_keys[key]](self,inputs,path)
