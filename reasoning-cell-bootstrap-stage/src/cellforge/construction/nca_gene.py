"""Authored local NCA gene: 48 shared microcells, no host neural model."""
import numpy as np
from ..ir.graph import Graph, Limits


class Nodes:
    def __init__(self): self.nodes=[]
    def add(self,use,inputs=None,**args):
        name=f'n{len(self.nodes)}'; node={'id':name,'use':use,'in':inputs or {}}
        if args: node['args']=args
        self.nodes.append(node); return name
    def bundle(self,inputs,outputs,role='cluster'):
        return {'kind':'bundle','role':role,'inputs':inputs,'nodes':self.nodes,'outputs':outputs}


def build_nca_gene(iterations=2, seed='identity'):
    if iterations not in (2,4,8) or seed not in ('identity','neighbor_majority'):
        raise ValueError('fixed T2/4/8 and authored seed required')
    graph=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':1,'state_dim':0,
                     'construction':'direct authored local NCA; no fitting','nca_recipe':{'iterations':iterations,'seed':seed,'channels':12,'features':108,'hidden':12},
                     'limitations':'fixed same-size gene; demonstration adaptation not implemented; no ARC score'})
    first=np.zeros((12,108)); second=np.eye(12)
    for channel in range(12):
        if seed=='identity' or channel>=10:
            first[channel,48+channel]=1.
        else:
            for neighbor in range(9): first[channel,neighbor*12+channel]=1/9
    # Background may not overwrite colored neighbors solely due to padding.
    if seed=='neighbor_majority': first[0] *= .25
    for row in range(12):
        for chunk in range(3):
            name=f'nca.h{row}.c{chunk}'
            graph.params[name+'.W']=first[row:row+1,chunk*36:(chunk+1)*36].copy()
            graph.params[name+'.b']=np.zeros(1)
            graph.defs[name]={'kind':'cell','params':{'W1':name+'.W','b1':name+'.b'},'act':'identity'}
        name=f'nca.o{row}'; graph.params[name+'.W']=second[row:row+1].copy(); graph.params[name+'.b']=np.zeros(1)
        graph.defs[name]={'kind':'cell','params':{'W1':name+'.W','b1':name+'.b'},'act':'identity'}
    nodes=Nodes(); patches=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            y=nodes.add('op:shift',{'x':'$x'},axis=0,k=dy)
            patches.append(nodes.add('op:shift',{'x':y},axis=1,k=dx))
    features=nodes.add('op:concat',{f'x{i}':v for i,v in enumerate(patches)},axis=2)
    slices=[nodes.add('op:slice',{'x':features},axis=2,start=i*36,stop=(i+1)*36) for i in range(3)]
    hidden=[]
    for row in range(12):
        values=[nodes.add(f'nca.h{row}.c{i}',{'x':slices[i]}) for i in range(3)]
        summed=nodes.add('op:add',{'a':values[0],'b':values[1]})
        summed=nodes.add('op:add',{'a':summed,'b':values[2]})
        hidden.append(nodes.add('op:relu',{'x':summed}))
    h=nodes.add('op:concat',{f'x{i}':v for i,v in enumerate(hidden)},axis=2)
    outputs=[nodes.add(f'nca.o{i}',{'x':h}) for i in range(12)]
    result=nodes.add('op:concat',{f'x{i}':v for i,v in enumerate(outputs)},axis=2)
    result=nodes.add('op:mul',{'a':result,'b':'$mask'})
    graph.defs['nca_step']=nodes.bundle(['x','mask'],{'x':result})
    nodes=Nodes(); current=nodes.add('op:pad',{'x':'$input'},axis=2,after=2)
    for _ in range(iterations): current=nodes.add('nca_step',{'x':current,'mask':'$input_mask'})
    colors=nodes.add('op:slice',{'x':current},axis=2,start=0,stop=10)
    graph.defs['nca_gene']=nodes.bundle(['input','input_mask','input_height','input_width'],{'colors':colors,'height':'$input_height','width':'$input_width'})
    nodes=Nodes(); gene=nodes.add('nca_gene',{name:'$'+name for name in ('input','input_mask','input_height','input_width')})
    colors=nodes.add('op:reshape',{'x':gene+'.colors'},shape=[1,30,30,10])
    height=nodes.add('op:reshape',{'x':gene+'.height'},shape=[1,30]); width=nodes.add('op:reshape',{'x':gene+'.width'},shape=[1,30])
    ports=['input','input_mask','input_height','input_width','output','output_mask','output_height','output_width','phase','index','state']
    graph.defs['grid_root']=nodes.bundle(ports,{'color_logits':colors,'height_logits':height,'width_logits':width,'state':'$state'},role='root')
    graph.entries['grid_step']='grid_root'; graph.entries['gene']='nca_gene'
    graph.validate(Limits(strict_microcells=True))
    return graph


def mutate_nca_gene(parent, seed):
    """Fixed active-slot genome: row blocks, cell choice or typed wiring swaps."""
    rng=np.random.default_rng(seed); child=parent.copy()
    slots=[node for node in child.defs['nca_step']['nodes'] if node['use'] in child.defs and child.defs[node['use']]['kind']=='cell']
    active={node['use'] for node in slots}
    names=sorted(child.defs[name]['params']['W1'] for name in active)
    operator=int(rng.integers(3)); details={'operator':('row_block','cell_activation','typed_wire_swap')[operator]}
    if operator==1:
        name=sorted(active)[int(rng.integers(len(active)))]; old=child.defs[name].get('act','identity')
        new=str(rng.choice([act for act in ('identity','relu','tanh') if act!=old])); child.defs[name]['act']=new
        details.update({'cell':name,'old':old,'new':new})
    elif operator==2:
        indices=rng.choice(36,2,replace=False); a,b=(slots[int(i)] for i in indices)
        a['use'],b['use']=b['use'],a['use']
        details.update({'slots':list(map(int,indices)),'width_contract':36})
    else:
        name=names[int(rng.integers(len(names)))]; array=child.params[name]
        start=int(rng.integers(array.size)); size=min(3,array.size-start)
        delta=float(rng.choice([-.25,-.0625,.0625,.25]))
        old=array.reshape(-1)[start:start+size].copy(); array.reshape(-1)[start:start+size]+=delta
        details.update({'parameter':name,'start':start,'size':size,'old':old.tolist(),'delta':delta})
    child.meta['nca_mutation']={'parent_hash':parent.hash(),'seed':seed,'details':details,'gradient_estimator':False}
    child.validate(Limits(strict_microcells=True))
    return child


def nca_genome_contract(graph):
    slots=[node for node in graph.defs['nca_step']['nodes'] if node['use'] in graph.defs and graph.defs[node['use']]['kind']=='cell']
    widths=[graph.params[graph.defs[node['use']]['params']['W1']].shape[1] for node in slots]
    if widths!=[36]*36+[12]*12: raise ValueError('fixed NCA slot width contract violated')
    return {'protocol':'nca-fixed-slots/1','active_cells':48,'hidden_chunk_widths':widths[:36],'output_widths':widths[36:],
            'outputs_per_cell':1,'cell_parameter_cap':40,'stored_inactive_ancestry_counts_separately':True}


def crossover_nca_gene(recipient, donor, row, prefix):
    """Typed shared-step module transfer, graph-native import_def ancestry."""
    from .transactions import import_def
    if not 0<=row<12 or prefix in recipient.defs: raise ValueError('row/prefix invalid')
    child=recipient.copy(); targets=[f'nca.h{row}.c{i}' for i in range(3)]+[f'nca.o{row}']
    for target in targets:
        if any(n.startswith(prefix+target) for n in child.defs): raise ValueError('fresh namespace required')
        # Corresponding stable node slots retain meaning across earlier imports.
        donor_uses=[n['use'] for n in donor.defs['nca_step']['nodes'] if n['use'] in donor.defs and donor.defs[n['use']]['kind']=='cell']
        recipient_nodes=[n for n in child.defs['nca_step']['nodes'] if n['use'] in child.defs and child.defs[n['use']]['kind']=='cell']
        slot=row*3+int(target.rsplit('c',1)[1]) if '.h' in target else 36+row
        source=donor_uses[slot]
        cell=donor.defs[source]
        recipient_cell=child.defs[recipient_nodes[slot]['use']]
        if any(donor.params[cell['params'][key]].shape!=child.params[recipient_cell['params'][key]].shape for key in ('W1','b1')):
            raise ValueError('typed slot shape mismatch')
        # Canonical temporary cell preserves exact donor values while bounding
        # namespace length across many generations; import_def performs transfer.
        canonical=Graph(); stable=f'slot_{slot}'
        canonical.params[stable+'.W']=donor.params[cell['params']['W1']].copy()
        canonical.params[stable+'.b']=donor.params[cell['params']['b1']].copy()
        canonical.defs[stable]={'kind':'cell','act':cell.get('act','identity'),'params':{'W1':stable+'.W','b1':stable+'.b'}}
        import_def(child,canonical,stable,prefix)
        recipient_nodes[slot]['use']=prefix+stable
    child.meta['nca_crossover']={'recipient_hash':recipient.hash(),'donor_hash':donor.hash(),'row':row,'prefix':prefix,'runtime_admission_required':True}
    child.validate(Limits(strict_microcells=True)); return child
