"""Separate 414-slot shallow NCA, inlined cluster suitable for ladder-micro."""
import numpy as np
from ..ir.graph import Graph, Limits
from ..ir.profiles import check_profile


class Nodes:
    def __init__(self): self.nodes=[]
    def add(self,use,inputs=None,**args):
        name=f'n{len(self.nodes)}'; node={'id':name,'use':use,'in':inputs or {}}
        if args: node['args']=args
        self.nodes.append(node); return name
    def bundle(self,inputs,outputs,role):
        return {'kind':'bundle','role':role,'inputs':inputs,'nodes':self.nodes,'outputs':outputs}


def build_shallow_nca(seed='identity',architecture='residual'):
    if seed not in ('identity','swap01') or architecture not in ('residual','direct'): raise ValueError('authored seed/architecture required')
    graph=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':1,'state_dim':0,
                     'construction':'direct-authored shallow NCA/1; no fitting or gradient methods',
                     'profile':'ladder-micro/1','nca_recipe':{'channels':10,'features':90,'hidden':4,'iterations':3,'seed':seed,'architecture':architecture,
                     'parameterization':'360hiddenW+4sharedbias+40outputW+10outputbias; hidden tiled W stored3x and summed/3 to share effective bias exactly'},
                     'limitations':('residual skip preserves all10colors at zero output weights; different from Claude direct NCA, not a replication' if architecture=='residual' else 'direct bottleneck retains only color0..3; colors4..9 collapse to0')+'; same-size, no demonstration adaptation; two identical experts exist for ladder dispatch compatibility, not distinct capability'})
    for row in range(4):
        graph.params[f'shallow.h{row}.b']=np.zeros(1)
        weights=np.zeros((1,90)); weights[0,40+row]=3.
        for chunk in range(3):
            name=f'shallow.h{row}.c{chunk}'
            graph.params[name+'.W']=weights[:,chunk*30:(chunk+1)*30].copy()
            graph.defs[name]={'kind':'cell','act':'identity','params':{'W1':name+'.W','b1':f'shallow.h{row}.b'}}
    for row in range(10):
        name=f'shallow.o{row}'; weights=np.zeros((1,4))
        if architecture=='direct' and row<4: weights[0,(1-row if seed=='swap01' and row<2 else row)]=1.
        elif architecture=='residual' and seed=='swap01' and row<2:
            weights[0,row]=-1.; weights[0,1-row]=1.
        graph.params[name+'.W']=weights; graph.params[name+'.b']=np.zeros(1)
        graph.defs[name]={'kind':'cell','act':'identity','params':{'W1':name+'.W','b1':name+'.b'}}
    nodes=Nodes(); current='$input'
    for _ in range(3):
        patches=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                y=nodes.add('op:shift',{'x':current},axis=0,k=dy)
                patches.append(nodes.add('op:shift',{'x':y},axis=1,k=dx))
        features=nodes.add('op:concat',{f'x{i}':v for i,v in enumerate(patches)},axis=2)
        slices=[nodes.add('op:slice',{'x':features},axis=2,start=i*30,stop=(i+1)*30) for i in range(3)]
        hidden=[]
        for row in range(4):
            values=[nodes.add(f'shallow.h{row}.c{i}',{'x':slices[i]}) for i in range(3)]
            summed=nodes.add('op:add',{'a':values[0],'b':values[1]}); summed=nodes.add('op:add',{'a':summed,'b':values[2]})
            scaled=nodes.add('op:scale',{'x':summed},c=1/3)
            hidden.append(nodes.add('op:relu',{'x':scaled}))
        h=nodes.add('op:concat',{f'x{i}':v for i,v in enumerate(hidden)},axis=2)
        outputs=[nodes.add(f'shallow.o{i}',{'x':h}) for i in range(10)]
        residual=nodes.add('op:concat',{f'x{i}':v for i,v in enumerate(outputs)},axis=2)
        current=nodes.add('op:add',{'a':current,'b':residual}) if architecture=='residual' else residual
        current=nodes.add('op:mul',{'a':current,'b':'$input_mask'})
    graph.defs['nca_shallow_gene']=nodes.bundle(['input','input_mask','input_height','input_width'],
                                             {'colors':current,'height':'$input_height','width':'$input_width'},'cluster')
    ports=['input','input_mask','input_height','input_width','output','output_mask','output_height','output_width','phase','index','state']
    nodes=Nodes(); gene=nodes.add('nca_shallow_gene',{name:'$'+name for name in ('input','input_mask','input_height','input_width')})
    colors=nodes.add('op:reshape',{'x':gene+'.colors'},shape=[1,30,30,10])
    height=nodes.add('op:reshape',{'x':gene+'.height'},shape=[1,30]); width=nodes.add('op:reshape',{'x':gene+'.width'},shape=[1,30])
    expert=nodes.bundle(ports,{'color_logits':colors,'height_logits':height,'width_logits':width,'state':'$state'},'expert')
    import copy
    graph.defs['shallow_expert0']=expert; graph.defs['shallow_expert1']=copy.deepcopy(expert)
    nodes=Nodes(); scores=nodes.add('op:coords',n=2,start=0.,step=0.)
    graph.defs['shallow_router']=nodes.bundle([],{'scores':scores},'router')
    nodes=Nodes(); router=nodes.add('shallow_router'); dispatch=nodes.add('dispatch',dict({name:'$'+name for name in ports},scores=router+'.scores'),branches=['shallow_expert0','shallow_expert1'],k=1)
    graph.defs['shallow_root']=nodes.bundle(ports,{name:dispatch+'.'+name for name in ('color_logits','height_logits','width_logits','state')},'root')
    graph.entries['grid_step']='shallow_root'
    graph.validate(Limits(strict_microcells=True)); check_profile(graph,'ladder-micro/1')
    return graph
