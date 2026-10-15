"""Independent typed cells and fail-closed development Red Queen selection."""
import copy
import math
from cellforge.construction.joint_weight_population import digest
from cellforge.construction.red_queen_history import OpponentHistory

def validate_literal(params):
    def walk(value):
        if isinstance(value,list):
            if not value:raise ValueError('empty tensor')
            for item in value:walk(item)
        elif type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite literal numerical tensors required')
    if not params or any(k.startswith('slot') for k in params):raise ValueError('weight cells required')
    for value in params.values():walk(value)

class CellPopulation:
    def __init__(self):
        self.cells={};self.live={};self.histories={};self.events=[];self.charges=[]
    def seed(self,structure,params,provenance,scores):
        validate_literal(params)
        cid=digest({'structure':structure,'params':params})
        if cid in self.cells:raise ValueError('duplicate founder')
        self.cells[cid]={'id':cid,'structure':structure,'params':copy.deepcopy(params),'parents':[],
                        'provenance':copy.deepcopy(provenance),'scores':copy.deepcopy(scores),'state':'live'}
        self.live.setdefault(structure,[]).append(cid)
        h=self.histories.setdefault(structure,OpponentHistory())
        h.register(graph_hash=cid,family=structure,structure_hash=digest(structure),provenance=provenance,scores=scores)
        self.events.append({'event':'seed','cell':cid,'family':structure})
        return cid
    def birth(self,structure,params,parents,provenance,scores,*,caps_passed,fixtures_passed,arm):
        if not parents or any(p not in self.cells for p in parents):raise ValueError('known parents required')
        recipient=self.cells[parents[0]]
        if recipient['state']!='live':raise ValueError('live recipient required')
        try:
            validate_literal(params)
            if structure!=recipient['structure'] and arm.startswith('llm'):
                if not all(provenance.get(k) for k in ('raw_response','model','newly_authored','typed_io')):
                    raise ValueError('structural repair requires charged author and typed IO provenance')
            cid=digest({'structure':structure,'params':params})
        except (ValueError,TypeError) as error:
            self.charges.append({'arm':arm,'parents':parents,'child':None,'charged':True})
            self.events.append({'event':'invalid','family':structure,'reason':str(error)})
            return {'accepted':False,'invalid':True,'reasons':[str(error)]}
        self.charges.append({'arm':arm,'parents':parents,'child':cid,'charged':True})
        if cid in self.cells:
            self.events.append({'event':'rejected','cell':cid,'reason':'duplicate'});return {'accepted':False,'reasons':['duplicate']}
        self.cells[cid]={'id':cid,'structure':structure,'params':copy.deepcopy(params),'parents':parents,
            'provenance':copy.deepcopy(provenance),'scores':copy.deepcopy(scores),'state':'born'}
        self.events.append({'event':'birth','cell':cid,'family':structure,'parents':parents})
        h=self.histories[recipient['structure']]
        result=h.admission(recipient['scores'],dict(scores,caps_passed=caps_passed,fixtures_passed=fixtures_passed))
        if result['accepted']:
            self.cells[cid]['state']='live';self.live[recipient['structure']].remove(parents[0])
            self.cells[parents[0]]['state']='retired'
            self.live.setdefault(structure,[]).append(cid)
            self.histories.setdefault(structure,h)
            h.register(graph_hash=cid,family=structure,structure_hash=digest(structure),provenance=provenance,scores=scores)
            self.events.extend([{'event':'admitted','cell':cid},{'event':'retired','cell':parents[0]}])
        else:
            self.cells[cid]['state']='rejected';self.events.append({'event':'rejected','cell':cid,'admission':result})
        return result
    def assemble(self,router,ids=None):
        ids=ids if ids is not None else sorted(cid for values in self.live.values() for cid in values)
        if not ids or len(set(ids))!=len(ids):raise ValueError('nonempty distinct selected cells required')
        for cid in ids:
            if cid not in self.cells or self.cells[cid]['state']!='live':raise ValueError('only live cells cooperate')
            if not any(self.cells[cid]['scores'][k]['outcomes'][t] for k in ('fresh','archive','anchor') for t in self.cells[cid]['scores'][k]['outcomes']):raise ValueError('assembly requires measured cell competence')
        if len(router['route.b'])!=len(ids) or len(router['route.W'])!=len(ids) or any(len(r)!=len(ids) for r in router['route.W']):
            raise ValueError('router must align selected live cells')
        return {'cell_ids':ids,'cells':[copy.deepcopy(self.cells[c]) for c in ids],
                'router':copy.deepcopy(router),'router_cell_order':ids}
    @classmethod
    def from_snapshot(cls,snapshot):
        p=cls()
        p.cells=copy.deepcopy(snapshot['cells']);p.live=copy.deepcopy(snapshot['live'])
        p.events=copy.deepcopy(snapshot['events']);p.charges=copy.deepcopy(snapshot['charges'])
        p.histories={k:OpponentHistory(v) for k,v in snapshot['histories'].items()}
        for ids in p.live.values():
            if any(cid not in p.cells or p.cells[cid]['state']!='live' for cid in ids):raise ValueError('invalid resumable live population')
        return p
    def snapshot(self):
        return {'cells':self.cells,'live':self.live,'events':self.events,'charges':self.charges,
                'histories':{k:v.records for k,v in self.histories.items()}}

