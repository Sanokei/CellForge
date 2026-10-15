"""Independent typed cells and fail-closed development Red Queen selection."""
import copy
from cellforge.construction.joint_weight_population import digest
from cellforge.construction.red_queen_history import OpponentHistory

class CellPopulation:
    def __init__(self):
        self.cells={};self.live={};self.histories={};self.events=[];self.charges=[]
    def seed(self,structure,params,provenance,scores):
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
        cid=digest({'structure':structure,'params':params})
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
    def snapshot(self):
        return {'cells':self.cells,'live':self.live,'events':self.events,'charges':self.charges,
                'histories':{k:v.records for k,v in self.histories.items()}}

