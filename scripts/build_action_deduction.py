"""Controlled positive-English clause parser and JSON actions in graph arithmetic.

Supported clauses: Fact atom. If atom then atom. Is atom entailed?
This is an authored bounded language, not a general English interpreter.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.text_codec import BYTE_CODEC,BYTE_VOCAB
from cellforge.construction.transactions import import_def


def action_builder():
    spec=importlib.util.spec_from_file_location('action_builder',Path(__file__).with_name('build_action_language.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Nodes:
    def __init__(self):
        self.nodes=[]
        self.params={'positive_W':np.ones((1,1)),'positive_b':np.zeros(1)}
    def node(self,use,inputs=None,**args):
        name=f'n{len(self.nodes)}';node={'id':name,'use':use,'in':inputs or {}}
        if args:node['args']=args
        self.nodes.append(node);return name
    def op(self,kind,**inputs):return self.node('op:'+kind,inputs)
    def u(self,kind,x,**args):return self.node('op:'+kind,{'x':x},**args)
    def p(self,value):
        name=f'p{len(self.params)}';self.params[name]=np.asarray(value,dtype=float)
        return self.node('param:'+name)
    def add(self,a,b):return self.op('add',a=a,b=b)
    def mul(self,a,b):return self.op('mul',a=a,b=b)
    def sub(self,a,b):return self.op('sub',a=a,b=b)
    def bias(self,x,c):return self.u('bias',x,c=c)
    def scale(self,x,c):return self.u('scale',x,c=c)
    def positive(self,x):return self.node('positive',{'x':x})
    def one(self,x):return self.bias(self.scale(x,0),1)
    def inv(self,x):return self.sub(self.one(x),x)
    def eq(self,a,b):return self.u('relu',self.bias(self.u('neg',self.u('abs',self.sub(a,b))),1))
    def reshape(self,x,shape):return self.u('reshape',x,shape=shape)
    def sl(self,x,start,stop,axis=0):return self.u('slice',x,axis=axis,start=start,stop=stop)
    def sum(self,x,axis=0):return self.u('sum',x,axis=axis,keepdims=True)
    def cat(self,refs,axis=0):return self.node('op:concat',{f'x{j}':ref for j,ref in enumerate(refs)},axis=axis)
    def char(self,c):return self.sl('$prev',ord(c)+3,ord(c)+4)
    def pattern(self,text):
        arr=np.zeros((len(text),259))
        for j,c in enumerate(text):arr[j,ord(c)+3]=1
        count=self.reshape(self.sum(self.sum(self.mul(self.sl('$ctx',64-len(text),64),self.p(arr)),1),0),[1])
        return self.positive(self.bias(count,-len(text)+0.5))


def build_deduction_action(availability=False):
    n=Nodes()
    # role, capturemode, slot, facts, rules, pendingrule, invalid, querydone,
    # queryready, generationactive/index, toolok/result, status, true/false.
    scalar_count=21 if availability else 17
    s=[n.sl('$state',j,j+1) for j in range(scalar_count)]
    role,cap,slot,fc,rc,pending,bad,qdone,qready,ga,gi,toolok,result,entailed,notproven,true,false=s[:17]
    if availability:contract,available,failed,withheld=s[17:]
    lengths=n.sl('$state',scalar_count,scalar_count+13)
    buffer=n.reshape(n.sl('$state',scalar_count+13,scalar_count+13+13*12*259),[13*12,259])
    role_matches=[n.pattern('"role":"'+r+'"') for r in ('system','user','tool','assistant')]
    changed=role_matches[0]
    for m in role_matches[1:]:changed=n.add(changed,m)
    newrole=n.mul(role,n.inv(changed))
    for j,m in enumerate(role_matches):newrole=n.add(newrole,n.scale(m,j+1))
    user=n.mul(n.eq(newrole,n.bias(n.scale(role,0),2)),n.inv(ga))
    tool=n.mul(n.eq(newrole,n.bias(n.scale(role,0),3)),n.inv(ga))
    if availability:
        system=n.mul(n.eq(newrole,n.bias(n.scale(role,0),1)),n.inv(ga))
        newcontract=n.mul(n.positive(n.add(contract,n.mul(system,n.pattern(' Available tools: ')))),n.inv(role_matches[0]))
        newavailable=n.mul(n.positive(n.add(available,n.mul(n.mul(system,newcontract),n.pattern('deduction: arguments ')))),n.inv(role_matches[0]))
    idle=n.eq(cap,n.scale(cap,0))
    fact=n.mul(n.mul(user,idle),n.pattern('Fact '))
    rule=n.mul(n.mul(user,idle),n.pattern('If '))
    query=n.mul(n.mul(user,idle),n.pattern('Is '))
    consequence=n.mul(n.mul(n.mul(user,idle),pending),n.pattern(' then '))
    start=n.add(n.add(fact,rule),n.add(query,consequence))
    fmode=n.eq(cap,n.bias(n.scale(cap,0),1));amode=n.eq(cap,n.bias(n.scale(cap,0),2))
    cmode=n.eq(cap,n.bias(n.scale(cap,0),3));qmode=n.eq(cap,n.bias(n.scale(cap,0),4))
    capturing=n.positive(cap)
    close=n.add(n.mul(n.add(fmode,cmode),n.char('.')),n.mul(n.add(amode,qmode),n.char(' ')))
    allowed=n.sum(n.mul('$prev',n.p([float(i>=3 and chr(i-3) in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') for i in range(259)])))
    initial=n.sum(n.mul('$prev',n.p([float(i>=3 and chr(i-3) in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_') for i in range(259)])))
    slot_coords=n.p(np.arange(13,dtype=float))
    smask=n.eq(slot_coords,slot)
    length=n.sum(n.mul(lengths,smask))
    room=n.positive(n.bias(n.u('neg',length),12))
    write=n.mul(n.mul(n.mul(capturing,allowed),room),user)
    invalidchar=n.mul(n.mul(capturing,n.inv(n.add(allowed,close))),user)
    invalidinitial=n.mul(n.mul(n.mul(capturing,allowed),n.eq(length,n.scale(length,0))),n.inv(initial))
    overflow=n.mul(n.mul(capturing,allowed),n.inv(room))
    empty=n.mul(close,n.eq(length,n.scale(length,0)))
    clauseoverflow=n.add(n.mul(fact,n.positive(n.bias(fc,-3.5))),n.mul(rule,n.positive(n.bias(rc,-3.5))))
    interrupt=n.mul(pending,n.add(n.add(fact,rule),query))
    unsupported=n.mul(user,n.add(n.pattern(' and '),n.add(n.pattern(' not '),n.pattern('Not '))))
    newbad=n.positive(n.add(bad,n.add(n.add(n.add(invalidchar,invalidinitial),n.add(overflow,empty)),n.add(n.add(clauseoverflow,interrupt),unsupported))))
    newslot=n.mul(slot,n.inv(start))
    newslot=n.add(newslot,n.add(n.mul(fact,fc),n.add(n.mul(rule,n.add(n.scale(rc,2),n.bias(n.scale(rc,0),4))),
       n.add(n.scale(query,12),n.mul(consequence,n.bias(n.scale(rc,2),5))))))
    newmode=n.mul(cap,n.inv(close))
    newmode=n.add(newmode,n.add(fact,n.add(n.scale(rule,2),n.add(n.scale(consequence,3),n.scale(query,4)))))
    newfc=n.add(fc,n.mul(close,fmode));newrc=n.add(rc,n.mul(close,cmode))
    newpending=n.mul(n.positive(n.add(pending,rule)),n.inv(n.mul(close,cmode)))
    newqdone=n.positive(n.add(qdone,n.mul(close,qmode)))
    newqready=n.positive(n.add(qready,n.mul(n.mul(user,newqdone),n.pattern(' entailed?'))))
    newmask=n.eq(slot_coords,newslot)
    newlengths=n.mul(n.add(lengths,n.mul(smask,write)),n.inv(n.mul(newmask,start)))
    positions=n.p(np.arange(13*12,dtype=float)[:,None])
    target=n.add(n.scale(slot,12),length)
    position=n.mul(n.eq(positions,n.reshape(target,[1,1])),n.reshape(write,[1,1]))
    delta=n.op('matmul',a=position,b=n.reshape('$prev',[1,259]))
    # Clear only newly opened atom row, allowing a repeated query to replace it.
    rows=n.reshape(n.p(np.repeat(np.arange(13),12)),[13*12,1])
    reset=n.mul(n.eq(rows,n.reshape(newslot,[1,1])),n.reshape(start,[1,1]))
    newbuffer=n.mul(n.add(buffer,delta),n.inv(reset))
    toolrole=role_matches[2]
    newok=n.mul(n.positive(n.add(toolok,n.mul(tool,n.pattern('"status":"ok"')))),n.inv(toolrole))
    newresult=n.mul(n.positive(n.add(result,n.mul(n.mul(tool,newok),n.pattern('"result":{')))),n.inv(toolrole))
    validtool=n.mul(n.mul(tool,newok),newresult)
    newent=n.mul(n.positive(n.add(entailed,n.mul(validtool,n.pattern('"status":"entailed"')))),n.inv(toolrole))
    newnot=n.mul(n.positive(n.add(notproven,n.mul(validtool,n.pattern('"status":"not_proven"')))),n.inv(toolrole))
    newtrue=n.mul(n.positive(n.add(true,n.mul(validtool,n.pattern('"entailed":true')))),n.inv(toolrole))
    newfalse=n.mul(n.positive(n.add(false,n.mul(validtool,n.pattern('"entailed":false')))),n.inv(toolrole))
    yes=n.mul(newent,newtrue);no=n.mul(newnot,newfalse)
    valid=n.mul(n.mul(newqready,n.inv(newbad)),n.positive(n.sl(newlengths,12,13)))
    if availability:
        error=n.add(n.pattern('"content":{"status":"invalid_action"}'),n.pattern('"content":{"status":"specialist_error"}'))
        newfailed=n.mul(n.positive(n.add(failed,n.mul(tool,error))),n.inv(toolrole))
        newwithheld=n.mul(n.positive(n.add(withheld,n.mul(tool,n.pattern('"content":{"status":"observation_withheld"}')))),n.inv(toolrole))
        failedreply=n.mul(valid,newfailed)
        withheldreply=n.mul(n.mul(valid,newwithheld),n.inv(newfailed))
        unavailable=n.mul(n.mul(n.mul(valid,n.inv(newavailable)),n.inv(newfailed)),n.inv(newwithheld))
        blocked=n.positive(n.add(n.add(failedreply,withheldreply),unavailable))
        yes=n.mul(yes,n.inv(blocked));no=n.mul(no,n.inv(blocked))
    observed=n.positive(n.add(yes,no))
    selected=n.positive(n.add(valid,observed))
    trigger=n.pattern('\nAction:')
    newga=n.positive(n.add(ga,trigger));index=n.mul(n.add(gi,ga),n.inv(trigger))
    zero=n.scale(index,0)
    def const(x):return n.bias(zero,x)
    def table(text):
        arr=np.zeros((len(text),259))
        for j,c in enumerate(text):arr[j,ord(c)+3]=1
        return n.p(arr)
    emissions=[]
    def literal(text,offset,active):
        if not text:return
        weights=n.eq(n.reshape(n.sub(index,offset),[1,1]),n.p(np.arange(len(text),dtype=float)[None]))
        emissions.append(n.mul(n.reshape(n.op('matmul',a=weights,b=table(text)),[259]),active))
    atom_positions=n.p(np.arange(12,dtype=float)[None])
    def atom(which,offset,active):
        local=n.sub(index,offset)
        length=n.sl(newlengths,which,which+1)
        inside=n.mul(n.positive(n.bias(local,0.5)),n.positive(n.bias(n.sub(length,local),-0.5)))
        weights=n.mul(n.eq(n.reshape(local,[1,1]),atom_positions),n.reshape(n.mul(inside,active),[1,1]))
        payload=n.sl(newbuffer,which*12,(which+1)*12)
        emissions.append(n.reshape(n.op('matmul',a=weights,b=payload),[259]))
        return length
    call=n.mul(valid,n.inv(observed))
    if availability:call=n.mul(n.mul(call,newavailable),n.inv(blocked))
    offset=zero
    prefix='{"type":"call","name":"deduction","arguments":{"facts":['
    literal(prefix,offset,call);offset=n.bias(offset,len(prefix))
    for j in range(4):
        active=n.mul(call,n.positive(n.bias(newfc,-j-0.5)))
        text='"' if j==0 else ',"'
        literal(text,offset,active)
        length=atom(j,n.bias(offset,len(text)),active)
        literal('"',n.add(n.bias(offset,len(text)),length),active)
        offset=n.add(offset,n.mul(n.bias(length,len(text)+1),n.positive(n.bias(newfc,-j-0.5))))
    text='],"rules":[';literal(text,offset,call);offset=n.bias(offset,len(text))
    for j in range(4):
        active=n.mul(call,n.positive(n.bias(newrc,-j-0.5)))
        text='{"if":["' if j==0 else ',{"if":["'
        literal(text,offset,active)
        ante=atom(4+2*j,n.bias(offset,len(text)),active)
        middle='"],"then":"'
        midpos=n.add(n.bias(offset,len(text)),ante)
        literal(middle,midpos,active)
        concl=atom(5+2*j,n.bias(midpos,len(middle)),active)
        end=n.add(n.bias(midpos,len(middle)),concl)
        literal('"}',end,active)
        size=n.bias(n.add(ante,concl),len(text)+len(middle)+2)
        offset=n.add(offset,n.mul(size,n.positive(n.bias(newrc,-j-0.5))))
    text='],"query":"';literal(text,offset,call);offset=n.bias(offset,len(text))
    querylen=atom(12,offset,call)
    end=n.add(offset,querylen);literal('"}}',end,call)
    callend=n.bias(end,3)
    replyyes='{"type":"respond","text":"entailed"}'
    replyno='{"type":"respond","text":"not_proven"}'
    literal(replyyes,zero,yes);literal(replyno,zero,no)
    eosgate=n.add(n.mul(call,n.positive(n.bias(n.sub(index,callend),0.5))),
         n.add(n.mul(yes,n.positive(n.bias(index,-len(replyyes)+0.5))),n.mul(no,n.positive(n.bias(index,-len(replyno)+0.5)))))
    if availability:
        for text,flag in [('tool_failed',failedreply),('observation_unavailable',withheldreply),('tool_unavailable',unavailable)]:
            reply='{"type":"respond","text":"'+text+'"}'
            literal(reply,zero,flag)
            eosgate=n.add(eosgate,n.mul(flag,n.positive(n.bias(index,-len(reply)+0.5))))
    emissions.append(n.mul(n.p(np.eye(259)[2]),eosgate))
    logits=emissions[0]
    for e in emissions[1:]:logits=n.add(logits,e)
    logits=n.scale(logits,20)
    scalars=[newrole,newmode,newslot,newfc,newrc,newpending,newbad,newqdone,newqready,newga,index,newok,newresult,newent,newnot,newtrue,newfalse]
    if availability:scalars.extend([newcontract,newavailable,newfailed,newwithheld])
    state=n.cat(scalars+[newlengths,n.reshape(newbuffer,[13*12*259])])
    graph=Graph(defs={'positive':{'kind':'cell','act':'step','params':{'W1':'positive_W','b1':'positive_b'}},
      'deduction_action_step':{'kind':'bundle','role':'cluster','inputs':['ctx','prev','state','step'],'nodes':n.nodes,
                              'outputs':{'logits':logits,'state':state,'selected':selected}}},params=n.params,entries={'step':'deduction_action_step'},
      meta={'task':'lm','text_codec':BYTE_CODEC,'vocab':BYTE_VOCAB,'ctx_len':64,'state_dim':scalar_count+13+13*12*259,
            'language_interface':'controlled-positive-English-actions/2' if availability else 'controlled-positive-English-actions/1','construction':'directly authored byte parser and JSON generation; no fitting',
            'limitations':'Fact atom. If atom then atom. Is atom entailed? only; four facts/four unary rules, twelve-byte positive ASCII atoms; no general English or negation'})
    graph.validate();return graph


def compose_deduction_action(parent,component,tile=True):
    """Add isolated parser state and graph-owned action-head selection."""
    if parent.meta.get('text_codec')!=BYTE_CODEC or parent.meta.get('ctx_len')!=64:
        raise ValueError('requires64context UTF8 parent')
    g=parent.copy()
    import_def(g,component,component.entries['step'],prefix='deduction_')
    root=g.defs[g.entries['step']]
    size=parent.meta['state_dim']
    old_logits,old_state=root['outputs']['logits'],root['outputs']['state']
    if old_state=='$state':old_state='deduction_parent_state'
    for node in root['nodes']:
        for port,ref in node.get('in',{}).items():
            if ref=='$state':node['in'][port]='deduction_parent_state'
    root['nodes']=[
       {'id':'deduction_parent_state','use':'op:slice','in':{'x':'$state'},'args':{'axis':0,'start':0,'stop':size}},
       {'id':'deduction_component_state','use':'op:slice','in':{'x':'$state'},'args':{'axis':0,'start':size,'stop':size+component.meta['state_dim']}}]+root['nodes']
    root['nodes'].extend([
       {'id':'deduction_component','use':'deduction_'+component.entries['step'],'in':{'ctx':'$ctx','prev':'$prev','state':'deduction_component_state','step':'$step'}},
       {'id':'deduction_priority_negative','use':'op:neg','in':{'x':'deduction_component.selected'}},
       {'id':'deduction_parent_priority','use':'op:bias','in':{'x':'deduction_priority_negative'},'args':{'c':1}},
       {'id':'deduction_component_logits','use':'op:mul','in':{'a':'deduction_component.logits','b':'deduction_component.selected'}},
       {'id':'deduction_parent_logits','use':'op:mul','in':{'a':old_logits,'b':'deduction_parent_priority'}},
       {'id':'deduction_composed_logits','use':'op:add','in':{'a':'deduction_component_logits','b':'deduction_parent_logits'}},
       {'id':'deduction_composed_state','use':'op:concat','in':{'x0':old_state,'x1':'deduction_component.state'},'args':{'axis':0}}])
    root['outputs'].update(logits='deduction_composed_logits',state='deduction_composed_state')
    g.meta.update(state_dim=size+component.meta['state_dim'],parent_hash=parent.hash(),deduction_component_hash=component.hash(),
      language_interface='unified-byte-controlled-deduction/2' if component.meta['language_interface'].endswith('/2') else 'unified-byte-controlled-deduction/1',
      limitations=component.meta['limitations']+'; original OCR and ordinary language paths unchanged where deduction parser inactive')
    g.validate()
    return action_builder().tile_action_constants(g) if tile else g


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--parent');p.add_argument('--component');p.add_argument('--availability',action='store_true');args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise SystemExit('output must be new')
    if bool(args.parent)!=bool(args.component):raise SystemExit('parent and component required together')
    if args.parent:
        raw=compose_deduction_action(Graph.load(args.parent),Graph.load(args.component),tile=False)
        raw.save(out/'UNTILED_DONOR')
        g=action_builder().tile_action_constants(raw)
    else:g=build_deduction_action(availability=args.availability)
    digest=g.save(out)
    source=Path(__file__).read_bytes();(out/'BUILDER.py').write_bytes(source)
    if args.parent:
        (out/'ACTION_BUILDER_DEPENDENCY.py').write_bytes(Path(__file__).with_name('build_action_language.py').read_bytes())
    (out/'MODEL_CARD.json').write_text(json.dumps({'hash':digest,'builder_sha256':hashlib.sha256(source).hexdigest(),
      'numeric_provenance':'authored scalar thresholds, ASCII masks, syntax templates; no data fit','limitations':g.meta['limitations']},indent=2))
    print(digest)


if __name__=='__main__':main()
