"""Authored raw-byte action grammar. No natural-language understanding claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.graph import array_hash
from cellforge.ir.profiles import check_profile
from cellforge.runtime.text_codec import BYTE_CODEC,BYTE_VOCAB
from cellforge.construction.transactions import import_def


def build_action_language():
    n,p = [],{}
    def op(kind, inputs, **args):
        name=f'n{len(n)}'
        node={'id':name,'use':'op:'+kind,'in':inputs}
        if args: node['args']=args
        n.append(node)
        return name
    def param(name,value):
        p[name]=np.asarray(value,dtype=float)
        node=f'n{len(n)}'
        n.append({'id':node,'use':'param:'+name})
        return node
    def unary(kind,x,**args):return op(kind,{'x':x},**args)
    def binary(kind,a,b):return op(kind,{'a':a,'b':b})
    def add(a,b):return binary('add',a,b)
    def mul(a,b):return binary('mul',a,b)
    def bias(x,c):return unary('bias',x,c=c)
    def eq(a,b):return unary('relu',bias(unary('neg',unary('abs',binary('sub',a,b))),1))
    def one(x):return bias(unary('scale',x,c=0),1)
    def negflag(x):return binary('sub',one(x),x)
    def reshape(x,shape):return unary('reshape',x,shape=shape)
    def sl(x,start,stop,axis=0):return unary('slice',x,axis=axis,start=start,stop=stop)
    def sum_(x,axis=0):return unary('sum',x,axis=axis,keepdims=True)
    def char(c):return sl('$prev',ord(c)+3,ord(c)+4)
    def pattern(text):
        k=len(text)
        encoded=np.zeros((k,259))
        for j,c in enumerate(text):encoded[j,ord(c)+3]=1
        match=sum_(sum_(mul(sl('$ctx',64-k,64),param('pattern'+str(len(p)),encoded)),1),0)
        return reshape(unary('step',bias(match,-k+0.5)),[1])
    # Scalars: role, id capture/count/done/invalid, text capture/count/done/invalid,
    # generation active/index. Two raw-byte capture buffers follow.
    s=[sl('$state',j,j+1) for j in range(11)]
    role,ic,il,idd,ib,tc,tl,td,tb,ga,gi=s
    idbuf=reshape(sl('$state',11,11+24*259),[24,259])
    txbuf=reshape(sl('$state',11+24*259,11+48*259),[24,259])
    # Role changes only at literal JSON role fields, not escaped strings.
    role_matches=[pattern('"role":"'+r+'"') for r in ('system','user','tool','assistant')]
    changed=role_matches[0]
    for m in role_matches[1:]:changed=add(changed,m)
    newrole=mul(role,negflag(changed))
    for j,m in enumerate(role_matches):newrole=add(newrole,unary('scale',m,c=j+1))
    user=eq(newrole,bias(unary('scale',role,c=0),2))
    tool=eq(newrole,bias(unary('scale',role,c=0),3))
    istart=mul(mul(user,pattern('OCR[')),negflag(ga))
    tstart=mul(mul(tool,pattern('"text":"')),negflag(ga))
    iclose=mul(ic,char(']'))
    tclose=mul(tc,char('"'))
    allowed_id=sum_(mul('$prev',param('id_allowed',[float(i>=3 and chr(i-3) in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-') for i in range(259)])))
    allowed_text=sum_(sl('$prev',ord('0')+3,ord('9')+4))
    iw=mul(mul(ic,allowed_id),unary('step',bias(unary('neg',il),24)))
    tw=mul(mul(tc,allowed_text),unary('step',bias(unary('neg',tl),24)))
    bad_i=mul(ic,negflag(add(char(']'),allowed_id)))
    bad_t=mul(tc,negflag(add(char('"'),allowed_text)))
    overflow_i=mul(mul(ic,allowed_id),unary('step',bias(il,-23.5)))
    overflow_t=mul(mul(tc,allowed_text),unary('step',bias(tl,-23.5)))
    coords=param('capture_positions',np.arange(24)[:,None])
    byte=reshape('$prev',[1,259])
    def write(buffer,count,gate,start):
        position=mul(eq(coords,reshape(count,[1,1])),reshape(gate,[1,1]))
        delta=op('matmul',{'a':position,'b':byte})
        return mul(add(buffer,delta),reshape(negflag(start),[1,1]))
    new_id=write(idbuf,il,iw,istart)
    new_tx=write(txbuf,tl,tw,tstart)
    new_ic=mul(add(ic,istart),negflag(iclose))
    new_tc=mul(add(tc,tstart),negflag(tclose))
    new_il=mul(add(il,iw),negflag(istart))
    new_tl=mul(add(tl,tw),negflag(tstart))
    new_ib=mul(unary('step',add(add(ib,bad_i),overflow_i)),negflag(istart))
    new_tb=mul(unary('step',add(add(tb,bad_t),overflow_t)),negflag(tstart))
    new_idd=mul(unary('step',add(idd,iclose)),negflag(istart))
    new_td=mul(unary('step',add(td,tclose)),negflag(tstart))
    trigger=pattern('\nAction:')
    new_ga=unary('step',add(ga,trigger))
    index=mul(add(gi,ga),negflag(trigger))
    call=mul(mul(new_idd,negflag(new_ib)),unary('step',new_il))
    respond=mul(new_td,negflag(new_tb))
    call=mul(call,negflag(respond))
    fallback=negflag(add(call,respond))
    def emit(prefix,suffix,buffer,length,label):
        def table(text,label):
            arr=np.zeros((len(text),259))
            for j,c in enumerate(text):arr[j,ord(c)+3]=1
            return param(label,arr)
        pre=table(prefix,label+'_prefix')
        post=table(suffix,label+'_suffix')
        pp=param(label+'_prepos',np.arange(len(prefix))[None])
        sp=param(label+'_postpos',np.arange(len(suffix))[None])
        a=op('matmul',{'a':eq(reshape(index,[1,1]),pp),'b':pre})
        ci=bias(index,-len(prefix))
        inside=mul(unary('step',bias(ci,0.5)),unary('step',bias(binary('sub',length,ci),-0.5)))
        cw=mul(eq(reshape(ci,[1,1]),unary('transpose',coords,perm=[1,0])),reshape(inside,[1,1]))
        b=op('matmul',{'a':cw,'b':buffer})
        si=binary('sub',ci,length)
        c=op('matmul',{'a':eq(reshape(si,[1,1]),sp),'b':post})
        eos=unary('step',bias(si,-len(suffix)+0.5))
        e=mul(reshape(eos,[1,1]),param(label+'_eos',np.eye(259)[[2]]))
        return reshape(add(add(a,b),add(c,e)),[259])
    calltokens=emit('{"type":"call","name":"ocr","arguments":{"image_id":"','"}}',new_id,new_il,'call')
    response=emit('{"type":"respond","text":"','"}',new_tx,new_tl,'respond')
    emptybuf=unary('scale',new_tx,c=0)
    emptylength=unary('scale',new_tl,c=0)
    abstain=emit('{"type":"respond","text":"','"}',emptybuf,emptylength,'abstain')
    logits=unary('scale',add(add(mul(calltokens,call),mul(response,respond)),mul(abstain,fallback)),c=20)
    scalars=[newrole,new_ic,new_il,new_idd,new_ib,new_tc,new_tl,new_td,new_tb,new_ga,index]
    state=op('concat',{f'x{j}':x for j,x in enumerate(scalars+[reshape(new_id,[24*259]),reshape(new_tx,[24*259])])},axis=0)
    graph=Graph(defs={'action_step':{'kind':'bundle','inputs':['ctx','prev','state','step'],'nodes':n,'outputs':{'logits':logits,'state':state}}},params=p,
      entries={'step':'action_step'},meta={'task':'lm','text_codec':BYTE_CODEC,'vocab':BYTE_VOCAB,'ctx_len':64,'state_dim':11+48*259,
       'language_interface':'raw-byte-action-grammar/1','construction':'authored recurrent raw-byte parser and output grammar; no fitting',
       'limitations':'explicit OCR[id] directive, alnum underscore hyphen IDs and digit OCR observations only; not natural-language reasoning or document understanding'})
    graph.validate()
    return graph


def integrate_action_language(parent,grammar):
    """Compose the frozen grammar with the inherited hierarchical byte MoE.

    Both components own recurrent state. Graph-owned explicit-command detection
    selects the grammar head; ordinary input retains the original byte head.
    """
    if parent.meta.get('text_codec')!=BYTE_CODEC or parent.meta.get('ctx_len')!=64:
        raise ValueError('requires canonical UTF8 64-context parent')
    g=parent.copy()
    import_def(g,grammar,grammar.entries['step'],prefix='action_')
    size=parent.meta['state_dim']
    root=g.defs[g.entries['step']]
    for node in root['nodes']:
        for port,ref in node.get('in',{}).items():
            if ref=='$state':node['in'][port]='base_state'
    previous_logits=root['outputs']['logits']
    previous_state=root['outputs']['state']
    if previous_state=='$state':previous_state='base_state'
    root['nodes']=[{'id':'base_state','use':'op:slice','in':{'x':'$state'},'args':{'axis':0,'start':0,'stop':size}},
                   {'id':'action_state','use':'op:slice','in':{'x':'$state'},'args':{'axis':0,'start':size,'stop':size+grammar.meta['state_dim']}}]+root['nodes']
    root['nodes'].extend([
        {'id':'action_grammar','use':'action_'+grammar.entries['step'],'in':{'ctx':'$ctx','prev':'$prev','step':'$step','state':'action_state'}},
        {'id':'action_id_done','use':'op:slice','in':{'x':'action_grammar.state'},'args':{'axis':0,'start':3,'stop':4}},
        {'id':'action_text_done','use':'op:slice','in':{'x':'action_grammar.state'},'args':{'axis':0,'start':7,'stop':8}},
        {'id':'action_detected','use':'op:add','in':{'a':'action_id_done','b':'action_text_done'}},
        {'id':'action_selected','use':'op:step','in':{'x':'action_detected'}},
        {'id':'action_negative','use':'op:neg','in':{'x':'action_selected'}},
        {'id':'base_selected','use':'op:bias','in':{'x':'action_negative'},'args':{'c':1}},
        {'id':'action_head','use':'op:mul','in':{'a':'action_grammar.logits','b':'action_selected'}},
        {'id':'base_head','use':'op:mul','in':{'a':previous_logits,'b':'base_selected'}},
        {'id':'composed_logits','use':'op:add','in':{'a':'action_head','b':'base_head'}},
        {'id':'composed_state','use':'op:concat','in':{'x0':previous_state,'x1':'action_grammar.state'},'args':{'axis':0}}])
    root['outputs'].update(logits='composed_logits',state='composed_state')
    g.meta.update(state_dim=size+grammar.meta['state_dim'],language_interface='unified-byte-action-grammar/1',
                  parent_hash=parent.hash(),grammar_hash=grammar.hash(),
                  limitations='authored priority for explicit OCR commands and digit observations; ordinary language competence unchanged; no learned tool selection')
    g.validate()
    return g


def tile_action_constants(source,cap=40):
    """Mechanically tile oversized parameter references, preserving each value.

    This does not compress or fit the constants. Concatenation restores their
    donor flattened order before a reshape restores the original tensor.
    """
    if cap!=40:raise ValueError('fixed micro profile cap required')
    g=source.copy()
    lineage={}
    for name,d in g.defs.items():
        if d['kind']!='bundle':continue
        if name.startswith('action_'):d['role']='cluster'
        newnodes=[]
        for node in d['nodes']:
            if not node['use'].startswith('param:'):
                newnodes.append(node);continue
            donor=node['use'][6:]
            arr=source.params[donor]
            if arr.size<=cap:
                newnodes.append(node);continue
            flat=arr.reshape(-1)
            chunks=[]
            inputs={}
            for j,start in enumerate(range(0,len(flat),cap)):
                stop=min(start+cap,len(flat))
                parameter=donor+f'__tile{j:04d}'
                identifier=node['id']+f'_tile{j:04d}'
                g.params[parameter]=flat[start:stop].copy()
                newnodes.append({'id':identifier,'use':'param:'+parameter})
                inputs[f'x{j}']=identifier
                chunks.append({'parameter':parameter,'donor_flat_start':start,'donor_flat_stop':stop,'sha256':array_hash(g.params[parameter])})
            joined=node['id']+'_tiled_flat'
            newnodes.append({'id':joined,'use':'op:concat','in':inputs,'args':{'axis':0}})
            newnodes.append({'id':node['id'],'use':'op:reshape','in':{'x':joined},'args':{'shape':list(arr.shape)}})
            lineage[donor]={'donor_sha256':array_hash(arr),'shape':list(arr.shape),'chunks':chunks}
        d['nodes']=newnodes
    referenced={p for name in g.defs for p in g.params_of(name)}
    for name in lineage:
        if name not in referenced:del g.params[name]
    g.meta.update(parent_hash=source.hash(),numeric_rewrite='lossless donor-flat-index tiling at40elements',
                  tile_lineage=lineage,profile='ladder-micro/1')
    g.validate()
    check_profile(g,'ladder-micro/1')
    return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--parent');p.add_argument('--grammar');p.add_argument('--tile');args=p.parse_args()
    target=Path(args.out)
    if target.exists():raise SystemExit('output must be new')
    if bool(args.parent)!=bool(args.grammar):raise SystemExit('parent and grammar required together')
    if args.tile and args.parent:raise SystemExit('tile and parent are separate modes')
    graph=tile_action_constants(Graph.load(args.tile)) if args.tile else (
        integrate_action_language(Graph.load(args.parent),Graph.load(args.grammar)) if args.parent else build_action_language())
    digest=graph.save(target)
    (target/'BUILDER.py').write_bytes(Path(__file__).read_bytes())
    (target/'MODEL_CARD.json').write_text(json.dumps({'hash':digest,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'numeric_provenance':'all thresholds, byte syntax templates and positions directly authored; no data fit',
      'limitations':graph.meta['limitations'],'architecture':'inherited hierarchical MoE with graph-owned grammar priority' if args.parent or args.tile else 'external grammar LM graph, not hierarchical MoE',
      'profile':graph.meta.get('profile'),'parent_hash':graph.meta.get('parent_hash'),'grammar_hash':graph.meta.get('grammar_hash')},indent=2))
    print(digest)


if __name__=='__main__':main()
