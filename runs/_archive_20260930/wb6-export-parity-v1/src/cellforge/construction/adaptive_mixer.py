"""Graph-owned posterior mixer for three byte probability components.

This exports the adaptive update equation, not the count/word/cache models
feeding it. The historical 1.2222 bpb experiment is not reproduced here.
"""
import hashlib
from pathlib import Path
import numpy as np
from ..ir.graph import Graph

STATE_DIM=771
REFERENCE_SHA256='1dc79a23dfef6bded802aedf01058d5d0d0b22e48bada9d02aaeb4ef293291f7'


def validate_mixer_recipe(eta,decay,prior):
    prior=np.asarray(prior,dtype=float)
    if prior.shape!=(3,) or not np.isfinite(prior).all() or not (prior>0).all() or not np.isclose(prior.sum(),1.,rtol=0,atol=1e-12):
        raise ValueError('three strictly positive normalized prior weights required')
    if not np.isfinite(eta) or eta<0:raise ValueError('finite nonnegative eta required')
    if not np.isfinite(decay) or not 0<=decay<=1:raise ValueError('decay in [0,1] required')
    return prior


def build_adaptive_mixer(eta=.5,decay=.97,prior=(.8,.15,.05),prefix='mixer'):
    """Return a reusable cluster, with encode/step/predict entries aliased.

    Inputs p0/p1/p2: [B,256] positive normalized byte distributions;
    observed: [B,256] raw byte onehot, all-zero at BOS;
    step: [B,1] (zero at document BOS); state: [B,771].
    Output probabilities: [B,256], weights: [B,3], state: [B,771].
    Step t first updates from stored p(t-1) and observed byte, then mixes p(t).
    """
    prior=validate_mixer_recipe(eta,decay,prior)
    nodes=[]
    def node(name,use,inputs=None,args=None):
        item={'id':name,'use':use}
        if inputs:item['in']=inputs
        if args is not None:item['args']=args
        nodes.append(item);return name
    def op(name,use,inputs,args=None):return node(name,'op:'+use,inputs,args)
    node('eta','param:'+prefix+'.eta');node('decay','param:'+prefix+'.decay');node('prior','param:'+prefix+'.prior')
    node('fresh',prefix+'.fresh',{'x':'$step'})
    op('negative_fresh','scale',{'x':'fresh.y'},{'c':-1})
    op('reset','bias',{'x':'negative_fresh'},{'c':1})
    op('old_weights','slice',{'x':'$state'},{'axis':0,'start':768,'stop':771})
    op('kept_weights','mul',{'a':'old_weights','b':'fresh.y'})
    op('reset_weights','mul',{'a':'prior','b':'reset'})
    op('weights_before','add',{'a':'kept_weights','b':'reset_weights'})
    for i in range(3):
        op('previous_'+str(i),'slice',{'x':'$state'},{'axis':0,'start':i*256,'stop':(i+1)*256})
        op('observed_'+str(i),'mul',{'a':'previous_'+str(i),'b':'$observed'})
        op('likelihood_'+str(i),'sum',{'x':'observed_'+str(i)},{'axis':0,'keepdims':True})
    op('likelihoods','concat',{'x'+str(i):'likelihood_'+str(i) for i in range(3)},{'axis':0})
    op('kept_likelihoods','mul',{'a':'likelihoods','b':'fresh.y'})
    # At BOS there is no observed byte: use likelihood 1 before exponentiation.
    op('reset_ones','scale',{'x':'prior'},{'c':0})
    op('ones','bias',{'x':'reset_ones'},{'c':1})
    op('reset_likelihoods','mul',{'a':'ones','b':'reset'})
    op('evidence','add',{'a':'kept_likelihoods','b':'reset_likelihoods'})
    op('tempered_evidence','power',{'a':'evidence','b':'eta'})
    op('unnormalized','mul',{'a':'weights_before','b':'tempered_evidence'})
    op('normalizer','sum',{'x':'unnormalized'},{'axis':0,'keepdims':True})
    op('posterior','divide',{'a':'unnormalized','b':'normalizer'})
    op('posterior_kept','mul',{'a':'posterior','b':'decay'})
    op('negative_decay','scale',{'x':'decay'},{'c':-1})
    op('shrink','bias',{'x':'negative_decay'},{'c':1})
    op('prior_shrunk','mul',{'a':'prior','b':'shrink'})
    op('weights_updated','add',{'a':'posterior_kept','b':'prior_shrunk'})
    op('weights_update_active','mul',{'a':'weights_updated','b':'fresh.y'})
    op('weights','add',{'a':'weights_update_active','b':'reset_weights'})
    for i in range(3):
        op('w'+str(i),'slice',{'x':'weights'},{'axis':0,'start':i,'stop':i+1})
        op('pweighted'+str(i),'mul',{'a':'$p'+str(i),'b':'w'+str(i)})
    op('p01','add',{'a':'pweighted0','b':'pweighted1'})
    op('probabilities','add',{'a':'p01','b':'pweighted2'})
    op('next_state','concat',{'x0':'$p0','x1':'$p1','x2':'$p2','x3':'weights'},{'axis':0})
    name=prefix+'.cluster'
    graph=Graph(defs={prefix+'.fresh':{'kind':'cell','role':'cell','act':'step',
                                     'params':{'W1':prefix+'.fresh.W1','b1':prefix+'.fresh.b1'}},
                     name:{'kind':'bundle','role':'cluster','inputs':['p0','p1','p2','observed','step','state'],
                           'nodes':nodes,'outputs':{'probabilities':'probabilities','weights':'weights','state':'next_state'}}},
                params={prefix+'.eta':np.array([eta]),prefix+'.decay':np.array([decay]),prefix+'.prior':prior.copy(),
                        prefix+'.fresh.W1':np.ones((1,1)),prefix+'.fresh.b1':np.zeros(1)},
                entries={'encode':name,'step':name,'predict':name},
                meta={'task':'component','component_kind':'adaptive_probability_mixer','state_dim':STATE_DIM,
                      'adaptive_mixer_recipe':{'protocol':'posterior-mixer/1','reference_script':'scripts/claude_adaptive_mixer.py',
                        'reference_sha256':REFERENCE_SHA256,'equation':'post_i = w_i * likelihood_i**eta / sum_j(w_j*likelihood_j**eta); w_i = decay*post_i+(1-decay)*prior_i',
                        'reset':'step=0 uses prior; state stores previous component probabilities and used weights',
                        'recipe_parameters':{prefix+'.eta':[1],prefix+'.decay':[1],prefix+'.prior':[3]},
                        'constraints':'eta>=0; 0<=decay<=1; prior positive sum1; components positive normalized256',
                        'scope':'update-equation parity only; no full count/word/cache corpus parity or 1.2222 claim'}})
    graph.validate();return graph
