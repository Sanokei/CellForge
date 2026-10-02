"""Versioned neural-score/argument ports; no host action or argument inference."""
import copy
import numpy as np
from cellforge.construction.tool_router_evolution import validate,features,ACTIONS
from cellforge.construction.action_argument_cell import infer

VERSION='tool-action-joint/2'
MAIN_LABELS=tuple('edit_replace' if name=='edit_in_place' else name for name in ACTIONS)

def compatible_artifact(router_tensors, argument_proposal, main_router):
    t=validate(router_tensors)
    return {'version':VERSION,'router':t,'router_feature22':'argument_cell_file_role_present',
            'router_feature22_original':'request-only filename syntax; explicitly superseded for this joint interface',
            'router_actions':list(ACTIONS),'main_action_ports':list(MAIN_LABELS),
            'argument_tensors':copy.deepcopy(argument_proposal['tensors']),
            'argument_vocab':copy.deepcopy(main_router['vocab']),
            'argument_probe_router':copy.deepcopy(main_router),
            'provenance':{'router':'actual GPT6.1Sol low proposal inherited unchanged','arguments':'main actual27position proposal inherited unchanged',
                          'new_weight_positions':0,'interface_change':'port22 now uses measured neural file-role presence; different model interface, prior phrase reports do not transfer'}}

def infer_compatible(request,artifact):
    if artifact['version']!=VERSION or artifact['router_actions']!=list(ACTIONS) or artifact['main_action_ports']!=list(MAIN_LABELS):
        raise ValueError('exact versioned output-port contract required')
    t=validate(artifact['router']);args=artifact['argument_tensors']
    # Main's numerical token scorer independently chooses spans; its probe action is discarded.
    probe=infer(request,artifact['argument_probe_router'],args['slot.W'],args['null.b'])
    v=features(request);v[22]=int(probe['slots']['file'] is not None)
    logits=(np.asarray(t['W'])@np.asarray(v)+np.asarray(t['b'])).tolist()
    # Mechanical score ports only. Zero W suppresses duplicate keyword scoring;
    # b here is runtime neural activation transport, NEVER a stored genome bias.
    transport={'actions':list(MAIN_LABELS),'vocab':artifact['argument_vocab'],
               'W':np.zeros((9,len(artifact['argument_vocab']))).tolist(),'b':logits}
    result=infer(request,transport,args['slot.W'],args['null.b'])
    if result['slots']!=probe['slots']:raise ValueError('argument selection changed under transport')
    return dict(result,version=VERSION,router_trace={'features':v,'literal_logits':logits,
                'source_row':int(np.argmax(logits)),'source_action':ACTIONS[int(np.argmax(logits))],
                'main_action':MAIN_LABELS[int(np.argmax(logits))],
                'semantic_change':'neural file-role presence feature; not old filename-syntax router'},
                transport_scope='ephemeral numerical score forwarding, not genome rewrite')
