import numpy as np
import pytest
from cellforge.construction.adaptive_mixer import build_adaptive_mixer,STATE_DIM,validate_mixer_recipe
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


@pytest.mark.parametrize('eta,decay',[(0.,1.),(.25,.9),(.5,.97),(1.,1.)])
def test_reference_equation_parity_batched_documents_and_explicit_reset(eta,decay):
    prior=np.array([.8,.15,.05]);graph=build_adaptive_mixer(eta,decay,prior)
    it=Interpreter(graph);compiled=CompiledInterpreter(graph)
    rng=np.random.default_rng(937);B=2
    rows=rng.random((13,B,3,256))+.01;rows/=rows.sum(axis=-1,keepdims=True)
    # The first component's likelihood is far below exp(-60), retaining exact
    # direct-power behavior rather than clipped-softmax behavior.
    rows[2,0,0,17]=1e-90;rows[2,0,0]/=rows[2,0,0].sum()
    observations=rng.integers(0,256,size=(13,B));observations[3,0]=17
    state=np.zeros((B,STATE_DIM));weights=np.broadcast_to(prior,(B,3)).copy()
    previous=None
    for t in range(13):
        observed=np.zeros((B,256))
        if t:observed[np.arange(B),observations[t]]=1.
        if t:
            likelihood=np.array([previous[j,:,observations[t,j]] for j in range(B)])
            unnormalized=weights*(likelihood**eta)
            posterior=unnormalized/unnormalized.sum(axis=1,keepdims=True)
            weights=decay*posterior+(1-decay)*prior
        expected=sum(weights[:,i:i+1]*rows[t,:,i] for i in range(3))
        inputs={**{'p'+str(i):rows[t,:,i] for i in range(3)},'observed':observed,'step':np.full((B,1),t,dtype=float),'state':state}
        outputs=it.run_bundle(graph.entries['step'],inputs)
        coutputs=compiled.run_bundle(graph.entries['step'],inputs)
        np.testing.assert_allclose(outputs['weights'],weights,rtol=1e-13,atol=1e-15)
        np.testing.assert_allclose(outputs['probabilities'],expected,rtol=1e-13,atol=1e-15)
        for key in outputs:np.testing.assert_array_equal(outputs[key],coutputs[key])
        np.testing.assert_array_equal(outputs['state'][:,:768],rows[t].reshape(B,768))
        state=outputs['state'];previous=rows[t]
    # BOS resets graph weights even when a preceding document state is supplied.
    inputs.update(step=np.zeros((B,1)),observed=np.zeros((B,256)),state=state)
    reset=it.run_bundle(graph.entries['step'],inputs)
    np.testing.assert_array_equal(reset['weights'],np.broadcast_to(prior,(B,3)))


def test_recipe_constraints_and_graph_owned_parameter_update():
    for recipe in [(-1,.9,[.8,.15,.05]),(.5,1.1,[.8,.15,.05]),(.5,.9,[1,0,0]),(.5,.9,[.8,.1,.05])]:
        with pytest.raises(ValueError):validate_mixer_recipe(*recipe)
    graph=build_adaptive_mixer(prefix='custom')
    assert graph.params['custom.eta'].shape==(1,)
    assert graph.params['custom.decay'].shape==(1,)
    assert graph.params['custom.prior'].shape==(3,)
    assert graph.meta['adaptive_mixer_recipe']['scope'].startswith('update-equation parity only')
