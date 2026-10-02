import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from build_grid_numeric_neighbor_gate import build_weight_gate, examples
from cellforge.construction.numeric_recipe import apply_numeric_recipe
from cellforge.construction.transactions import TransactionError
from cellforge.runtime.grid import GridPredictor
from cellforge.construction.maker_operator import MakerOperator


def test_numeric_crossover_assembles_gate_and_records_exact_parents():
    left = build_weight_gate((0, 0, 1, 0), -1.5)
    right = build_weight_gate((0, 0, 0, 1), -1.5)
    hashes = [left.hash(), right.hash()]
    child = apply_numeric_recipe(left, right, [dict(
        op='homologous_mask', name='neighbor_gate.W', mask=[[0, 0, 0, 1]])])
    assert child.meta['numeric_recipe']['parents'] == hashes
    assert child.meta['numeric_recipe']['status'] == 'unadmitted'
    np.testing.assert_array_equal(child.params['neighbor_gate.W'], [[0, 0, 1, 1]])
    assert [left.hash(), right.hash()] == hashes
    for backend in ('reference', 'compiled'):
        predict = GridPredictor(child, backend=backend)
        for task in examples(4717000):
            assert task['test'][0]['output'] in predict(task['train'], task['test'][0]['input'])


def test_structured_scaling_then_crossover_is_ordered_and_atomic():
    left = build_weight_gate(); right = build_weight_gate((0, 0, 1, 1))
    before = left.hash()
    child = apply_numeric_recipe(left, right, [
        dict(op='scale_rows', name='neighbor_gate.W', rows=[0], factor=2),
        dict(op='homologous_mask', name='neighbor_gate.W', mask=[[1, 1, 0, 0]])])
    np.testing.assert_array_equal(child.params['neighbor_gate.W'], [[0, 0, 2, 2]])
    with pytest.raises(TransactionError):
        apply_numeric_recipe(left, right, [
            dict(op='scale_rows', name='neighbor_gate.W', rows=[0], factor=2),
            dict(op='scale_rows', name='neighbor_gate.W', rows=[5], factor=1)])
    assert left.hash() == before


@pytest.mark.parametrize('mask', [[[0, 1]], [[0, 0, 0, 0.5]], [[0, 0, 0, float('nan')]]])
def test_rejects_invalid_masks(mask):
    with pytest.raises(TransactionError):
        apply_numeric_recipe(build_weight_gate(), build_weight_gate(), [
            dict(op='homologous_mask', name='neighbor_gate.W', mask=mask)])


def test_rejects_missing_provenance_and_nonhomologous_cells():
    left = build_weight_gate(); right = build_weight_gate()
    del right.meta['gradients']
    recipe = [dict(op='scale_rows', name='neighbor_gate.W', rows=[0], factor=1)]
    with pytest.raises(TransactionError): apply_numeric_recipe(left, right, recipe)


def test_charged_maker_recipe_exact_resolution_and_freeze(tmp_path):
    left = build_weight_gate((0, 0, 1, 0), -1.5)
    right = build_weight_gate((0, 0, 0, 1), -1.5)
    operator = MakerOperator(tmp_path / 'ledger.jsonl', budget=3)
    for graph in (left, right):
        graph.defs['numeric_fixture'] = dict(kind='bundle', role='cluster', inputs=['x'],
            nodes=[dict(id='gate', use='neighbor_gate', **{'in':dict(x='$x')})],
            outputs=dict(y='gate.y'))
    # Execute the changed cell itself as a typed fixture. Full grid parity is
    # checked above; a fixture is a contract check, not hidden-label scoring.
    fixtures = [dict(entry='numeric_fixture', inputs=dict(x=[[0, 0, 1, 1]]),
        output_shapes=dict(y=[1, 1]))]
    recipe = [dict(op='homologous_mask', name='neighbor_gate.W', mask=[[0, 0, 0, 1]])]
    args = dict(diagnostics=dict(partition='development', evidence='train-pairs'),
        provenance=dict(actor='test-author', model='fixture-only', method='manual-llm-maker'),
        fixtures=fixtures, evaluate=lambda _: dict(reconstructed=1), accept=lambda _: False)
    ops = [dict(op='numeric_recipe', **{'from':right.hash()}, recipe=recipe)]
    _, missing = operator.submit(left, ops=ops, **args)
    assert missing['status'] == 'invalid' and missing['evaluation_calls'] == 0
    _, stale = operator.submit(left, ops=ops, resolver=lambda _: left, **args)
    assert stale['status'] == 'invalid' and 'hash mismatch' in stale['error']
    child, valid = operator.submit(left, ops=ops, resolver=lambda _: right, **args)
    assert child is not None and valid['evaluation_calls'] == 1
    assert valid['status'] == 'rejected'  # no accidental archive admission
    assert valid['candidate_hash'] == child.hash()
    operator.freeze(child)
    assert operator.sealed_once(child, lambda _: dict(score=0)) == dict(score=0)
    with pytest.raises(ValueError, match='already called'):
        operator.sealed_once(child, lambda _: dict(score=1))
    assert len(MakerOperator(operator.path, budget=3).rows) == 5
    right = build_weight_gate(); right.defs['neighbor_gate']['act'] = 'relu'
    with pytest.raises(TransactionError): apply_numeric_recipe(left, right, recipe)
