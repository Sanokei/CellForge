from pathlib import Path
import numpy as np
import pytest
from cellforge.ir.graph import Graph
from cellforge.construction.fitted_byte_seed import fitted_byte_seed
from cellforge.runtime.interp import Interpreter


def test_fitted_seed_count_method_and_graph_lookup():
    founder = Graph.load(Path('runs/unified-byte/artifact'))
    before = founder.hash()
    partition = {'partition': 'dev_train_selection', 'corpus_sha256': 'fixture',
                 'items': [{'family': 'fixture-a', 'text': 'ababa'}]}
    seed, provenance = fitted_byte_seed(founder, partition)
    assert founder.hash() == before and provenance['artifact_class'] == 'fitted'
    assert provenance['fit_content_bytes'] == 5
    prev = np.zeros((1,259)); prev[0, ord('a')+3] = 1
    output = Interpreter(seed).run_bundle('fitted_ngram_trunk', {'prev': prev})['logits']
    assert np.argmax(output) == ord('b')+3
    np.testing.assert_allclose(np.exp(output).sum(), 1., atol=1e-12)


@pytest.mark.parametrize('partition', ['report', 'final', 'selection'])
def test_fitted_seed_rejects_non_training_partition(partition):
    founder = Graph.load(Path('runs/unified-byte/artifact'))
    with pytest.raises(ValueError, match='dev_train'):
        fitted_byte_seed(founder, {'partition': partition, 'items': []})
