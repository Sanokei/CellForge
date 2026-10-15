import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.interp import Interpreter


def test_memory_detects_two_distant_continuations_only():
    graph = Graph.load('runs/unified-byte-memory/artifact')
    vocab = graph.meta['vocab']
    width = len(vocab)
    context = np.zeros((1, 64, width))
    # Two abc->d observations more than eight positions back; query abc.
    for offset in (0, 16):
        for i, value in enumerate(b'abcd'):
            context[0, offset + i, vocab.index(f'<byte:{value:02x}>')] = 1
    for i, value in enumerate(b'abc'):
        context[0, 61 + i, vocab.index(f'<byte:{value:02x}>')] = 1
    runtime = Interpreter(graph)
    penalty = runtime.run_bundle('long_ngram_memory', {'ctx': context}, '/memory')['penalty']
    assert penalty.shape == (1, width)
    assert penalty[0, vocab.index('<byte:64>')] == 2
    assert np.count_nonzero(penalty) == 1
    context[0, :4] = 0
    assert not np.any(runtime.run_bundle('long_ngram_memory', {'ctx': context}, '/memory')['penalty'])
