# SPDX-License-Identifier: MIT-0
"""Fresh variable-size cases for the declared marker-edge hypothesis."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from build_grid_marker_mirror_gene import build_marker_mirror
from cellforge.ir.profiles import check_profile
from cellforge.runtime.grid import GridPredictor


from cellforge.reference.marker_mirror_curriculum import mirror_example


def test_marker_population_corpora_are_balanced_and_selection_has_no_queries():
    from run_grid_population_matched import frozen_expanded_corpora
    selection, report, paths, commitments = frozen_expanded_corpora(
        1, 4507000, 4517000, 2, include_marker=True)
    assert len(selection) == 13 and len(report) == 26 and paths == []
    assert all(set(task) == {'train'} for task in selection)
    assert commitments['selection_demo_pairs'] == 37
    families = [task.get('family', task.get('metadata', {}).get('family'))
                for task in report]
    assert len(set(families)) == 13
    assert all(families.count(name) == 2 for name in set(families))


@pytest.mark.parametrize('backend', ['reference', 'compiled'])
def test_fresh_marker_edge_mirrors(backend):
    graph = build_marker_mirror()
    check_profile(graph, 'ladder-micro/1')
    predictor = GridPredictor(graph, backend=backend)
    for seed in range(4478000, 4478100):
        image, expected = mirror_example(seed)
        # The standalone gene predicts geometry with background zero. It does
        # not learn the source ARC task's output-background recoloring.
        assert predictor([{'input': [[0]], 'output': [[0]]}], image)[0] == expected


@pytest.mark.parametrize('backend', ['reference', 'compiled'])
def test_moe_learns_background_and_resets(backend):
    from cellforge.ir.graph import Graph
    from cellforge.construction.transactions import apply_ops
    parent = Graph.load(Path(__file__).parents[1] /
                        'runs/arc-grid-genes/expanded-breeding-v2/base')
    donor = build_marker_mirror()
    child, _ = apply_ops(parent, [dict(
        op='import_slot_gene', slot=8, prefix='test_mirror_',
        name='repeat.gene', **{'from': donor.hash()})],
        resolver=lambda ref: donor)
    predictor = GridPredictor(child, backend=backend)
    original_hash = child.hash()
    for background in (3, 9, 3):
        pairs = []
        for seed in range(4478200, 4478204):
            image, geometry = mirror_example(seed)
            output = [[background if v == 0 else v for v in row]
                      for row in geometry]
            pairs.append(dict(input=image, output=output))
        image, geometry = mirror_example(4478210)
        expected = [[background if v == 0 else v for v in row]
                    for row in geometry]
        assert expected in predictor(pairs, image)
        assert expected in predictor(list(reversed(pairs)), image)
    assert child.hash() == original_hash
