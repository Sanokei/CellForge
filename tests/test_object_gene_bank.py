import numpy as np
import pytest

from cellforge.construction.object_gene_bank import build_object_gene_bank
from cellforge.ir.graph import Graph, Limits
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


@pytest.fixture(scope='module', params=[Interpreter, CompiledInterpreter])
def runtime(request):
    return request.param(build_object_gene_bank())


def run(runtime, name, **inputs):
    # A heterogeneous batch also checks independence, not just duplicated data.
    return runtime.run_bundle(name, {k: np.asarray(v, dtype=float) for k, v in inputs.items()})


def test_every_definition_reachable_and_literal_export(tmp_path):
    graph = build_object_gene_bank()
    graph.validate(Limits(strict_microcells=True))
    reachable = set()
    def walk(name):
        if name in reachable:
            return
        reachable.add(name)
        for child in graph.children(name):
            walk(child)
    for name in graph.entries.values():
        walk(name)
    assert reachable == set(graph.defs)
    assert len(graph.meta['port_contracts']) == 53
    assert set(graph.meta['cell_port_contracts']) == {k for k, d in graph.defs.items() if d['kind'] == 'cell'}
    graph.save(tmp_path / 'bank')
    assert Graph.load(tmp_path / 'bank').hash() == graph.hash()


def test_selection_ties_padding_empty_and_uniqueness(runtime):
    valid = np.zeros((2, 16)); valid[0, :4] = 1
    values = np.zeros((2, 16)); values[0, :4] = [2, 7, 7, 1]
    for field in ('area', 'row', 'column', 'width', 'height'):
        for direction, expected in [('largest', [0, 1, 1, 0]), ('smallest', [0, 0, 0, 1])]:
            for tie in ('all', 'first'):
                got = run(runtime, f'select_{direction}_{field}_{tie}', valid=valid, values=values)['selected']
                want = np.array(expected)
                if tie == 'first' and direction == 'largest':
                    want = [0, 1, 0, 0]
                np.testing.assert_array_equal(got[0, :4], want)
                assert not got[0, 4:].any() and not got[1].any()
    for field, width in [('area', 1), ('color', 1), ('shape', 900)]:
        features = np.zeros((2, 16, width))
        features[0, :4, 0] = [1, 0, 0, 2] if width == 1 else [1, 0, 0, 0]
        if width == 900:
            features[0, 3, 1] = 1
        got = run(runtime, 'select_unique_'+field, features=features, valid=valid)['selected']
        np.testing.assert_array_equal(got[0, :4], [1, 0, 0, 1])
        assert not got[1].any()


def test_correspondence_priorities_padding_and_empty(runtime):
    v = np.zeros((2, 16)); v[0, :2] = 1
    sv = np.zeros((2, 16)); sv[:, 0] = 1
    source = np.zeros((2, 16, 900)); source[:, 0, 0] = 1
    target = np.zeros_like(source); target[0, 1, 0] = 1
    colors = np.zeros((2, 16)); colors[:, 0] = 3
    tc = np.zeros_like(colors); tc[0, 0] = 3; tc[0, 1] = 4
    positions = np.zeros((2, 16, 2)); positions[:, 0] = [5, 5]
    tp = np.zeros_like(positions); tp[0, 0] = [5, 5]; tp[0, 1] = [8, 8]
    inputs = dict(source_shapes=source, target_shapes=target, source_colors=colors,
                  target_colors=tc, source_positions=positions, target_positions=tp,
                  source_valid=sv, target_valid=v)
    for label, expected in [('shape', 1), ('color', 0), ('position', 0),
                            ('shape_color', 0), ('shape_position', 1),
                            ('color_position', 0), ('shape_color_position', 0)]:
        got = run(runtime, 'match_'+label, **inputs)['matches']
        assert got[0, 0, expected] == 1 and got[0].sum() == 1
        assert not got[1].any()


def scene():
    colors = np.zeros((2, 30, 30, 10)); colors[:, :4, :4, 0] = 1
    colors[:, 1, 1] = 0; colors[:, 1, 1, 3] = 1
    colors[:, 2, 1] = 0; colors[:, 2, 1, 4] = 1
    masks = np.zeros((2, 16, 900)); masks[:, 0, 31] = 1
    selected = np.zeros((2, 16)); selected[:, 0] = 1
    canvas = np.zeros((2, 30, 30, 1)); canvas[:, :4, :4] = 1
    bg = np.zeros((2, 10)); bg[:, 0] = 1
    return dict(colors=colors, masks=masks, selected=selected, canvas=canvas, background=bg)


def test_editing_overlap_signed_offsets_and_clipping(runtime):
    inputs = scene()
    for operation in ('delete', 'keep', 'copy', 'move'):
        args = dict(inputs)
        if operation in ('move', 'copy'):
            args['offset'] = [[1, 0], [-1, -1]]
        got = run(runtime, 'object_'+operation, **args)['colors']
        assert got[0, 1, 1].argmax() == (3 if operation in ('keep', 'copy') else 0)
        if operation in ('move', 'copy'):
            assert got[0, 2, 1].argmax() == 3  # destination overwrite
            assert got[1, 0, 0].argmax() == 3
        elif operation == 'keep':
            assert got[0, 2, 1].argmax() == 0
        else:
            assert got[0, 2, 1].argmax() == 4
        assert not got[:, 4:].any()
        np.testing.assert_array_equal(got.sum(-1), inputs['canvas'][..., 0])
    clipped = run(runtime, 'object_move', **inputs, offset=[[-2, 0], [0, 29]])['colors']
    assert not clipped[..., 3].any()
    stationary = run(runtime, 'object_move', **inputs, offset=[[0, 0], [0, 0]])['colors']
    np.testing.assert_array_equal(stationary, inputs['colors'])


def test_gravity_and_counts(runtime):
    inputs = scene()
    for direction, offset in [('down', [1, 0]), ('up', [-1, 0]), ('left', [0, -1]), ('right', [0, 1])]:
        got = run(runtime, 'gravity_by_clearance_'+direction, **inputs, clearance=[[1], [0]])['colors']
        expected = run(runtime, 'object_move', **inputs, offset=[offset, [0, 0]])['colors']
        np.testing.assert_array_equal(got, expected)
    for kind, width in [('objects', 16), ('pixels', 900)]:
        items = np.zeros((2, width)); items[0, :5] = 1
        np.testing.assert_array_equal(run(runtime, 'count_'+kind, items=items)['count'], [[5], [0]])
    for count in range(31):
        ins = [[count], [30-count]]
        strip = run(runtime, 'count_to_strip', count=ins)['output']
        hot = run(runtime, 'count_to_onehot', count=ins)['output']
        np.testing.assert_array_equal(strip.sum(1), [count, 30-count])
        np.testing.assert_array_equal(strip[0], np.arange(30) < count)
        np.testing.assert_array_equal(hot.argmax(1), [count, 30-count])
        np.testing.assert_array_equal(hot.sum(1), [1, 1])


def test_computed_gravity_obstacles_border_and_empty(runtime):
    inputs = scene()
    for direction, expected in [('down', 0), ('up', 1), ('left', 1), ('right', 2)]:
        got = run(runtime, 'gravity_'+direction, **inputs)
        np.testing.assert_array_equal(got['clearance'], [[expected], [expected]])
        np.testing.assert_array_equal(got['colors'].sum(-1), inputs['canvas'][..., 0])
    inputs['selected'][1] = 0
    got = run(runtime, 'gravity_right', **inputs)
    assert got['clearance'][1, 0] == 0
    np.testing.assert_array_equal(got['colors'][1], inputs['colors'][1])
    # An obstacle two spaces away blocks the path even with free space beyond it.
    inputs = scene()
    inputs['colors'][:, 2, 1] = 0; inputs['colors'][:, 2, 1, 0] = 1
    inputs['colors'][:, 3, 1] = 0; inputs['colors'][:, 3, 1, 4] = 1
    assert (run(runtime, 'gravity_down', **inputs)['clearance'] == 1).all()


def test_colored_count_output(runtime):
    color = np.zeros((2, 10)); color[:, 6] = 1
    background = np.zeros((2, 10)); background[:, 9] = 1
    for count in (0, 1, 16, 30):
        got = run(runtime, 'count_to_color_strip', count=[[count], [count]],
                  color=color, background=background)
        np.testing.assert_array_equal(got['width'], [[max(1, count)]]*2)
        assert got['colors'][..., 6].sum() == 2*count
        assert got['colors'][..., 9].sum() == (2 if count == 0 else 0)
        assert not got['colors'][:, 1:].any()


def test_count_compositions(runtime):
    for kind, width in [('objects', 16), ('pixels', 900)]:
        items = np.zeros((2, width)); items[0, :3] = 1
        for encoding in ('strip', 'onehot', 'color_strip'):
            args = {'items': items}
            count_args = {'count': [[3], [0]]}
            if encoding == 'color_strip':
                color = np.zeros((2, 10)); color[:, 2] = 1
                bg = np.zeros((2, 10)); bg[:, 0] = 1
                args.update(color=color, background=bg)
                count_args.update(color=color, background=bg)
            got = run(runtime, f'count_{kind}_to_{encoding}', **args)
            expected = run(runtime, 'count_to_'+encoding, **count_args)
            for key in got:
                np.testing.assert_array_equal(got[key], expected[key])
