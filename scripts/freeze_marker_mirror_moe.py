# SPDX-License-Identifier: MIT-0
"""Freeze matched marker-mirror activation and inactive-donor graphs."""
import json
from pathlib import Path

from cellforge.ir.graph import Graph
from cellforge.construction.transactions import apply_ops
from cellforge.ir.profiles import check_profile
from expanded_grid_genetics import canonical


def main():
    out = Path('runs/marker-mirror-moe-v1')
    if out.exists():
        raise ValueError('fresh output required')
    parent = Graph.load('runs/arc-grid-genes/expanded-breeding-v2/base')
    donor = Graph.load('runs/arc-grid-genes/marker-mirror-v1')
    child, _ = apply_ops(parent, [dict(
        op='import_slot_gene', slot=8, prefix='marker_mirror_',
        name='repeat.gene', **{'from': donor.hash()})],
        resolver=lambda ref: donor)
    control, _ = apply_ops(child, [dict(
        op='set_slot_gene', slot=8, gene='geometry_0')])
    canonical(control, child)
    for graph in (control, child):
        check_profile(graph, 'ladder-micro/1')
    out.mkdir()
    hashes = {name: graph.save(out / name) for name, graph in (
        ('inactive-mirror-control', control), ('marker-mirror-moe', child))}
    (out / 'policy.json').write_text(json.dumps(dict(
        hashes=hashes, parent_hash=parent.hash(), donor_hash=donor.hash(),
        scope='same imported bank and parameters; only slot8 activation differs',
        provenance='authored from 2bcee788 training demonstrations; no query labels',
        gradients=False), indent=2))
    print(json.dumps(hashes))


if __name__ == '__main__':
    main()
