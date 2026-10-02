"""Head-center skill-cell core: the frozen pop1 templates (10-colour palette) plus new typed skill structures with literal tensors.

A cell is a template name + literal tensors. The structure (forward function) is fixed code; every number that selects behaviour lives in
the tensors, which are written by the maker. Routing between cells is the existing demonstration-error router (route.W/route.b).

tile_orient: output is a pr x pc arrangement of tiles, tile (i,j) = dihedral transform k(i,j) of the input grid.
  per[0..3], pec[0..3]: pr = argmax(per)+1, pc = argmax(pec)+1.   orient[i][j][0..7]: k(i,j) = argmax(orient[i][j]).
  transforms k: 0 identity, 1 fliplr, 2 flipud, 3 rot180, 4 transpose, 5 rot90 clockwise, 6 rot90 counter-clockwise, 7 anti-transpose.
  Transposing transforms need a square input (otherwise the cell is invalid on that grid and the router sees maximal error).
"""
import importlib.util

import numpy as np

OPS = [lambda a: a, np.fliplr, np.flipud, lambda a: np.rot90(a, 2), lambda a: a.T, lambda a: np.rot90(a, -1),
       lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2).T]
MAXT = 4
SHAPES = {"tile_orient": {"per": (MAXT,), "pec": (MAXT,), "orient": (MAXT, MAXT, 8)}}
DOCS = {"tile_orient": ("pr = argmax(per)+1 tile rows, pc = argmax(pec)+1 tile columns (each at most 4). Tile (i,j) of the output is the input grid "
                        "transformed by k = argmax(orient[i][j]) (a list of 8 numbers) from: 0 identity, 1 fliplr, 2 flipud, 3 rot180, 4 transpose, "
                        "5 rot90 clockwise, 6 rot90 counter-clockwise, 7 anti-transpose. Only entries with i<pr and j<pc are used. Transposing "
                        "transforms (4-7) require a square input.")}


def tile_orient(params, grid, trace=False):
    a = np.asarray(grid)
    pr, pc = int(np.argmax(params["per"])) + 1, int(np.argmax(params["pec"])) + 1
    orient = np.asarray(params["orient"], float)
    rows = []
    for i in range(pr):
        tiles = []
        for j in range(pc):
            k = int(np.argmax(orient[i][j]))
            if k >= 4 and a.shape[0] != a.shape[1]:
                raise ValueError("transposing tile transform needs a square input")
            tiles.append(OPS[k](a))
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0).astype(int).tolist(), {"cells": 1, "pixels": 0}


def make_head_core(pop1_core_path):
    from cellforge.construction.arc_weight_moe import make_core
    core = make_core(pop1_core_path)
    core.SHAPES.update(SHAPES)
    core.PREDICTORS["tile_orient"] = tile_orient
    docs = getattr(core, "DOCS", {})
    docs.update(DOCS)
    core.DOCS = docs
    return core


def load_by_path(path):
    spec = importlib.util.spec_from_file_location("head_core_" + str(abs(hash(str(path)))), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
