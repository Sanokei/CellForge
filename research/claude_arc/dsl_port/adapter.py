"""Adapter between numpy grids and Hodel ARC-DSL tuple-of-tuples grids, plus safe wrappers.

The DSL (MIT, Michael Hodel) lives in third_party/arc_dsl and is imported as the top-level module `dsl`
(its own `from arc_types import *` needs that directory on sys.path).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_DSL_DIR = os.path.join(HERE, "third_party", "arc_dsl")
_PARENT = os.path.dirname(HERE)            # research/claude_arc (cells.py, fitted.py, ...)
for _p in (_DSL_DIR, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dsl as D  # noqa: E402  (Hodel dsl.py, unmodified)

MAXD = 30


def to_h(g):
    """numpy 2-D int array -> tuple of tuples of Python ints."""
    return tuple(tuple(int(v) for v in r) for r in np.asarray(g))


def from_h(g):
    """tuple-of-tuples -> numpy int array (raises on ragged/empty)."""
    a = np.array(g, dtype=int)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("not a 2-D grid")
    return a


def ok_grid(a):
    return a is not None and a.ndim == 2 and a.size > 0 and a.shape[0] <= MAXD and a.shape[1] <= MAXD


def safe_grid(fn, *args):
    """Call fn(*args) (DSL grid function), return numpy grid or None on any error / size violation."""
    try:
        o = fn(*args)
        if not isinstance(o, tuple) or len(o) == 0 or len(o) > MAXD or len(o[0]) == 0 or len(o[0]) > MAXD:
            return None
        a = np.array(o, dtype=int)
        return a if ok_grid(a) else None
    except Exception:
        return None


def wrap_cell(fn):
    """Wrap a DSL function Grid->Grid as a numpy cell g -> ndarray (None when it fails or exceeds 30x30)."""
    def cell(g):
        return safe_grid(fn, to_h(g))
    cell.__name__ = getattr(fn, "__name__", "cell")
    return cell
