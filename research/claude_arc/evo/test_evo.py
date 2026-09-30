import numpy as np
from common import *
import archive, redqueen


def test_hash_and_signature():
    g = np.arange(6).reshape(2, 3)
    assert ghash(g) == ghash(g.copy()) and ghash(g) != ghash(g.T) and ghash(None) == "N"
    probes = [g, np.eye(4, dtype=int)]
    s1 = archive.signature(("flipud", "flipud"), probes)
    s0 = archive.signature(("id",), probes)
    assert s1 == s0
    assert archive.signature(("flipud",), probes) != s0
    assert archive.similarity(s0, archive.signature(("flipud",), probes)) < 1.0


def test_dedupe():
    probes = [np.random.RandomState(0).randint(0, 4, (5, 5))]
    a = archive.NoveltyArchive(probes, near=0.98)
    assert a.add(("id",)) == "new"
    assert a.add(("flipud", "flipud")) == "dup"
    assert a.add(("fliplr",)) == "new"


def test_niche_archive():
    na = archive.NicheArchive(k=2)
    for c, n in [(("a",), 3), (("b",), 2), (("c",), 1)]:
        na.update("n1", c, n)
    assert na.top("n1") == [("a",), ("b",)]


def test_generator_shapes_and_determinism():
    rng = np.random.RandomState(1)
    for _ in range(20):
        g = redqueen.random_generator(rng)
        t1, t2 = redqueen.make_task(g, 4, 5), redqueen.make_task(g, 4, 5)
        assert (t1 is None) == (t2 is None)
        if t1:
            assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(t1, t2))
            assert all(max(x.shape) <= 20 and y.ndim == 2 for x, y in t1)


def test_solver_solves_own_generator():
    g = redqueen.Generator(("rot1",), {1: 2})
    tk = redqueen.make_task(g, 3, 6)
    assert tk and redqueen.solves(("rot1",), tk)       # chain + fitted colour map
    assert not redqueen.solves(("rot3",), tk)


def test_evolve_smoke():
    r = redqueen.evolve(3, 0)
    assert r["evals"] > 0 and "solvers" in r


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
