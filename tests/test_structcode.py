"""Maker-proposed structures run only behind a whitelist, a worker process and acceptance controls. These tests try to escape each layer."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import structcode as X  # noqa: E402

FLIP = "def run(params, grid, demos):\n    return np.flipud(grid)\n"
LUT = ("def run(params, grid, demos):\n"
       "    lut = np.round(params['lut']).astype(int)\n"
       "    return lut[grid]\n")
HARDCODED = ("def run(params, grid, demos):\n"
             "    out = grid.copy()\n"
             "    out[grid == 1] = 2\n"
             "    return out\n")


def task(pairs, test=None):
    t = {"train": [{"input": a, "output": b} for a, b in pairs]}
    t["test"] = [{"input": a, "output": b} for a, b in (test or [])]
    return t


class StaticWhitelist(unittest.TestCase):
    def test_a_plain_numpy_structure_is_accepted_and_demo_use_is_detected(self):
        self.assertEqual(X.validate(FLIP), (True, ""))
        self.assertEqual(X.validate(LUT), (True, ""))
        self.assertFalse(X.uses_demos(FLIP))
        self.assertTrue(X.uses_demos("def run(params, grid, demos):\n    return demos[0][1]\n"))

    def test_every_escape_attempt_is_rejected(self):
        wrap = lambda body: "def run(params, grid, demos):\n" + "".join("    %s\n" % line for line in body.splitlines())  # noqa: E731
        attempts = {
            "import": "import os\nreturn grid",
            "import from": "from os import system\nreturn grid",
            "__import__": "m = __import__('os')\nreturn grid",
            "dunder attribute": "c = ().__class__.__bases__\nreturn grid",
            "dunder name": "x = __builtins__\nreturn grid",
            "open": "f = open('/etc/passwd')\nreturn grid",
            "exec": "exec('1')\nreturn grid",
            "eval": "eval('1')\nreturn grid",
            "getattr": "f = getattr(np, 'load')\nreturn grid",
            "np.load": "a = np.load('x.npy')\nreturn grid",
            "np.save": "np.save('x', grid)\nreturn grid",
            "tofile": "grid.tofile('x')\nreturn grid",
            "np.ctypeslib": "c = np.ctypeslib\nreturn grid",
            "np.lib": "c = np.lib\nreturn grid",
            "np.random": "r = np.random.rand(3)\nreturn grid",
            "f-string": "s = f'{grid}'\nreturn grid",
            "try": "try:\n    x = 1\nexcept Exception:\n    x = 2\nreturn grid",
            "with": "with open('x') as f:\n    pass\nreturn grid",
            "global": "global z\nreturn grid",
            "lambda calling eval": "f = lambda s: eval(s)\nreturn grid",
            "str format escape": "s = '{0.__class__}'.format(grid)\nreturn grid",
            "type": "t = type(grid)\nreturn grid",
            "sys": "sys.exit(1)\nreturn grid",
            "bytes constant": "b = b'abc'\nreturn grid",
        }
        for name, body in attempts.items():
            ok, why = X.validate(wrap(body))
            self.assertFalse(ok, "%s was accepted" % name)
        for source, name in (("x = 1\ndef run(params, grid, demos):\n    return grid\n", "top-level assignment"),
                             ("class A:\n    pass\ndef run(params, grid, demos):\n    return grid\n", "class"),
                             ("@property\ndef run(params, grid, demos):\n    return grid\n", "decorator"),
                             ("def helper(grid):\n    return grid\n", "no run"),
                             ("def run(grid):\n    return grid\n", "wrong signature"),
                             ("def run(params, grid, demos):\n    return grid\n" + "# " + "x" * X.MAX_SOURCE, "too long"),
                             ("", "empty")):
            self.assertFalse(X.validate(source)[0], name + " was accepted")

    def test_shapes_and_tensor_budget_are_enforced(self):
        self.assertTrue(X.check_shapes({"a": [2, 2]}, {"a": [[0, 1], [2, 3]]})[0])
        self.assertFalse(X.check_shapes({"a": [20, 20]}, {"a": [[0] * 20] * 20})[0], "answer smuggled as a big table")
        self.assertFalse(X.check_shapes({"a": [2]}, {"b": [0, 0]})[0])
        self.assertFalse(X.check_shapes({"a": [2]}, {"a": [0, float("nan")]})[0])
        self.assertFalse(X.check_shapes({"a": [2]}, {"a": [0, 5000]})[0])


class Runtime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = X.Sandbox(timeout=3.0)

    @classmethod
    def tearDownClass(cls):
        cls.box.close()

    def test_a_structure_runs_in_the_worker_and_reads_its_tensors(self):
        out, error = self.box.call(FLIP, {"a": [0]}, [[1, 2], [3, 4]], [])
        self.assertEqual((out, error), ([[3, 4], [1, 2]], None))
        lut = [0, 2, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(self.box.call(LUT, {"lut": lut}, [[1, 3]], [])[0], [[2, 3]])

    def test_an_endless_loop_is_killed_and_the_next_call_still_works(self):
        spin = "def run(params, grid, demos):\n    while True:\n        pass\n"
        out, error = X.Sandbox(timeout=1.0).call(spin, {"a": [0]}, [[1]], [])
        self.assertIsNone(out)
        self.assertIn("timeout", error)
        box = X.Sandbox(timeout=1.0)
        self.assertIn("timeout", box.call(spin, {"a": [0]}, [[1]], [])[1])
        self.assertEqual(box.call(FLIP, {"a": [0]}, [[1], [2]], [])[0], [[2], [1]])
        box.close()

    def test_bad_outputs_are_refused(self):
        for body, why in (("return np.full((31, 31), 1)", "too big"), ("return grid + 20", "values above 9"), ("return grid[0]", "not 2-D"),
                          ("return grid * 0.5", "not integers"), ("return grid - 5", "negative"), ("return 7", "scalar")):
            out, error = self.box.call("def run(params, grid, demos):\n    %s\n" % body, {"a": [0]}, [[1, 2], [3, 4]], [])
            self.assertIsNone(out, why)
            self.assertTrue(error, why)

    def test_errors_inside_a_structure_are_reported_never_raised(self):
        out, error = self.box.call("def run(params, grid, demos):\n    return grid[99]\n", {"a": [0]}, [[1]], [])
        self.assertIsNone(out)
        self.assertIn("IndexError", error)

    @unittest.skipUnless(sys.platform.startswith("linux"), "address-space limits are Linux only")
    def test_a_memory_bomb_dies_inside_the_worker(self):
        bomb = "def run(params, grid, demos):\n    a = np.zeros((40000, 40000))\n    return grid\n"
        out, error = X.Sandbox(timeout=5.0).call(bomb, {"a": [0]}, [[1]], [])
        self.assertIsNone(out)
        self.assertTrue(error)


class Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = X.Sandbox(timeout=3.0)
        a, b = [[1, 3], [0, 1]], [[2, 3], [0, 2]]
        c, d = [[1, 1], [3, 0]], [[2, 2], [3, 0]]
        e, f = [[0, 1], [1, 3]], [[0, 2], [2, 3]]
        cls.task = task([(a, b), (c, d)], test=[(e, f)])
        cls.lut = [0, 2, 2, 3, 4, 5, 6, 7, 8, 9]
        cls.shapes = {"lut": [10]}

    @classmethod
    def tearDownClass(cls):
        cls.box.close()

    def test_a_raw_lut_with_invalid_random_controls_is_rejected(self):
        res = X.evaluate(self.box, LUT, self.shapes, {"lut": self.lut}, self.task)
        self.assertTrue(res["valid"])
        self.assertEqual(res["demos_fit"], res["n_demos"])
        self.assertFalse(res["controls_ok"], res)
        self.assertEqual({error["seed"] for error in res["control_errors"]}, {11, 12, 13})
        self.assertTrue(all("output values" in error["error"] for error in res["control_errors"]))
        self.assertTrue(res["solved_test"])

    def test_a_structure_that_ignores_its_tensors_fails_the_controls(self):
        res = X.evaluate(self.box, HARDCODED, {"a": [1]}, {"a": [0]}, self.task)
        self.assertEqual(res["demos_fit"], res["n_demos"])
        self.assertFalse(res["controls_ok"], "zeroed tensors still solved it: the tensors do not matter")

    def test_wrong_tensors_fit_nothing_and_stop_early(self):
        res = X.evaluate(self.box, LUT, self.shapes, {"lut": [0] * 10}, self.task)
        self.assertLess(res["demos_fit"], res["n_demos"])
        self.assertIsNone(res["zero_fit"])

    def test_rejected_source_never_reaches_the_worker(self):
        res = X.evaluate(self.box, "import os\n", self.shapes, {"lut": self.lut}, self.task)
        self.assertFalse(res["valid"])


if __name__ == "__main__":
    unittest.main()
