"""Safety and plumbing tests for ARC-3 policy cells, the maker meter and the manager's no-idle-slot guarantee. Run: python -m unittest tests.test_arc3cell -v"""
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import arc3cell as C  # noqa: E402
import maker as M  # noqa: E402

GOOD = """def act(params, mem, frame, avail, info):
    mem['n'] = mem.get('n', 0) + 1
    order = [int(round(v)) for v in params['order']]
    a = order[mem['n'] % len(order)]
    if a not in avail:
        a = avail[0]
    return (a, int(frame.shape[1] // 2), int(params['y'][0]))
"""
PARAMS = {"order": [1, 2, 3], "y": [7]}


class Validate(unittest.TestCase):
    def test_good_policy_is_accepted(self):
        self.assertEqual(C.validate(GOOD), (True, ""))

    def test_rejections(self):
        bad = {
            "import": "import os\ndef act(params, mem, frame, avail, info):\n    return (1, 0, 0)\n",
            "dunder": "def act(params, mem, frame, avail, info):\n    return (frame.__class__, 0, 0)\n",
            "open": "def act(params, mem, frame, avail, info):\n    open('x')\n    return (1, 0, 0)\n",
            "print": "def act(params, mem, frame, avail, info):\n    print(1)\n    return (1, 0, 0)\n",
            "signature": "def act(params, frame):\n    return (1, 0, 0)\n",
            "no act": "def run(params, mem, frame, avail, info):\n    return (1, 0, 0)\n",
            "toplevel statement": "x = 1\ndef act(params, mem, frame, avail, info):\n    return (1, 0, 0)\n",
            "attribute": "def act(params, mem, frame, avail, info):\n    return (frame.tofile, 0, 0)\n",
            "long string": "def act(params, mem, frame, avail, info):\n    s = '%s'\n    return (1, 0, 0)\n" % ("a" * 60),
            "lambda import trick": "def act(params, mem, frame, avail, info):\n    return (getattr(frame, 'sum'), 0, 0)\n",
            "decorator": "@property\ndef act(params, mem, frame, avail, info):\n    return (1, 0, 0)\n",
        }
        for name, source in bad.items():
            ok, _ = C.validate(source)
            self.assertFalse(ok, name)


class Sandbox(unittest.TestCase):
    def test_memory_persists_and_params_are_read(self):
        p = C.Policy(GOOD, PARAMS)
        self.assertIsNone(p.error)
        frame = np.zeros((64, 64), dtype=int)
        seen = []
        for _ in range(4):
            action, x, y, err = p.step(frame, [1, 2, 3, 4], {"levels": 0, "resets": 0, "step": 0, "key": "k"})
            self.assertIsNone(err)
            seen.append((action, x, y))
        p.close()
        self.assertEqual([a for a, _, _ in seen], [2, 3, 1, 2])  # mem['n'] counts across steps, params['order'] drives the choice
        self.assertTrue(all(x == 32 and y == 7 for _, x, y in seen))

    def test_zero_tensors_change_behaviour(self):
        p = C.Policy(GOOD, C.zeros_like(PARAMS))
        action, x, y, err = p.step(np.zeros((64, 64), dtype=int), [1, 2, 3], {"levels": 0, "resets": 0, "step": 0, "key": "k"})
        p.close()
        self.assertIsNone(err)
        self.assertEqual(y, 0)

    def test_endless_loop_is_killed(self):
        loop = "def act(params, mem, frame, avail, info):\n    while True:\n        pass\n    return (1, 0, 0)\n"
        p = C.Policy(loop, {"a": [1]}, timeout=1.5)
        _, _, _, err = p.step(np.zeros((64, 64), dtype=int), [1], {"levels": 0, "resets": 0, "step": 0, "key": "k"})
        p.close()
        self.assertIsNotNone(err)

    def test_bad_return_is_an_error_not_a_crash(self):
        bad = "def act(params, mem, frame, avail, info):\n    return 5\n"
        p = C.Policy(bad, {"a": [1]})
        _, _, _, err = p.step(np.zeros((64, 64), dtype=int), [1], {"levels": 0, "resets": 0, "step": 0, "key": "k"})
        p.close()
        self.assertIn("act must return", err)

    def test_rejected_source_never_reaches_a_worker(self):
        p = C.Policy("import os\n", {"a": [1]})
        self.assertIn("rejected", p.error)
        _, _, _, err = p.step(np.zeros((64, 64), dtype=int), [1], {})
        self.assertIsNotNone(err)

    def test_tensor_limit(self):
        ok, _ = C.check_shapes({"w": [300]}, {"w": [0.0] * 300})
        self.assertFalse(ok)


class Meter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (M.LEDGER, M.CAP_FILE)
        M.LEDGER = pathlib.Path(self.tmp.name) / "BUDGET.jsonl"
        M.CAP_FILE = pathlib.Path(self.tmp.name) / "CAP.json"

    def tearDown(self):
        M.LEDGER, M.CAP_FILE = self.saved
        self.tmp.cleanup()

    def test_default_cap_and_last_reading(self):
        self.assertEqual(M.cap(), 85.0)
        self.assertEqual(M.last_used(), (None, None))
        M.log({"utc": "t", "role": "writer", "used_percent": 61.5, "resets_at": 5})
        M.log({"utc": "t", "role": "writer", "used_percent": None, "resets_at": None})  # unreadable meter keeps the last good value
        self.assertEqual(M.last_used(), (61.5, 5))

    def test_ask_refuses_at_cap_before_calling_anything(self):
        M.CAP_FILE.write_text(json.dumps({"stop_at_percent": 60}))
        M.log({"utc": "t", "role": "writer", "used_percent": 60.0, "resets_at": 5})
        with self.assertRaises(M.BudgetExhausted):
            M.ask("never sent", pathlib.Path(self.tmp.name) / "w", role="writer")
        self.assertFalse((pathlib.Path(self.tmp.name) / "w" / "START.json").exists())


class ManagerGuarantee(unittest.TestCase):
    """The manager may fail in every way; slots must still be filled and weights still requested."""

    def test_fallback_fills_every_slot(self):
        import lab_manager as L
        with tempfile.TemporaryDirectory() as tmp:
            lab = L.Lab.__new__(L.Lab)
            lab.root = pathlib.Path(tmp)
            lab.state = {"cycle": 0, "games": {g: {"tags": [], "n_levels": 6, "attempts": 0, "best_levels": 0, "credited": 0, "best_cell": None, "history": []}
                                               for g in ("aa11", "bb22", "cc33", "dd44")}}
            lab.state["games"]["aa11"]["attempts"] = L.MAX_ATTEMPTS_PER_GAME  # exhausted game must never be ordered

            def broken(*args, **kwargs):
                raise RuntimeError("maker down")

            saved = L.M.ask
            L.M.ask = broken
            try:
                orders = lab.orders(3)
            finally:
                L.M.ask = saved
            self.assertEqual(len(orders), 3)
            self.assertNotIn("aa11", [o["game"] for o in orders])
            self.assertTrue(all(o["by"] == "fallback" for o in orders))

    def test_manager_orders_for_unknown_games_are_dropped(self):
        import lab_manager as L
        with tempfile.TemporaryDirectory() as tmp:
            lab = L.Lab.__new__(L.Lab)
            lab.root = pathlib.Path(tmp)
            lab.state = {"cycle": 0, "games": {g: {"tags": [], "n_levels": 6, "attempts": 0, "best_levels": 0, "credited": 0, "best_cell": None, "history": []}
                                               for g in ("aa11", "bb22")}}
            saved = L.M.ask
            L.M.ask = lambda *a, **k: {"orders": [{"game": "zz99", "house": "Ember", "instruction": "x"}, {"game": "bb22", "house": "Nobody", "instruction": "y" * 900}],
                                       "summary": "s"}
            try:
                orders = lab.orders(2)
            finally:
                L.M.ask = saved
            games = [o["game"] for o in orders]
            self.assertEqual(sorted(games), ["aa11", "bb22"])
            managed = [o for o in orders if o["by"] == "manager"]
            self.assertEqual(managed[0]["game"], "bb22")
            self.assertIn(managed[0]["house"], L.HOUSES)
            self.assertLessEqual(len(managed[0]["instruction"]), 400)


class CrossoverPrompt(unittest.TestCase):
    def test_second_cell_appears_in_the_prompt_and_caps_double_with_credit(self):
        import lab_manager as L
        cell = {"source_lines": ["def act(params, mem, frame, avail, info):", "    return (1, 0, 0)"], "shapes": {"a": [1]}, "params": {"a": [1]},
                "result": {"levels": 1, "n_levels": 6, "level_steps": [9], "used": 1500, "resets": 0, "noops": 0, "distinct": 5, "state": "NOT_FINISHED", "action_hist": {"1": 1500}}}
        probe = {"game": "zz99", "tags": [], "available": [1], "n_levels": 6, "human_baseline_actions": [5], "colours": {0: 1}, "actions": {}, "clicks": [], "first_frame": "."}
        other = dict(cell, source_lines=["def act(params, mem, frame, avail, info):", "    return (2, 0, 0)"])
        text = L.writer_prompt("Ember", probe, cell, "", other)
        self.assertIn("CROSSOVER", text)
        self.assertIn("return (2, 0, 0)", text)
        self.assertNotIn("CROSSOVER", L.writer_prompt("Ember", probe, cell, "", None))
        donor = L.writer_prompt("Ember", probe, cell, "", None, ("lp85", other))
        self.assertIn("DONOR CELL", donor)
        self.assertIn("(lp85)", donor)
        self.assertNotIn("DONOR CELL", L.writer_prompt("Ember", probe, cell, "", None, None))
        self.assertEqual(L.attempt_cap({"credited": 0}), L.MAX_ATTEMPTS_PER_GAME)
        self.assertEqual(L.attempt_cap({"credited": 1}), 2 * L.MAX_ATTEMPTS_PER_GAME)


if __name__ == "__main__":
    unittest.main()
