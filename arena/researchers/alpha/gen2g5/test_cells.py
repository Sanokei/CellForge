"""Unit test for gen-4 cell proposals: import into cells.py namespace, run with
a fake GameState over a synthetic frame. Checks: generic behavior, no crash
on empty frame, auto-walk order, probed-set sharing, level reset."""
import sys, os
import numpy as np

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"

import cells as C
from arcengine import GameAction, GameState as GState

# load proposal functions into the cells namespace
src = open("/srv/cellforge-arena/researchers/alpha/gen2g5/cell_proposals_gen4.py").read()
ns = dict(vars(C))
ns["__name__"] = "proposal_test"
exec(compile(src, "proposal", "exec"), ns)
c_Clicking, c_scanning = ns["c_Clicking"], ns["c_scanning"]

class FakeResp:
    def __init__(self): self.state = GState.NOT_FINISHED; self.levels_completed = 0
    @property
    def available_actions(self): return [1, 2, 3, 4, 5, 6, 7]

class FakeSt:
    def __init__(self, frame):
        self._f = frame; self.resp = FakeResp(); self.budget = 100
        self.steps_used = 0; self.level = 0; self.tested = set(); self.prev_frame = None
    def frame(self): return self._f.copy()
    def _step(self, action, data=None):
        self.prev_frame = self._f.copy(); self.steps_used += 1
        return ""

# synthetic frame: two objects (big color-3 square, small color-5 square)
f = np.zeros((64, 64), dtype=np.uint8)
f[10:20, 10:20] = 3
f[40:44, 40:44] = 5

st = FakeSt(f)
print("objs:", [(o["c"], o["x0"], o["y0"]) for o in C._objs(f)])
r1 = c_Clicking(st, {})          # auto-walk -> obj0 (color 3)
print("clickprobe1:", r1[:80]); assert st.steps_used == 1
r2 = c_Clicking(st, {})          # auto-walk -> obj1 (color 5)
print("clickprobe2:", r2[:80]); assert st.steps_used == 2
r3 = c_Clicking(st, {})          # all probed -> fresh pass, obj0 again
print("clickprobe3:", r3[:80]); assert st.steps_used == 3
r4 = c_Clicking(st, {"which": 1})  # explicit which
print("clickprobe4:", r4[:80]); assert "obj1" in r4 and st.steps_used == 4
r5 = c_Clicking(st, {"which": 99}) # out of range -> auto-walk (fresh pass done, obj1 untested in new pass)
print("clickprobe5:", r5[:80])
# scanning on fresh level state
st2 = FakeSt(f)
r6 = c_scanning(st2, {})
print("scan1:", r6[:120]); assert st2.steps_used == 2, st2.steps_used
r7 = c_scanning(st2, {})  # everything probed -> nothing new
print("scan2:", r7[:80]); assert st2.steps_used == 2
# empty frame
st3 = FakeSt(np.zeros((64, 64), dtype=np.uint8))
print("empty probe:", c_Clicking(st3, {})[:60])
print("empty scan:", c_scanning(st3, {})[:60])
# level change resets probed set
st.level = 1
r8 = c_Clicking(st, {})
print("new-level probe:", r8[:80]); assert st.steps_used == 6
print("ALL CELL UNIT TESTS PASSED")
