"""Operational stop-race reproductions; no ARC plays, services or production edits."""
import pathlib,sys,unittest
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import test_review as F
from common import read

class StopDuringLaunch(unittest.TestCase):
    setUp=F.ControllerCrashPoints.setUp
    tearDown=F.ControllerCrashPoints.tearDown
    enable=F.ControllerCrashPoints.enable
    def during_gate(self,action):
        self.enable()
        with patch.object(self.controller,'gates',side_effect=lambda *_:action()):
            self.controller.tick()
        self.assertEqual(self.backend.starts,0,'a stop/disable arriving during gates must be rechecked before launch')
        self.assertIsNone(read(self.root/'REGISTRY.json')['active'])
    def test_stop_marker_arriving_during_gates_prevents_launch(self):
        self.during_gate(lambda:(self.root/'STOP').write_text('fixture STOP'))
    def test_dispatch_removed_during_gates_prevents_launch(self):
        self.during_gate(lambda:(self.root/'DISPATCH_ENABLED').unlink())
    def test_sigterm_arriving_during_gates_prevents_launch(self):
        self.during_gate(lambda:setattr(self.controller,'halt',True))
    def test_stop_arriving_during_launch_is_terminal_before_tick_returns(self):
        self.enable();start=self.backend.start
        def stopped_start(*args):
            result=start(*args)
            self.controller.halt=True
            (self.root/'STOP').write_text('fixture STOP during RPC')
            return result
        with patch.object(self.backend,'start',side_effect=stopped_start):self.controller.tick()
        state=read(self.root/'REGISTRY.json')
        self.assertIsNone(state['active'],'the main loop exits after this tick when halt=True; an owned job must already be cancelled')
        self.assertEqual(state['conditions'][self.fp]['status'],'cancelled')
        self.assertEqual(state['sequence'],1)

if __name__=='__main__':unittest.main(verbosity=2)