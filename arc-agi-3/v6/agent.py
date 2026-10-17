"""ARC-AGI-3 agent contract adapter: is_done(frames, latest_frame) / choose_action(frames, latest_frame).

The v5 explorer + v6 director is written as a loop that *pulls* observations (env.step). The contract
is push-style, so the loop runs in a daemon thread against a proxy env; each choose_action call hands the
newest frame to the thread and returns the action it requests next. Behaviour is identical to run.py.
"""
from __future__ import annotations

import os
import queue
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


class _ProxyEnv:
    def __init__(self, first, to_agent, from_agent):
        self.observation_space = first
        self._q_out, self._q_in = from_agent, to_agent

    def step(self, action, data=None):
        self._q_out.put((action, data))
        nxt = self._q_in.get()
        if nxt is None:
            raise SystemExit
        return nxt


class Arc3AgentV6:
    def __init__(self, budget: int = 3000, seed: int = 0, flags=("pr", "nm", "ck", "gv", "d6")):
        self.budget, self.seed, self.flags = budget, seed, list(flags)
        self._in, self._out = queue.Queue(), queue.Queue()
        self._thread = None
        self.actions = 0

    def is_done(self, frames, latest_frame) -> bool:
        from arcengine import GameState
        return latest_frame.state == GameState.WIN or self.actions >= self.budget

    def choose_action(self, frames, latest_frame):
        from arcengine import GameAction
        from explorer5 import Explorer
        if self._thread is None:
            env = _ProxyEnv(latest_frame, self._in, self._out)

            def run():
                try:
                    Explorer(env, 10 ** 9, seed=self.seed, flags=self.flags).run()
                except SystemExit:
                    pass
            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()
        else:
            self._in.put(latest_frame)
        action, data = self._out.get()
        self.actions += 1
        if data:
            action.set_data(data)
        return action

    def close(self):
        if self._thread is not None:
            self._in.put(None)
