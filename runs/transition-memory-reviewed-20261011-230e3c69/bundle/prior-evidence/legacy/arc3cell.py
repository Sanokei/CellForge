"""ARC-AGI-3 policy cells: restricted Python written by the maker, plus literal tensors, run in a sandbox worker against a game.

Same safety model as scripts/structcode.py (static AST whitelist, separate worker process, no network via `unshare -n`, rlimits, wall-clock kill), but the
entry point is a step function instead of a grid function:

    def act(params, mem, frame, avail, info):
        # params: dict tensor_name -> float numpy array (the literal weights)
        # mem:    dict that persists across the steps of ONE game run (previous frames, plans, counters, visited keys ...)
        # frame:  2-D numpy int array 64x64, colours 0..15, the settled frame after the last action
        # avail:  list of legal action ids (1..5 are simple actions, 6 is a click at (x, y), 0 is RESET and is issued by the harness only)
        # info:   dict(levels=int, resets=int, step=int, key=str)  key = digest of the frame (use it to detect revisited states)
        # return: (action_id, x, y)   x and y only matter for action 6 (0..63)

Why a policy cell and not a script: every task-specific number (which colour is the agent, what is a wall, step size, the order in which actions are tried,
click targets) must come from `params`. A cell is credited only if (a) it completes more levels than zeroed tensors and than three random-tensor draws at
the same positions, and (b) it also does better than the same cell run BLIND (frame replaced by zeros), so an open-loop replay of memorised actions earns nothing.
"""
import ast
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structcode as X  # noqa: E402

MAX_SOURCE = 20000
MAX_POSITIONS = 256
STEP_TIMEOUT = 4.0
ALLOWED_ATTRS = set(X.ALLOWED_ATTRS) | {"setdefault", "unravel_index", "ravel_multi_index", "bit_count", "argpartition", "lexsort", "cumsum", "any", "all",
                                         "clear", "popitem", "issubset", "issuperset", "isdisjoint", "array_equal", "isclose", "median", "hypot", "bit_length",
                                         "symmetric_difference", "difference_update", "intersection_update", "symmetric_difference_update",
                                         "rint", "trunc", "exp", "log", "log2", "power", "square", "arctan2", "sin", "cos", "tan", "degrees", "radians"}
ALLOWED_NODES = X.ALLOWED_NODES
EXTRA_BUILTINS = ("next", "iter", "pow", "divmod", "filter", "frozenset", "StopIteration")


def restricted_globals():
    import builtins
    g = X._restricted_globals()
    g["__builtins__"].update({n: getattr(builtins, n) for n in EXTRA_BUILTINS if hasattr(builtins, n)})
    return g


def validate(source):
    """Static whitelist. Returns (True, '') or (False, reason). Only function definitions at top level; `act(params, mem, frame, avail, info)` must exist."""
    if not isinstance(source, str) or not source.strip():
        return False, "empty source"
    if len(source) > MAX_SOURCE:
        return False, "source longer than %d characters" % MAX_SOURCE
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return False, "syntax error: %s" % error.msg
    functions = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            return False, "only function definitions are allowed at top level"
        functions[node.name] = node
        if node.decorator_list:
            return False, "decorators are not allowed"
    if "act" not in functions:
        return False, "no function named act"
    if [a.arg for a in functions["act"].args.args] != ["params", "mem", "frame", "avail", "info"]:
        return False, "act must be def act(params, mem, frame, avail, info)"
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return False, "node %s is not allowed" % type(node).__name__
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in X.FORBIDDEN_NAMES:
                return False, "name %r is not allowed" % node.id
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRS:
                return False, "attribute .%s is not allowed" % node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex)):
            return False, "bytes and complex constants are not allowed"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 40:
            return False, "long string constants are not allowed"
        elif isinstance(node, ast.arg) and node.arg.startswith("__"):
            return False, "argument names may not start with __"
    return True, ""


def frame_key(frame):
    return hashlib.blake2b(np.asarray(frame, dtype=np.uint8).tobytes(), digest_size=8).hexdigest()


# ---------------------------------------------------------------- worker (runs in the sandbox process)
def _worker_main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2048 * 1024 ** 2, 2048 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    fn, params, mem = None, None, {}
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["op"] == "load":
                ok, why = validate(req["source"])  # validated again on this side of the boundary
                if not ok:
                    raise ValueError("rejected: " + why)
                g = restricted_globals()
                exec(compile(req["source"], "<policy>", "exec"), g)  # noqa: S102 (validated, restricted globals, sandbox process)
                fn, mem = g["act"], {}
                params = {k: np.array(v, dtype=float) for k, v in req["params"].items()}
                reply = {"id": req["id"], "ok": True}
            elif req["op"] == "act":
                frame = np.array(req["frame"], dtype=int).reshape(64, 64)
                out = fn(params, mem, frame, list(req["avail"]), dict(req["info"]))
                if not (isinstance(out, (tuple, list)) and len(out) == 3):
                    raise ValueError("act must return (action_id, x, y)")
                action, x, y = int(out[0]), int(out[1]), int(out[2])
                reply = {"id": req["id"], "action": action, "x": max(0, min(63, x)), "y": max(0, min(63, y)), "mem": len(mem)}
            else:
                raise ValueError("unknown op")
        except BaseException as error:  # noqa: BLE001 the worker must always answer
            reply = {"id": req["id"], "error": "%s: %s" % (type(error).__name__, str(error)[:200])}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------- client (parent process)
def _pump(proc, lines):
    for line in proc.stdout:
        lines.put(line)
    lines.put(None)


class Policy:
    """One sandboxed worker holding one loaded policy. `step` never raises for anything the policy does; it returns (action, x, y, error)."""

    def __init__(self, source, params, timeout=STEP_TIMEOUT):
        self.source, self.params, self.timeout = source, params, timeout
        self.proc, self.lines, self.n, self.network_isolated, self.error = None, None, 0, False, None
        ok, why = validate(source)
        if not ok:
            self.error = "rejected: " + why
            return
        self._start()
        _, err = self._call({"op": "load", "source": source, "params": {k: np.asarray(v, float).tolist() for k, v in params.items()}})
        if err:
            self.error = err

    def _start(self):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker"]
        if sys.platform.startswith("linux") and shutil.which("unshare"):
            try:
                if subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=5).returncode == 0:
                    cmd = ["unshare", "-n"] + cmd
                    self.network_isolated = True
            except Exception:  # noqa: BLE001
                pass
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        if os.name == "nt":
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                                     cwd=os.path.dirname(os.path.abspath(__file__)))
        self.lines = queue.Queue()
        threading.Thread(target=_pump, args=(self.proc, self.lines), daemon=True).start()

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def _call(self, request):
        if self.proc is None or self.proc.poll() is not None:
            return None, "worker not running"
        self.n += 1
        request["id"] = self.n
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            line = self.lines.get(timeout=self.timeout)
        except (queue.Empty, BrokenPipeError, OSError):
            self.close()
            return None, "timeout or crash (worker killed)"
        if line is None:
            self.close()
            return None, "worker exited"
        reply = json.loads(line)
        return reply, reply.get("error")

    def step(self, frame, avail, info):
        if self.error:
            return None, 0, 0, self.error
        reply, err = self._call({"op": "act", "frame": np.asarray(frame, dtype=int).reshape(-1).tolist(), "avail": list(avail), "info": info})
        if err:
            return None, 0, 0, err
        return reply["action"], reply["x"], reply["y"], None


def zeros_like(params):
    return X.zeros_like(params)


def random_like(params, seed):
    return X.random_like(params, seed)


def check_shapes(shapes, params):
    return X.check_shapes(shapes, params)


def positions(shapes):
    return X.positions(shapes)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
