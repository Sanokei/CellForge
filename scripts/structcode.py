"""Maker-proposed STRUCTURES: restricted Python that the maker writes, validated and run in a sandbox. Owner decision 2026-10-02 ("do all three").

A proposed structure is a parametric function `run(params, grid, demos)`: the code is the architecture, the literal tensors in `params` carry every
task-specific number. It is accepted only if it passes the static whitelist below, reproduces the demonstrations leave-one-out, FAILS with zeroed and with
random tensors (so the tensors really matter), and has at most MAX_POSITIONS tensor entries (so the answer cannot be smuggled in as a lookup table).

Safety layers (model-written code never touches the server unchecked):
  1. static: AST whitelist (no imports, no dunder names or attributes, no file/network/introspection names, numpy functions by name only);
  2. runtime: a separate worker process (cleared environment, no network via `unshare -n` when the OS allows it, address-space/file-size/core limits on
     Linux, a hard wall-clock kill by the parent, output validated to a 2-D grid of 0..9 no larger than 30x30);
  3. the worker never receives secrets, ARC test outputs or any path: only the grid, the other demonstrations and the tensors.
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

MAX_SOURCE = 9000
MAX_POSITIONS = 256
MAX_GRID = 30
TIMEOUT = 5.0

SAFE_BUILTINS = ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "map", "max", "min", "range", "reversed", "round", "set",
                 "sorted", "sum", "tuple", "zip", "isinstance", "True", "False", "None", "IndexError", "ValueError", "KeyError", "ZeroDivisionError")
FORBIDDEN_NAMES = {"exec", "eval", "compile", "open", "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "input", "breakpoint", "memoryview",
                   "bytearray", "bytes", "type", "object", "super", "classmethod", "staticmethod", "property", "help", "exit", "quit", "print", "id", "hash",
                   "os", "sys", "subprocess", "socket", "importlib", "builtins"}
# attributes reachable on `np` and on arrays/lists/dicts/sets; nothing that touches files, memory maps, ctypes, random state or the interpreter
ALLOWED_ATTRS = set("""
shape size ndim T dtype astype copy reshape flatten ravel tolist item any all sum prod max min argmax argmin mean std nonzero sort argsort clip round
cumsum cumprod diagonal trace transpose swapaxes squeeze repeat take fill dot
array asarray zeros ones full empty arange eye zeros_like ones_like full_like where argwhere roll rot90 flip fliplr flipud unique abs absolute
concatenate stack vstack hstack tile kron bincount minimum maximum pad count_nonzero diff isin equal not_equal logical_and logical_or logical_not
logical_xor greater less greater_equal less_equal bitwise_and bitwise_or bitwise_xor invert sign floor ceil sqrt mod floor_divide divide multiply add
subtract linspace meshgrid indices ix_ histogram searchsorted split array_split select choose compress extract triu tril diag flatnonzero
int64 int32 int8 uint8 bool_ float64 pi
append extend insert pop index count remove reverse sort get items keys values update add discard union intersection difference
""".split())
ALLOWED_NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If, ast.Expr, ast.Pass,
                 ast.Break, ast.Continue, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword, ast.Attribute, ast.Subscript,
                 ast.Slice, ast.Tuple, ast.List, ast.Set, ast.Dict, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Name,
                 ast.Constant, ast.Starred, ast.Lambda, ast.Load, ast.Store, ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                 ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.Invert, ast.Not, ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt,
                 ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.MatMult)


def validate(source):
    """Static whitelist. Returns (True, '') or (False, reason). The only top-level statements are function definitions; `run` must exist."""
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
    if "run" not in functions:
        return False, "no function named run"
    if [a.arg for a in functions["run"].args.args] != ["params", "grid", "demos"]:
        return False, "run must be def run(params, grid, demos)"
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return False, "node %s is not allowed" % type(node).__name__
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                return False, "name %r is not allowed" % node.id
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRS:
                return False, "attribute .%s is not allowed" % node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex)):
            return False, "bytes and complex constants are not allowed"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 40:
            return False, "long string constants are not allowed"
        elif isinstance(node, ast.arg) and (node.arg.startswith("__")):
            return False, "argument names may not start with __"
    return True, ""


def uses_demos(source):
    """Does the structure read the other demonstrations? (Only then is it a demonstration-aware template.)"""
    tree = ast.parse(source)
    body_names = {n.id for f in tree.body for n in ast.walk(f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return "demos" in body_names


def positions(shapes):
    return int(sum(int(np.prod(s)) for s in shapes.values()))


def check_shapes(shapes, params):
    if not isinstance(shapes, dict) or not shapes or not all(isinstance(k, str) and isinstance(v, list) and v and all(isinstance(i, int) and i > 0 for i in v)
                                                              for k, v in shapes.items()):
        return False, "shapes must map tensor names to lists of positive ints"
    if positions(shapes) > MAX_POSITIONS:
        return False, "more than %d tensor entries" % MAX_POSITIONS
    if not isinstance(params, dict) or set(params) != set(shapes):
        return False, "params and shapes name different tensors"
    try:
        for k, s in shapes.items():
            arr = np.asarray(params[k], dtype=float)
            if list(arr.shape) != list(s) or not np.isfinite(arr).all() or np.abs(arr).max() > 1e3:
                return False, "tensor %s does not match its shape, is not finite, or exceeds 1e3" % k
    except Exception as error:
        return False, "tensor %s is not numeric: %s" % (k, type(error).__name__)
    return True, ""


# ---------------------------------------------------------------- worker (runs in the sandbox process)
def _restricted_globals():
    import builtins
    safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS if hasattr(builtins, n)}
    return {"__builtins__": safe, "np": np}


def _grid_ok(out):
    arr = np.asarray(out)
    if arr.ndim != 2 or arr.size == 0 or arr.shape[0] > MAX_GRID or arr.shape[1] > MAX_GRID:
        raise ValueError("output must be a non-empty 2-D grid of at most %dx%d" % (MAX_GRID, MAX_GRID))
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr.astype(float)).all() or (arr != np.round(arr)).any() or arr.min() < 0 or arr.max() > 9:
        raise ValueError("output values must be integers 0..9")
    return arr.astype(int).tolist()


def _worker_main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024 ** 2, 1536 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:  # not Linux: the parent's wall-clock kill is the only runtime limit
        pass
    cache = {}
    for line in sys.stdin:
        req = json.loads(line)
        try:
            key = hashlib.sha256(req["source"].encode()).hexdigest()
            if key not in cache:
                ok, why = validate(req["source"])  # validated again on this side of the boundary: the worker trusts nothing it receives
                if not ok:
                    raise ValueError("rejected: " + why)
                g = _restricted_globals()
                exec(compile(req["source"], "<structure>", "exec"), g)  # noqa: S102 (validated, restricted globals, sandbox process)
                cache[key] = g["run"]
            params = {k: np.array(v, dtype=float) for k, v in req["params"].items()}
            demos = [(np.array(d["input"], dtype=int), np.array(d["output"], dtype=int)) for d in req["demos"]]
            out = _grid_ok(cache[key](params, np.array(req["grid"], dtype=int), demos))
            reply = {"id": req["id"], "out": out}
        except BaseException as error:  # noqa: BLE001 the worker must always answer
            reply = {"id": req["id"], "error": "%s: %s" % (type(error).__name__, str(error)[:160])}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------- client (parent process)
def _pump(proc, lines):
    for line in proc.stdout:
        lines.put(line)
    lines.put(None)  # end of stream: the worker exited


class Sandbox:
    """One long-lived worker process; a timeout or crash kills it and the next call starts a fresh one."""

    def __init__(self, timeout=TIMEOUT):
        self.timeout, self.proc, self.lines, self.n = timeout, None, None, 0
        self.network_isolated = False

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
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
        self.lines = queue.Queue()
        threading.Thread(target=_pump, args=(self.proc, self.lines), daemon=True).start()

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def call(self, source, params, grid, demos):
        """Returns (grid, None) or (None, error string). Never raises for anything the structure does."""
        ok, why = validate(source)
        if not ok:
            return None, "rejected: " + why
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        self.n += 1
        req = {"id": self.n, "source": source, "params": {k: np.asarray(v, float).tolist() for k, v in params.items()}, "grid": grid,
               "demos": [{"input": d["input"], "output": d["output"]} for d in demos]}
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.lines.get(timeout=self.timeout)
        except (queue.Empty, BrokenPipeError, OSError):
            self.close()
            return None, "timeout or crash (worker killed)"
        if line is None:
            self.close()
            return None, "worker exited"
        reply = json.loads(line)
        return reply.get("out"), reply.get("error")


def zeros_like(params):
    return {k: np.zeros(np.asarray(v, float).shape).tolist() for k, v in params.items()}


def random_like(params, seed):
    rng = np.random.default_rng(seed)
    return {k: rng.uniform(-9, 9, np.asarray(v, float).shape).tolist() for k, v in params.items()}


def fit(box, source, params, task, extra=None):
    """Leave-one-out fit on the demonstrations: how many demonstrations the structure reproduces exactly when the others are the context."""
    demos = task["train"]
    hits, errors = 0, []
    for i, d in enumerate(demos):
        out, error = box.call(source, params, d["input"], [e for j, e in enumerate(demos) if j != i])
        if error:
            errors.append(error)
        elif out == d["output"]:
            hits += 1
    return hits, errors


def evaluate(box, source, shapes, params, task):
    """Full acceptance test. Returns a dict: valid, demos_fit (of n), zero_fit, random_fit (max over 3 draws), controls_ok, solved_test (None when the
    test outputs are not available), errors. Control outcomes retain arm/seed
    attribution separately; an execution error is never a successful ablation."""
    res = {"valid": False, "demos_fit": 0, "n_demos": len(task["train"]), "zero_fit": None, "random_fit": None, "controls_ok": False, "solved_test": None,
           "errors": [], "control_results": [], "control_errors": []}
    ok, why = validate(source)
    if not ok:
        res["errors"].append(why)
        return res
    ok, why = check_shapes(shapes, params)
    if not ok:
        res["errors"].append(why)
        return res
    res["valid"] = True
    res["demos_fit"], errs = fit(box, source, params, task)
    res["errors"] += errs[:2]
    if res["demos_fit"] == res["n_demos"]:
        controls = [("zero", None, zeros_like(params))]
        controls.extend(("random", seed, random_like(params, seed)) for seed in (11, 12, 13))
        for arm, seed, control_params in controls:
            hits, errors = fit(box, source, control_params, task)
            res["control_results"].append({"arm": arm, "seed": seed, "fit": hits, "errors": errors})
            res["control_errors"].extend({"arm": arm, "seed": seed, "error": error} for error in errors)
        res["zero_fit"] = res["control_results"][0]["fit"]
        res["random_fit"] = max(control["fit"] for control in res["control_results"][1:])
        res["controls_ok"] = (not res["control_errors"] and res["zero_fit"] < res["n_demos"]
                              and res["random_fit"] < res["n_demos"])
        if all("output" in t for t in task["test"]):
            res["solved_test"] = all(box.call(source, params, t["input"], task["train"])[0] == t["output"] for t in task["test"])
    return res


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
