"""Parameter-free primitive operations.

Every tensor flowing through the interpreter carries a leading batch axis.
Axis arguments in graph definitions refer to the *per-example* axes, so the
interpreter shifts non-negative axes by one before calling numpy.  None of
these operations contains recognition logic; they are generic array math.
Numerical constants passed through ``args`` are counted as constant slots by
``ir.stats`` so that no weights hide inside op arguments.
"""
from __future__ import annotations

import numpy as np

EXP_CLIP = 60.0


def _ax(axis: int) -> int:
    return axis + 1 if axis >= 0 else axis


def _unary(fn):
    def op(inputs, args):
        return fn(inputs["x"])
    return op


def _softmax(x, axis, temp):
    z = x / temp
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -EXP_CLIP, EXP_CLIP))
    return e / np.sum(e, axis=axis, keepdims=True)


def op_patchify(inputs, args):
    x = inputs["x"]
    ph, pw = int(args["ph"]), int(args["pw"])
    b, h, w = x.shape
    if h % ph or w % pw:
        raise ValueError(f"patchify: {h}x{w} not divisible by {ph}x{pw}")
    x = x.reshape(b, h // ph, ph, w // pw, pw).transpose(0, 1, 3, 2, 4)
    return x.reshape(b, h // ph, w // pw, ph * pw)


def op_reshape(inputs, args):
    x = inputs["x"]
    return x.reshape((x.shape[0],) + tuple(int(s) for s in args["shape"]))


def op_transpose(inputs, args):
    perm = [0] + [int(p) + 1 for p in args["perm"]]
    return np.transpose(inputs["x"], perm)


def op_add(inputs, args):
    return inputs["a"] + inputs["b"]


def op_sub(inputs, args):
    return inputs["a"] - inputs["b"]


def op_mul(inputs, args):
    return inputs["a"] * inputs["b"]


def op_maximum(inputs, args):
    return np.maximum(inputs["a"], inputs["b"])


def op_scale(inputs, args):
    return inputs["x"] * float(args["c"])


def op_bias(inputs, args):
    return inputs["x"] + float(args["c"])


def op_reduce(kind):
    fn = {"sum": np.sum, "mean": np.mean, "max": np.max, "min": np.min}[kind]

    def op(inputs, args):
        return fn(inputs["x"], axis=_ax(int(args["axis"])), keepdims=bool(args.get("keepdims", False)))
    return op


def op_cumsum(inputs, args):
    return np.cumsum(inputs["x"], axis=_ax(int(args["axis"])))


def op_cummax(inputs, args):
    return np.maximum.accumulate(inputs["x"], axis=_ax(int(args["axis"])))


def op_softmax(inputs, args):
    return _softmax(inputs["x"], _ax(int(args.get("axis", -1))), float(args.get("temp", 1.0)))


def op_matmul(inputs, args):
    return np.matmul(inputs["a"], inputs["b"])


def op_concat(inputs, args):
    keys = sorted((k for k in inputs if k.startswith("x")), key=lambda k: int(k[1:]))
    arrs = [inputs[k] for k in keys]
    b = max(a.shape[0] for a in arrs)
    arrs = [np.broadcast_to(a, (b,) + a.shape[1:]) for a in arrs]
    return np.concatenate(arrs, axis=_ax(int(args["axis"])))


def op_slice(inputs, args):
    x = inputs["x"]
    ax = _ax(int(args["axis"]))
    idx = [slice(None)] * x.ndim
    idx[ax] = slice(args.get("start"), args.get("stop"), args.get("step"))
    return x[tuple(idx)]


def op_pad(inputs, args):
    x = inputs["x"]
    ax = _ax(int(args["axis"]))
    widths = [(0, 0)] * x.ndim
    widths[ax] = (int(args.get("before", 0)), int(args.get("after", 0)))
    return np.pad(x, widths, constant_values=float(args.get("value", 0.0)))


def op_shift(inputs, args):
    """Shift along an axis by k, filling vacated positions with ``value``."""
    x = inputs["x"]
    ax = _ax(int(args["axis"]))
    k = int(args["k"])
    out = np.full_like(x, float(args.get("value", 0.0)))
    n = x.shape[ax]
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim
    if k >= 0:
        src[ax], dst[ax] = slice(0, n - k), slice(k, n)
    else:
        src[ax], dst[ax] = slice(-k, n), slice(0, n + k)
    out[tuple(dst)] = x[tuple(src)]
    return out


def op_coords(inputs, args):
    """Constant coordinate ramp of length n (public shape information)."""
    n = int(args["n"])
    start = float(args.get("start", 0.0))
    step = float(args.get("step", 1.0))
    return (start + step * np.arange(n, dtype=np.float64))[None, :]


def op_onehot_argmax(inputs, args):
    x = inputs["x"]
    ax = _ax(int(args.get("axis", -1)))
    idx = np.argmax(x, axis=ax)
    out = np.zeros_like(x)
    np.put_along_axis(out, np.expand_dims(idx, ax), 1.0, axis=ax)
    return out


def op_expand(inputs, args):
    return np.expand_dims(inputs["x"], _ax(int(args["axis"])))


def op_broadcast_to(inputs, args):
    """Broadcast x against ``like`` (shape-only use of the second input)."""
    x, like = inputs["x"], inputs["like"]
    return np.broadcast_to(x, np.broadcast_shapes(x.shape, like.shape)).copy()


OPS = {
    "patchify": (op_patchify, ["x"]),
    "reshape": (op_reshape, ["x"]),
    "transpose": (op_transpose, ["x"]),
    "add": (op_add, ["a", "b"]),
    "sub": (op_sub, ["a", "b"]),
    "mul": (op_mul, ["a", "b"]),
    "maximum": (op_maximum, ["a", "b"]),
    "scale": (op_scale, ["x"]),
    "bias": (op_bias, ["x"]),
    "relu": (_unary(lambda x: np.maximum(x, 0.0)), ["x"]),
    "tanh": (_unary(np.tanh), ["x"]),
    "sigmoid": (_unary(lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -EXP_CLIP, EXP_CLIP)))), ["x"]),
    "abs": (_unary(np.abs), ["x"]),
    "square": (_unary(np.square), ["x"]),
    "neg": (_unary(np.negative), ["x"]),
    "exp": (_unary(lambda x: np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))), ["x"]),
    "step": (_unary(lambda x: (x > 0).astype(np.float64)), ["x"]),
    "sum": (op_reduce("sum"), ["x"]),
    "mean": (op_reduce("mean"), ["x"]),
    "max": (op_reduce("max"), ["x"]),
    "min": (op_reduce("min"), ["x"]),
    "cumsum": (op_cumsum, ["x"]),
    "cummax": (op_cummax, ["x"]),
    "softmax": (op_softmax, ["x"]),
    "matmul": (op_matmul, ["a", "b"]),
    "concat": (op_concat, None),  # variadic x0..xn
    "slice": (op_slice, ["x"]),
    "pad": (op_pad, ["x"]),
    "shift": (op_shift, ["x"]),
    "coords": (op_coords, []),
    "onehot_argmax": (op_onehot_argmax, ["x"]),
    "expand": (op_expand, ["x"]),
    "broadcast_to": (op_broadcast_to, ["x", "like"]),
}

# Args that carry numerical constants (counted as constant slots).
CONST_ARGS = {"scale": ["c"], "bias": ["c"], "softmax": ["temp"], "pad": ["value"],
              "shift": ["value"], "coords": ["start", "step"]}

ACTIVATIONS = {
    "identity": lambda x: x,
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -EXP_CLIP, EXP_CLIP))),
    "abs": np.abs,
    "square": np.square,
    "step": lambda x: (x > 0).astype(np.float64),
}
