"""Tiny numpy MLP policy head: 100 -> 64 -> 32 -> {action(8), click_x(64), click_y(64)}.
~13K params. Manual forward/backward + Adam. No torch needed.
"""
import numpy as np

DIMS = (100, 64, 32)


def init_params(rng):
    ps = {}
    d_in = DIMS[0]
    for i, d_out in enumerate(DIMS[1:], start=1):
        ps[f"W{i}"] = (rng.standard_normal((d_in, d_out)) *
                       np.sqrt(2.0 / d_in)).astype(np.float64)
        ps[f"b{i}"] = np.zeros(d_out, dtype=np.float64)
        d_in = d_out
    # heads
    ps["Wa"] = (rng.standard_normal((d_in, 8)) * np.sqrt(2.0 / d_in)).astype(np.float64)
    ps["ba"] = np.zeros(8, dtype=np.float64)
    ps["Wx"] = (rng.standard_normal((d_in, 64)) * np.sqrt(2.0 / d_in)).astype(np.float64)
    ps["bx"] = np.zeros(64, dtype=np.float64)
    ps["Wy"] = (rng.standard_normal((d_in, 64)) * np.sqrt(2.0 / d_in)).astype(np.float64)
    ps["by"] = np.zeros(64, dtype=np.float64)
    return ps


def param_count(ps):
    return sum(v.size for v in ps.values())


def forward(ps, X):
    """X: (B,100) float64. Returns cache dict with logits."""
    c = {"X": X}
    h = X
    for i in range(1, 3):
        z = h @ ps[f"W{i}"] + ps[f"b{i}"]
        h = np.maximum(z, 0)
        c[f"z{i}"] = z
        c[f"h{i}"] = h
    c["la"] = h @ ps["Wa"] + ps["ba"]
    c["lx"] = h @ ps["Wx"] + ps["bx"]
    c["ly"] = h @ ps["Wy"] + ps["by"]
    return c


def softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def train_step(ps, opt, X, a, cx, cy, lr=3e-3, l2=1e-5):
    """One Adam step. a: (B,) int action; cx,cy: (B,) int, -1 = not a click.
    Returns (loss, acc)."""
    B = X.shape[0]
    c = forward(ps, X)
    pa = softmax(c["la"])
    # action loss
    loss_a = -np.log(pa[np.arange(B), a] + 1e-12).mean()
    da = pa.copy()
    da[np.arange(B), a] -= 1
    da /= B
    # click heads (only on click samples)
    m = (cx >= 0)
    loss_x = loss_y = 0.0
    dx = dy = None
    if m.sum() > 0:
        px = softmax(c["lx"][m])
        py = softmax(c["ly"][m])
        loss_x = -np.log(px[np.arange(m.sum()), cx[m]] + 1e-12).mean()
        loss_y = -np.log(py[np.arange(m.sum()), cy[m]] + 1e-12).mean()
        dx = np.zeros_like(c["lx"]); dy = np.zeros_like(c["ly"])
        ddx = px.copy(); ddx[np.arange(m.sum()), cx[m]] -= 1; ddx /= m.sum()
        ddy = py.copy(); ddy[np.arange(m.sum()), cy[m]] -= 1; ddy /= m.sum()
        dx[m] = ddx; dy[m] = ddy
    loss = loss_a + loss_x + loss_y
    acc = (pa.argmax(axis=1) == a).mean()

    # backward
    h2 = c["h2"]
    gWa = h2.T @ da + l2 * ps["Wa"]; gba = da.sum(axis=0)
    dh = da @ ps["Wa"].T
    if dx is not None:
        gWx = h2.T @ dx + l2 * ps["Wx"]; gbx = dx.sum(axis=0)
        gWy = h2.T @ dy + l2 * ps["Wy"]; gby = dy.sum(axis=0)
        dh = dh + dx @ ps["Wx"].T + dy @ ps["Wy"].T
    else:
        gWx = l2 * ps["Wx"]; gbx = np.zeros(64)
        gWy = l2 * ps["Wy"]; gby = np.zeros(64)
    dh[c["z2"] <= 0] = 0
    gW2 = c["h1"].T @ dh + l2 * ps["W2"]; gb2 = dh.sum(axis=0)
    dh1 = dh @ ps["W2"].T
    dh1[c["z1"] <= 0] = 0
    gW1 = X.T @ dh1 + l2 * ps["W1"]; gb1 = dh1.sum(axis=0)
    grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
             "Wa": gWa, "ba": gba, "Wx": gWx, "bx": gbx,
             "Wy": gWy, "by": gby}
    # Adam
    b1, b2, eps = 0.9, 0.999, 1e-8
    opt["t"] += 1
    t = opt["t"]
    for k in ps:
        g = grads[k]
        opt["m"][k] = b1 * opt["m"][k] + (1 - b1) * g
        opt["v"][k] = b2 * opt["v"][k] + (1 - b2) * g * g
        mh = opt["m"][k] / (1 - b1 ** t)
        vh = opt["v"][k] / (1 - b2 ** t)
        ps[k] -= lr * mh / (np.sqrt(vh) + eps)
    return float(loss), float(acc)


def new_opt(ps):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in ps.items()},
            "v": {k: np.zeros_like(v) for k, v in ps.items()}}


def save(ps, path):
    np.savez(path, **{k: v.astype(np.float32) for k, v in ps.items()})


def load(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def predict(ps, X):
    """X: (B,100). Returns (pa, px, py) probabilities."""
    c = forward(ps, X.astype(np.float64))
    return softmax(c["la"]), softmax(c["lx"]), softmax(c["ly"])
