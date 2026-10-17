"""Train the tiny policy head with behavioral cloning.

Usage: python train.py <out.npz> <game1> <game2> ...
Loads model/data/<game>.npz, featurizes, trains MLP, saves weights.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import batch_features
from mlp import init_params, train_step, new_opt, save, param_count

BASE = "/srv/arc3-claude/v6run/model"


def main():
    out = sys.argv[1]
    games = sys.argv[2:]
    Xs, As, CXs, CYs = [], [], [], []
    for g in games:
        z = np.load(os.path.join(BASE, "data", f"{g}.npz"))
        frames = z["frames"]
        av = z["avails"]
        avails = [row[row >= 0].tolist() for row in av]
        acts = z["acts"]
        keep = acts != 0  # RESET is a hardcoded rule at inference; drop it
        frames, acts = frames[keep], acts[keep]
        avails = [avails[i] for i in np.nonzero(keep)[0]]
        cx, cy = z["cxs"][keep], z["cys"][keep]
        print(f"{g}: {len(acts)} samples", flush=True)
        if len(acts) == 0:
            continue
        t0 = time.time()
        Xs.append(batch_features(frames, avails))
        print(f"  featurized in {time.time()-t0:.1f}s", flush=True)
        As.append(acts)
        CXs.append(cx)
        CYs.append(cy)
    X = np.concatenate(Xs).astype(np.float64)
    A = np.concatenate(As)
    CX = np.concatenate(CXs)
    CY = np.concatenate(CYs)
    n = len(A)
    print(f"total {n} samples, X {X.shape}", flush=True)
    uniq, cnt = np.unique(A, return_counts=True)
    print("action dist:", dict(zip(uniq.tolist(), cnt.tolist())), flush=True)

    rng = np.random.default_rng(0)
    ps = init_params(rng)
    print(f"params: {param_count(ps)}", flush=True)
    opt = new_opt(ps)
    bs, epochs, lr = 1024, 25, 3e-3
    for ep in range(epochs):
        idx = rng.permutation(n)
        tl, ta, nb = 0.0, 0.0, 0
        for s in range(0, n, bs):
            b = idx[s:s+bs]
            l, a_ = train_step(ps, opt, X[b], A[b], CX[b], CY[b], lr=lr)
            tl += l * len(b); ta += a_ * len(b); nb += len(b)
        print(f"ep {ep+1}/{epochs} loss {tl/nb:.4f} acc {ta/nb:.4f}", flush=True)
    save(ps, out)
    meta = {"games": games, "n_samples": n, "params": int(param_count(ps)),
            "epochs": epochs}
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
