"""MNIST-1D splits (procedural generator, seeded). Requires the ``mnist1d`` package.

Authors' machines may hold this package; makers are instructed not to read its
source (it contains the ten ground-truth templates).  Final evaluations draw
fresh signals with an operator-side secret seed on the evaluation server.
"""
from __future__ import annotations

import numpy as np


def make_split(seed: int, num_samples: int, train_split: float, part: str):
    from mnist1d.data import get_dataset_args, make_dataset
    a = get_dataset_args()
    a.seed = seed
    a.num_samples = num_samples
    a.train_split = train_split
    d = make_dataset(a)
    if part == "x":
        return np.asarray(d["x"], dtype=np.float64), np.asarray(d["y"], dtype=np.int64)
    return np.asarray(d["x_test"], dtype=np.float64), np.asarray(d["y_test"], dtype=np.int64)


def split_from_protocol(proto: dict, name: str):
    s = proto[name]
    return make_split(s["seed"], s["num_samples"], s["train_split"], s["part"])
