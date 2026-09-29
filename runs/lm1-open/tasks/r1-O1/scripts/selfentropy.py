"""Entropy of my own model: sample docs from the model itself and score them (no corpus access)."""
import json, sys
import numpy as np
sys.path.insert(0, r"C:\Users\wkeif\Documents\GitHub\CellForge\src")
from cellforge.construction.run import Run
from cellforge.construction.transactions import apply_ops
from cellforge.runtime.lm import score_texts, generate
r = Run(r"C:\Users\wkeif\Documents\GitHub\CellForge\runs\lm1-open")
task = json.load(open(r"..\task.json"))
g = r.load_graph(task["base"])
g, _ = apply_ops(g, json.load(open(sys.argv[1]))["ops"])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
txt, info = generate(g, n, max_len=400, temperature=1.0, seed=7)
res, _ = score_texts(g, txt)
print("self-entropy bpc %.4f  mean len %.1f" % (res["bits"].sum() / res["n_tokens"].sum(), np.mean([len(t) for t in txt])))
