"""Mechanics check on sentences I type myself (not corpus text)."""
import json, sys
import numpy as np
sys.path.insert(0, r"C:\Users\wkeif\Documents\GitHub\CellForge\src")
from cellforge.construction.run import Run
from cellforge.construction.transactions import apply_ops
from cellforge.runtime.lm import score_texts, generate

r = Run(r"C:\Users\wkeif\Documents\GitHub\CellForge\runs\lm1-open")
task = json.load(open(r"..\task.json"))
g = r.load_graph(task["base"])
prop = json.load(open(sys.argv[1]))
g, _ = apply_ops(g, prop["ops"])
texts = sys.argv[2:] if len(sys.argv) > 2 else ["the cat sleeps."]
res, info = score_texts(g, texts)
V = g.meta["vocab"]
for i, t in enumerate(texts):
    pb = info["per_pos_bits"][i][:len(t)+1]
    print(repr(t), "bpc=%.3f" % (res["bits"][i]/res["n_tokens"][i]), "failed", res["failed"][i])
    print(" ".join(f"{(t+'$')[k]}:{pb[k]:.1f}" for k in range(len(t)+1)))
txt, _ = generate(g, 3, temperature=1.0, seed=1)
print(txt)
