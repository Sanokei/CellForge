"""Unit-test a proposal on sentences I type myself (NOT corpus text):
prints per-character bits and the routed expert, to debug the parser FSM."""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from cellforge.construction.run import Run
from cellforge.construction.transactions import validate_proposal
from cellforge.runtime.lm import score_texts

r = Run("runs/lm1") if not hasattr(Run, "open") else Run.open("runs/lm1")
task = json.load(open("runs/lm1/tasks/r1-L2/task.json"))
prop = json.load(open(sys.argv[1]))
prop["base"] = task["base"]
v = validate_proposal(r.load_graph(task["base"]), prop, r.sample_imgs(), r.limits(), kind=r.kind, profile=r.profile)
assert v["ok"], v["error"]
g = v["graph"]
MY_TEXTS = [
    "the girls that this brave king that a mouse follows follows help some cats often. a child likes a red girl.",
    "these dogs near a farmer under a girl jump, and the cat sees every fox quickly. every mouse sleeps.",
    "many birds that the old king chases see the horse near the children. the fox runs.",
]
texts = MY_TEXTS if len(sys.argv) < 3 else [sys.argv[2]]
res, info = score_texts(g, texts)
pb = info["per_pos_bits"]
for i, t in enumerate(texts):
    print(f"doc {i}: bpc {res['bits'][i] / res['n_tokens'][i]:.3f}")
    s = t + "#"
    print("".join(f"{c}" for c in s))
    line = ""
    for j, c in enumerate(s):
        b = pb[i, j]
        line += " " if b < 0.5 else ("." if b < 2 else ("o" if b < 5 else "X"))
    print(line)
    bad = [(j, s[j], round(pb[i, j], 2)) for j in range(len(s)) if pb[i, j] > 4]
    print("  >4 bits:", bad)

if "--route" in sys.argv:
    from cellforge.runtime.interp import Interpreter
    info_e = json.load(open(sys.argv[1].replace(".json", "_info.json")))
    names = list(info_e["experts"])
    t = texts[0]
    vocab = g.meta["vocab"]; ix = {c: i for i, c in enumerate(vocab)}
    toks = [1] + [ix[c] for c in t]
    ctx = np.zeros((1, 8, 32)); state = np.zeros((1, g.meta["state_dim"]))
    for j, tk in enumerate(toks):
        ctx = np.roll(ctx, -1, axis=1); ctx[0, 7] = 0; ctx[0, 7, tk] = 1
        it = Interpreter(g, trace=True)
        out = it.run_bundle(g.entries["step"], {"ctx": ctx, "prev": ctx[:, 7], "step": np.array([[float(j)]]), "state": state}, "/s")
        state = out["state"]
        rk = [k for k in it.trace if k.endswith(".route")][0]
        print(repr(vocab[tk]), names[int(it.trace[rk][0])])
