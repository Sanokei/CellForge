"""Analyze fwd_dev6b.npz: choose the stuck label, derive top-40 stuck neurons.
Selection criterion: neurons separating FUTILE repetition from BOTH rest and
productive-repetition (dev-safe: must not fire on pre-solve routines).
Writes stuck_top40.json: {label, neurons:[{idx,sign,d_futile_rest,d_futile_prod}], sep stats}
"""
import json, numpy as np

WD = "/srv/cellforge-arena/researchers/gamma/gen3"
z = np.load(WD + "/fwd_dev6b.npz")
gidx, didx, neu = z["gidx"], z["didx"], z["neu"].astype(np.float32)
games = [g for g in z["games"]]
dec_map = {}
for l in open(WD + "/decisions_dev6b.jsonl"):
    d = json.loads(l); dec_map[(d["game"], d["dec"])] = d
N = len(gidx)
flab = {}
for lab in ["v1", "v2", "futile", "prodrep", "futile2"]:
    flab[lab] = np.array([bool(dec_map[(games[gidx[i]], int(didx[i]))][lab]) for i in range(N)])

def contrast(pos, neg):
    a = neu[pos]; b = neu[neg]
    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0), b.var(0)
    d = (ma - mb) / np.sqrt((va + vb) / 2 + 1e-12)
    return d, ma, mb

f = flab["futile"]; p = flab["prodrep"]; r = ~(f | p)
print(f"forwards: futile={f.sum()} prodrep={p.sum()} rest={r.sum()}", flush=True)
d_fr, ma_f, mb_r = contrast(f, r)
d_fp, _, mb_p = contrast(f, p)
# safety score: min separation vs rest and vs prodrep
safe = np.minimum(np.abs(d_fr), np.abs(d_fp)) * np.sign(d_fr)
order = np.argsort(-np.abs(safe))
print("top by min(|d_futile_rest|,|d_futile_prodrep|):", flush=True)
for i in order[:15]:
    print(f"  neu {i:3d} d_fr={d_fr[i]:+.2f} d_fp={d_fp[i]:+.2f} "
          f"mean_f={ma_f[i]:+.4f} mean_r={mb_r[i]:+.4f} mean_p={mb_p[i]:+.4f}", flush=True)

top40 = []
for i in order[:40]:
    sgn = 1.0 if d_fr[i] >= 0 else -1.0
    top40.append({"idx": int(i), "sign": sgn,
                  "d_fr": float(d_fr[i]), "d_fp": float(d_fp[i])})
# separability of the summed score S = sum sign_i * neu_i
S = np.stack([t["sign"] * neu[:, t["idx"]] for t in top40]).sum(0)
def st(a): return (float(a.mean()), float(a.std()), int(a.size))
print("S=futile-score separability (mean,std,n):", flush=True)
print("  futile :", st(S[f]), flush=True)
print("  prodrep:", st(S[p]), flush=True)
print("  rest   :", st(S[r]), flush=True)
# also v2-based futile2
f2 = flab["futile2"]
d2, _, _ = contrast(f2, ~(f2 | p))
print("futile2 n=", int(f2.sum()), " top|d|:", sorted(np.abs(d2), reverse=True)[:5], flush=True)

json.dump({"label": "futile(v1)+safety-vs-prodrep", "neurons": top40,
           "sep": {"futile": st(S[f]), "prodrep": st(S[p]), "rest": st(S[r])}},
          open(WD + "/stuck_top40.json", "w"), indent=1)
print("WROTE stuck_top40.json", flush=True)
