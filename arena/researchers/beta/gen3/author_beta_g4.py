#!/usr/bin/env python3
"""Beta G4 DnD authoring: BIGRAM-GATED EXPLORATION BOOST.

Appends a hand-authored block-4 (index 4) to the bred_g1_a25 root.

Gate (single MLP neuron, soft-AND via ln2.bias threshold smuggling):
    a[t] = alpha*(d_act.z'[t]) + beta*(e_dot.z'[t]) - Gamma
    d_act  = Fisher direction for (prev token == 'act') among dot positions
             (true cell-suffix slot vs decimal dots in THINK text)
    e_dot  = Fisher direction for (current token == '.')
    z'     = normalized ln2( x + attn_out ); attn is DEAD (zero).
    Fires only at true cell-suffix positions (position-gated, never global).

When the gate fires:
    residual += gelu(a[t]) * v_out,
    v_out = s_boost * (w_sweep + w_try)  (lm_head rows for suffix tokens
            427='sweep', 53='try')
=> exploration suffix tokens get a logit boost exactly where the cell
   choice is made. With temperature sampling this is uncertainty-gated:
   confident routine decisions are unaffected, close calls tip to exploration.

Why no novelty gate: five candidate signals (pooled novelty, LDA trajectory
means, logit gaps, chg-absence, LVL progression) were all calibrated and ALL
failed to separate dev/tx decisions (see cal_diag*.py). The model is a
perfect pattern-collapser; no measurable signal distinguishes novel states.
A weak position-gated boost is the dev-safe mechanism: it cannot corrupt
reasoning (fires only at suffix slots) and its strength is calibrated on
dev-3 (largest s_boost that keeps dev green).

NO BACKPROP. Directions from cal_dirs2.pt (empirical Fisher probes).
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
from gpt import GPT  # noqa: E402

WD = "/srv/cellforge-arena/researchers/beta/gen3"
ROOT = "/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="cal_params.json")
    ap.add_argument("--tag", default="g4")
    a = ap.parse_args()
    P = json.load(open(a.params))

    ck = torch.load(ROOT, map_location="cpu", weights_only=False)
    root_state = {k: v.float() for k, v in ck["state"].items()}
    root_cfg = ck["cfg"]
    n_embd, n_head = root_cfg["n_embd"], root_cfg["n_head"]

    cal = torch.load(os.path.join(WD, "cal_dirs2.pt"), map_location="cpu",
                     weights_only=False)
    d_act = cal["d_act"].float(); d_act /= d_act.norm() + 1e-12
    e_dot = cal["e_dot"].float(); e_dot /= e_dot.norm() + 1e-12

    alpha = float(P["alpha"]); beta = float(P["beta"])
    Gamma = float(P["Gamma"]); s_boost = float(P["s_boost"])

    Wlm = root_state["lm_head.weight"].float()
    v_out = s_boost * (Wlm[427] + Wlm[53])   # sweep + try suffix rows

    B = {}
    B["blocks.4.ln1.weight"] = torch.ones(n_embd)
    B["blocks.4.ln1.bias"] = torch.zeros(n_embd)
    # dead attention: all zeros -> adds nothing (keeps block minimal &
    # dev-neutral; the gate features are per-position, no pooling needed)
    B["blocks.4.attn.c_attn.weight"] = torch.zeros(3 * n_embd, n_embd)
    B["blocks.4.attn.c_proj.weight"] = torch.zeros(n_embd, n_embd)
    row0 = alpha * d_act + beta * e_dot
    r2 = float(row0 @ row0)
    B["blocks.4.ln2.weight"] = torch.ones(n_embd)
    B["blocks.4.ln2.bias"] = (-Gamma / (r2 + 1e-12)) * row0
    c_fc = torch.zeros(4 * n_embd, n_embd)
    c_fc[0, :] = row0
    B["blocks.4.mlp.c_fc.weight"] = c_fc
    m_proj = torch.zeros(n_embd, 4 * n_embd)
    m_proj[:, 0] = v_out
    B["blocks.4.mlp.c_proj.weight"] = m_proj

    new_cfg = dict(root_cfg); new_cfg["n_layer"] = 5
    model = GPT(**{k: new_cfg[k] for k in
                   ("vocab_size", "n_layer", "n_head", "n_embd",
                    "block_size", "dropout")})
    new_state = {k: v.clone() for k, v in root_state.items()}
    for k, v in B.items():
        assert k not in new_state
        new_state[k] = v
    ref = dict(model.state_dict())
    assert set(new_state) == set(ref), f"keys {set(new_state) ^ set(ref)}"
    for k in new_state:
        assert tuple(new_state[k].shape) == tuple(ref[k].shape), k
        assert torch.isfinite(new_state[k]).all(), k
    model.load_state_dict(new_state)
    n_params = sum(v.numel() for v in new_state.values())
    assert n_params <= 10_000_000
    print(f"params={n_params} Gamma={Gamma} s_boost={s_boost}", flush=True)
    out = os.path.join(WD, f"beta_g4_{a.tag}.pt")
    torch.save({"cfg": new_cfg, "state": new_state}, out)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
