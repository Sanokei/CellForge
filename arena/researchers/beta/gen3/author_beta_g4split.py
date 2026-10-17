#!/usr/bin/env python3
"""Beta G4 split: boost ONLY sweep (427) or ONLY try (53), to test if one
is less disruptive to dev solves than the combined boost."""
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
    ap.add_argument("--params", required=True)
    ap.add_argument("--which", choices=["sweep", "try"], required=True)
    ap.add_argument("--tag", default="split")
    a = ap.parse_args()
    P = json.load(open(a.params))

    ck = torch.load(ROOT, map_location="cpu", weights_only=False)
    root_state = {k: v.float() for k, v in ck["state"].items()}
    root_cfg = ck["cfg"]
    n_embd = root_cfg["n_embd"]

    cal = torch.load(os.path.join(WD, "cal_dirs2.pt"), map_location="cpu",
                     weights_only=False)
    d_act = cal["d_act"].float(); d_act /= d_act.norm() + 1e-12
    e_dot = cal["e_dot"].float(); e_dot /= e_dot.norm() + 1e-12

    alpha = float(P["alpha"]); beta = float(P["beta"])
    Gamma = float(P["Gamma"]); s_boost = float(P["s_boost"])
    Wlm = root_state["lm_head.weight"].float()
    tok = 427 if a.which == "sweep" else 53
    # scale up 2x to match combined-boost magnitude on the single token
    v_out = s_boost * 2.0 * Wlm[tok]

    B = {}
    B["blocks.4.ln1.weight"] = torch.ones(n_embd)
    B["blocks.4.ln1.bias"] = torch.zeros(n_embd)
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
    assert set(new_state) == set(ref)
    for k in new_state:
        assert tuple(new_state[k].shape) == tuple(ref[k].shape), k
        assert torch.isfinite(new_state[k]).all(), k
    model.load_state_dict(new_state)
    print(f"params={sum(v.numel() for v in new_state.values())} which={a.which}",
          flush=True)
    out = os.path.join(WD, f"beta_g4_{a.tag}.pt")
    torch.save({"cfg": new_cfg, "state": new_state}, out)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
