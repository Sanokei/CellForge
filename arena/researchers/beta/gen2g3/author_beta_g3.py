#!/usr/bin/env python3
"""Beta G3 DnD authoring: audit-cited LoRA deltas on bred_g1_a25.
NO BACKPROP. All deltas authored from the reference audit + targeted probe.

Audit citations:
 [F1] wrong-circuit firing / pattern collapse (tx matches dominant dev routine
      cos 0.96-0.99) -> LEVER C: damp dominant-routine attractor neurons.
 [F2] 169/2048 novelty separators at L0 unread downstream -> LEVER A: wire
      novelty direction to a distinct output branch at lm_head (boost
      reachable explore cells act.sweep/act.try + CALL on novelty, suppress
      macro action cells). NOTE: perceive.* cells are unspellable (<unk>) so
      the distinct branch uses sweep/try, verified by encoding.
 [F4] late-layer wash (novelty strongest L0/L1, drowned by L3) -> LEVER D:
      L2->L3 gate: amplify L2 novelty-separator direction into residual stream
      along n_dir so it survives to the output layer.
 [F5] CALL emitter breaks on novel states (malformed 5%->50%) -> LEVER B:
      state-independent hardening of CALL template tokens at lm_head.

Usage: author_beta_g3.py --a1 0.4 --a2 0.4 --b 0.25 --eps 0.25 --eta 0.7
       --k 48 --tag iter1
"""
import argparse
import json
import os
import sys

import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
from tok import load_vocab, encode  # noqa: E402

WD = "/srv/cellforge-arena/researchers/beta/gen2g3"
ROOT = "/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt"


def tokset(stoi, strings):
    ids = set()
    for s in strings:
        for t in encode(s, stoi):
            ids.add(t)
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", type=float, default=0.4)   # novelty->perceive/CALL boost
    ap.add_argument("--a2", type=float, default=0.4)   # novelty->macro suppress
    ap.add_argument("--b", type=float, default=0.25)   # template harden
    ap.add_argument("--eps", type=float, default=0.25)  # attractor damp factor
    ap.add_argument("--eta", type=float, default=0.7)  # L2->L3 novelty gate gain
    ap.add_argument("--k", type=int, default=48)       # attractor neurons damped
    ap.add_argument("--no-damp", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-branch", action="store_true")
    ap.add_argument("--no-harden", action="store_true")
    ap.add_argument("--tag", default="iter1")
    a = ap.parse_args()

    ck = torch.load(ROOT, map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in ck["state"].items()}
    probe = torch.load(os.path.join(WD, "probe_dirs.pt"), map_location="cpu",
                       weights_only=False)
    n_dir = probe["n_dir"].float()
    m_dir = probe["m_dir"].float()
    l2_nov = probe["l2_nov"].float()
    attractor = probe["attractor"]  # list of (key, idx), L2 then L3 by act
    print(f"dominant dev routine: {probe['dominant_dev']}", flush=True)

    # gate direction: novelty orthogonal to the mean residual direction, so the
    # branch is ~silent on typical dev states and active on novel states.
    g = n_dir - (n_dir @ m_dir) * m_dir
    g = g / (g.norm() + 1e-12)

    vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))

    # Token sets by explicit id (verified by encoding on the box):
    #  - the model can only SPELL 7 cells; all perceive.* and act.key_seq
    #    contain <unk> and are unreachable by generation (the parse-fail
    #    fallback calls perceive.describe by direct string, not by spelling).
    #  - the novelty "distinct branch" must therefore be built from reachable
    #    cells: exploration-like {act.sweep, act.try} vs the dominant macro
    #    routine {act.repeat_click, act.ghost_follow, act.pick_drop}.
    #    act.click_at / act.reset stay neutral (basic primitives).
    STRUCT_IDS = [18, 19, 5, 9, 4, 33, 35]      # { } " : , cell args
    CALL_IDS = [32]                              # CALL
    EXPLORE_IDS = [427, 53]                      # sweep, try
    MACRO_IDS = [1170, 1519, 145]                # repeat_click, ghost_follow, pick_drop
    structural, call_tok = STRUCT_IDS, CALL_IDS
    explore_distinct, macro_distinct = EXPLORE_IDS, MACRO_IDS

    deltas = {}

    # ---- LEVER A+B: lm_head rank-3 ----
    # col0: +a1 * (g.h) on explore/CALL tokens   [F2 distinct branch]
    # col1: -a2 * (g.h) on macro-cell tokens     [F2 suppress wrong routine]
    # col2: +b  * (m.h) on structural tokens     [F5 template hardening]
    V = state["lm_head.weight"].shape[0]
    A = torch.zeros(V, 3)
    if not a.no_branch:
        for t in explore_distinct + call_tok:
            A[t, 0] = a.a1
        for t in macro_distinct:
            A[t, 1] = -a.a2
    if not a.no_harden:
        for t in structural:
            A[t, 2] = a.b
    B = torch.stack([g, g, m_dir], dim=0)  # 3 x 128
    deltas["lm_head.weight"] = (A, B)

    # ---- LEVER C: damp dominant-routine attractor neurons [F1] ----
    if not a.no_damp:
        # group attractor entries by layer key, take top-k overall by order
        by_key = {}
        for key, idx in attractor[: a.k * 2]:
            by_key.setdefault(key, []).append(idx)
        for key, idxs in by_key.items():
            idxs = idxs[: a.k]
            W = state[key]  # (512, 128)
            k = len(idxs)
            A = torch.zeros(W.shape[0], k)
            Bm = torch.zeros(k, W.shape[1])
            for j, i in enumerate(idxs):
                A[i, j] = -a.eps
                Bm[j, :] = W[i, :]
            deltas[key] = (A, Bm)
            print(f"damp {key}: {k} neurons eps={a.eps}", flush=True)

    # ---- LEVER D: L2->L3 novelty gate [F4] ----
    if not a.no_gate:
        # blocks.2.mlp.c_proj: (128, 512); delta = eta * n_dir outer l2_nov
        A = (a.eta * n_dir).unsqueeze(1)   # 128 x 1
        Bm = l2_nov.unsqueeze(0)           # 1 x 512
        deltas["blocks.2.mlp.c_proj.weight"] = (A, Bm)

    # ---- apply ----
    for key, (A, Bm) in deltas.items():
        W = state[key]
        assert A.shape[0] == W.shape[0] and Bm.shape[1] == W.shape[1], \
            f"shape {key}: A{A.shape} B{Bm.shape} W{W.shape}"
        dW = A @ Bm
        assert torch.isfinite(dW).all(), f"non-finite {key}"
        state[key] = W + dW
        print(f"{key}: rank {A.shape[1]} ||dW||/||W|| = "
              f"{dW.norm().item() / W.norm().item():.4f}", flush=True)

    out = os.path.join(WD, f"beta_g3_{a.tag}.pt")
    torch.save({"cfg": ck["cfg"], "state": state}, out)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
