#!/usr/bin/env python3
"""weight_author.py — DnD-style LoRA delta authoring for router_v3.

NO BACKPROP. The frontier LLM authors directional weight nudges by reasoning
about failure modes; the base model is only MEASURED (hidden-state directions
for context gating), never gradient-updated.

Prompt bundle: model/METHODS.md + model/notes/* + traces_grammar.jsonl stats
  + the three router_v3 failure modes (wasteful, parse-fails, zero transfer).

Authored failure-mode -> mechanism -> delta spec (by the frontier LLM):

  FM1 WASTEFUL (s5i5 L1 in 166 actions vs 13-action grammar solve; RHAE
      punishes quadratically): the router dithers with single click_at calls
      instead of committing to macro cells. Nudge: raise the prior of macro
      cells (act.repeat_click, act.ghost_follow) gated on the "choosing a
      cell name" hidden-state direction (+0.9 nats typical), keep click_at
      available (+0.25), boost digit tokens for repeat counts (+0.45) gated
      on the "emitting an arg value" direction. Plus attention sharpening
      (amplify dominant q/k/v directions 5%): peakier attention -> less
      dithering, more decisive commits.
  FM2 PARSE FAILS (20-50% of CALL generations unparseable): the router rambles
      in THINK and emits sloppy CALL lines. Nudge: make CALL emission reliable
      by boosting the CALL token (+0.6) gated on the "thinking is done" hidden
      direction; tighten the JSON template by boosting structural tokens
      ({ } " : , cell args, +0.35) gated on the generic "inside a CALL line"
      direction. English THINK is untouched (no alignment there -> no shift).
  FM3 ZERO TRANSFER (0.00% on 12 unseen games): not fixable with dev-specific
      nudges; all deltas above are mechanism-level (grammar + macros +
      decisiveness), game-agnostic by construction. Transfer is measured
      honestly; no transfer-game knowledge enters the deltas.

Output: structured JSON {state_dict_key: {"A": [[...]], "B": [[...]]}},
  rank 4, delta = A @ B, W_new = W + delta. Applied by apply_lora.py.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tok import load_vocab, encode, render_trace  # noqa: E402
from gpt import GPT  # noqa: E402

MODEL_DIR = "/srv/arc3-claude/v6run/model"
LM_DIR = os.path.join(MODEL_DIR, "lm")
BASE_PT = os.path.join(LM_DIR, "router_v3.pt")
VOCAB_JSON = os.path.join(LM_DIR, "vocab.json")
METHODS_MD = os.path.join(MODEL_DIR, "METHODS.md")
NOTES_DIR = os.path.join(MODEL_DIR, "notes")
GRAMMAR_TRACES = os.path.join(LM_DIR, "traces_grammar.jsonl")

FAILURE_MODES = """\
FM1 WASTEFUL: s5i5 L1 solved in 166 actions vs 13-action grammar solve (RHAE punishes quadratically).
FM2 PARSE FAILS: 20-50% of CALL generations unparseable; call grammar overfit to seen games.
FM3 ZERO TRANSFER: 0.00% RHAE on 12 unseen games; 0 levels on keyboard games (ls20, g50t)."""

# Authored boost magnitudes (nats of logit shift at typical context alignment).
B_CALL = 0.6      # CALL token when thinking is done
B_STRUCT = 0.35   # JSON structural tokens inside a CALL line
B_MACRO = 0.9     # act.repeat_click / act.ghost_follow at cell-name position
B_CLICK = 0.25    # act.click_at kept available, below macros
B_DIGIT = 0.45    # digit tokens at arg-value position (repeat counts)
SHARPEN_EPS = 0.05  # attention/MLP dominant-direction amplification
RANK = 4


def load_bundle():
    """Assemble the prompt bundle the deltas are authored against."""
    bundle = {"failure_modes": FAILURE_MODES}
    with open(METHODS_MD) as f:
        bundle["methods"] = f.read()
    notes = {}
    for fn in sorted(os.listdir(NOTES_DIR)):
        if fn.endswith(".md"):
            with open(os.path.join(NOTES_DIR, fn)) as f:
                notes[fn] = f.read()
    bundle["notes"] = notes
    traces = []
    with open(GRAMMAR_TRACES) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    bundle["grammar_traces"] = traces
    bundle["stats"] = {
        "n_grammar_traces": len(traces),
        "macro_cells_in_traces": sorted({d["call"]["cell"] for d in traces}),
    }
    return bundle


def hidden_at(model, idx):
    feats = {}

    def hook(m, inp, out):
        feats["h"] = out.detach()

    handle = model.ln_f.register_forward_hook(hook)
    with torch.no_grad():
        model(idx)
    handle.remove()
    return feats["h"][:, -1, :]


def context_directions(model, vocab, stoi, traces):
    """Measure (not train) hidden-state directions for three generation
    contexts. Returns {name: (u[4,128] unit dirs, align[4] typical proj)}."""
    H = {"thinkend": [], "cellname": [], "args": [], "callline": []}
    for d in traces:
        state, think = d["state"], d["think"]
        call = d["call"]
        base = f"STATE {state}\nTHINK {think}"
        # position where the next token should be CALL ("thinking is done")
        H["thinkend"].append(base)
        # position right before the cell-name tokens begin
        H["cellname"].append(base + '\nCALL {"cell":"')
        H["callline"].append(base + '\nCALL {"cell":"')
        # position right before the first numeric arg value
        args = call.get("args", {}) or {}
        numkey = next((k for k, v in args.items()
                       if isinstance(v, int)), None)
        if numkey is not None:
            prefix = (base + "\nCALL " +
                      json.dumps({"cell": call["cell"], "args": {}},
                                 separators=(",", ":"))[:-1] +
                      f'"{numkey}":')
            H["args"].append(prefix)
            H["callline"].append(prefix)
    out = {}
    for name, ctxs in H.items():
        hs = []
        for c in ctxs:
            ids = encode(c, stoi)
            if len(ids) > 192:
                ids = ids[-192:]
            idx = torch.tensor([ids], dtype=torch.long)
            hs.append(hidden_at(model, idx))
        Hm = torch.cat(hs, dim=0).float()  # (n,128)
        U, S, Vh = torch.linalg.svd(Hm, full_matrices=False)
        u = Vh[:RANK]  # (4,128) unit directions
        align = (Hm @ u.T).mean(dim=0)  # typical projection per direction
        out[name] = (u, align)
        print(f"dir {name}: n={Hm.shape[0]} top_sv={S[0]:.2f} "
              f"align={align[0]:.2f}", flush=True)
    return out


def author_deltas(dirs, stoi):
    """The authored numbers. Every magnitude below is a deliberate choice
    documented in the module docstring (failure mode -> mechanism)."""
    u_think, a_think = dirs["thinkend"]
    u_cell, a_cell = dirs["cellname"]
    u_args, a_args = dirs["args"]
    u_line, a_line = dirs["callline"]
    # B rows: [thinkend, cellname, args, callline] top-1 directions
    B = torch.stack([u_think[0], u_cell[0], u_args[0], u_line[0]], dim=0)
    a = torch.tensor([a_think[0], a_cell[0], a_args[0], a_line[0]])

    def row(boosts):
        # boosts: (b_think, b_cell, b_args, b_line) in nats -> calibrated row
        return [b / aj for b, aj in zip(boosts, a.tolist())]

    groups = {
        "CALL": ([stoi["CALL"]], (B_CALL, 0, 0, 0)),
        "STRUCT": ([stoi[t] for t in ["{", "}", '"', ":", ",", "cell", "args"]],
                   (0, 0, 0, B_STRUCT)),
        "MACRO": ([stoi["repeat_click"], stoi["ghost_follow"]],
                  (0, B_MACRO, 0, 0)),
        "CLICK": ([stoi["click_at"]], (0, B_CLICK, 0, 0)),
        "DIGIT": ([stoi[str(i)] for i in range(1, 21)], (0, 0, B_DIGIT, 0)),
    }
    A = torch.zeros(2209, RANK)
    for gname, (tids, boosts) in groups.items():
        r = torch.tensor(row(boosts))
        for t in tids:
            A[t] += r
        print(f"lm_head group {gname}: {len(tids)} tokens row={r.tolist()}",
              flush=True)
    deltas = {"lm_head.weight": {"A": A.tolist(), "B": B.tolist()}}

    # Attention / MLP sharpening: amplify the dominant directions of the
    # already-trained matrices -> peakier attention, more decisive routing.
    ckpt = torch.load(BASE_PT, map_location="cpu", weights_only=False)
    for key in ["blocks.2.attn.c_attn.weight",
                "blocks.3.attn.c_attn.weight",
                "blocks.3.mlp.c_proj.weight"]:
        W = ckpt["state"][key].float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        s = (SHARPEN_EPS * S[:RANK]).sqrt()
        A2 = U[:, :RANK] * s.unsqueeze(0)
        B2 = s.unsqueeze(1) * Vh[:RANK]
        check = A2 @ B2
        rel = check.norm().item() / W.norm().item()
        print(f"sharpen {key}: rel_norm={rel:.4f}", flush=True)
        deltas[key] = {"A": A2.tolist(), "B": B2.tolist()}
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(LM_DIR, "dnd1_deltas.json"))
    a = ap.parse_args()
    torch.manual_seed(0)

    bundle = load_bundle()
    print(f"bundle: methods={len(bundle['methods'])} chars, "
          f"notes={list(bundle['notes'])}", flush=True)
    print(f"bundle: {bundle['stats']}", flush=True)

    vocab, stoi = load_vocab(VOCAB_JSON)
    ckpt = torch.load(BASE_PT, map_location="cpu", weights_only=False)
    model = GPT(**ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()

    dirs = context_directions(model, vocab, stoi, bundle["grammar_traces"])
    deltas = author_deltas(dirs, stoi)
    with open(a.out, "w") as f:
        json.dump(deltas, f)
    n_el = sum(len(v["A"]) * len(v["A"][0]) + len(v["B"]) * len(v["B"][0])
               for v in deltas.values())
    print(f"WROTE {a.out} layers={list(deltas)} delta_params={n_el}",
          flush=True)


if __name__ == "__main__":
    main()
