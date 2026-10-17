#!/usr/bin/env python3
"""gamma-g2 v2 deltas - Authored, Not Trained (rev C).

Zero gradient steps. Single rank-1 LoRA on lm_head.weight: a DISCOVERY GATE.
B = one discriminative novelty direction measured at the cell-suffix emission
position (after CALL {"cell":"act.), residualized against the top-32
recognized-pattern subspace. A rows: probe cells UP (sweep +1.0, try +0.7
nats at typical novelty), habitual macro cells DOWN (repeat_click -0.6,
ghost_follow -0.6): on unrecognized contexts the model explores instead of
committing to the wrong answer routine. On recognized contexts the gate is
~0 by construction (dnd1's answer routines untouched -> anti-forgetting).
Measured, never trained, on the v3+dnd1 composed base. Dev-game text only;
transfer games never touched.
"""
import json, os, random, sys
import torch

sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from tok import load_vocab, encode
from gpt import GPT
from weight_author import hidden_at

LM = "/srv/arc3-claude/v6run/model/lm"
BASE_PT = os.path.join(LM, "router_v3+dnd1.pt")
OUT = "/srv/cellforge-arena/researchers/gamma/gen2/gamma_v2_deltas.json"
SUFFIX = '\nCALL {"cell":"act.'
K_REC = 32
BOOSTS = {"sweep": 1.0, "try": 0.7, "repeat_click": -0.6, "ghost_follow": -0.6}

def main():
    global stoi, model
    torch.manual_seed(0); random.seed(0)
    vocab, stoi = load_vocab(os.path.join(LM, "vocab.json"))
    ckpt = torch.load(BASE_PT, map_location="cpu", weights_only=False)
    model = GPT(**ckpt["cfg"]); model.load_state_dict(ckpt["state"]); model.eval()

    def h_at(state, think):
        ids = encode("STATE " + state + "\nTHINK " + think + SUFFIX, stoi)
        if len(ids) > 192: ids = ids[-192:]
        return hidden_at(model, torch.tensor([ids], dtype=torch.long))

    recog = [json.loads(l) for l in open(os.path.join(LM, "traces_grammar.jsonl")) if l.strip()]
    w2 = [json.loads(l) for l in open(os.path.join(LM, "traces_w2.jsonl")) if l.strip()]
    random.shuffle(w2)
    Rfit, Rhold = recog + w2[:300], w2[300:460]
    pool = recog + w2[:300]
    novels = []
    for d in random.sample(pool, 70):
        toks = d["state"].split(); random.shuffle(toks)
        novels.append((" ".join(toks), d["think"]))
    for d in random.sample(pool, 70):
        e = random.choice(pool)
        if e is not d: novels.append((d["state"], e["think"]))

    with torch.no_grad():
        Hr = torch.cat([h_at(d["state"], d["think"]) for d in Rfit]).float()
        Hh = torch.cat([h_at(d["state"], d["think"]) for d in Rhold]).float()
        Hn = torch.cat([h_at(s, t) for s, t in novels]).float()
    U, S, Vh = torch.linalg.svd(Hr, full_matrices=False)
    Uk = Vh[:K_REC]
    def res(H): return H - H @ Uk.T @ Uk
    Rr, Rh, Rn = res(Hr), res(Hh), res(Hn)

    w = Rn.mean(0) - Rr.mean(0); w = w / w.norm()
    if float((Rn @ w).mean()) < 0: w = -w
    B = w.unsqueeze(0)
    a = float((Rn @ w).mean())
    dp = abs(float((Rn @ w).mean()) - float((Rr @ w).mean())) / float(torch.sqrt((Rn @ w).var() + (Rr @ w).var()) / 2 + 1e-9)
    print("fit: recog n=%d novel n=%d a=%.3f d-prime=%.2f" % (Hr.shape[0], Hn.shape[0], a, dp), flush=True)

    for name, H in [("recog-fit", Hr), ("recog-heldout", Hh), ("novel", Hn)]:
        g = float(((H @ B.T) / a).mean())
        shifts = " ".join("%s%+.2f" % (t, g * b) for t, b in BOOSTS.items())
        print("  %s: mean gate=%.3f shifts: %s" % (name, g, shifts), flush=True)

    A = torch.zeros(2209, 1)
    for tok, b in BOOSTS.items():
        A[stoi[tok], 0] = b / a
    deltas = {"lm_head.weight": {"A": A.tolist(), "B": B.tolist()}}
    with open(OUT, "w") as f: json.dump(deltas, f)
    W = ckpt["state"]["lm_head.weight"].float(); dW = A @ B
    print("WROTE %s ||dW||/||W||=%.5f nonzero_A_rows=%d" %
          (OUT, dW.norm().item() / W.norm().item(), int((A.abs().sum(1) > 0).sum())), flush=True)

if __name__ == "__main__":
    main()
