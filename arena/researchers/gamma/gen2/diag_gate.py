import json, os, random, sys
import torch
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from tok import load_vocab, encode
from gpt import GPT
from weight_author import hidden_at

LM = "/srv/arc3-claude/v6run/model/lm"
torch.manual_seed(0); random.seed(0)
vocab, stoi = load_vocab(os.path.join(LM, "vocab.json"))
ckpt = torch.load(os.path.join(LM, "router_v3+dnd1.pt"), map_location="cpu", weights_only=False)
model = GPT(**ckpt["cfg"]); model.load_state_dict(ckpt["state"]); model.eval()

def hpos(state, think, suffix):
    ctx = f"STATE {state}\nTHINK {think}" + suffix
    ids = encode(ctx, stoi)
    if len(ids) > 192: ids = ids[-192:]
    return hidden_at(model, torch.tensor([ids], dtype=torch.long))

recog = [json.loads(l) for l in open(os.path.join(LM,"traces_grammar.jsonl")) if l.strip()]
w2 = [json.loads(l) for l in open(os.path.join(LM,"traces_w2.jsonl")) if l.strip()]
random.shuffle(w2)
R_all, H_all = recog + w2[:300], w2[300:420]
novels = []
pool = recog + w2[:300]
for d in random.sample(pool, 60):
    toks = d["state"].split(); random.shuffle(toks); novels.append((" ".join(toks), d["think"]))
for d in random.sample(pool, 60):
    e = random.choice(pool)
    if e is not d: novels.append((d["state"], e["think"]))

for pos_name, suffix in [("cellname", '\nCALL {"cell":"'), ("thinkend", ""), ("stateend", "")]:
    # stateend: suffix empty but think removed -> measure pure state novelty
    with torch.no_grad():
        if pos_name == "stateend":
            Hr = torch.cat([hidden_at(model, torch.tensor([encode(f"STATE {d['state']}\n", stoi)[-192:]], dtype=torch.long)) for d in R_all]).float()
            Hh = torch.cat([hidden_at(model, torch.tensor([encode(f"STATE {d['state']}\n", stoi)[-192:]], dtype=torch.long)) for d in H_all]).float()
            Hn = torch.cat([hidden_at(model, torch.tensor([encode(f"STATE {s}\n", stoi)[-192:]], dtype=torch.long)) for s,t in novels]).float()
        else:
            Hr = torch.cat([hpos(d["state"], d["think"], suffix) for d in R_all]).float()
            Hh = torch.cat([hpos(d["state"], d["think"], suffix) for d in H_all]).float()
            Hn = torch.cat([hpos(s, t, suffix) for s,t in novels]).float()
    for k in [4, 12, 32]:
        U,S,Vh = torch.linalg.svd(Hr, full_matrices=False)
        Uk = Vh[:k]
        Rr = Hr - Hr@Uk.T@Uk; Rh = Hh - Hh@Uk.T@Uk; Rn = Hn - Hn@Uk.T@Uk
        # discriminative dir: residualized mean difference
        mu_r, mu_n = Rr.mean(0), Rn.mean(0)
        w = mu_n - mu_r; w = w / w.norm()
        pr, ph, pn = (Rr@w), (Rh@w), (Rn@w)
        dp = abs(pn.mean()-pr.mean())/torch.sqrt((pn.var()+pr.var())/2+1e-9)
        print(f"{pos_name} k={k}: d'={float(dp):.2f} recog_fit={float(pr.mean()):.3f}±{float(pr.std()):.3f} "
              f"heldout={float(ph.mean()):.3f}±{float(ph.std()):.3f} novel={float(pn.mean()):.3f}±{float(pn.std()):.3f}", flush=True)
