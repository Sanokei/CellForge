#!/usr/bin/env python3
"""Beta G3 TARGETED probe of bred_g1_a25 (elitism root).
Verifies the pieces of the reference audit relevant to beta's direction:
 (a) L0 novelty separators exist (effect sizes),
 (b) dominant dev routine + its attractor neurons (L2/L3 top-200),
 (c) novelty direction in the FINAL residual stream (n_dir) + mean resid (m_dir),
 (d) CALL-emitter breakage on novel states (parse-fail split dev vs tx).
Labels only in outputs. Transfer = MEASURE-ONLY.
"""
import json, os, sys, time, re
import torch
import torch.nn.functional as F

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
import logging
from cells import GameState, call_cell
from tok import load_vocab, encode, decode
from gpt import GPT
from rhae_scorer import environment_score
from arc_agi import Arcade
from arcengine import GameState as GState
# CRITICAL: use the runner's own parse_call (whitespace-strip + REGISTRY
# validation). A local copy silently miscounts parse fails and drops real
# actions, diverging the whole game trajectory.
from fixed_runner import parse_call

os.environ.setdefault("OMP_NUM_THREADS", "1")
WD = "/srv/cellforge-arena/researchers/beta/gen2g3"

DEV = ["lf52", "s5i5", "ls20", "g50t", "tu93", "bp35"]
DEV_LABELS = [f"DEV-{i+1}" for i in range(6)]
TX = [g["game"] for g in
      json.load(open("/srv/cellforge-arena/gen2/judge_transfer_beta.json"))["results"]]
TX_LABELS = [f"TX-{chr(65+i)}" for i in range(len(TX))]

CKPT = "/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt"
ck = torch.load(CKPT, map_location="cpu")
cfg = ck["cfg"]
model = GPT(**{k: cfg[k] for k in
               ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ck["state"])
model.eval()

mlp_cap, res_cap = {}, {}


def _mkmlp(li):
    def h(module, inp, out):
        mlp_cap[li] = F.gelu(out)[:, -1, :].detach()
    return h


def _mkres(li):
    def h(module, inp, out):
        res_cap[li] = out[:, -1, :].detach()
    return h


for li, block in enumerate(model.blocks):
    block.mlp.c_fc.register_forward_hook(_mkmlp(li))
    block.register_forward_hook(_mkres(li))
model.ln_f.register_forward_hook(lambda m, i, o: res_cap.__setitem__("lnf", o[:, -1, :].detach()))


@torch.no_grad()
def generate_hooked(model, idx, max_new_tokens, temperature=0.7, top_k=40):
    mlp_steps, res_steps = [], []
    for _ in range(max_new_tokens):
        idx_c = idx[:, -model.block_size:]
        logits, _ = model(idx_c)
        mlp_steps.append([mlp_cap[li].clone() for li in range(4)])
        res_steps.append(res_cap["lnf"].clone())
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
    return idx, mlp_steps, res_steps


vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
lg = logging.getLogger("q")
lg.addHandler(logging.NullHandler())
lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")


def run_probe(game, seed, budget, maxdec):
    torch.manual_seed(seed)
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == game)
    baselines = [int(x) for x in info.baseline_actions]
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, budget)
    hist, ndec = [], 0
    per_level, prev_lvl, prev_steps = {}, 0, 0
    mlp_acc = [torch.zeros(512) for _ in range(4)]
    res_acc = torch.zeros(128)
    n_steps, n_parse_fail = 0, 0
    while ndec < maxdec and st.steps_used < budget:
        if st.resp.state == GState.WIN:
            break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:])
        prompt += "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > 192:
            idx = idx[:, -192:]
        out, mlp_steps, res_steps = generate_hooked(model, idx, max_new_tokens=64)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        for ms, rs in zip(mlp_steps, res_steps):
            for li in range(4):
                mlp_acc[li] += ms[li][0].cpu()
            res_acc += rs[0].cpu()
            n_steps += 1
        name, args = parse_call(gen)
        if name is None:
            n_parse_fail += 1
            name, args = "perceive.describe", {}
        call_cell(st, name, args)
        lvl = int(st.resp.levels_completed)
        if lvl > prev_lvl:
            for L in range(prev_lvl + 1, lvl + 1):
                per_level[L] = st.steps_used - prev_steps
            prev_lvl, prev_steps = lvl, st.steps_used
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    env_score, _ = environment_score(baselines, per_level)
    return {"rhae": round(env_score * 100, 4), "levels": prev_lvl,
            "n_steps": n_steps, "parse_fail": n_parse_fail,
            "mlp": [(a / max(1, n_steps)).tolist() for a in mlp_acc],
            "res": (res_acc / max(1, n_steps)).tolist()}


results = {}
t0 = time.time()
for label, game in zip(DEV_LABELS, DEV):
    r = run_probe(game, 0, 3000, 30)
    results[label] = r
    print(f"{label}: rhae={r['rhae']} lv={r['levels']} pf={r['parse_fail']} "
          f"steps={r['n_steps']} t={time.time()-t0:.0f}s", flush=True)
for label, game in zip(TX_LABELS, TX):
    r = run_probe(game, 0, 3000, 16)
    results[label] = r
    print(f"{label}: rhae={r['rhae']} lv={r['levels']} pf={r['parse_fail']} "
          f"steps={r['n_steps']} t={time.time()-t0:.0f}s", flush=True)
json.dump({l: {k: results[l][k] for k in ("rhae", "levels", "n_steps", "parse_fail")}
           for l in results},
          open(os.path.join(WD, "probe_summary.json"), "w"), indent=1)

# ---------- analysis ----------
labels = DEV_LABELS + TX_LABELS
MLP = {li: torch.tensor([results[l]["mlp"][li] for l in labels]) for li in range(4)}
RES = torch.tensor([results[l]["res"] for l in labels])  # 18 x 128
dev_idx, tx_idx = list(range(6)), list(range(6, 18))

# (a) L0 novelty separators: effect size per neuron
dm = MLP[0][dev_idx].mean(0); tm = MLP[0][tx_idx].mean(0)
psd = torch.sqrt((MLP[0][dev_idx].var(0) + MLP[0][tx_idx].var(0)) / 2 + 1e-12)
es = ((tm - dm) / psd)
strong = (es.abs() > 2.0).sum().item()
print(f"(a) L0 |effect|>2.0 neurons: {strong}/512, max|es|={es.abs().max():.2f}", flush=True)

# (b) dominant dev routine: which DEV game do TX cluster to (L2+L3 residual cos)
rn = RES / (RES.norm(dim=1, keepdim=True) + 1e-12)
cos = rn @ rn.t()
nn = {}
for t in TX_LABELS:
    ti = labels.index(t)
    nn[t] = max(DEV_LABELS, key=lambda d: cos[ti, labels.index(d)].item())
from collections import Counter
dom_dev, dom_count = Counter(nn.values()).most_common(1)[0]
dom_cos = [round(cos[labels.index(t), labels.index(dom_dev)].item(), 4) for t in TX_LABELS]
print(f"(b) dominant dev routine: {dom_dev} ({dom_count}/12 tx nearest), cos={dom_cos}", flush=True)
# attractor neurons: top-200 by mean act on dominant dev game at L2+L3
dvec = torch.cat([MLP[2][labels.index(dom_dev)], MLP[3][labels.index(dom_dev)]])  # 1024
top200 = torch.topk(dvec, 200).indices.tolist()
attractor = [("blocks.2.mlp.c_fc.weight", i) if i < 512 else ("blocks.3.mlp.c_fc.weight", i - 512)
             for i in top200]

# (c) directions in final residual
n_dir = (RES[tx_idx].mean(0) - RES[dev_idx].mean(0))
n_dir = n_dir / (n_dir.norm() + 1e-12)
m_dir = RES[dev_idx].mean(0)
m_dir = m_dir / (m_dir.norm() + 1e-12)
# novelty-gain direction at L2 hidden (for L2->L3 gate)
l2_nov = MLP[2][tx_idx].mean(0) - MLP[2][dev_idx].mean(0)
l2_nov = l2_nov / (l2_nov.norm() + 1e-12)
print(f"(c) n_dir norm=1 saved; m_dir saved; l2_nov saved", flush=True)

# (d) parse-fail split
dev_pf = sum(results[l]["parse_fail"] for l in DEV_LABELS)
dev_st = sum(results[l]["n_steps"] for l in DEV_LABELS)
tx_pf = sum(results[l]["parse_fail"] for l in TX_LABELS)
tx_st = sum(results[l]["n_steps"] for l in TX_LABELS)
print(f"(d) parse-fail rate dev={dev_pf}/{dev_st}={dev_pf/max(1,dev_st):.3f} "
      f"tx={tx_pf}/{tx_st}={tx_pf/max(1,tx_st):.3f}", flush=True)

torch.save({
    "n_dir": n_dir, "m_dir": m_dir, "l2_nov": l2_nov,
    "attractor": attractor, "dominant_dev": dom_dev,
    "l0_effect": es,
}, os.path.join(WD, "probe_dirs.pt"))
print("PROBE DONE", flush=True)
