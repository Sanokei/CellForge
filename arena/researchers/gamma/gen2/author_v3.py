"""gamma-g2 v3 deltas - Authored, Not Trained (rev E).

Rev C/D post-mortem: a novelty direction with small typical projection
needs huge A rows (b/a ~ 4), which inject probe/macro tokens into mid-THINK
generation -> derailed CALL JSON -> parse fails (lf52 10 -> 45). A
think-orthogonalized gate still leaves max|w.h| = 3.3 on think states.
Static LoRA gates cannot be position-specific; any small-a direction is
unsafe on lm_head.

Rev E: ONLY large-a gating directions (top SVD dirs of natural emission
contexts -> small A rows -> no think injection, verified). Two mechanisms:

1. <unk> SUPPRESSION at cell-emission positions. Observed failure: the
   model emits `"act.<unk>"` / `"<unk>.<unk>"` as the cell name -> instant
   parse fail (dnd1: 10-49 parse fails/game). <unk> is NEVER a valid cell
   suffix -> suppressing it is pure win. Gated on cellname + suffix
   position directions, -1.2 nats.

2. PROBE REBALANCE (the transfer play, honest weak version): at the
   suffix position, sweep +0.5 / try +0.4, repeat_click -0.3 /
   ghost_follow -0.3. Net on dev: macros still +0.6 (dnd1's +0.9 minus
   0.3) so answer routines keep winning; probes move from far-behind to
   near-parity, so on unfamiliar contexts the model explores more. No
   novelty detection (proved unsafe); the bet is that a standing probe
   prior beats a standing macro prior on unseen games.

Both use large-a directions only. A-row magnitudes printed and asserted
small. Think-injection probe + dev-6 decide.
"""
import json, os, random, sys
import torch

sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
from tok import load_vocab, encode
from gpt import GPT
from weight_author import hidden_at

LM = '/srv/arc3-claude/v6run/model/lm'
BASE_PT = os.path.join(LM, 'router_v3+dnd1.pt')
OUT = '/srv/cellforge-arena/researchers/gamma/gen2/gamma_v3_deltas.json'
SUF_CELL = '\nCALL {"cell":"'
SUF_SUF = '\nCALL {"cell":"act.'
B_UNK_CELL, B_UNK_SUF = -2.0, -4.0
B_SWEEP, B_TRY = 0.5, 0.4
B_MACRO = -0.3

def main():
    global stoi, model
    torch.manual_seed(0); random.seed(0)
    vocab, stoi = load_vocab(os.path.join(LM, 'vocab.json'))
    ckpt = torch.load(BASE_PT, map_location='cpu', weights_only=False)
    model = GPT(**ckpt['cfg']); model.load_state_dict(ckpt['state']); model.eval()

    def h_at(state, think, suffix):
        ids = encode('STATE ' + state + '\nTHINK ' + think + suffix, stoi)
        if len(ids) > 192: ids = ids[-192:]
        return hidden_at(model, torch.tensor([ids], dtype=torch.long))

    recog = [json.loads(l) for l in open(os.path.join(LM, 'traces_grammar.jsonl')) if l.strip()]
    w2 = [json.loads(l) for l in open(os.path.join(LM, 'traces_w2.jsonl')) if l.strip()]
    random.shuffle(w2)
    Rfit = recog + w2[:1500]
    with torch.no_grad():
        Hc = torch.cat([h_at(dd['state'], dd['think'], SUF_CELL) for dd in Rfit]).float()
        Hs = torch.cat([h_at(dd['state'], dd['think'], SUF_SUF) for dd in Rfit]).float()
    Uc, Sc, Vhc = torch.linalg.svd(Hc, full_matrices=False)
    Us, Ss, Vhs = torch.linalg.svd(Hs, full_matrices=False)
    u_cell, u_suf = Vhc[0].clone(), Vhs[0].clone()
    a_cell = float((Hc @ u_cell).mean())
    a_suf = float((Hs @ u_suf).mean())
    if a_cell < 0:
        u_cell = -u_cell; a_cell = -a_cell
    if a_suf < 0:
        u_suf = -u_suf; a_suf = -a_suf
    print('a_cell=%.2f a_suf=%.2f (large -> small A rows -> safe)' % (a_cell, a_suf), flush=True)
    B = torch.stack([u_cell, u_suf])

    A = torch.zeros(2209, 2)
    unk = stoi['<unk>']
    A[unk, 0] = B_UNK_CELL / a_cell
    A[unk, 1] = B_UNK_SUF / a_suf
    A[stoi['sweep'], 1] = B_SWEEP / a_suf
    A[stoi['try'], 1] = B_TRY / a_suf
    A[stoi['repeat_click'], 1] = B_MACRO / a_suf
    A[stoi['ghost_follow'], 1] = B_MACRO / a_suf
    print('A row max abs = %.4f (must be small)' % float(A.abs().max()), flush=True)
    assert float(A.abs().max()) < 0.5, 'A rows too large - injection risk'
    for tok in ['<unk>', 'sweep', 'try', 'repeat_click', 'ghost_follow']:
        print('  A[%s] = %s' % (tok, [round(float(x), 4) for x in A[stoi[tok]]]), flush=True)

    deltas = {'lm_head.weight': {'A': A.tolist(), 'B': B.tolist()}}
    with open(OUT, 'w') as f: json.dump(deltas, f)
    W = ckpt['state']['lm_head.weight'].float(); dW = A @ B
    print('WROTE %s ||dW||/||W||=%.5f' % (OUT, dW.norm().item() / W.norm().item()), flush=True)

if __name__ == '__main__':
    main()
