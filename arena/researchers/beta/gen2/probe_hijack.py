import sys
sys.path.insert(0, '/srv/arc3-claude/v6run/model')
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
import torch
from gpt_rmt import RMTGPT
from tok import load_vocab, encode

vocab, stoi = load_vocab('/srv/arc3-claude/v6run/model/lm/vocab.json')

def load(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    m = RMTGPT(**ck['cfg'])
    m.load_state_dict(ck['state'])
    m.eval()
    return m

# untrained warm-start vs my v5 trained
untrained = load('/srv/arc3-claude/v6run/model/lm/router_rmt_dnd1_untrained.pt')
trained = load('/srv/cellforge-arena/researchers/beta/gen2/router_beta_g2e.pt')

prompt = ('STATE LVL0 STEP0 OBJ1 c5@0,0-63,62a4032 AVAIL[3, 4, 6, 7]\nTHINK '
          'Probing click at (8,25).\nCALL {"cell":"act.click_at","args":{"x":8,"y":25}}\n')
idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)

for name, model in [('untrained', untrained), ('v5-trained', trained)]:
    with torch.no_grad():
        # evolve memory over 3 rows then measure
        mem = None
        for _ in range(3):
            _, _, mem = model(idx, mem)
        # mem norm
        print('%s: mem_out norm mean=%.3f max=%.3f' % (
            name, mem.norm(dim=-1).mean().item(), mem.norm(dim=-1).max().item()))
        # attention: hook block 0 attn weights, measure text->mem vs text->text
        attn_w = {}
        def hook(mod, inp, out):
            # out is (y, att) if we patch; instead recompute manually below
            pass
        # manual: run first block's attention
        B, T = idx.size()
        n_mem = model.n_mem
        pos = torch.arange(n_mem + T)
        x = torch.cat([mem + model.wpe(pos[:n_mem]).unsqueeze(0),
                       model.wte(idx) + model.wpe(pos[n_mem:]).unsqueeze(0)], dim=1)
        blk = model.blocks[0]
        h = blk.ln1(x)
        # replicate Block.attn forward to get weights
        attn = blk.attn
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(128, dim=2)
        # reshape for 4 heads
        def split_heads(t):
            return t.view(B, -1, 4, 32).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (32 ** 0.5))
        # causal mask
        L = att.size(-1)
        mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        # text positions are n_mem: ; measure their attention to mem (0:8) vs text
        text_att = att[:, :, n_mem:, :]          # (B,H,Tt,L)
        to_mem = text_att[:, :, :, :n_mem].sum(-1).mean().item()
        to_text = text_att[:, :, :, n_mem:].sum(-1).mean().item()
        print('%s: block0 text->mem attn=%.4f text->text attn=%.4f' % (name, to_mem, to_text))
