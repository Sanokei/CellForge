import sys
sys.path.insert(0, '/srv/arc3-claude/v6run/model')
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
import torch, json
from gpt_rmt import RMTGPT, rmt_generate
from tok import load_vocab, encode, decode
from fixed_runner import parse_call

vocab, stoi = load_vocab('/srv/arc3-claude/v6run/model/lm/vocab.json')
rck = torch.load('/srv/cellforge-arena/researchers/beta/gen2/router_beta_g2c.pt',
                 map_location='cpu', weights_only=False)
rmt = RMTGPT(**rck['cfg'])
rmt.load_state_dict(rck['state'])
rmt.eval()

rows = [json.loads(l) for l in open('/srv/arc3-claude/v6run/model/lm/traces_fc.jsonl')][:5]

def triple(d):
    return ("STATE " + d['state'] + "\nTHINK " + d['think'] + "\nCALL " +
            json.dumps(d['call'], separators=(',', ':')) + "\n")

prompt = 'STATE LVL0 STEP0 OBJ1 c5@0,0-63,62a4032 AVAIL[3, 4, 6, 7]\nTHINK '
idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)

for mem_kind in ['fresh', 'evolved']:
    ok = 0
    for s in range(8):
        torch.manual_seed(200 + s)
        with torch.no_grad():
            mem = None
            if mem_kind == 'evolved':
                for d in rows:
                    t = torch.tensor([encode(triple(d), stoi)], dtype=torch.long)
                    _, _, mem = rmt(t, mem)
            out, _ = rmt_generate(rmt, mem, idx, max_new_tokens=64,
                                  temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        nm, ag = parse_call(gen)
        if nm:
            ok += 1
        elif s < 2:
            print('  %s FAIL ex: %r' % (mem_kind, gen[:150]))
    print('%s memory: parsed %d/8' % (mem_kind, ok))
