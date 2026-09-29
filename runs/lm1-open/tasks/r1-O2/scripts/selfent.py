"""Sample documents from my own typed grammar (no corpus access) and report its
self-entropy in bits/char, as a reference floor if the typed grammar were exact."""
import math, random, sys
sys.path.insert(0, 'scripts')
import build as b
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
tot_bits = 0; tot_chars = 0
for _ in range(2000):
    ph, st, ci = ('CS', '-', '-'), (), 0
    text = ''
    bits = 0
    while True:
        ud = b.unit_dist(ph, st, ci)
        units = list(ud.items())
        r = random.random() * sum(q for _, q in units)
        for (u, q) in units:
            r -= q
            if r <= 0: break
        w, term = u
        bits -= math.log2(q)
        text += w + term
        cls = b.WORDCLASS[w]
        tk = b.topkind(st)
        st = b.apply_op(st, b.op_of(ph, cls, term))
        ci2 = b.trans_ci(ci, cls, term)
        ph = b.trans_phase(ph, cls, term, tk); ci = ci2
        if term == '.':
            L = len(text)
            pe = 1 / (1 + math.exp(-b.EOS_SLOPE * (L - b.EOS_THR)))
            if L >= 150:   # my current guess of the generator's stopping rule
                bits -= math.log2(pe); break
            bits -= math.log2(1 - pe)
            text += ' '
    tot_bits += bits; tot_chars += len(text) + 1
print('self bpc %.4f' % (tot_bits / tot_chars), 'avg len', tot_chars / 2000)
print(text)
