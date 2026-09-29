import json, sys
r = json.load(open(sys.argv[1]))
if 'bpc' not in r: print(r); sys.exit()
print(r['bpc'], r['top1'], r['dev_evals_used'])
for k, v in r['bits_by_context'].items(): print(' ', k, v)
for w in r['worst_positions']: print(w['bits'], repr(w['context']), w['target'])
for g in r['generation']['sampled_t0.8'][:3]: print(g)
