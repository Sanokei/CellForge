import json,glob,sys
f=sorted(glob.glob('runs/real1-open/tasks/s1-solo/evals/*/report.json'))[-1];print(f)
d=json.load(open(f))
print(d['bpc'],d['top1']);print({k:(v['bpc'],v['n']) for k,v in d['bits_by_context'].items()})
for w in d['worst_positions'][:int(sys.argv[1]) if len(sys.argv)>1 else 8]:print(w)
