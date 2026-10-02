"""Held-out transfer check for an ARC cell bank (read-only, no maker). Run on the server with the venv python; reads weights from the head run dir.
Held-out = public ARC TRAINING tasks minus the 100 declared DEV ids minus the reserved SHA256-mod-5 audit population (excluded if EITHER the bare id or
"<id>.json" hashes into it). Evaluation split is never touched."""
import json,hashlib,pathlib,sys
import numpy as np
sys.path.insert(0,"/srv/cellforge-inbox/two-loop-20261001/runs/weights-arc-moe-smallskill-v2/src")
from cellforge.construction.arc_weight_moe import make_core,score
RUN=pathlib.Path("/srv/cellforge-inbox/two-loop-20261001/runs/weights-arc-moe-smallskill-v2")
HEAD=pathlib.Path("/srv/cellforge-inbox/head-20261001/runs/weights-head-arc-cells")
DATA=pathlib.Path("/srv/cellforge-private/arcagi2/data/training")
core=make_core(HEAD/"pop1_core.py")
dev={x["id"] for x in json.load(open(RUN/"TASKS.json"))["development"]}
def audit(tid):  # conservative: exclude if EITHER hashing variant lands in the reserved mod-5 audit population
    return any(int(hashlib.sha256(s.encode()).hexdigest(),16)%5==0 for s in (tid, tid+".json"))
ids=sorted(p.stem for p in DATA.glob("*.json"))
held=[i for i in ids if i not in dev and not audit(i)]
print("training tasks",len(ids),"dev",len(dev),"held-out after audit exclusion",len(held))
rows=[{"id":i,"task":json.load(open(DATA/(i+".json")))} for i in held]
champ=json.load(open(HEAD/"FINAL_WEIGHTS.json"))["tensors"]
base=json.load(open(HEAD/"parent.json"))["tensors"]
zero={k:np.zeros_like(np.asarray(v)).tolist() for k,v in champ.items()}
out={}
for name,w in (("baseline_5_cells",base),("head_bank_plus_3_authored_cells",champ),("zero_weights",zero)):
    r=score(w,rows,core); out[name]=r["solved"]; print(name,r["solved"],"/",len(rows),"oracle_ceiling",r["oracle_solved_diagnostic_only"],"uniform_router",r["uniform_solved"])
    if name.startswith("head"): print("solved ids:",[k for k,v in r["outcomes"].items() if v])
# also confirm dev numbers for the same champion
dev_rows=[{"id":i,"task":json.load(open(DATA/(i+".json")))} for i in sorted(dev)]
r=score(champ,dev_rows,core); print("DEV (authored-on) solved",r["solved"],"/",len(dev_rows),"oracle",r["oracle_solved_diagnostic_only"])
