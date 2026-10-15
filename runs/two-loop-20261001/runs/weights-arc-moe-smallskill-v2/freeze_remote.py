from pathlib import Path
import json,hashlib,shutil
root=Path(__file__).resolve().parent
source=Path("/srv/cellforge-private/arcagi2/data/training")
excluded=set(json.loads((root/"EXCLUDED_REPORTING_IDS.json").read_text()))
paths=sorted((hashlib.sha256(("two-loop-moe-dev-20261001:"+p.stem).encode()).hexdigest(),p) for p in source.glob("*.json") if p.stem not in excluded)[:100]
if len(paths)!=100:raise ValueError("100 existing public training tasks required")
out=root/"data/training";out.mkdir(parents=True)
rows=[]
for _,p in paths:
 raw=p.read_bytes();(out/p.name).write_bytes(raw)
 rows.append({"id":p.stem,"sha256":hashlib.sha256(raw).hexdigest()})
(root/"TASKS.json").write_text(json.dumps({"source":"existing ARC-AGI-2 public training","development":rows,"reporting":[]},sort_keys=True,indent=2))
dependencies={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and p.name not in ("DECLARATION.json","freeze.log")}
(root/"DECLARATION.json").write_text(json.dumps({"schema":"prospective-ARC-MoE/1","dependencies":dependencies,
 "scope":"Independent slower ARC MoE competition, three rounds; no held-out claims, no model APIs"},sort_keys=True,indent=2))
print("MOE_INITIAL_DECLARATION_SHA256",hashlib.sha256((root/"DECLARATION.json").read_bytes()).hexdigest())
print("BOUND_TASKS",len(rows),"BOUND_FILES",len(dependencies))
