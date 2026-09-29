#!/bin/bash
# usage: mk.sh cfgfile tag  -> builds proposal, runs dev-eval, prints summary
cd /c/Users/wkeif/Documents/GitHub/CellForge
export CELLFORGE_FONT_DIR=$PWD/fonts
T=runs/real1/tasks/a1-architect
python $T/scripts/integrate.py $T/scripts/$1 $T/p_$2.json || exit 1
cellforge dev-eval --run runs/real1 --task a1-architect --proposal $T/p_$2.json > $T/scripts/out_$2.json 2>&1
python - <<PY
import json
d=json.load(open("$T/scripts/out_$2.json"))
print("bpc",d.get("bpc"),"top1",d.get("top1"),"fail",d.get("failures"))
print({k:(v["bpc"],v["n"]) for k,v in d["bits_by_context"].items()})
print(d["exec"]); print(d["route_counts"])
for w in d["worst_positions"][:6]: print(w)
PY
