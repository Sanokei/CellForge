"""Read-only engineering decision gate; never promotes or admits a plan."""
import argparse
import pathlib
from common import *
from worker import compare_complete

COMPARATORS=("zero","matched_random","action_off","memory_off","context_off","blind","frozen_spatial","scripted_explorer")
def decision(plan,results):
    output=[]
    for f in plan["founders"]:
        configs=[c for c in plan["conditions"] if c["label"]==f["name"]]
        complete=len(configs)==3 and all(c["fingerprint"] in results and compare_complete(c,results[c["fingerprint"]]) for c in configs)
        gains={g:[] for g in GAMES}; no_regression=True
        if complete:
            for c in configs:
                by={(r["arm"],r["result"]["game"]):r for r in results[c["fingerprint"]]}
                for game in GAMES:
                    full=by["full",game]
                    refs=[by[a,game] for a in COMPARATORS]
                    multiple=full["max_legal_actions"]>=2
                    levels=full["result"]["levels"]
                    gains[game].append(bool(multiple and all(levels>r["result"]["levels"] for r in refs)))
                    no_regression &= levels>=by["frozen_spatial",game]["result"]["levels"]
        passed=complete and no_regression and any(sum(v)>=2 for v in gains.values())
        output.append({"founder":f["name"],"comparisons_complete":complete,"paired_multiaction_gains":gains,
                       "no_reference_regression":no_regression if complete else None,"engineering_gate_passed":passed})
    return {"qualified":False,"automatic_promotion":False,"founders":output}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle",type=pathlib.Path,required=True); p.add_argument("--runs",type=pathlib.Path,required=True)
    args=p.parse_args(); plan=read(args.bundle/"PLAN.json"); results={}
    for c in plan["conditions"]:
        path=args.runs/"jobs"/c["fingerprint"]/"RESULTS.json"
        if path.exists(): results[c["fingerprint"]]=read(path)
    print(canonical(decision(plan,results)).decode())
if __name__=="__main__": main()

