"""Independent replay of bounded action trials; never executes artifact code."""
import hashlib,json,pathlib,tempfile
from .action_argument_cell import infer,execute_packet

SCHEMA="cellforge-verified-action-trial/1"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(run_dir):
    root=pathlib.Path(run_dir); errors=[]; binding={}
    try:
        manifest=json.loads((root/"MANIFEST.json").read_text(encoding="utf-8-sig"))
        if manifest.get("schema")=="bounded-action-growth/1":
            from .action_growth_verification import verify_growth
            return verify_growth(root)
        if manifest.get("schema")=="action-population-manifest/1":
            from .action_population_verification import verify_population
            return verify_population(root)
    except (OSError,ValueError):pass
    names=("MANIFEST.json","FROZEN.json","baseline.json","proposal.json","RESULT.json")
    try:
        data={name:json.loads((root/name).read_text(encoding="utf-8-sig")) for name in names}
        binding={name:sha(root/name) for name in names}
        integrity=json.loads((root/"INTEGRITY.json").read_text(encoding="utf-8-sig"))
        if any(integrity.get(n)!=binding[n] for n in names):errors.append("stale_or_tampered_integrity")
        cases=data["MANIFEST.json"]["cases"]; candidates=data["FROZEN.json"]; actual=data["RESULT.json"]["results"]
        if not 1<=len(cases)<=100 or not 1<=len(candidates)<=20:raise ValueError("bounded replay required")
        if len(actual)!=len(candidates):raise ValueError("candidate count differs")
        for candidate,record in zip(candidates,actual):
            if (record["arm"],record["seed"])!=(candidate["arm"],candidate["seed"]):raise ValueError("candidate identity differs")
            expected_rows=[]
            for case in cases:
                with tempfile.TemporaryDirectory(prefix="action_verify_") as d:
                    p=pathlib.Path(d)
                    for name,body in case["initial"].items():
                        if pathlib.Path(name).name!=name or "/" in name or "\\" in name or ".." in name:raise ValueError("fixture initial path invalid")
                        (p/name).write_text(body,encoding="utf-8")
                    prediction=infer(case["request"],data["baseline.json"]["router"],candidate["W"],candidate["thresholds"],legacy=candidate["legacy"])
                    execution=execute_packet(prediction["packet"],allowed_root=p)
                    effects={f.name:f.read_text(encoding="utf-8") for f in p.iterdir() if f.is_file()}
                    expected_rows.append({"id":case["id"],"split":case["split"],"action_correct":prediction["action"]==case["action"],
                       "state_correct":effects==case["expected"],"prediction":prediction,"execution":execution,"actual":effects})
            if record["rows"]!=expected_rows:errors.append("prediction_or_effect_replay_differs:"+str(candidate["arm"])+":"+str(candidate["seed"]))
            scores={split:{"state_correct":sum(r["state_correct"] for r in expected_rows if r["split"]==split),
                          "action_correct":sum(r["action_correct"] for r in expected_rows if r["split"]==split),
                          "total":sum(r["split"]==split for r in expected_rows)} for split in ("development","reporting")}
            if scores!=record["scores"]:errors.append("score_replay_differs")
        from . import action_argument_cell,action_packet
        binding["source:action_argument_cell.py"]=sha(pathlib.Path(action_argument_cell.__file__))
        binding["source:action_packet.py"]=sha(pathlib.Path(action_packet.__file__))
    except (OSError,ValueError,KeyError,TypeError) as e:
        errors.append("missing_or_invalid_evidence:"+str(e))
    return {"schema":SCHEMA,"artifact_binding":binding,"mechanism_replay":{"passed":not errors,"errors":errors},
            "scientific_acceptance":False,"scientific_headline_allowed":False,
            "scope":"Copied lexical spans and fixture effects only; no independent-population or ARC acceptance",
            "verification_method":"Known installed tensor inference and restricted fixture executor replay; artifact code never executed"}

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("run_dir");args=parser.parse_args()
    receipt=verify(args.run_dir);print(json.dumps(receipt,indent=2))
    return 0 if receipt["mechanism_replay"]["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
