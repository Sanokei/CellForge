import pathlib,json,hashlib,tarfile
root=pathlib.Path("/srv/cellforge-inbox/weights-logic-circuit-population-g1-20261001")
r=json.loads((root/"RESULT.json").read_text());started=json.loads((root/"START.json").read_text())
summary={"start":started,"seconds":r["seconds"],"live_founders":r["live_founder_ids"],"contexts":[],"scientific_acceptance":False,
"runtime_acceptance":False,"reason":"Prototype ran before new circuit admission tests; no independent circuit-replay gate yet. Scoped selection provisional, source trials preserved."}
for c in r["contexts"]:
 summary["contexts"].append({"seed":c["seed"],"selected":{arm:p["live"] for arm,p in c["population"].items()},
 "admitted":{arm:sum(e["event"]=="admitted" for e in p["events"]) for arm,p in c["population"].items()},
 "charged_positions":{arm:sum(q["positions"] for q in p["charges"]) for arm,p in c["population"].items()},
 "report":{arm:sum(p["outcomes"].values()) for arm,p in c["reporting"].items()}})
(root/"SUMMARY.json").write_text(json.dumps(summary,indent=2))
(root/"VERIFICATION.json").write_text(json.dumps({"schema":"cellforge-verified-action-trial/1","kind":"logic-circuit-prototype","mechanism_replay":{"passed":False,"errors":["independent circuit replay not implemented","pre-run circuit gate not executed"]},"runtime_acceptance":False,"scientific_acceptance":False,"scientific_headline_allowed":False,"scope":"development-selected numerical topology/glue prototype, not verified runtime admission"}))
h={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and p.name not in ("bundle.tar.gz","FINAL_INTEGRITY.json")}
(root/"FINAL_INTEGRITY.json").write_text(json.dumps(h,indent=2))
with tarfile.open("/srv/cellforge-inbox/logic-circuit-population-results.tar.gz","w:gz") as t:
 for p in root.iterdir():
  if p.name!="bundle.tar.gz":t.add(p,arcname=p.name)
print(json.dumps(summary,indent=2))

