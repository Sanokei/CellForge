import pathlib,json,hashlib,tarfile,datetime
root=pathlib.Path("/srv/cellforge-inbox/weights-action-population-corrected-g3-20261001")
assert (root/"RESULT.json").exists()
(root/"FINAL_SEAL.json").write_text(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"stdout_closed_before_hashing":True,"previous_run_preserved":"weights-action-population-v2-g3","prior_admissions_quarantined":True}))
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and p.name not in ("bundle.tar.gz","FINAL_INTEGRITY.json")}
(root/"FINAL_INTEGRITY.json").write_text(json.dumps(files,indent=2))
with tarfile.open("/srv/cellforge-inbox/action-population-corrected-results.tar.gz","w:gz") as archive:
 for p in root.iterdir():
  if p.name!="bundle.tar.gz":archive.add(p,arcname=p.name)
r=json.loads((root/"RESULT.json").read_text())
print(json.dumps({"pid":json.loads((root/"START.json").read_text())["pid"],"seconds":r["seconds"],"hashes":len(files),"runtime_gate":json.loads((root/"RUNTIME_GATE_RESULT.json").read_text())["passed"],"contexts":[{"seed":c["seed"],"report":c["report_correct"],"router_random":c["router_random"]["accepted"],"arg_random":[s["random_decision"]["accepted"] for s in c["stages"]]} for c in r["contexts"]]},indent=2))

