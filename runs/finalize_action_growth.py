import pathlib,json,hashlib,tarfile,ast
root=pathlib.Path("/srv/cellforge-inbox/weights-action-growth-g2-20261001")
assert (root/"GENERATION-2.json").exists() and (root/"REPORTING.json").exists()
manifest=json.loads((root/"MANIFEST.json").read_text())
source=(root/"run.py").read_bytes()
with tarfile.open(root/"bundle.tar.gz") as t:
 member=next(m for m in t.getmembers() if m.name.lstrip("./")=="run.py")
 assert t.extractfile(member).read()==source,"source changed from beforephase1 bundle"
tree=ast.parse(source.decode("utf-8-sig"))
report_ast=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="report" for t in n.targets))
assert isinstance(report_ast,ast.List)
cases=[]
for n in report_ast.elts:
 assert isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="c" and not n.keywords
 args=[ast.literal_eval(a) for a in n.args];assert len(args) in (4,5)
 initial=args[4] if len(args)==5 else {}
 cases.append({"id":args[0],"request":args[1],"initial":initial,"expected":{args[2]:args[3]},"action":"edit_in_place" if initial else "write_file"})
recovery={"schema":"cellforge-reporting-declaration-recovery/1","source_sha256":hashlib.sha256(source).hexdigest(),"source_equals_original_prephase_bundle":True,
"reason":"Duplicate reporting key in MANIFEST constructor overwrote case list with note. Original manifest and results retained. Cases recovered through literal-only AST inspection of prephase run.py, no artifact code execution or score relabeling.",
"cases":cases,"original_manifest_reporting_type":type(manifest["reporting"]).__name__,"pristine_manifest":False}
(root/"REPORTING_MANIFEST_RECOVERY.json").write_text(json.dumps(recovery,indent=2))
(root/"FINAL_SEAL.json").write_text(json.dumps({"stdout_closed_before_hashing":True,"phase2_complete":True,"original_protocol_defect_preserved":True}))
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and p.name not in ("bundle.tar.gz","FINAL_INTEGRITY.json")}
(root/"FINAL_INTEGRITY.json").write_text(json.dumps(hashes,indent=2))
with tarfile.open("/srv/cellforge-inbox/action-growth-results.tar.gz","w:gz") as t:
 for p in root.iterdir():
  if p.name!="bundle.tar.gz":t.add(p,arcname=p.name)
print(json.dumps({"start1":json.loads((root/"START-1.json").read_text()),"start2":json.loads((root/"START-2.json").read_text()),"hashes":len(hashes),"report_cases_recovered":len(cases),"manifest_defect":True,"metrics_not_claimed_until_replay":True},indent=2))

