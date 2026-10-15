import base64,json,pathlib,hashlib,sys
root=pathlib.Path(".")
plan=json.loads((root/"PLAN.json").read_text())
expected=json.loads((root/"LEGACY_REGISTRY_PREVIEW.json").read_text())
payload={"common":(root/"common.py").read_text(),"plan":plan,
         "legacy_ids":sorted(expected["conditions"]),"desktop_python":sys.version,
         "evaluator_sha256":hashlib.sha256((root/"arc3_env.py").read_bytes()).hexdigest()}
encoded=base64.b64encode(json.dumps(payload,separators=(",",":")).encode()).decode()
runner="""import base64,json,sys,pathlib,hashlib,datetime
payload=json.loads(base64.b64decode(PAYLOAD))
ns={"__name__":"cellforge_cross_version_probe"}
exec(compile(payload["common"],"<review common>","exec"),ns)
plan=payload["plan"]
ns["validate_plan"](plan)
computed=[ns["science_identity"](c) for c in plan["conditions"]]
expected=[c["fingerprint"] for c in plan["conditions"]]
e=pathlib.Path("/srv/cellforge-inbox/continuous-spatial-20261011/arc3_env.py")
assert hashlib.sha256(e.read_bytes()).hexdigest()==payload["evaluator_sha256"]
assert all(c["evaluator_ast"]==ns["normalized_source"](e.read_text()) for c in plan["conditions"])
legacy=ns["import_legacy"](ns["fresh_registry"](),"/srv/cellforge-inbox/continuous-spatial-20261011")
assert sorted(legacy["conditions"])==payload["legacy_ids"]
assert computed==expected
print(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "scope":"read-only metadata validation; zero ARC plays/model calls",
 "desktop_python":payload["desktop_python"],"server_python":sys.version,
 "normalizer":"cellforge-semantic-ast/1, frozen3.10 fields, cellforge-science/3",
 "conditions_validated":len(computed),"all_fingerprints_equal":computed==expected,
 "all_evaluator_bindings_equal":True,"legacy_identities_equal":True,
 "fingerprints":computed,"plan_canonical_sha256":ns["digest"](plan)},sort_keys=True))
"""
(root/"cross_version_probe.py").write_text(runner.replace("PAYLOAD",repr(encoded)),encoding="utf-8")
(root/"CROSS_VERSION_EXPECTED.json").write_text(json.dumps({"desktop_python":sys.version,
 "fingerprints":[c["fingerprint"] for c in plan["conditions"]],"legacy_identities":payload["legacy_ids"]},sort_keys=True)+"\n")
print("Prepared read-only server stream: "+str((root/"cross_version_probe.py").stat().st_size)+" bytes")

