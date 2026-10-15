import pathlib,json
base=pathlib.Path("/srv/cellforge-inbox/two-loop-20261001");control=base/"control";control.mkdir(exist_ok=True)
root=base/"runs/weights-action-continuation"
anchor="bcfdbec09bc4464c8f147517633e76f831ddf0af34121c41d3592ff27a6caeb9"
phase="9a739609359579ce0aaa47f74d5a56d45a89a9985f06be68e7ef35b0991c8445"
(control/"cell-anchors.json").write_text(json.dumps({"initial_anchor":anchor,"phase_anchors":{"1":phase},"recorded_by_parent_utc":"2026-10-01T15:34:00Z"}),encoding="utf-8")
state=json.loads((root/"STATE-0.json").read_text())
p=state["contexts"][0]["arms"]["llm"];cid=p["live"]["action-arguments-lexical"][0];cell=p["cells"][cid]
(root/"FOUNDER_WEIGHTS.json").write_text(json.dumps({"tensors":cell["params"]}),encoding="utf-8")
standard={"arms":{"llm":{"maker":"configured Codex GPT-6.1 Sol low, offline JSON literal proposals"},"random":{"operator":"full39positions uniform[-9,9], independent survivors"}},
"seeds":[401,402,403],"seeds_declared_before_run":True,"budget_per_arm":3,"niches":["action-arguments-lexical"],
"founders":[{"hash":cid,"niche":"action-arguments-lexical","author":"llm","actor":"imported source-argument-author.json (historical ancestry uncertified)",
"rationale":"Existing raw literal weights imported and independently remeasured under anchored development; no validated historical lineage claim","weights_file":"FOUNDER_WEIGHTS.json"}],
"genomes":[],"proposals":[],"status":"prospective bounded trial, budget incomplete until three generations",
"scope":"small non-grid copied-span tool-use cell competition; not full architecture or ARC transfer","scientific_acceptance":False}
if not (root/"STANDARD.json").exists():(root/"STANDARD.json").write_text(json.dumps(standard,indent=2),encoding="utf-8")
print("parent recorded phase1 controller installed; no evaluation in this script")
