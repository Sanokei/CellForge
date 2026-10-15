"""Freeze the initial nine-experiment study. Run only before supervisor activation."""
import hashlib
import json
import pathlib
from supervisor import CODE, atomic, digest, filehash, fingerprint

root = pathlib.Path(__file__).resolve().parent
if (root / "PLAN.json").exists():
    raise SystemExit("existing plan preserved; preparation refuses overwrite")
source = "def act" + (root / "policy_v2.py").read_text().split("def act", 1)[1]
hypotheses = {
    "rare_region_exploration": {
        "action": [0, 6, 5, 2, 1, 4, 3, 1],
        "pixel": [6, 1, 2, 6, 0, 0], "rates": [0, 1]},
    "boundary_exploration": {
        "action": [0, 6, 5, 2, 1, 4, 3, 1],
        "pixel": [2, 6, 2, 6, 0, 0], "rates": [0, 1]},
    "stagnation_recovery": {
        "action": [0, 3, 8, 4, 3, 8, 6, 1],
        "pixel": [4, 3, 4, 8, 0, 0], "rates": [0, 2]}}
files = {}
base = pathlib.Path("/srv/arc3-claude/environment_files")
for game in ("ar25", "lp85", "sc25"):
    directories = [p for p in base.iterdir() if p.is_dir() and (p.name == game or p.name.startswith(game + "-"))]
    if len(directories) != 1:
        raise SystemExit("ambiguous development environment: " + game)
    for path in sorted(directories[0].rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(path)] = filehash(path)
if not files:
    raise SystemExit("missing official offline development environments")
code = {name: filehash(root / name) for name in CODE}
stops = {}
for parent in (pathlib.Path("/srv/cellforge-inbox/head-20261001"),
               pathlib.Path("/srv/cellforge-inbox/two-loop-20261001"),
               pathlib.Path("/srv/cellforge-inbox/research-20261008/constraints")):
    for path in parent.rglob("STOP"):
        if path.is_file():
            stops[str(path)] = filehash(path)
provenance = {
    "author": "Codex assistant in the current user-authorized conversation",
    "source_thread_id": "01a118b6-0d88-716e-a8e2-859c0d95701e",
    "kind": "explicit assistant-authored literals; not a remote maker completion",
    "actual_platform_context_sha256": None,
    "protocol_sha256": filehash(root / "PROTOCOL.md"),
    "remote_author_calls": 0, "external_paid_API_calls": 0}
plan = {"schema": "cellforge-spatial-study/1", "hypotheses": list(hypotheses),
        "code_sha256": code, "dataset_files": files, "dataset_sha256": digest(files),
        "protected_stops": stops, "remote_maker_call_allowance": 0,
        "scientific_context": "public-development diagnostics; no held-out/general-capability claim",
        "experiments": [], "study_cap": 9, "automatic_expansion": False,
        "quota_policy": "no remote calls; any future maker route requires fresh account RPC and existing 85% cap"}
for seed in (0, 2, 3):
    for index, (name, params) in enumerate(hypotheses.items()):
        config = {"hypothesis": name, "params": params, "games": ["ar25", "lp85", "sc25"],
                  "seed": seed, "control_seed": 101 + seed * 3 + index,
                  "actions": 1500, "remote_maker_calls": 0,
                  "code_sha256": code, "dataset_sha256": plan["dataset_sha256"],
                  "policy_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "provenance": provenance}
        config["fingerprint"] = fingerprint(config)
        plan["experiments"].append(config)
(root / "runs").mkdir(exist_ok=True)
(root / "cancel").mkdir(exist_ok=True)
(root / "notes").mkdir(exist_ok=True)
atomic(root / "PLAN.json", plan)
print(json.dumps({"experiments": len(plan["experiments"]), "code_sha256": code,
                  "protected_stops": len(stops), "dataset_files": len(files)}, indent=2))

