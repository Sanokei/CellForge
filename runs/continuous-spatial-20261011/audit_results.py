"""Read-only result interpretation. Writes an independent completeness audit."""
import importlib.metadata
import json
import pathlib
import sys
from supervisor import atomic, command, filehash, now, read

root=pathlib.Path(__file__).resolve().parent
audit={"captured_utc":now(),"scientific_scope":"public development only",
       "scoring_source_sha256":filehash(root/"arc3_env.py"),"runs":[]}
for run in sorted((root/"runs").iterdir()):
    if not (run/"CONFIG.json").exists():
        continue
    config=read(run/"CONFIG.json")
    rows=read(run/"RESULTS.json") if (run/"RESULTS.json").exists() else []
    terminal=read(run/"TERMINAL.json") if (run/"TERMINAL.json").exists() else None
    expected={(a,g) for a in ("assistant_authored","matched_random","zero","spatial_off","action_off","blind")
              for g in config["games"]}
    actual={(r["arm"],r["result"]["game"]) for r in rows}
    binding=all(r["result"]["seed"]==config["seed"] and r["result"]["budget"]==config["actions"] for r in rows)
    played=all(not r["result"]["error"] and
               (r["result"]["used"]==config["actions"] or r["result"]["state"]=="WIN") for r in rows)
    complete=bool(terminal and terminal["outcome"]=="completed" and actual==expected and
                  len(rows)==len(expected) and binding and played)
    item={"fingerprint":config["fingerprint"],"hypothesis":config["hypothesis"],"seed":config["seed"],
          "dataset_sha256":config["dataset_sha256"],"action_budget":config["actions"],
          "terminal_outcome":terminal["outcome"] if terminal else "in_progress",
          "comparison_complete":complete,"matching_seed_budget_scoring":binding,
          "missing_comparisons":[list(pair) for pair in sorted(expected-actual)],
          "scores":[]}
    for row in rows:
        result=row["result"]
        item["scores"].append({"arm":row["arm"],"game":result["game"],
               "levels":result["levels"],"noops":result["noops"],"distinct":result["distinct"],
               "used":result["used"],"budget":result["budget"],"seed":result["seed"],
               "error":result["error"],"state":result["state"],
               "valid_budget_completion":not result["error"] and
                        (result["used"]==result["budget"] or result["state"]=="WIN")})
    if not complete:
        item["interpretation"]="INCOMPLETE: no fair complete-arm capability comparison"
    else:
        item["interpretation"]="complete development diagnostic; no qualification or promotion"
    audit["runs"].append(item)
atomic(root/"COMPARISON_AUDIT.json",audit)
versions={}
for package in ("numpy","arc-agi","arcengine"):
    try:
        versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package]="unavailable"
atomic(root/"ENVIRONMENT.json",{"utc":now(),"python":sys.version,"packages":versions})
# Finish the crash-recovery proof after the previous invocation has a terminal receipt.
if (root/"RESTART_BEFORE.json").exists():
    before=read(root/"RESTART_BEFORE.json")
    fp=before["state"]["active"]
    receipt=root/"runs"/fp/"TERMINAL.json"
    if receipt.exists():
        terminal=read(receipt)
        events=[json.loads(line) for line in (root/"EVENTS.jsonl").read_text().splitlines()]
        count=sum(e["kind"]=="launched" and e.get("fingerprint")==fp for e in events)
        proof={"captured_utc":now(),"recovered_fingerprint":fp,"launch_count":count,
               "old_invocation":before["job"]["InvocationID"],
               "terminal_invocation":terminal["systemd"].get("InvocationID"),
               "same_recovered_invocation":before["job"]["InvocationID"]==terminal["systemd"].get("InvocationID"),
               "outcome":terminal["outcome"],
               "supervisor":command(["systemctl","show","cellforge-continuous.service",
                                    "--property=MainPID,NRestarts,ActiveEnterTimestamp"]).stdout}
        assert proof["same_recovered_invocation"] and count==1,proof
        atomic(root/"RESTART_COMPLETED_EVIDENCE.json",proof)
print(json.dumps({"completed_comparisons":[{"hypothesis":r["hypothesis"],"seed":r["seed"],
      "complete":r["comparison_complete"]} for r in audit["runs"]],
      "state":read(root/"STATE.json")},indent=2))

