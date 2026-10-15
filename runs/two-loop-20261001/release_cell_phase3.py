from pathlib import Path
import json,os
root=Path("/srv/cellforge-inbox/two-loop-20261001/runs/weights-action-continuation")
controller=root.parents[1]/"control/cell-anchors.json"
value=json.loads(controller.read_text());value["phase_anchors"]["3"]="cb87891047cbc6d94049c779ddad3151408a9d4cb37fc0d812e565fe9c66cadc"
value["phase3_recorded_by_parent_utc"]="2026-10-01T16:23:00Z"
pending=controller.with_suffix(".pending");pending.write_text(json.dumps(value),encoding="utf-8");os.replace(pending,controller)
print("parent-recorded phase3 released atomically")

