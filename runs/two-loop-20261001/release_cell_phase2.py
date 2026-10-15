from pathlib import Path
import json,os
root=Path("/srv/cellforge-inbox/two-loop-20261001/runs/weights-action-continuation")
controller=root.parents[1]/"control/cell-anchors.json"
value=json.loads(controller.read_text());value["phase_anchors"]["2"]="c2c2ad132ccd0ce3f583baef12ee765fb3258b396b3b7beebdfeabe773d6e4a9"
value["phase2_recorded_by_parent_utc"]="2026-10-01T16:01:00Z"
pending=controller.with_suffix(".pending");pending.write_text(json.dumps(value),encoding="utf-8");os.replace(pending,controller)
print("parent-recorded phase2 released atomically")
