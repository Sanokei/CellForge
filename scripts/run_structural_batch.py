"""One serial maker driver for the revised structural niches."""
from pathlib import Path
import sys
import run_bfcl_skill as runner

base=Path(__file__).resolve().parent
for module,run in [('anchor_assign_skill','weights-anchor-assign4'),('frame_infer_skill','weights-frame-corner3'),
                   ('reference_paint_skill','weights-reference-paint3'),('corridor_skill','weights-corridor-marker2')]:
    sys.argv=['run_bfcl_skill.py',str(base.parent/'runs'/run),'--source',str(base/(module+'.py')),
              '--module',module,'--rounds','8']
    runner.main()
