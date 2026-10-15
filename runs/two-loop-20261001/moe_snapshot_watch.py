"""Observable bounded MoE scheduler: no redundant router rounds on unchanged pool."""
import pathlib,json,time,os,datetime
base=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001');root=base/'runs/weights-arc-moe-smallskill-v2';state=json.loads((root/'STATE-3.json').read_text());control=base/'control';deadline=time.monotonic()+1500
status=control/'MOE-SCHEDULER.json';expected_pool=state['pool_hash']
def write(kind,**kw):
 value={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':kind,'last_completed_round':3,'last_pool_hash':expected_pool,'training_active':False,**kw}
 pending=status.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2));os.replace(pending,status)
while time.monotonic()<deadline:
 signal=control/'NEW-GRID-CELL-SNAPSHOT.json'
 if signal.exists():
  value=json.loads(signal.read_text())
  if value.get('pool_hash')!=expected_pool:
   write('new_snapshot_requires_typed_admission',snapshot=value,reason='verify real numeric cells and declared small-skill challenges before opening next immutable MoE epoch');break
 write('waiting_for_new_eligible_grid_cell_snapshot',reason='previous ARC oracle0/100; avoid repeated router-only rounds. Horn count/Boolean interfaces are incompatible with grid IO.',cell_run=str(base/'runs/weights-reasoning-cell-epochs'))
 time.sleep(10)
else:write('bounded_scheduler_deadline',reason='no new compatible earned grid snapshot arrived; no capability advancement claimed')

