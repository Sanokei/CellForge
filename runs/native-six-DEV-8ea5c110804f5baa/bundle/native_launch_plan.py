"""Read-only renderer/validator for the exact prospective native handoff. Never dispatches."""
import argparse,hashlib,json,pathlib,os,stat
from study_accounting import Accounting,Hold,digest,read
LOCK='/run/cellforge-continuous.lock'
UNIT='cellforge-continuous-job.service'
WATCHER='cellforge-continuous-v2.service'
def launch_token(study):return digest({'study_id':study['study_id'],'cohort':study['six_input_freeze_sha256'],'one_job_slot':1})
def launch_argv(study):
    token=launch_token(study);state=study['control_state_root'];bundle=study['bundle_root'];review=study['review_root']
    properties={'RemainAfterExit':'yes','Restart':'no','CPUQuota':'100%','CPUAffinity':'0','MemoryMax':'1G','MemorySwapMax':'0','MemoryAccounting':'yes','CPUAccounting':'yes','TasksMax':'64','RuntimeMaxSec':'595','TimeoutStopSec':'5','KillMode':'control-group','SendSIGKILL':'yes','OOMPolicy':'kill','StandardOutput':'null','StandardError':'null','PrivateNetwork':'yes','PrivateTmp':'yes','ProtectSystem':'strict','ProtectHome':'yes','ReadWritePaths':state+' '+LOCK,'ReadOnlyPaths':bundle+' '+review,'InaccessiblePaths':'/srv/cellforge-private','LimitFSIZE':'16M','LimitCORE':'0','NoNewPrivileges':'yes','CapabilityBoundingSet':'','WorkingDirectory':bundle}
    argv=['systemd-run','--quiet','--unit='+UNIT,'--description=cellforge-native4:'+study['study_id']+':'+token,'--service-type=exec']
    argv += ['--property='+k+'='+v for k,v in properties.items()]
    argv += ['--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=PYTHONIOENCODING=utf-8','/usr/bin/python3',bundle+'/study_worker.py','--bundle',bundle,'--control-root',state,'--approval',review+'/APPROVAL.json','--audit',review+'/AUDIT_REPORT.json','--launch-token',token,'--commit-reviewed']
    return argv

def check_before_stop(actual,pinned):
    keys=('MainPID','InvocationID','ActiveState','SubState','UnitFileState','FragmentPath','ControlGroup','Restart','KillSignal')
    if any(actual['v2'].get(k)!=pinned['v2'].get(k) for k in keys):raise Hold('v2 owner/config changed; no stop authorization')
    for k in ('unit_file_sha256','controller_sha256'):
        if actual.get(k)!=pinned.get(k):raise Hold('reviewed v2 code/unit changed')
    if actual['old_root_STOP_present']:raise Hold('preexisting new STOP is unconditional; do not clear it')
    if actual['old_registry']!={'active':None,'attempted':9,'cap':9,'status':'paused_exhausted'}:raise Hold('old cap/active/accounting changed')
    for key in ('device','inode','nlink','path'):
        if actual['global_lock'].get(key)!=pinned['global_lock'].get(key):raise Hold('lock inode/path changed')
    records=actual['global_lock']['kernel_records'];pid=actual['v2']['MainPID'];identity=actual['global_lock'];kernel_key=pinned['global_lock']['kernel_records'][0].split()[5]
    if len(records)!=1 or ' FLOCK ' not in records[0] or ' WRITE '+pid+' ' not in records[0] or kernel_key not in records[0].split():raise Hold('sole kernel lock holder is not the exact reviewed watcher')
    if actual.get('old_native_proposal_root_exists'):raise Hold('prior native proposal root appeared; no re-root/reset')

def check_stopped(watcher,lock_identity,kernel_records,pinned,job_empty,stops_preserved):
    if watcher!={'ActiveState':'inactive','MainPID':'0','UnitFileState':'disabled'}:raise Hold('watcher has not stopped disabled')
    if not job_empty or not stops_preserved:raise Hold('full tree/protected history not preserved')
    if kernel_records:raise Hold('some process still holds the global inode')
    if any(lock_identity.get(k)!=pinned['global_lock'].get(k) for k in ('device','inode','nlink','path')):raise Hold('lock file changed during stop')

def check_new_handoff_stop(identity,payload,stop_started,stop_returned):
    if identity.get('nlink')!=1 or identity.get('regular_nonsymlink') is not True:raise Hold('new v2 STOP is not a regular unique file')
    if set(payload)!={'epoch','reason'} or payload['reason']!='explicit reviewed service stop' or not stop_started<=payload['epoch']<=stop_returned:raise Hold('new STOP is not attributable to the reviewed standard stop window; preserve and HOLD')
    return identity

def check_owned_terminal(unit,description,invocation,populated,remaining_pids):
    if unit.get('LoadState')!='not-found' and (unit.get('Description')!=description or unit.get('InvocationID')!=invocation):raise Hold('foreign unit; never stop/reset it')
    if populated or remaining_pids:raise Hold('whole cgroup still populated; no terminal certification')
    if unit.get('MainPID') not in ('0',0):raise Hold('main process still running')
    if unit.get('LoadState') not in ('loaded','not-found') or unit.get('ActiveState') in ('activating','deactivating','reloading'):raise Hold('unit lifecycle not terminal')
    if unit.get('ActiveState')=='active' and unit.get('SubState')!='exited':raise Hold('active unit is not retained terminal metadata')

def render(study,checkpoint):
    return {'phase':'REVIEW_HOLD_NO_DISPATCH','cohort_role':'EXPOSED_PUBLIC_DEV','qualified':False,'control_root':study['control_root'],'state_root':study['control_state_root'],'launch_token':launch_token(study),'argv':launch_argv(study),'job_reserved_before_RPC':True,'lock':checkpoint['global_lock'],'watcher_owner':{k:checkpoint['v2'][k] for k in ('MainPID','InvocationID','FragmentPath')},'old_cap':9,'old_attempted':9,'RuntimeMaxSec':595,'TimeoutStopSec':5,'whole_job_seconds_ceiling':600,'provisioning':'must be explicitly authorized after review; absent roots are never automatically created','post_job_watcher':'left stopped/disabled with its newly generated STOP intact; no restart/STOP removal','transport_or_launch_uncertainty':'HOLD with permanently spent one-job claim; never re-dispatch'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=pathlib.Path,required=True);args=parser.parse_args();study=read(args.bundle/'STUDY.json');checkpoint=read(args.bundle/'HANDOFF_CHECKPOINT.json');print(json.dumps(render(study,checkpoint),sort_keys=True))
if __name__=='__main__':main()