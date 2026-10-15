"""Read-only dispatch observations. An observation deadline never changes dispatch."""
import argparse,json,pathlib,time,math
import supervisor as V
from common import Blocked
RESOURCE_PROPERTIES=('CPUQuotaPerSecUSec','CPUAffinity','MemoryMax','MemorySwapMax','RuntimeMaxUSec','TasksMax','PrivateNetwork')
EXPECTED={'CPUQuotaPerSecUSec':'1s','CPUAffinity':'0','MemoryMax':'1073741824','MemorySwapMax':'0','RuntimeMaxUSec':'10min','TasksMax':'64','PrivateNetwork':'yes'}
def resources():
    r=V.command(['systemctl','show',V.UNIT,'--property='+','.join(RESOURCE_PROPERTIES)])
    if r.returncode:raise Blocked('resource policy observation unavailable')
    return dict(x.split('=',1) for x in r.stdout.splitlines())
def observe(root,backend=None,resource_reader=None):
    root=pathlib.Path(root);backend=backend or V.Systemd();resource_reader=resource_reader or resources
    try:
        before=(root/'REGISTRY.json').read_bytes();state=json.loads(before);fp=state['active'];info=backend.show()
        out={'observed_epoch':time.time(),'registry_status':state['status'],'active':fp,'stop_present':(root/'STOP').exists(),'dispatch_present':(root/'DISPATCH_ENABLED').exists(),'consecutive_failures':state['consecutive_failures'],'counts':{k:sum(not x.get('legacy_unqualified') and x['status']==k for x in state['conditions'].values()) for k in ('pending','completed','cancelled','failed','launch_intent')},'unit':info,'observer_mutations':0}
        if fp:
            entry=state['conditions'][fp]
            if not entry.get('invocation'):out['phase']='intent_reconciling'
            elif backend.owned(info,entry['token'],entry['invocation']):
                out['invocation']=entry['invocation']
                if info['SubState']=='running':
                    actual=resource_reader()
                    if actual!=EXPECTED:raise Blocked('actual resource policy differs or is incomplete')
                    fresh=backend.show()
                    if not backend.owned(fresh,entry['token'],entry['invocation']):raise Blocked('owned unit disappeared during resource observation')
                    fields=('InvocationID','Description','MainPID','LoadState','ActiveState','SubState','ControlGroup')
                    if any(fresh[k]!=info[k] for k in fields):raise Blocked('unit changed during resource observation')
                    out['actual_resources']=actual
                    out['phase']='owned_running_stop_requested' if out['stop_present'] or not out['dispatch_present'] else 'running_verified'
                else:out['phase']='terminal_reconciling'
            else:out['phase']='active_entry_reconciling'
        elif info['LoadState']!='not-found' or not backend.empty(info):raise Blocked('unowned fixed unit or tree prevents an idle claim')
        else:out['phase']='held_idle' if out['stop_present'] or not out['dispatch_present'] else 'idle_observing'
        # Marker reads must be fresh after resources/unit transport, too.
        stop_now=(root/'STOP').exists();dispatch_now=(root/'DISPATCH_ENABLED').exists()
        marker_changed=(stop_now!=out['stop_present'] or dispatch_now!=out['dispatch_present'])
        out['stop_present']=stop_now;out['dispatch_present']=dispatch_now
        if out.get('phase')=='running_verified' and (stop_now or not dispatch_now):
            out['phase']='owned_running_stop_requested'
        elif marker_changed:
            return {'phase':'snapshot_changed','observer_mutations':0,'observed_epoch':time.time()}
        if (root/'REGISTRY.json').read_bytes()!=before:return {'phase':'snapshot_changed','observer_mutations':0,'observed_epoch':time.time()}
        return out
    except Exception as exc:return {'phase':'blocked_unknown','reason':str(exc)[:500],'observer_mutations':0,'observed_epoch':time.time()}
def watch(root,seconds=0,emit=None,sleeper=time.sleep,clock=time.monotonic,observer=observe):
    try: finite=(type(seconds) in (int,float) and math.isfinite(seconds))
    except (OverflowError,TypeError): finite=False
    if not finite or seconds<0 or seconds>600:raise ValueError('observation slice must be finite and 0..600 seconds')
    emit=emit or (lambda item:print(json.dumps(item,sort_keys=True),flush=True));end=clock()+seconds
    while True:
        snapshot=observer(root);emit(snapshot)
        if snapshot['phase']=='running_verified' or seconds==0:return snapshot
        if clock()>=end:
            result={'phase':'observation_pending','last_snapshot':snapshot,'observer_mutations':0,'action':'return snapshot; no stop, cancellation, marker change or controller deadline'}
            emit(result);return result
        sleeper(min(2,max(0,end-clock())))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=pathlib.Path,required=True);parser.add_argument('--watch-seconds',type=float,default=0)
    args=parser.parse_args();watch(args.root,args.watch_seconds)
if __name__=='__main__':main()