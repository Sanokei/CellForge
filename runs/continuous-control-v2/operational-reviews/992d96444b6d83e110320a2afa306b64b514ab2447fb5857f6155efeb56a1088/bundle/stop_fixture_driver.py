"""Isolated actual-main wrapper. Sleep-only workers; no research admission/play."""
import os,pathlib,signal,sys,time
root=pathlib.Path(sys.argv[1]);unit=sys.argv[2];base=pathlib.Path(sys.argv[3]);stop='--stop' in sys.argv[4:]
sys.path.insert(0,str(base));import supervisor as V
V.UNIT=unit;V.GROUP='/system.slice/'+unit;V.LOCK=str(root/'fixture-dispatcher.lock')
original=V.command
mode=(root/'MODE').read_text().strip()
def wait(path,seconds=20):
    end=time.monotonic()+seconds
    while not path.exists():
        if time.monotonic()>=end:raise RuntimeError('fixture checkpoint release expired: '+str(path))
        time.sleep(.05)
def bounded(argv):
    if argv[0]=='systemd-run' and mode=='rpc':
        (root/'RPC_ENTERED').write_text('durable intent precedes actual RPC');wait(root/'RELEASE_RPC')
    result=original([a.replace('RuntimeMaxSec=600','RuntimeMaxSec=40') for a in argv])
    if argv[0]=='systemd-run' and result.returncode==0:wait(pathlib.Path(argv[-2])/'PROBE_PIDS.json',5)
    if argv[0]=='systemd-run' and result.returncode==0 and mode=='crash-rpc':V.request_stop(root);os.kill(os.getpid(),signal.SIGKILL)
    return result
V.Systemd.__init__.__defaults__=(bounded,'/sys/fs/cgroup')
class FixtureController(V.Controller):
    def gates(self,bundle,plan):
        if mode=='gate':
            (root/'GATE_ENTERED').write_text('real registry lock held during gates');wait(root/'RELEASE_GATE')
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if (root/'CRASH_POINT').exists():
            target=(root/'CRASH_POINT').read_text().strip()
            def crash(point):
                if point==target:V.request_stop(root);os.kill(os.getpid(),signal.SIGKILL)
            self.crash=crash
V.Controller=FixtureController
sys.argv=['supervisor','--root',str(root)]+(['--stop'] if stop else []);V.main()