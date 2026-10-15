"""Isolated real v2 main, with research gates bypassed only for operational fixtures."""
import os,pathlib,signal,sys
root=pathlib.Path(sys.argv[1]); unit=sys.argv[2]; base=pathlib.Path(sys.argv[3])
sys.path.insert(0,str(base))
import supervisor as V
V.UNIT=unit; V.GROUP="/system.slice/"+unit; V.LOCK=str(root/"fixture-dispatcher.lock")
original_command=V.command
def bounded(argv):
    return original_command([arg.replace("RuntimeMaxSec=600","RuntimeMaxSec=40") for arg in argv])
V.Systemd.__init__.__defaults__=(bounded,"/sys/fs/cgroup")
original_controller=V.Controller
class FixtureController(original_controller):
    def gates(self,bundle,plan): pass
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if (root/"CRASH_AFTER_LAUNCH").exists():
            def crash(point):
                if point=="after_launch": os.kill(os.getpid(),signal.SIGKILL)
            self.crash=crash
V.Controller=FixtureController
stop="--stop" in sys.argv[4:]
sys.argv=["supervisor","--root",str(root)]+(["--stop"] if stop else [])
V.main()

