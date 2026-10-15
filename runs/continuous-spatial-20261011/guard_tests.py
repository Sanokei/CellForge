"""Real systemd aggregate CPU, timeout, process-tree and OOM acceptance probes."""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from supervisor import atomic, command

ROOT = pathlib.Path(__file__).resolve().parent
UNIT = "cellforge-continuous-guard-test.service"

def show():
    p = command(["systemctl","show",UNIT,
                 "--property=ActiveState,SubState,Result,ExecMainStatus,ControlGroup,CPUUsageNSec,MemoryPeak,MemoryMax,CPUQuotaPerSecUSec"])
    return dict(line.split("=",1) for line in p.stdout.splitlines() if "=" in line)

def empty(info):
    group = info.get("ControlGroup","")
    if not group:
        return True
    path = pathlib.Path("/sys/fs/cgroup") / group.lstrip("/")
    if not path.exists():
        return True
    events = dict(line.split() for line in (path/"cgroup.events").read_text().splitlines())
    return events.get("populated") == "0"

def child(mode):
    # Test descendants explicitly detach process sessions: cgroup must still kill them.
    for _ in range(2):
        pid = os.fork()
        if pid == 0:
            os.setsid()
            fd=os.open(ROOT/("guard-"+mode+".pids"),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
            os.write(fd,(str(os.getpid())+"\n").encode())
            os.close(fd)
            if mode == "memory":
                block=bytearray(600*1024**2)
                for i in range(0,len(block),4096):
                    block[i]=1
            while True:
                if mode=="cpu":
                    sum(range(10000))
                else:
                    time.sleep(1)
    while True:
        time.sleep(1)

def start(mode, timeout):
    command(["systemctl","stop",UNIT])
    command(["systemctl","reset-failed",UNIT])
    argv=["systemd-run","--quiet","--unit="+UNIT,"--service-type=exec",
          "--property=RemainAfterExit=yes","--property=CPUQuota=100%",
          "--property=CPUAffinity=0","--property=MemoryMax=1G",
          "--property=MemorySwapMax=0","--property=MemoryAccounting=yes",
          "--property=CPUAccounting=yes","--property=TasksMax=16",
          "--property=RuntimeMaxSec="+str(timeout),"--property=TimeoutStopSec=3",
          "--property=KillMode=control-group","--property=OOMPolicy=kill",
          "--property=SendSIGKILL=yes","--property=StandardOutput=null",
          "--property=StandardError=null","--setenv=PYTHONDONTWRITEBYTECODE=1",
          "/usr/bin/python3",str(ROOT/"guard_tests.py"),mode]
    p=command(argv)
    assert p.returncode==0,p.stderr

def gone(mode):
    path=ROOT/("guard-"+mode+".pids")
    pids=[int(v) for v in path.read_text().splitlines()]
    assert len(pids)==2,pids
    for pid in pids:
        stat=pathlib.Path("/proc")/str(pid)/"stat"
        assert not stat.exists() or stat.read_text().split()[2]=="Z",str(pid)+" survived"
    return pids

def probe():
    result={}
    start("cpu",10)
    time.sleep(1)
    first=show()
    tick=time.monotonic()
    time.sleep(4)
    second=show()
    ratio=(int(second["CPUUsageNSec"])-int(first["CPUUsageNSec"]))/1e9/(time.monotonic()-tick)
    assert 0.4 < ratio < 1.20,ratio
    assert second["MemoryMax"]==str(1073741824)
    assert second["CPUQuotaPerSecUSec"]=="1s"
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        info=show()
        if info.get("ActiveState")=="failed":
            break
        time.sleep(0.25)
    assert info["Result"]=="timeout",info
    assert empty(info),info
    pids=gone("cpu")
    result["timeout_cpu"]={"systemd":info,"aggregate_cpu_seconds_per_wall_second":ratio,
                           "detached_descendants":pids,"all_dead":True,"cgroup_empty":True}
    start("memory",30)
    deadline=time.monotonic()+35
    while time.monotonic()<deadline:
        info=show()
        if info.get("ActiveState")=="failed":
            break
        time.sleep(0.25)
    assert info["Result"]=="oom-kill",info
    assert empty(info),info
    result["aggregate_memory"]={"systemd":info,"detached_descendants":gone("memory"),
                               "all_dead":True,"cgroup_empty":True}
    command(["systemctl","stop",UNIT])
    command(["systemctl","reset-failed",UNIT])
    atomic(ROOT/"GUARD_EVIDENCE.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    if len(sys.argv)>1:
        child(sys.argv[1])
    else:
        probe()

