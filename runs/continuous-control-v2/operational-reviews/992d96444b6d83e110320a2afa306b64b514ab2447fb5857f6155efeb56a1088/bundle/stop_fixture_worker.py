"""Detached sleep-only worker fixture: no imports of policy, ARC or model tools."""
import json,os,pathlib,sys,time
run=pathlib.Path(sys.argv[1]);pids=[]
for _ in range(2):
    pid=os.fork()
    if pid==0:
        os.setsid()
        while True:time.sleep(.1)
    pids.append(pid)
tmp=run/'PROBE_PIDS.json.tmp';tmp.write_text(json.dumps([os.getpid()]+pids));tmp.replace(run/'PROBE_PIDS.json')
while True:time.sleep(.1)