"""Operational fixture only: detached sleepers/CPU burners/allocators, no ARC."""
import json,os,pathlib,sys,time
run=pathlib.Path(sys.argv[1])
mode=(run/"PROBE_MODE").read_text() if (run/"PROBE_MODE").exists() else "sleep"
pids=[]
for _ in range(2):
    pid=os.fork()
    if pid==0:
        os.setsid()
        if mode=="memory":
            block=bytearray(600*1024**2)
            for i in range(0,len(block),4096): block[i]=1
        while True:
            if mode=="cpu": sum(range(10000))
            else: time.sleep(.1)
    pids.append(pid)
tmp=run/"PROBE_PIDS.json.tmp"; tmp.write_text(json.dumps(pids)); tmp.replace(run/"PROBE_PIDS.json")
while True: time.sleep(.1)

