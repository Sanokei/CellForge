"""Detached exit/sleep fixture. No model, ARC environment or policy inference."""
import json,os,pathlib,sys,time
def compare_complete(config,rows): return False
def main():
    run=pathlib.Path(sys.argv[1]); config=json.loads((run/"CONFIG.json").read_text())
    failing=config["fingerprint"]==FAIL_FP
    pids=[]
    for _ in range(2):
        pid=os.fork()
        if pid==0:
            os.setsid()
            while True: time.sleep(.1)
        pids.append(pid)
    tmp=run/"PROBE_PIDS.json.tmp"; tmp.write_text(json.dumps(pids)); tmp.replace(run/"PROBE_PIDS.json")
    if failing:
        time.sleep(.6)
        raise SystemExit(2)
    while True: time.sleep(.1)
if __name__=="__main__": main()

