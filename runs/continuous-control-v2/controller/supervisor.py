"""Review-only sequential controller. No admission/activation in this bundle."""
import argparse
import os
import pathlib
import shutil
import signal
import subprocess
import time
import uuid
from common import *

UNIT="cellforge-continuous-job.service"
GROUP="/system.slice/"+UNIT
LOCK="/run/cellforge-continuous.lock"
PYTHON="/srv/arc3-claude/venv/bin/python"
ACTIVE={"active","activating","deactivating","reloading"}
PROPERTIES=("LoadState","ActiveState","SubState","Result","ExecMainStatus","MainPID",
            "ControlGroup","InvocationID","Description","CPUUsageNSec","MemoryPeak")

def command(argv):
    try: return subprocess.run(argv,text=True,capture_output=True,timeout=15)
    except (OSError,subprocess.TimeoutExpired) as exc: raise Blocked("systemd transport unavailable") from exc

class Systemd:
    def __init__(self,runner=command,cgroup_root="/sys/fs/cgroup"):
        self.runner=runner; self.cgroup_root=pathlib.Path(cgroup_root)
    def show(self):
        r=self.runner(["systemctl","show",UNIT,"--no-pager","--property="+",".join(PROPERTIES)])
        if r.returncode: raise Blocked("systemctl show failed; no ownership/emptiness certification")
        info={}
        for line in r.stdout.splitlines():
            if "=" not in line: raise Blocked("malformed systemctl property")
            k,v=line.split("=",1)
            if k in info: raise Blocked("duplicate systemctl property")
            info[k]=v
        if any(k not in info for k in PROPERTIES): raise Blocked("incomplete systemctl status")
        if info["LoadState"] not in {"loaded","not-found"}: raise Blocked("unknown unit load state")
        if info["ActiveState"] not in ACTIVE|{"inactive","failed"}: raise Blocked("unknown active state")
        try:
            if int(info["MainPID"])<0 or int(info["ExecMainStatus"])<0: raise ValueError()
        except ValueError: raise Blocked("invalid PID/exit status")
        group=info["ControlGroup"]
        if group and group!=GROUP: raise Blocked("unexpected fixed cgroup path")
        if info["SubState"]=="running" and (not group or not info["InvocationID"] or int(info["MainPID"])==0):
            raise Blocked("running status lacks ownership evidence")
        if info["LoadState"]=="not-found" and (info["ActiveState"]!="inactive" or info["SubState"]!="dead" or int(info["MainPID"]) or group or info["InvocationID"]):
            raise Blocked("inconsistent absent-unit status")
        info["_verified"]=True
        return info
    def empty(self,info):
        if info.get("_verified") is not True or any(k not in info for k in PROPERTIES):
            raise Blocked("unverified status cannot certify an empty tree")
        if info["ActiveState"] in ACTIVE and info["SubState"]!="exited": return False
        if int(info["MainPID"]): return False
        # Never infer emptiness just because ControlGroup is absent/blank.
        path=self.cgroup_root/GROUP.lstrip("/")
        try:
            path.stat()
        except FileNotFoundError:
            return True
        except OSError as exc: raise Blocked("cgroup status unreadable") from exc
        try:
            pairs=[line.split() for line in (path/"cgroup.events").read_text().splitlines()]
            if any(len(pair)!=2 for pair in pairs): raise ValueError()
            events=dict(pairs)
            if events.get("populated") not in {"0","1"}: raise ValueError()
        except (OSError,ValueError) as exc: raise Blocked("cgroup populated status unknown") from exc
        return events["populated"]=="0"
    def owned(self,info,token,invocation=None):
        if info.get("_verified") is not True: raise Blocked("unverified ownership")
        if info["LoadState"]=="not-found":
            if not self.empty(info): raise Blocked("absent unit has an orphan cgroup")
            return False
        if info["Description"]!="cellforge-v2:"+token: raise Blocked("unowned fixed job unit")
        if invocation and info["InvocationID"] and info["InvocationID"]!=invocation:
            raise Blocked("unit invocation changed")
        return True
    def stop(self,token=None,invocation=None):
        before=self.show()
        if token:
            self.owned(before,token,invocation)
        elif before["LoadState"]!="not-found":
            raise Blocked("refusing to stop an unowned unit")
        r=self.runner(["systemctl","stop",UNIT])
        after=self.show()
        if r.returncode and after["LoadState"]!="not-found": raise Blocked("job stop failed")
        if not self.empty(after): raise Blocked("full cgroup still populated")
        if after["ActiveState"] in ACTIVE: raise Blocked("unit not terminal after stop")
        return after
    def collect(self,token,invocation=None,checkpoint=None):
        """Collect only an owned empty failed unit; absence is verified before reuse."""
        checkpoint=checkpoint or (lambda _:None)
        info=self.show()
        if info["LoadState"]=="not-found":
            if not self.empty(info): raise Blocked("absent unit has populated fixed cgroup")
            return info
        if not token: raise Blocked("collection requires durable owner token")
        self.owned(info,token,invocation)
        if not self.empty(info) or info["ActiveState"] in ACTIVE:
            raise Blocked("collection requires a verified terminal empty tree")
        expected=invocation or info["InvocationID"]
        if not expected or info["InvocationID"]!=expected:
            raise Blocked("failed-unit collection lacks matching invocation")
        if info["ActiveState"]=="failed":
            checkpoint("before_failed_unit_collection")
            # Recheck after a checkpoint/possible pause; never reset an unowned replacement.
            fresh=self.show()
            self.owned(fresh,token,expected)
            if fresh["ActiveState"]!="failed" or not self.empty(fresh) or fresh["InvocationID"]!=expected:
                raise Blocked("failed-unit ownership/state changed before collection")
            result=self.runner(["systemctl","reset-failed",UNIT])
            checkpoint("after_failed_unit_reset")
            info=self.show()
            if result.returncode and info["LoadState"]!="not-found":
                raise Blocked("owned failed-unit collection RPC failed")
        # Inactive transient metadata may take a short time to be garbage-collected.
        deadline=time.monotonic()+3
        while info["LoadState"]!="not-found":
            self.owned(info,token,expected)
            if info["InvocationID"]!=expected or not self.empty(info) or info["ActiveState"] in ACTIVE:
                raise Blocked("unit changed during collection")
            if time.monotonic()>=deadline: raise Blocked("owned transient metadata not yet collected")
            time.sleep(.1)
            info=self.show()
        if not self.empty(info): raise Blocked("collected unit has populated cgroup")
        return info


    def start(self,bundle,run,token):
        before=self.show()
        if not self.empty(before) or before["ActiveState"] in ACTIVE:
            raise Blocked("fixed unit/cgroup prevents launch")
        if before["LoadState"]!="not-found":
            raise Blocked("previous transient unit has not been collected")
        argv=["systemd-run","--quiet","--unit="+UNIT,"--description=cellforge-v2:"+token,
              "--service-type=exec","--property=RemainAfterExit=yes","--property=Restart=no",
              "--property=CPUQuota=100%","--property=CPUAffinity=0","--property=MemoryMax=1G",
              "--property=MemorySwapMax=0","--property=MemoryAccounting=yes","--property=CPUAccounting=yes",
              "--property=TasksMax=64","--property=RuntimeMaxSec=600","--property=TimeoutStopSec=5",
              "--property=KillMode=control-group","--property=SendSIGKILL=yes","--property=OOMPolicy=kill",
              "--property=StandardOutput=null","--property=StandardError=null",
              "--property=PrivateNetwork=yes","--property=PrivateTmp=yes","--property=ProtectSystem=strict",
              "--property=ProtectHome=yes","--property=ReadWritePaths="+str(run),
              "--property=InaccessiblePaths=/srv/cellforge-private","--property=LimitFSIZE=16M",
              "--property=LimitCORE=0","--property=WorkingDirectory="+str(bundle)]
        for k,v in {"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
                    "NUMEXPR_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1",
                    "TMPDIR":str(run/"scratch"),"ARC3_DIR":"/srv/arc3-claude"}.items():
            argv.append("--setenv="+k+"="+v)
        argv += [PYTHON,str(bundle/"worker.py"),str(run),str(bundle)]
        r=self.runner(argv)
        if r.returncode: raise Blocked("launch RPC failed; durable intent must be reconciled")
        info=self.show()
        self.owned(info,token)
        return info

def request_stop(root):
    """Create a durable STOP once. Existing marker bytes are never replaced."""
    root=pathlib.Path(root)
    try: fd=os.open(root/"STOP",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError: return
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(canonical({"reason":"explicit reviewed service stop","epoch":time.time()})+b"\n")
            stream.flush(); os.fsync(stream.fileno())
        if os.name=="posix":
            directory=os.open(root,os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
    except Exception:
        # A partial marker still forbids launch. Never remove it to retry a stop.
        raise

def tree_size(root):
    return sum(p.stat().st_size for p in pathlib.Path(root).rglob("*") if p.is_file() and not p.is_symlink())

class Controller:
    def __init__(self,root,backend=None,crash=None):
        self.root=pathlib.Path(root); self.backend=backend or Systemd(); self.crash=crash or (lambda _:None)
        self.halt=False
    def persist(self,state):
        atomic(self.root/"REGISTRY.json",state)
    def stop_requested(self,run=None,fp=None):
        return (self.halt or (self.root/"STOP").exists() or
                not (self.root/"DISPATCH_ENABLED").exists() or
                (run is not None and (run/"STOP").exists()) or
                (fp is not None and (self.root/"cancel"/fp).exists()))
    def pause(self,state,run=None):
        state["status"]=("paused_stop" if self.halt or (self.root/"STOP").exists() or
                         (run is not None and (run/"STOP").exists()) else "paused_dispatch_disabled")
        self.persist(state)
    def verify_stopped(self):
        # Shares the registry lock with launch admission and --stop reconciliation.
        with registry_lock(self.root):
            state=read(self.root/"REGISTRY.json")
            if state["active"] is not None: raise Blocked("stop still has a durable active entry")
            info=self.backend.show()
            if info["LoadState"]!="not-found" or not self.backend.empty(info):
                raise Blocked("stop cannot certify fixed unit absence and an empty tree")
        return True
    def context(self,state,fp):
        ph=state["conditions"][fp]["study"]
        bundle=self.root/"studies"/ph
        plan=read(bundle/"PLAN.json")
        if digest(plan)!=ph: raise Blocked("admitted plan changed")
        config=next(c for c in plan["conditions"] if c["fingerprint"]==fp)
        return bundle,plan,config,pathlib.Path(plan["lab_runs_root"])/("weights-transition-"+ph)/"jobs"/fp
    def gates(self,bundle,plan):
        validate_plan(plan)
        manifest=read(bundle/"MANIFEST.json")
        approval=read(bundle/"APPROVAL.json")
        if approval["plan_sha256"]!=digest(plan) or approval["bundle_manifest_sha256"]!=filehash(bundle/"MANIFEST.json"):
            raise Blocked("approval binding changed")
        if approval["decision"]!="permit_baseline_development" or approval["qualified"] is not False or approval["remote_model_calls"]!=0:
            raise Blocked("approval scope changed")
        if filehash(bundle/"AUDIT_REPORT.md")!=approval["audit_report_sha256"]: raise Blocked("audit report changed")
        for name,h in manifest["files"].items():
            if pathlib.PurePath(name).is_absolute() or ".." in pathlib.PurePath(name).parts or filehash(bundle/name)!=h:
                raise Blocked("approved bundle changed: "+name)
        for name,h in {**plan["protected_stops"],**plan["dataset_files"]}.items():
            if filehash(name)!=h: raise Blocked("protected STOP/data changed")
        import importlib.metadata
        for name,version in plan["packages"].items():
            if importlib.metadata.version(name)!=version: raise Blocked("runtime package version changed")
        registry=read(self.root/"REGISTRY.json")
        all_run_bytes=sum(tree_size(pathlib.Path(plan["lab_runs_root"])/("weights-transition-"+ph)) for ph in registry["studies"])
        if shutil.disk_usage(self.root).free<LIMITS["disk_floor"] or tree_size(self.root)+all_run_bytes>LIMITS["total_bytes_max"]:
            raise Blocked("disk/artifact gate")
    def finish_receipt(self,state,fp,run,receipt):
        entry=state["conditions"][fp]
        self.crash("after_receipt_before_collection")
        expected_invocation=entry.get("invocation") or receipt["evidence"]["systemd_before_stop"].get("InvocationID") or None
        absence=self.backend.collect(entry["token"],expected_invocation,checkpoint=self.crash)
        if absence["LoadState"]!="not-found" or not self.backend.empty(absence):
            raise Blocked("terminal accounting requires collected absent unit")
        self.crash("after_collection_before_registry")
        evidence={"fingerprint":fp,"terminal_receipt_id":receipt["receipt_id"],
                  "owner_token":entry["token"],"invocation":expected_invocation,
                  "unit_absent":True,"job_tree_empty":True,"verified_systemd":absence}
        target=run/"COLLECTION.json"
        if target.exists():
            prior=read(target)
            if any(prior.get(k)!=evidence[k] for k in ("fingerprint","terminal_receipt_id","owner_token","invocation","unit_absent","job_tree_empty")):
                raise Blocked("conflicting collection evidence")
        else: atomic(target,evidence)
        self.crash("after_receipt_before_registry")
        replay_terminal(state,receipt); self.persist(state)
        self.crash("after_registry_commit")


    def terminal(self,state,fp,info,outcome):
        bundle,plan,config,run=self.context(state,fp)
        active=state["conditions"][fp]
        self.backend.owned(info,active["token"],active.get("invocation"))
        after=self.backend.stop(active["token"],active.get("invocation") or info["InvocationID"] or None)
        if not self.backend.empty(after): raise Blocked("cannot certify terminal empty tree")
        if filehash(run/"CONFIG.json")!=active["config_sha256"]: raise Blocked("run configuration changed")
        receipt=make_terminal(state,fp,outcome,{"systemd_before_stop":info,"systemd_after_stop":after,
             "job_tree_empty":True,"config_sha256":filehash(run/"CONFIG.json"),
             "results_sha256":filehash(run/"RESULTS.json") if (run/"RESULTS.json").exists() else None})
        target=run/"TERMINAL.json"
        if target.exists(): receipt=read(target)
        else: atomic(target,receipt)
        self.finish_receipt(state,fp,run,receipt)
    def tick(self):
        with registry_lock(self.root):
            state=read(self.root/"REGISTRY.json")
            fp=state["active"]
            info=self.backend.show()  # unknown transport blocks all dispatch and receipts
            if fp:
                bundle,plan,config,run=self.context(state,fp)
                entry=state["conditions"][fp]
                receipt_path=run/"TERMINAL.json"
                if receipt_path.exists():
                    receipt=read(receipt_path)
                    expected=entry.get("invocation") or receipt["evidence"]["systemd_before_stop"].get("InvocationID") or None
                    self.backend.stop(entry["token"],expected)
                    self.finish_receipt(state,fp,run,receipt)
                    return
                owned=self.backend.owned(info,entry["token"],entry.get("invocation"))
                stopped=self.halt or (self.root/"STOP").exists() or (run/"STOP").exists()
                disabled=not (self.root/"DISPATCH_ENABLED").exists()
                if stopped or disabled or (self.root/"cancel"/fp).exists():
                    self.terminal(state,fp,info,"cancelled")
                else:
                    try: self.gates(bundle,plan)
                    except Exception:
                        self.terminal(state,fp,info,"resource_blocked"); return
                    if tree_size(run)>LIMITS["run_bytes_max"]:
                        self.terminal(state,fp,info,"resource_blocked")
                    elif not owned:
                        self.terminal(state,fp,info,"interrupted")
                    elif info["SubState"]=="running":
                        if not entry.get("invocation"):
                            entry["invocation"]=info["InvocationID"]; self.persist(state)
                    elif info["SubState"]=="exited" or info["ActiveState"] in {"failed","inactive"}:
                        summary=read(run/"SUMMARY.json") if (run/"SUMMARY.json").exists() else {}
                        from worker import compare_complete
                        rows=read(run/"RESULTS.json") if (run/"RESULTS.json").exists() else []
                        complete=(info["Result"]=="success" and info["ExecMainStatus"]=="0" and
                                  summary.get("fingerprint")==fp and summary.get("plays")==27 and
                                  summary.get("comparison_complete") is True and compare_complete(config,rows))
                        self.terminal(state,fp,info,"completed" if complete else "failed")
                    else: raise Blocked("unknown owned unit lifecycle")
                return
            if self.halt or (self.root/"STOP").exists():
                state["status"]="paused_stop"
            elif not (self.root/"DISPATCH_ENABLED").exists():
                state["status"]="paused_dispatch_disabled"
            elif not self.backend.empty(info) or info["ActiveState"] in ACTIVE or info["LoadState"]!="not-found":
                raise Blocked("unowned fixed unit blocks dispatch")
            elif state["consecutive_failures"]>=3:
                state["status"]="blocked_failure_circuit"
            elif time.time()<state["not_before"]:
                state["status"]="failure_backoff"
            else:
                for fp,entry in state["conditions"].items():
                    if entry["status"]!="pending": continue
                    bundle,plan,config,run=self.context(state,fp)
                    self.crash("before_gates")
                    self.gates(bundle,plan)
                    self.crash("after_gates_before_admission")
                    # Expensive gates cannot authorize a launch after STOP/disable/SIGTERM.
                    # The check and reservation share the registry lock with --stop.
                    if self.stop_requested(run):
                        self.pause(state,run); return
                    # A pending per-condition cancellation still consumes the finite cap.
                    reserve(state,fp)
                    run.mkdir(parents=True,exist_ok=True); (run/"scratch").mkdir(exist_ok=True)
                    if (run/"CONFIG.json").exists() and read(run/"CONFIG.json")!=config:
                        raise Blocked("conflicting partial run")
                    atomic(run/"CONFIG.json",config)
                    entry["config_sha256"]=filehash(run/"CONFIG.json")
                    entry["token"]=fp+":"+uuid.uuid4().hex
                    self.persist(state)
                    self.crash("after_launch_intent")
                    if self.stop_requested(run,fp):
                        self.terminal(state,fp,self.backend.show(),"cancelled"); return
                    self.crash("before_launch_rpc")
                    # A signal or marker may arrive while the durable intent is written.
                    if self.stop_requested(run,fp):
                        self.terminal(state,fp,self.backend.show(),"cancelled"); return
                    try:
                        unit=self.backend.start(bundle,run,entry["token"])
                    except Exception:
                        # Never guess whether a failed launch RPC created a tree.
                        fresh=self.backend.show()
                        owned=self.backend.owned(fresh,entry["token"])
                        if owned:
                            entry["invocation"]=fresh["InvocationID"]; self.persist(state)
                        if self.stop_requested(run,fp):
                            self.terminal(state,fp,fresh,"cancelled")
                        elif not owned:
                            self.terminal(state,fp,fresh,"failed")
                        return
                    entry["invocation"]=unit["InvocationID"]
                    self.persist(state); self.crash("after_launch")
                    # The stop may race the RPC itself: persist identity before draining it.
                    if self.stop_requested(run,fp):
                        self.terminal(state,fp,self.backend.show(),"cancelled")
                    return
                state["status"]="paused_exhausted"
            self.persist(state)

def run_loop(c,sleeper=time.sleep):
    while True:
        try:
            if c.halt: request_stop(c.root)
            c.tick()
            if c.halt:
                request_stop(c.root)
                c.verify_stopped()
                return
        except Exception as exc:
            atomic(c.root/"STATUS.json",{"status":"blocked_unknown","reason":str(exc)[:500],"utc_epoch":time.time()})
            # Unknown ownership/transport/collection is never a successful stop.
            # Keep the dispatcher lock and retry reconciliation instead of returning.
        sleeper(2)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=pathlib.Path,required=True)
    parser.add_argument("--stop",action="store_true")
    args=parser.parse_args()
    if args.stop:
        if os.environ.get("SERVICE_RESULT","success")!="success" and not (args.root/"STOP").exists(): return
        request_stop(args.root)
        c=Controller(args.root); c.halt=True
        # tick holds REGISTRY.lock; it cannot read an idle snapshot halfway through launch.
        c.tick(); c.verify_stopped()
        return
    if os.name!="posix": raise Blocked("production requires Linux/systemd")
    import fcntl
    stream=open(LOCK,"a+")
    try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise Blocked("existing continuous supervisor holds the global lock")
    c=Controller(args.root)
    def halt(*_): c.halt=True
    signal.signal(signal.SIGTERM,halt); signal.signal(signal.SIGINT,halt)
    try: run_loop(c)
    finally: stream.close()
if __name__=="__main__": main()
