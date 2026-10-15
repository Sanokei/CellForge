"""Authorized isolated Linux v2 operational fixtures; no research admission/play."""
import base64,copy,datetime,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,tempfile,time,uuid
payload=json.loads(base64.b64decode(PAYLOAD))
parent=pathlib.Path("/srv/cellforge-inbox")
base=pathlib.Path(tempfile.mkdtemp(prefix="cellforge-v2-review-",dir=parent))
units=[]; processes=[]
report={"scope":"operational fixtures only, no research jobs/admissions/game plays/model calls",
        "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"fixture_root":str(base),
        "tightened_runtime_limits_seconds":[8,30,40],"research_runtime_limit_seconds":600,"cases":{}}
for name in ("common.py","supervisor.py"): (base/name).write_text(payload["files"][name])
sys.path.insert(0,str(base))
import supervisor as V
from common import atomic,read,digest,filehash,fresh_registry
original_command=V.command
worker_source=payload["fixture_worker"]
driver_source=payload["fixture_driver"]

def until(fn,seconds=20):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        value=fn()
        if value: return value
        time.sleep(.2)
    raise RuntimeError("bounded fixture wait expired")

def select(unit,wall):
    V.UNIT=unit; V.GROUP="/system.slice/"+unit
    def bounded(argv):
        tightened=[("--property=RuntimeMaxSec="+str(wall)) if arg=="--property=RuntimeMaxSec=600" else arg for arg in argv]
        return original_command(tightened)
    return V.Systemd(runner=bounded)

def case(name):
    root=base/name; root.mkdir()
    unit="cellforge-v2-review-"+base.name.rsplit("-",1)[-1]+"-"+name+".service"; units.append(unit)
    plan=copy.deepcopy(payload["plan"]); plan["lab_runs_root"]=str(root/"lab"/"runs")
    ph=digest(plan); fp=min(c["fingerprint"] for c in plan["conditions"])
    bundle=root/"studies"/ph; bundle.mkdir(parents=True)
    atomic(bundle/"PLAN.json",plan)
    (bundle/"worker.py").write_text(worker_source)
    state=fresh_registry(); state["scope"]="isolated operational fixture, not research admission"
    state["studies"][ph]={"cap":9,"attempted":0,"condition_ids":[c["fingerprint"] for c in plan["conditions"]]}
    state["conditions"]={c["fingerprint"]:{"study":ph,"status":"pending" if c["fingerprint"]==fp else "unused_probe_fixture"} for c in plan["conditions"]}
    atomic(root/"REGISTRY.json",state)
    run=pathlib.Path(plan["lab_runs_root"])/("weights-transition-"+ph)/"jobs"/fp
    return root,unit,bundle,fp,run

def dead_pids(run):
    pids=read(run/"PROBE_PIDS.json")
    for pid in pids:
        path=pathlib.Path("/proc")/str(pid)/"stat"
        assert not path.exists() or path.read_text().split()[2]=="Z", "fixture descendant survived: "+str(pid)
    return pids

try:
    root,unit,bundle,fp,run=case("cancel-restart")
    (root/"DISPATCH_ENABLED").write_text("isolated operational fixture only")
    (root/"CRASH_AFTER_LAUNCH").write_text("real SIGKILL at committed launch boundary")
    driver=base/"probe_driver.py"; driver.write_text(driver_source)
    argv=[sys.executable,str(driver),str(root),unit,str(base)]
    child=subprocess.Popen(argv,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True); processes.append(child)
    assert child.wait(timeout=20)==-signal.SIGKILL
    backend=select(unit,40); before=backend.show()
    state=read(root/"REGISTRY.json"); token=state["conditions"][fp]["token"]
    invocation=before["InvocationID"]
    assert backend.owned(before,token,invocation) and not backend.empty(before)
    until(lambda: (run/"PROBE_PIDS.json").exists())
    (root/"CRASH_AFTER_LAUNCH").unlink()
    adopted=subprocess.Popen(argv,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True); processes.append(adopted)
    until(lambda: backend.show()["InvocationID"]==invocation and read(root/"REGISTRY.json")["active"]==fp)
    # The existing scoped flock must reject another real v2 main process.
    time.sleep(.6)
    duplicate=subprocess.run(argv,text=True,capture_output=True,timeout=5)
    assert duplicate.returncode!=0 and "global lock" in duplicate.stderr
    assert backend.show()["InvocationID"]==invocation
    assert read(root/"REGISTRY.json")["studies"][digest(read(bundle/"PLAN.json"))]["attempted"]==1
    (root/"cancel").mkdir(); (root/"cancel"/fp).write_text("active cancellation fixture")
    cancelled=until(lambda: read(root/"REGISTRY.json")["active"] is None and (run/"TERMINAL.json").exists())
    receipt=read(run/"TERMINAL.json"); after=backend.show()
    assert receipt["outcome"]=="cancelled" and receipt["evidence"]["job_tree_empty"] is True and backend.empty(after)
    state=read(root/"REGISTRY.json")
    assert state["sequence"]==1 and state["consecutive_failures"]==0
    # Durable cancellation is not replayed; no second research/fixture condition is pending.
    time.sleep(2.2)
    assert read(root/"REGISTRY.json")["sequence"]==1
    report["cases"]["real_crash_duplicate_active_cancel"]={"same_invocation_adopted":invocation,
       "launch_attempts":1,"duplicate_main_refused":True,"outcome":"cancelled","terminal_sequence":1,
       "failure_count":0,"whole_tree_empty":True,"detached_descendants_dead":dead_pids(run),
       "systemd_before":before,"systemd_after":after}
    (root/"STOP").write_text("fixture cleanup")
    adopted.terminate(); adopted.wait(timeout=8)

    root,unit,bundle,fp,run=case("explicit-stop")
    backend=select(unit,40)
    (root/"DISPATCH_ENABLED").write_text("isolated operational fixture")
    child=subprocess.Popen([sys.executable,str(driver),str(root),unit,str(base)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True); processes.append(child)
    until(lambda: read(root/"REGISTRY.json")["active"]==fp and (run/"PROBE_PIDS.json").exists())
    stopped=subprocess.run([sys.executable,str(driver),str(root),unit,str(base),"--stop"],text=True,capture_output=True,timeout=20)
    assert stopped.returncode==0,stopped.stderr
    until(lambda: read(root/"REGISTRY.json")["active"] is None)
    assert (root/"STOP").exists() and backend.empty(backend.show())
    r=read(run/"TERMINAL.json"); assert r["outcome"]=="cancelled"
    report["cases"]["real_explicit_stop"]={"persistent_STOP":True,"outcome":r["outcome"],
         "whole_tree_empty":True,"detached_descendants_dead":dead_pids(run)}
    child.terminate(); child.wait(timeout=8)

    root,unit,bundle,fp,run=case("cpu-timeout")
    run.mkdir(parents=True); (run/"scratch").mkdir(); (run/"PROBE_MODE").write_text("cpu")
    backend=select(unit,8); token="review-probe:"+uuid.uuid4().hex
    initial=backend.start(bundle,run,token); until(lambda:(run/"PROBE_PIDS.json").exists())
    time.sleep(.5); first=backend.show(); tick=time.monotonic()
    time.sleep(3); second=backend.show()
    ratio=(int(second["CPUUsageNSec"])-int(first["CPUUsageNSec"]))/1e9/(time.monotonic()-tick)
    assert .35<ratio<1.20,ratio
    ended=until(lambda: (q if (q:=backend.show())["ActiveState"]=="failed" else None),seconds=15)
    assert ended["Result"]=="timeout" and backend.empty(ended)
    props=original_command(["systemctl","show",unit,"--property=MemoryMax,CPUQuotaPerSecUSec,KillMode,MemorySwapMax,RuntimeMaxUSec"])
    assert props.returncode==0
    settings=dict(x.split("=",1) for x in props.stdout.splitlines())
    assert settings["MemoryMax"]=="1073741824" and settings["CPUQuotaPerSecUSec"]=="1s"
    assert settings["KillMode"]=="control-group" and settings["MemorySwapMax"]=="0"
    report["cases"]["real_cpu_timeout"]={"systemd":ended,"aggregate_cpu_per_wall_second":ratio,
         "actual_settings":settings,"whole_tree_empty":True,"detached_descendants_dead":dead_pids(run)}
    backend.stop(token,initial["InvocationID"])

    root,unit,bundle,fp,run=case("memory-oom")
    run.mkdir(parents=True); (run/"scratch").mkdir(); (run/"PROBE_MODE").write_text("memory")
    backend=select(unit,30); token="review-probe:"+uuid.uuid4().hex
    initial=backend.start(bundle,run,token)
    ended=until(lambda: (q if (q:=backend.show())["ActiveState"]=="failed" else None),seconds=25)
    assert ended["Result"]=="oom-kill" and backend.empty(ended),ended
    report["cases"]["real_memory_oom"]={"systemd":ended,"requested_detached_allocations_bytes":[600*1024**2,600*1024**2],
        "memory_cap_bytes":1073741824,"whole_tree_empty":True,"detached_descendants_dead":dead_pids(run)}
    backend.stop(token,initial["InvocationID"])
    report["passed"]=True
except Exception as exc:
    report["passed"]=False; report["error"]=type(exc).__name__+": "+str(exc)
finally:
    cleanup=[]
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=8)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)
    for unit in units:
        backend=select(unit,8)
        try:
            status=backend.show()
            if status["LoadState"]=="loaded":
                description=status["Description"]
                if not description.startswith("cellforge-v2:"): raise RuntimeError("unowned isolated unit")
                backend.stop(description.split("cellforge-v2:",1)[1],status["InvocationID"] or None)
            final=backend.show()
            if final["ActiveState"]=="failed" and backend.empty(final):
                cleared=original_command(["systemctl","reset-failed",unit])
                if cleared.returncode: raise RuntimeError("isolated failed-unit metadata cleanup failed")
                final=backend.show()
            cleanup.append({"unit":unit,"empty":backend.empty(final),"active":final["ActiveState"]})
        except Exception as exc:
            cleanup.append({"unit":unit,"error":str(exc)}); report["passed"]=False
    report["cleanup"]=cleanup
    report["old_service"]=dict(x.split("=",1) for x in original_command(["systemctl","show","cellforge-continuous.service","--property=MainPID,ActiveState"]).stdout.splitlines())
    report["old_job"]=dict(x.split("=",1) for x in original_command(["systemctl","show","cellforge-continuous-job.service","--property=MainPID,LoadState,ActiveState"]).stdout.splitlines())
    oldroot=pathlib.Path("/srv/cellforge-inbox/continuous-spatial-20261011")
    oldplan=read(oldroot/"PLAN.json")
    report["protected_stop_mismatches"]=[p for p,h in oldplan["protected_stops"].items() if filehash(p)!=h]
    report["old_code_mismatches"]=[n for n,h in oldplan["code_sha256"].items() if filehash(oldroot/n)!=h]
    report["old_state"]=read(oldroot/"STATE.json")["status"]
    if report["protected_stop_mismatches"] or report["old_code_mismatches"] or report["old_job"].get("MainPID")!="0":
        report["passed"]=False
    # Preserve all data in the report, remove only this generated fixture directory.
    if base.resolve().parent!=parent.resolve() or not base.name.startswith("cellforge-v2-review-"):
        raise RuntimeError("refusing unexpected cleanup target")
    if all(q.get("empty") for q in cleanup):
        shutil.rmtree(base); report["fixture_files_cleaned"]=True
    else: report["fixture_files_cleaned"]=False
print(json.dumps(report,sort_keys=True))
raise SystemExit(0 if report.get("passed") else 1)

