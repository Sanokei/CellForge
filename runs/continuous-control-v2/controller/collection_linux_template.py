"""Actual isolated failure→backoff→next integration, including real collection crashes."""
import base64,copy,datetime,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,tempfile,time,uuid
payload=json.loads(base64.b64decode(PAYLOAD))
parent=pathlib.Path("/srv/cellforge-inbox")
base=pathlib.Path(tempfile.mkdtemp(prefix="cellforge-v2-collection-",dir=parent))
for name in ("common.py","supervisor.py"): (base/name).write_text(payload["files"][name])
(base/"worker.py").write_text("def compare_complete(config,rows): return False\n")
(base/"probe_driver.py").write_text(payload["driver"])
sys.path.insert(0,str(base))
import supervisor as V
from common import atomic,read,digest,filehash,fresh_registry,Blocked
original_command=V.command
processes=[]; unit_owners=[]
report={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "scope":"isolated operational fixtures; zero research admission/game play/model calls",
 "supervisor_sha256":hashlib.sha256(payload["files"]["supervisor.py"].encode()).hexdigest(),
 "common_sha256":hashlib.sha256(payload["files"]["common.py"].encode()).hexdigest(),
 "fixture_root":str(base),"cases":{},"manual_reset_failed_calls":0}

def until(fn,seconds=20):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=fn()
        if value: return value
        time.sleep(.2)
    raise RuntimeError("bounded fixture wait expired")
def select(unit):
    V.UNIT=unit; V.GROUP="/system.slice/"+unit
    def bounded(argv):
        return original_command([arg.replace("RuntimeMaxSec=600","RuntimeMaxSec=40") for arg in argv])
    return V.Systemd(runner=bounded)
def setup(name,prior_failures=0):
    root=base/name; root.mkdir()
    unit="cellforge-v2-collection-"+base.name.rsplit("-",1)[-1]+"-"+name+".service"
    p=copy.deepcopy(payload["plan"]); p["lab_runs_root"]=str(root/"lab"/"runs")
    ph=digest(p); fps=sorted(c["fingerprint"] for c in p["conditions"])
    bundle=root/"studies"/ph; bundle.mkdir(parents=True)
    atomic(bundle/"PLAN.json",p)
    (bundle/"worker.py").write_text(payload["worker"].replace("FAIL_FP",repr(fps[0])))
    s=fresh_registry(); s["scope"]="operational fixture, not scientific admission"
    s["studies"][ph]={"cap":9,"attempted":0,"condition_ids":fps}
    s["conditions"]={fp:{"study":ph,"status":"pending"} for fp in fps}
    s["consecutive_failures"]=prior_failures
    atomic(root/"REGISTRY.json",s)
    (root/"DISPATCH_ENABLED").write_text("isolated fixture marker only")
    def run(fp): return pathlib.Path(p["lab_runs_root"])/("weights-transition-"+ph)/"jobs"/fp
    unit_owners.append({"unit":unit,"root":root})
    return root,unit,ph,fps,run
def launch(root,unit):
    proc=subprocess.Popen([sys.executable,str(base/"probe_driver.py"),str(root),unit,str(base)],
                          stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    processes.append(proc); return proc
def dead(run):
    pids=read(run/"PROBE_PIDS.json")
    for pid in pids:
        stat=pathlib.Path("/proc")/str(pid)/"stat"
        assert not stat.exists() or stat.read_text().split()[2]=="Z"
    return pids

try:
    scenarios=[("baseline",None,0),("before-reset","before_failed_unit_collection",0),
               ("after-reset","after_failed_unit_reset",0),("after-collection","after_collection_before_registry",0),
               ("third-failure","after_failed_unit_reset",2)]
    for name,crashpoint,prior in scenarios:
        root,unit,ph,fps,run=setup(name,prior)
        backend=select(unit)
        if crashpoint: (root/"CRASH_POINT").write_text(crashpoint)
        driver=launch(root,unit)
        crash_record=None
        if crashpoint:
            assert driver.wait(timeout=25)==-signal.SIGKILL
            pre=read(root/"REGISTRY.json"); receipt=read(run(fps[0])/"TERMINAL.json")
            status=backend.show()
            assert pre["active"]==fps[0] and pre["consecutive_failures"]==prior and pre["sequence"]==0
            assert receipt["outcome"]=="failed"
            crash_record={"real_SIGKILL":True,"checkpoint":crashpoint,"registry_before_recovery":pre["consecutive_failures"],
                          "unit_at_crash":status,"receipt_id":receipt["receipt_id"]}
            (root/"CRASH_POINT").unlink(); driver=launch(root,unit)
        until(lambda: read(root/"REGISTRY.json")["active"] is None and read(root/"REGISTRY.json")["sequence"]==1)
        state=read(root/"REGISTRY.json"); receipt=read(run(fps[0])/"TERMINAL.json")
        receipt_hash=filehash(run(fps[0])/"TERMINAL.json")
        assert receipt["evidence"]["systemd_after_stop"]["LoadState"]=="loaded"
        assert receipt["evidence"]["systemd_after_stop"]["ActiveState"]=="failed"
        assert state["consecutive_failures"]==prior+1 and state["not_before"]==receipt["not_before_after"]
        assert sum(e["status"]=="pending" for e in state["conditions"].values())==8
        assert backend.show()["LoadState"]=="not-found" and backend.empty(backend.show())
        collection=read(run(fps[0])/"COLLECTION.json")
        assert collection["terminal_receipt_id"]==receipt["receipt_id"] and collection["unit_absent"]
        evidence={"crash":crash_record,"original_receipt":receipt,"collection":collection,
                  "failed_metadata_retained_before_collection":True,"pending_after_failure":8,
                  "receipt_sha256":receipt_hash,"failed_descendants_dead":dead(run(fps[0]))}
        if prior==2:
            until(lambda: read(root/"REGISTRY.json")["status"]=="blocked_failure_circuit")
            time.sleep(2.2)
            state=read(root/"REGISTRY.json")
            assert state["studies"][ph]["attempted"]==1 and state["consecutive_failures"]==3
            evidence.update(circuit_closed=True,launch_attempts=1,next_condition_not_launched=True)
        else:
            until(lambda: read(root/"REGISTRY.json")["status"]=="failure_backoff")
            assert time.time()<state["not_before"]
            before=read(root/"REGISTRY.json")
            assert before["studies"][ph]["attempted"]==1
            # Wait for the real original 30-second backoff; never edit deadlines/state.
            until(lambda: read(root/"REGISTRY.json")["active"]==fps[1],seconds=40)
            observed=time.time()
            second=read(root/"REGISTRY.json")
            assert observed>=receipt["not_before_after"]
            assert second["consecutive_failures"]==1 and second["sequence"]==1
            assert second["studies"][ph]["attempted"]==2
            assert fps[0]!=second["active"]
            second_invocation=backend.show()["InvocationID"]
            assert second_invocation!=receipt["evidence"]["systemd_before_stop"]["InvocationID"]
            until(lambda:(run(fps[1])/"PROBE_PIDS.json").exists())
            (root/"DISPATCH_ENABLED").unlink()
            until(lambda:read(root/"REGISTRY.json")["active"] is None)
            assert backend.show()["LoadState"]=="not-found" and backend.empty(backend.show())
            assert filehash(run(fps[0])/"TERMINAL.json")==receipt_hash
            evidence.update(real_backoff_seconds=receipt["not_before_after"]-receipt["terminal_epoch"],
              observed_next_start_epoch=observed,deadline_epoch=receipt["not_before_after"],
              next_distinct_fingerprint=fps[1],next_invocation=second_invocation,launch_attempts=2,
              next_descendants_dead=dead(run(fps[1])),first_receipt_unchanged=True)
        (root/"STOP").write_text("isolated fixture cleanup")
        driver.terminate(); driver.wait(timeout=8)
        report["cases"][name]=evidence
        print(json.dumps({"progress":name,"passed":True,"launch_attempts":evidence["launch_attempts"]}),flush=True)

    root,unit,ph,fps,run=setup("refusals")
    backend=select(unit); bundle=root/"studies"/ph
    location=run(fps[0]); location.mkdir(parents=True); (location/"scratch").mkdir()
    config=next(c for c in read(bundle/"PLAN.json")["conditions"] if c["fingerprint"]==fps[0])
    atomic(location/"CONFIG.json",config)
    token="isolated-refusal:"+uuid.uuid4().hex
    initial=backend.start(bundle,location,token)
    failed=until(lambda:(q if (q:=backend.show())["ActiveState"]=="failed" else None))
    backend.stop(token,initial["InvocationID"]); retained=backend.show()
    assert retained["LoadState"]=="loaded" and retained["ActiveState"]=="failed" and backend.empty(retained)
    refusals=[]
    for label,owner,invocation in [("unowned","someone-else",initial["InvocationID"]),
                                    ("invocation-mismatch",token,"different-invocation")]:
        try: backend.collect(owner,invocation); raise AssertionError("unsafe collection accepted")
        except Blocked: pass
        assert backend.show()["LoadState"]=="loaded" and backend.show()["InvocationID"]==initial["InvocationID"]
        refusals.append(label)
    # Inject only a failed read RPC; disrupting the shared real bus is outside scope.
    calls=[]
    def unknown(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv,1,"","injected read transport failure")
    unverified=V.Systemd(runner=unknown)
    try: unverified.collect(token,initial["InvocationID"]); raise AssertionError("unknown status accepted")
    except Blocked: pass
    assert calls and all(cmd[:2]==["systemctl","show"] for cmd in calls)
    assert backend.show()["LoadState"]=="loaded" and backend.show()["InvocationID"]==initial["InvocationID"]
    clear=backend.collect(token,initial["InvocationID"])
    assert clear["LoadState"]=="not-found" and backend.empty(clear)
    report["cases"]["refusals"]={"actual_retained_unit":retained,"unowned_refused":True,
        "invocation_mismatch_refused":True,"unknown_read_refused":True,"unknown_read_mutation_calls":0,
        "owned_collection_verified_absent":True,"detached_descendants_dead":dead(location)}
    report["passed"]=True
except Exception as exc:
    report["passed"]=False; report["error"]=type(exc).__name__+": "+str(exc)
finally:
    for owner in unit_owners:
        (owner["root"]/"STOP").write_text("isolated cleanup")
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=8)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)
    cleanup=[]
    for owner in unit_owners:
        backend=select(owner["unit"])
        try:
            status=backend.show()
            if status["LoadState"]=="loaded":
                assert status["Description"].startswith("cellforge-v2:")
                token=status["Description"].split("cellforge-v2:",1)[1]
                invocation=status["InvocationID"]
                backend.stop(token,invocation)
                backend.collect(token,invocation)
            status=backend.show(); assert status["LoadState"]=="not-found" and backend.empty(status)
            cleanup.append({"unit":owner["unit"],"absent":True,"empty":True})
        except Exception as exc:
            cleanup.append({"unit":owner["unit"],"error":str(exc)}); report["passed"]=False
    report["cleanup"]=cleanup
    old=pathlib.Path("/srv/cellforge-inbox/continuous-spatial-20261011"); plan=read(old/"PLAN.json")
    report["old_code_mismatches"]=[n for n,h in plan["code_sha256"].items() if filehash(old/n)!=h]
    report["protected_stop_mismatches"]=[n for n,h in plan["protected_stops"].items() if filehash(n)!=h]
    report["old_state"]=read(old/"STATE.json")["status"]
    raw=original_command(["systemctl","show","cellforge-continuous-job.service","--property=LoadState,MainPID,ActiveState"])
    report["old_job"]=dict(s.split("=",1) for s in raw.stdout.splitlines())
    if report["old_code_mismatches"] or report["protected_stop_mismatches"] or report["old_job"]["MainPID"]!="0":
        report["passed"]=False
    if base.resolve().parent!=parent.resolve() or not base.name.startswith("cellforge-v2-collection-"):
        raise RuntimeError("unexpected cleanup root")
    if all(c.get("absent") and c.get("empty") for c in cleanup):
        shutil.rmtree(base); report["fixture_files_cleaned"]=True
    else: report["fixture_files_cleaned"]=False
    report["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(report,sort_keys=True),flush=True)
raise SystemExit(0 if report.get("passed") else 1)

