"""Focused offline regressions. Fixtures do not play ARC3 or start systemd."""
import copy
import contextlib
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import common
import supervisor as V
import policy_transition as P
import scripted_explorer as S
import arc3cell as C
from worker import compare_complete
from analyze_results import decision
from common import *

HERE=pathlib.Path(__file__).parent
PLAN=read(HERE/"PLAN.json")
def info(**values):
    d={k:"" for k in V.PROPERTIES}
    d.update(LoadState="not-found",ActiveState="inactive",SubState="dead",Result="success",
             ExecMainStatus="0",MainPID="0",CPUUsageNSec="0",MemoryPeak="0")
    d.update(values)
    return d
def result(d,code=0):
    return subprocess.CompletedProcess([],code,"\n".join(k+"="+v for k,v in d.items())+"\n","fixture error" if code else "")
def approval(plan,manifest="m"):
    return {"plan_sha256":digest(plan),"bundle_manifest_sha256":manifest,"audit_report_sha256":"a"*64,
            "decision":"permit_baseline_development","qualified":False,"remote_model_calls":0}

class StatusSafety(unittest.TestCase):
    def test_empty_dict_is_unknown(self):
        with self.assertRaises(Blocked): V.Systemd().empty({})
    def test_failed_show_never_certifies_empty(self):
        with self.assertRaises(Blocked): V.Systemd(runner=lambda _:result(info(),1)).show()
    def test_missing_properties_are_unknown(self):
        for field in V.PROPERTIES:
            d=info(); del d[field]
            with self.subTest(field=field),self.assertRaises(Blocked):
                V.Systemd(runner=lambda _:result(d)).show()
    def test_running_requires_cgroup_pid_invocation(self):
        for field in ("ControlGroup","MainPID","InvocationID"):
            d=info(LoadState="loaded",ActiveState="active",SubState="running",MainPID="22",
                   ControlGroup=V.GROUP,InvocationID="inv")
            d[field]="0" if field=="MainPID" else ""
            with self.subTest(field=field),self.assertRaises(Blocked):
                V.Systemd(runner=lambda _:result(d)).show()
    def test_blank_controlgroup_still_checks_expected_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=V.Systemd(runner=lambda _:result(info()),cgroup_root=tmp)
            d=b.show(); p=pathlib.Path(tmp)/V.GROUP.lstrip("/"); p.mkdir(parents=True)
            (p/"cgroup.events").write_text("populated 1\nfrozen 0\n")
            self.assertFalse(b.empty(d))
            (p/"cgroup.events").write_text("populated 0\nfrozen 0\n")
            self.assertTrue(b.empty(d))
    def test_existing_tree_missing_events_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=V.Systemd(runner=lambda _:result(info()),cgroup_root=tmp); d=b.show()
            p=pathlib.Path(tmp)/V.GROUP.lstrip("/"); p.mkdir(parents=True)
            with self.assertRaises(Blocked): b.empty(d)
            (p/"cgroup.events").write_text("frozen 0\n")
            with self.assertRaises(Blocked): b.empty(d)
    def test_transport_timeout_is_unknown(self):
        with patch("supervisor.subprocess.run",side_effect=subprocess.TimeoutExpired([],15)):
            with self.assertRaises(Blocked): V.command(["systemctl"])
    def test_unowned_description_and_invocation_block(self):
        d=info(LoadState="loaded",Description="cellforge-v2:other",InvocationID="inv"); d["_verified"]=True
        with self.assertRaises(Blocked): V.Systemd().owned(d,"owner")
        d["Description"]="cellforge-v2:owner"
        with self.assertRaises(Blocked): V.Systemd().owned(d,"owner","different")
    def test_bad_cgroup_path_blocks(self):
        with self.assertRaises(Blocked):
            V.Systemd(runner=lambda _:result(info(ControlGroup="/other"))).show()

class ScienceAndAdmission(unittest.TestCase):
    def test_protocol_labels_paths_provenance_not_identity(self):
        c=copy.deepcopy(PLAN["conditions"][0]); fp=science_identity(c)
        c.update(label="new title",timestamp="later",supervisor_sha256="new",provenance={"new":"packet"},output="/other")
        for a in c["arms"]:
            a["name"]="renamed:"+a["name"]; a["source"]="# new packaging comment\n"+a["source"]
        c["arms"].reverse()
        self.assertEqual(fp,science_identity(c))
    def test_actual_parameter_seed_data_version_change_identity(self):
        original=PLAN["conditions"][0]
        for field in ("params","seed","data","version"):
            c=copy.deepcopy(original)
            if field=="params": c["arms"][0]["params"]["action"][1]+=0.1
            if field=="seed": c["seed"]=3
            if field=="data": c["dataset_sha256"]="different"
            if field=="version": c["packages"]["numpy"]="different"
            self.assertNotEqual(science_identity(original),science_identity(c))
    def test_global_duplicate_blocks_new_study_title(self):
        s=fresh_registry(); admit(s,PLAN,approval(PLAN),"m")
        p=copy.deepcopy(PLAN); p["status"]="retitled packaging"
        with self.assertRaises(Blocked): admit(s,p,approval(p),"m")
    def test_consumed_duplicate_blocks_new_root_plan(self):
        s=fresh_registry(); s["conditions"][PLAN["conditions"][0]["fingerprint"]]={"status":"completed"}
        with self.assertRaises(Blocked): admit(s,PLAN,approval(PLAN),"m")
    def test_plan_cap_and_resource_fields_frozen(self):
        for key,value in (("study_cap",10),("remote_model_calls",1),("qualified",True)):
            p=copy.deepcopy(PLAN); p[key]=value
            with self.assertRaises(Blocked): validate_plan(p)
    def test_approval_exact_binding_required(self):
        for key,value in (("plan_sha256","bad"),("bundle_manifest_sha256","bad"),("decision","approve"),
                          ("qualified",True),("remote_model_calls",1),("audit_report_sha256",None)):
            a=approval(PLAN); a[key]=value
            with self.subTest(key=key),self.assertRaises(Blocked): admit(fresh_registry(),PLAN,a,"m")
    def test_invalid_draw_is_rejected_without_repair(self):
        p=copy.deepcopy(PLAN["founders"][0]["params"]); p["action"][0]=9
        with self.assertRaises(Blocked): validate_params(p)
        p["action"][0]=0; p["pixel"][0]=5
        with self.assertRaises(Blocked): validate_params(p)
    def test_random_and_primary_support_masks_match(self):
        for f in PLAN["founders"]:
            validate_params(f["params"]); validate_params(f["random_params"])
            for k in ("pixel","memory_gate","context_gate"): self.assertEqual(f["params"][k],f["random_params"][k])
    def test_cap_enforced_at_reservation(self):
        s=fresh_registry(); ph=admit(s,PLAN,approval(PLAN),"m"); s["studies"][ph]["attempted"]=9
        with self.assertRaises(Blocked): reserve(s,PLAN["conditions"][0]["fingerprint"])
    def test_failure_circuit_survives_new_plan(self):
        s=fresh_registry(); s["consecutive_failures"]=3
        with self.assertRaises(Blocked): admit(s,PLAN,approval(PLAN),"m")
    def test_manifest_discovery_and_schema(self):
        paths=list((HERE/"preregistration").glob("weights-*/STANDARD.json")); self.assertEqual(len(paths),1)
        m=read(paths[0])
        for k in ("contract","evaluator","arms","seeds","seeds_declared_before_run","budget_per_arm",
                  "niches","founders","genomes","proposals","eval_seeds"): self.assertIn(k,m)
        for arm in ("llm","random"):
            for seed in SEEDS:
                self.assertEqual(sum(p["charged"] for p in m["proposals"] if p["arm"]==arm and p["seed"]==seed),1)
        self.assertEqual(m["literal_positions_charged"],{"llm":96,"random":96})
        for e in m["genomes"]: self.assertEqual(digest(read(paths[0].parent/e["weights_file"])),e["hash"])
    def test_semantic_adapter_explicitly_unqualified(self):
        import standard_interactive_adapter as adapter
        with self.assertRaises(adapter.InteractiveContractUnsupported): adapter.make_tasks(0,1)

class ReceiptReplay(unittest.TestCase):
    def setup_state(self):
        s=fresh_registry(); admit(s,PLAN,approval(PLAN),"m")
        return s
    def test_third_failed_receipt_opens_circuit_exactly_once(self):
        s=self.setup_state(); s["consecutive_failures"]=2; fp=PLAN["conditions"][0]["fingerprint"]; s["active"]=fp
        r=make_terminal(s,fp,"failed",{},epoch=1000)
        old=copy.deepcopy(s)
        self.assertTrue(replay_terminal(old,r))
        self.assertEqual(old["consecutive_failures"],3); self.assertEqual(old["not_before"],1120)
        self.assertFalse(replay_terminal(old,r)); self.assertEqual(old["consecutive_failures"],3)
    def test_backoff_uses_original_terminal_epoch(self):
        s=self.setup_state(); fp=PLAN["conditions"][0]["fingerprint"]
        r=make_terminal(s,fp,"failed",{},epoch=50)
        with patch("common.time.time",return_value=9000): replay_terminal(s,r)
        self.assertEqual(s["not_before"],80)
    def test_sequence_gap_and_digest_conflict_block(self):
        s=self.setup_state(); fp=PLAN["conditions"][0]["fingerprint"]; r=make_terminal(s,fp,"failed",{})
        r["outcome"]="completed"
        with self.assertRaises(Blocked): replay_terminal(s,r)
        r=make_terminal(s,fp,"failed",{}); r["previous_sequence"]=10; r["receipt_id"]=digest({k:v for k,v in r.items() if k!="receipt_id"})
        with self.assertRaises(Blocked): replay_terminal(s,r)
    def test_completed_resets_failures_cancel_does_not(self):
        s=self.setup_state(); fp=PLAN["conditions"][0]["fingerprint"]; s["consecutive_failures"]=2
        replay_terminal(s,make_terminal(s,fp,"cancelled",{})); self.assertEqual(s["consecutive_failures"],2)
        replay_terminal(s,make_terminal(s,fp,"completed",{})); self.assertEqual(s["consecutive_failures"],0)

class Fake(V.Systemd):
    """Model real stopped failures: failed metadata persists until reset-failed."""
    def __init__(self,tmp):
        super().__init__(runner=self.command,cgroup_root=tmp)
        self.value=info(); self.value["_verified"]=True
        self.starts=0; self.stops=0; self.collections=0
    def show(self): return copy.deepcopy(self.value)
    def command(self,argv):
        if argv[:2]==["systemctl","stop"]:
            self.stops+=1
            if self.value["ActiveState"]=="failed":
                self.value.update(MainPID="0",ControlGroup="")
            else:
                self.value=info(); self.value["_verified"]=True
        elif argv[:2]==["systemctl","reset-failed"]:
            self.collections+=1
            self.value=info(); self.value["_verified"]=True
        else: raise Blocked("unsupported fixture command")
        return subprocess.CompletedProcess(argv,0,"","")
    def start(self,bundle,run,token):
        self.starts+=1
        invocation="fixture-inv" if self.starts==1 else "fixture-inv-"+str(self.starts)
        self.value=info(LoadState="loaded",ActiveState="active",SubState="running",MainPID="42",
                       ControlGroup=V.GROUP,InvocationID=invocation,Description="cellforge-v2:"+token)
        self.value["_verified"]=True
        return self.show()


class Crash(RuntimeError): pass
class ControllerCrashPoints(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=pathlib.Path(self.tmp.name)
        p=copy.deepcopy(PLAN); p["lab_runs_root"]=str(self.root/"lab"/"runs")
        self.plan=p; self.ph=digest(p); self.fp=min(c["fingerprint"] for c in p["conditions"])
        (self.root/"studies"/self.ph).mkdir(parents=True)
        atomic(self.root/"studies"/self.ph/"PLAN.json",p)
        self.state=fresh_registry(); admit(self.state,p,approval(p),"m"); atomic(self.root/"REGISTRY.json",self.state)
        self.backend=Fake(self.root/"cgroups")
        self.lock=patch("supervisor.registry_lock",lambda root:contextlib.nullcontext()); self.lock.start()
        self.gate=patch.object(V.Controller,"gates",return_value=None); self.gate.start()
        self.controller=V.Controller(self.root,self.backend)
    def tearDown(self):
        self.gate.stop(); self.lock.stop(); self.tmp.cleanup()
    def enable(self): (self.root/"DISPATCH_ENABLED").write_text("fixture only")
    def fail_after(self,point):
        def crash(actual):
            if actual==point: raise Crash(actual)
        return crash
    def test_no_marker_means_no_launch(self):
        self.controller.tick(); self.assertEqual(self.backend.starts,0)
        self.assertEqual(read(self.root/"REGISTRY.json")["status"],"paused_dispatch_disabled")
    def test_stop_priority_means_no_launch(self):
        self.enable(); (self.root/"STOP").write_text("fixture")
        self.controller.tick(); self.assertEqual(self.backend.starts,0)
    def test_crash_after_intent_consumes_once_without_relaunch(self):
        self.enable(); self.controller.crash=self.fail_after("after_launch_intent")
        with self.assertRaises(Crash): self.controller.tick()
        self.assertEqual(self.backend.starts,0)
        V.Controller(self.root,self.backend).tick()
        s=read(self.root/"REGISTRY.json")
        self.assertEqual(s["conditions"][self.fp]["status"],"interrupted")
        self.assertEqual(s["studies"][self.ph]["attempted"],1); self.assertEqual(self.backend.starts,0)
    def test_crash_after_launch_adopts_same_invocation(self):
        self.enable(); self.controller.crash=self.fail_after("after_launch")
        with self.assertRaises(Crash): self.controller.tick()
        V.Controller(self.root,self.backend).tick()
        self.assertEqual(self.backend.starts,1)
        self.assertEqual(read(self.root/"REGISTRY.json")["conditions"][self.fp]["invocation"],"fixture-inv")
    def test_crash_after_third_failure_receipt_before_state(self):
        self.enable(); self.controller.tick()
        s=read(self.root/"REGISTRY.json"); s["consecutive_failures"]=2; atomic(self.root/"REGISTRY.json",s)
        self.backend.value.update(ActiveState="failed",SubState="failed",MainPID="0",Result="exit-code",ExecMainStatus="2")
        self.controller.crash=self.fail_after("after_receipt_before_registry")
        with self.assertRaises(Crash): self.controller.tick()
        run=self.controller.context(s,self.fp)[3]; r=read(run/"TERMINAL.json")
        self.assertEqual(read(self.root/"REGISTRY.json")["consecutive_failures"],2)
        recovered=V.Controller(self.root,self.backend); recovered.tick(); recovered.tick()
        fresh=read(self.root/"REGISTRY.json")
        self.assertEqual(fresh["consecutive_failures"],3); self.assertEqual(fresh["not_before"],r["not_before_after"])
        self.assertEqual(fresh["status"],"blocked_failure_circuit"); self.assertEqual(self.backend.starts,1)
    def test_crash_after_state_commit_does_not_recount(self):
        self.enable(); self.controller.tick()
        self.backend.value.update(ActiveState="failed",SubState="failed",MainPID="0",Result="exit-code",ExecMainStatus="1")
        self.controller.crash=self.fail_after("after_registry_commit")
        with self.assertRaises(Crash): self.controller.tick()
        V.Controller(self.root,self.backend).tick()
        s=read(self.root/"REGISTRY.json"); self.assertEqual(s["consecutive_failures"],1); self.assertEqual(s["sequence"],1)
    def test_unknown_status_blocks_all_dispatch_and_receipts(self):
        self.enable()
        with patch.object(self.backend,"show",side_effect=Blocked("transport")):
            with self.assertRaises(Blocked): self.controller.tick()
        self.assertEqual(self.backend.starts,0); self.assertEqual(read(self.root/"REGISTRY.json")["sequence"],0)
    def test_pending_cancel_consumes_cap_without_launch(self):
        self.enable(); (self.root/"cancel").mkdir(); (self.root/"cancel"/self.fp).write_text("fixture")
        self.controller.tick(); s=read(self.root/"REGISTRY.json")
        self.assertEqual(self.backend.starts,0); self.assertEqual(s["conditions"][self.fp]["status"],"cancelled")
        self.assertEqual(s["studies"][self.ph]["attempted"],1)
    def test_unowned_unit_not_stopped_or_replaced(self):
        self.enable()
        self.backend.value.update(LoadState="loaded",ActiveState="active",SubState="running",MainPID="22",
                                  Description="other",ControlGroup=V.GROUP,InvocationID="other-inv")
        with self.assertRaises(Blocked): self.controller.tick()
        self.assertEqual(self.backend.stops,0); self.assertEqual(self.backend.starts,0)

class PolicyFixtures(unittest.TestCase):
    def setUp(self):
        self.params={k:np.array(v,dtype=float) for k,v in PLAN["founders"][0]["params"].items()}
        self.frame=np.zeros((64,64),int); self.frame[10:20,15:25]=2
        self.info={"resets":0,"levels":0,"step":0,"key":"fixture"}
    def test_existing_sandbox_accepts_all_new_sources(self):
        for a in PLAN["conditions"][0]["arms"]:
            self.assertEqual(C.validate(a["source"]),(True,""),a["name"])
    def test_colour_renaming_invariant_features(self):
        f=self.frame.copy(); f[f==2]=7; f[f==0]=9
        np.testing.assert_allclose(P.observation_features(self.frame),P.observation_features(f))
    def test_dynamic_legal_actions_and_reset(self):
        mem={}
        P.act(self.params,mem,self.frame,[1],self.info)
        output=P.act(self.params,mem,self.frame,[1,7],dict(self.info,step=1))
        self.assertIn(output[0],[1,7]); self.assertEqual(len(mem["global"]),8)
        P.act(self.params,mem,self.frame,[6],dict(self.info,resets=1,step=2))
        self.assertEqual(len(mem["eligibility"]),1); self.assertEqual(len(mem["nodes"]),1)
    def test_context_off_is_exact_aggregate_path(self):
        p=copy.deepcopy(self.params); p["action"]=np.array([0,0,8,0,0,0,0,0]); p["lookahead"]=np.zeros(4)
        mem={}; P.act(p,mem,self.frame,[1,2],self.info)
        mem.pop("prev_features"); node=next(iter(mem["nodes"].values()))
        mem["global"][2,9]=1; node["rows"][1,9]=5; node["rows"][:,10]=100
        contextual=P.act(p,copy.deepcopy(mem),self.frame,[1,2],self.info)
        p["context_gate"]=np.array([0])
        aggregate=P.act(p,copy.deepcopy(mem),self.frame,[1,2],self.info)
        self.assertEqual(contextual[0],1); self.assertEqual(aggregate[0],2)
    def test_delayed_public_progress_credit(self):
        mem={}
        for step in range(4):
            P.act(self.params,mem,self.frame,[1,2],dict(self.info,step=step,levels=int(step==3)))
        self.assertGreater(mem["global"][:,9].sum(),0)
    def test_scripted_legal_and_spatial_match_single_action(self):
        a=P.act(self.params,{},self.frame,[6],self.info)
        b=S.act({"pixel":np.array(PIXEL)}, {},self.frame,[6],self.info)
        self.assertEqual(a,b)

class ComparisonGate(unittest.TestCase):
    def rows(self,c,full_levels=0,multi=True):
        rows=[]
        for a in c["arms"]:
            for g in c["games"]:
                rows.append({"arm":a["name"],"initial":{"valid":True,"frame_sha256":"fixture","legal_actions":[1,2]},
                   "max_legal_actions":2 if multi else 1,
                   "result":{"game":g,"seed":c["seed"],"budget":1500,"used":1500,"state":"NOT_FINISHED",
                             "error":None,"levels":full_levels if a["name"]=="full" and g=="ar25" else 0}})
        return rows
    def test_missing_arm_timeout_mismatched_initial_are_incomplete(self):
        c=PLAN["conditions"][0]
        for reason in ("missing","timeout","initial"):
            rows=self.rows(c)
            if reason=="missing": rows.pop()
            if reason=="timeout": rows[0]["result"].update(error="wall",used=20)
            if reason=="initial": rows[0]["initial"]["frame_sha256"]="different"
            self.assertFalse(compare_complete(c,rows))
    def test_single_action_gains_cannot_pass(self):
        results={c["fingerprint"]:self.rows(c,full_levels=1,multi=False) for c in PLAN["conditions"]}
        self.assertFalse(any(f["engineering_gate_passed"] for f in decision(PLAN,results)["founders"]))
    def test_same_founder_two_seed_all_comparator_gate(self):
        results={c["fingerprint"]:self.rows(c,full_levels=1) for c in PLAN["conditions"]}
        output=decision(PLAN,results)
        self.assertTrue(all(f["engineering_gate_passed"] for f in output["founders"]))
        self.assertFalse(output["qualified"]); self.assertFalse(output["automatic_promotion"])
        # Equal performance with any required comparator defeats the gate.
        for c in PLAN["conditions"]:
            for row in results[c["fingerprint"]]:
                if row["arm"]=="context_off" and row["result"]["game"]=="ar25": row["result"]["levels"]=1
        self.assertFalse(any(f["engineering_gate_passed"] for f in decision(PLAN,results)["founders"]))


class FinalControllerGuards(ControllerCrashPoints):
    # Four added cases and nine inherited controller regressions are executed.
    def test_success_summary_without_results_is_failed(self):
        self.enable(); self.controller.tick()
        s=read(self.root/"REGISTRY.json"); run=self.controller.context(s,self.fp)[3]
        atomic(run/"SUMMARY.json",{"fingerprint":self.fp,"plays":27,"comparison_complete":True})
        self.backend.value.update(SubState="exited",MainPID="0")
        self.controller.tick()
        self.assertEqual(read(self.root/"REGISTRY.json")["conditions"][self.fp]["status"],"failed")
    def test_complete_records_can_finish_and_reset_counter(self):
        self.enable(); self.controller.tick()
        s=read(self.root/"REGISTRY.json"); s["consecutive_failures"]=2; atomic(self.root/"REGISTRY.json",s)
        config=next(c for c in self.plan["conditions"] if c["fingerprint"]==self.fp)
        run=self.controller.context(s,self.fp)[3]
        atomic(run/"RESULTS.json",ComparisonGate().rows(config))
        atomic(run/"SUMMARY.json",{"fingerprint":self.fp,"plays":27,"comparison_complete":True})
        self.backend.value.update(SubState="exited",MainPID="0")
        self.controller.tick(); s=read(self.root/"REGISTRY.json")
        self.assertEqual(s["conditions"][self.fp]["status"],"completed")
        self.assertEqual(s["consecutive_failures"],0)
    def test_explicit_stop_persists_and_cancels_owned_tree(self):
        self.enable(); self.controller.tick()
        with patch("supervisor.Systemd",return_value=self.backend),patch("sys.argv",["supervisor","--root",str(self.root),"--stop"]),patch.dict("os.environ",{"SERVICE_RESULT":"success"}):
            V.main()
        self.assertTrue((self.root/"STOP").exists()); self.assertEqual(self.backend.starts,1)
        self.controller.tick()
        self.assertEqual(read(self.root/"REGISTRY.json")["conditions"][self.fp]["status"],"cancelled")
    def test_crash_execstop_does_not_synthesize_stop(self):
        self.enable(); self.controller.tick()
        with patch("supervisor.Systemd",return_value=self.backend),patch("sys.argv",["supervisor","--root",str(self.root),"--stop"]),patch.dict("os.environ",{"SERVICE_RESULT":"signal"}):
            V.main()
        self.assertFalse((self.root/"STOP").exists()); self.assertEqual(self.backend.stops,0)
        self.controller.tick(); self.assertEqual(self.backend.starts,1)

class ImmutableAdmission(unittest.TestCase):
    def test_bundle_and_audit_bytes_bound(self):
        import admission
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp); atomic(p/"PLAN.json",PLAN)
            manifest={"files":{"PLAN.json":filehash(p/"PLAN.json")}}; atomic(p/"MANIFEST.json",manifest)
            (p/"AUDIT_REPORT.md").write_text("fixture audit")
            a=approval(PLAN,filehash(p/"MANIFEST.json")); a["audit_report_sha256"]=filehash(p/"AUDIT_REPORT.md"); atomic(p/"review.json",a)
            admission.verify_bundle(p,p/"review.json",p/"AUDIT_REPORT.md")
            (p/"AUDIT_REPORT.md").write_text("tampered")
            with self.assertRaises(Blocked): admission.verify_bundle(p,p/"review.json",p/"AUDIT_REPORT.md")
            (p/"PLAN.json").write_text("{}")
            with self.assertRaises(Blocked): admission.verify_bundle(p,p/"review.json",p/"AUDIT_REPORT.md")

if __name__=="__main__": unittest.main(verbosity=2)

