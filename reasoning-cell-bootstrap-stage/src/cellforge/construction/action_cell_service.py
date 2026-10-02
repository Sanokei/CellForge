"""Bounded persistent cell-only service. Maker/anchor waiting is explicitly not training."""
import argparse,json,os,pathlib,time,datetime
from .action_cell_continuation import atomic,initial_state,prompts_for,compute_phase,successor,append_charge
from .experiment_manifest import load_json_strict
from .joint_weight_population import digest

def supervise(root,anchor,control_path,max_seconds=1800,poll_seconds=2):
    root=pathlib.Path(root).resolve();control_path=pathlib.Path(control_path).resolve()
    if control_path.is_relative_to(root):raise ValueError("external controller anchors cannot be run artifacts")
    if not 1<=max_seconds<=1800 or not 1<=poll_seconds<=60:raise ValueError("bounded supervisor resources")
    lock=root/"SERVICE.lock";fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    deadline=time.monotonic()+max_seconds
    waiting=None
    def status(kind,**details):
        receipt={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),
                 "status":kind,"initial_anchor":anchor,**details}
        atomic(root/"SERVICE_STATUS.json",receipt)
        print(json.dumps(receipt),flush=True)
    try:
        if not (root/"STATE-0.json").exists():
            state=initial_state(root,anchor);atomic(root/"STATE-0.json",state);atomic(root/"PROMPTS-1.json",prompts_for(state))
            status("founders_remeasured",phase=0,state_hash=digest(state))
        phases=[i for i in range(4) if (root/f"STATE-{i}.json").exists()]
        state=load_json_strict(root/f"STATE-{max(phases)}.json")
        while state["phase"]<3 and time.monotonic()<deadline:
            phase=state["phase"]+1
            if not (root/f"PROMPTS-{phase}.json").exists():atomic(root/f"PROMPTS-{phase}.json",prompts_for(state))
            if not control_path.exists() or not (root/f"RESPONSES-{phase}.json").exists():
                waiting="literal_maker_and_external_anchor"
                status("awaiting_literal_maker_and_external_anchor",next_phase=phase)
                time.sleep(poll_seconds);continue
            control=load_json_strict(control_path)
            if control.get("initial_anchor")!=anchor:raise ValueError("external controller initial anchor mismatch")
            phase_anchor=control.get("phase_anchors",{}).get(str(phase))
            if not phase_anchor:
                waiting="external_phase_anchor"
                status("awaiting_external_phase_anchor",next_phase=phase);time.sleep(poll_seconds);continue
            if (root/f"GENERATION-{phase}.json").exists():
                result=load_json_strict(root/f"GENERATION-{phase}.json")
                if result["phase_anchor"]!=phase_anchor or result["initial_anchor"]!=anchor:raise ValueError("recovery commitment mismatch")
                state=successor(state,result);atomic(root/f"STATE-{phase}.json",state)
                if phase<3:atomic(root/f"PROMPTS-{phase+1}.json",prompts_for(state))
                status("committed_generation_recovered",phase=phase);continue
            if (root/f"CHARGES-{phase}.jsonl").exists():
                raise ValueError("partial charged evaluation has no committed result; never recharge")
            status("evaluating_charged_cell_generation",phase=phase)
            started=time.monotonic()
            result=compute_phase(root,anchor,phase,phase_anchor,state,lambda e:append_charge(root,phase,e))
            atomic(root/f"GENERATION-{phase}.json",result);state=successor(state,result);atomic(root/f"STATE-{phase}.json",state)
            if phase<3:atomic(root/f"PROMPTS-{phase+1}.json",prompts_for(state))
            status("generation_complete",phase=phase,evaluation_seconds=time.monotonic()-started,
                   state_hash=digest(state),decisions=[{"seed":c["seed"],"arms":{a:v["decision"] for a,v in c["arms"].items()}}
                                                    for c in result["contexts"]])
        status("declared_generation_budget_complete" if state["phase"]==3 else "bounded_service_deadline",phase=state["phase"],waiting_for=waiting if state["phase"]<3 else None)
    except Exception as error:
        status("blocked",error=type(error).__name__+":"+str(error));raise
    finally:lock.unlink()

def main():
    p=argparse.ArgumentParser();p.add_argument("root",type=pathlib.Path);p.add_argument("--anchor",required=True)
    p.add_argument("--external-controller",type=pathlib.Path,required=True);p.add_argument("--max-seconds",type=int,default=1800)
    p.add_argument("--poll-seconds",type=int,default=2);a=p.parse_args()
    supervise(a.root,a.anchor,a.external_controller,a.max_seconds,a.poll_seconds)
if __name__=="__main__":main()
