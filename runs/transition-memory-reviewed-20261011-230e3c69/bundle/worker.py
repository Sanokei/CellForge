"""One frozen public-development comparison; this module does not run on import."""
import hashlib
import pathlib
import sys
import time
import numpy as np
import arc3_env as E
import arc3cell as C
from common import *

def sha(data): return hashlib.sha256(data).hexdigest()
def compare_complete(config,rows):
    if len(rows)!=len(config["arms"])*len(config["games"]): return False
    if {(r["arm"],r["result"]["game"]) for r in rows}!={(a["name"],g) for a in config["arms"] for g in config["games"]}: return False
    for row in rows:
        r=row["result"]
        if r["error"] or r["budget"]!=config["actions"] or r["seed"]!=config["seed"]: return False
        if r["used"]!=config["actions"] and r["state"]!="WIN": return False
        if not row["initial"] or not row["initial"]["valid"]: return False
    for game in config["games"]:
        initial=[r["initial"] for r in rows if r["result"]["game"]==game]
        if any(x!=initial[0] for x in initial[1:]): return False
    return True

def main(run,bundle):
    run=pathlib.Path(run).resolve(); bundle=pathlib.Path(bundle).resolve()
    config=read(run/"CONFIG.json")
    if config["fingerprint"]!=science_identity(config): raise Blocked("science identity mismatch")
    plan=read(bundle/"PLAN.json")
    if config not in plan["conditions"]: raise Blocked("config not in approved frozen plan")
    validate_plan(plan)
    rows=[]
    start=time.monotonic()
    for arm in config["arms"]:
        ok,why=C.validate(arm["source"])
        if not ok: raise Blocked("invalid proposal/control (charged, no redraw): "+why)
        shapes={k:[len(v)] for k,v in arm["params"].items()}
        ok,why=C.check_shapes(shapes,arm["params"])
        if not ok: raise Blocked("invalid proposal/control: "+why)
        for game in config["games"]:
            # The service controller also watches global STOP and dispatch marker continuously.
            if (run/"STOP").exists(): raise Blocked("run STOP")
            metadata={"initial":None,"legal_masks":{}}
            original_env=E.new_env; original_policy=C.Policy
            def observed_env(game_arg,seed_arg=0):
                arc,card,env=original_env(game_arg,seed_arg)
                resp=env.observation_space
                if resp is None:
                    metadata["initial"]={"valid":False}
                else:
                    frame=E.last_frame(resp)
                    metadata["initial"]={"valid":True,"frame_sha256":sha(np.asarray(frame,dtype=np.uint8).tobytes()),
                      "shape":list(frame.shape),"state":resp.state.name,"levels":int(resp.levels_completed),
                      "legal_actions":[int(a) for a in resp.available_actions if a!=0]}
                return arc,card,env
            class ObservedPolicy(original_policy):
                def step(self,frame,avail,info):
                    key=",".join(str(a) for a in avail)
                    metadata["legal_masks"][key]=metadata["legal_masks"].get(key,0)+1
                    return super().step(frame,avail,info)
            E.new_env=observed_env; C.Policy=ObservedPolicy
            try:
                result=E.play(arm["source"],arm["params"],game,config["actions"],seed=config["seed"],
                              blind=bool(arm.get("blind",False)),wall=45,keep_trace=True)
            finally:
                E.new_env=original_env; C.Policy=original_policy
            final=result.pop("final_frame",None)
            result["final_frame_sha256"]=sha((final or "").encode())
            row={"arm":arm["name"],"result":result,"initial":metadata["initial"],
                 "legal_masks":metadata["legal_masks"],
                 "max_legal_actions":max((len(k.split(",")) if k else 0 for k in metadata["legal_masks"]),default=0)}
            rows.append(row); atomic(run/"RESULTS.json",rows)
            atomic(run/"PROGRESS.json",{"completed_plays":len(rows),"expected_plays":27,"last_arm":arm["name"],"last_game":game})
    complete=compare_complete(config,rows)
    atomic(run/"SUMMARY.json",{"fingerprint":config["fingerprint"],"comparison_complete":complete,
           "plays":len(rows),"seconds":time.monotonic()-start,
           "qualified":False,"promoted":False,"remote_model_calls":0})
    return 0 if complete else 2

if __name__=="__main__":
    try: raise SystemExit(main(sys.argv[1],sys.argv[2]))
    except Exception as exc:
        atomic(pathlib.Path(sys.argv[1])/"FAILURE.json",{"error":type(exc).__name__+": "+str(exc)[:800]})
        raise SystemExit(1)

