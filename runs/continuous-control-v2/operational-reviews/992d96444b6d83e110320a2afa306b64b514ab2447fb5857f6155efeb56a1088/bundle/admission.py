"""Review-gated intake; defaults to validation only. Never creates dispatch marker."""
import argparse
import pathlib
import shutil
from common import *
from build_plan import LAB_RUNS

def verify_bundle(bundle,approval_path,audit_path):
    bundle=pathlib.Path(bundle)
    manifest=read(bundle/"MANIFEST.json")
    for name,h in manifest["files"].items():
        if pathlib.PurePath(name).is_absolute() or ".." in pathlib.PurePath(name).parts or filehash(bundle/name)!=h:
            raise Blocked("immutable bundle mismatch")
    plan=read(bundle/"PLAN.json"); validate_plan(plan)
    if plan["lab_runs_root"]!=LAB_RUNS: raise Blocked("canonical lab discovery root differs")
    approval=read(approval_path)
    if filehash(audit_path)!=approval.get("audit_report_sha256"): raise Blocked("audit bytes mismatch")
    # Never accept a self-generated approval in this preparation workflow.
    admit(fresh_registry(),plan,approval,filehash(bundle/"MANIFEST.json"))
    return plan,manifest,approval

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",required=True,type=pathlib.Path)
    parser.add_argument("--root",required=True,type=pathlib.Path)
    parser.add_argument("--approval",required=True,type=pathlib.Path)
    parser.add_argument("--audit",required=True,type=pathlib.Path)
    parser.add_argument("--legacy-root",required=True,type=pathlib.Path)
    parser.add_argument("--commit-reviewed-admission",action="store_true")
    args=parser.parse_args()
    plan,manifest,approval=verify_bundle(args.bundle,args.approval,args.audit)
    # Read-only default, usable only after actual audit/approval has been supplied.
    if not args.commit_reviewed_admission:
        state=read(args.root/"REGISTRY.json") if (args.root/"REGISTRY.json").exists() else fresh_registry()
        import_legacy(state,args.legacy_root); admit(state,plan,approval,filehash(args.bundle/"MANIFEST.json"))
        print("VALIDATED_ONLY; no admission, dispatch or service mutation"); return
    if os.name!="posix": raise Blocked("admission requires Linux/POSIX")
    if not args.root.is_dir(): raise Blocked("reviewed control root must be provisioned by operator")
    with registry_lock(args.root):
        state=read(args.root/"REGISTRY.json") if (args.root/"REGISTRY.json").exists() else fresh_registry()
        import_legacy(state,args.legacy_root)
        ph=admit(state,plan,approval,filehash(args.bundle/"MANIFEST.json"))
        target=args.root/"studies"/ph
        if target.exists(): raise Blocked("partial/conflicting admission requires review")
        target.mkdir(parents=True)
        for name in manifest["files"]:
            dst=target/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(args.bundle/name,dst)
        shutil.copyfile(args.bundle/"MANIFEST.json",target/"MANIFEST.json")
        shutil.copyfile(args.approval,target/"APPROVAL.json"); shutil.copyfile(args.audit,target/"AUDIT_REPORT.md")
        # New weights-* directory in the existing test scanner's actual RUNS root.
        lab=pathlib.Path(plan["lab_runs_root"])/("weights-transition-"+ph)
        if lab.exists(): raise Blocked("canonical run directory already exists")
        preview=args.bundle/"preregistration"/("weights-transition-"+ph)
        shutil.copytree(preview,lab)
        standard=read(lab/"STANDARD.json"); standard["status"]="admitted_unqualified_interactive"
        atomic(lab/"STANDARD.json",standard)
        adapter=pathlib.Path(plan["lab_runs_root"]).parent/"standard_interactive_adapter.py"
        source=target/"standard_interactive_adapter.py"
        if adapter.exists() and filehash(adapter)!=filehash(source): raise Blocked("existing evaluator adapter differs")
        if not adapter.exists(): shutil.copyfile(source,adapter)
        state["status"]="paused_dispatch_disabled"; atomic(args.root/"REGISTRY.json",state)
        print("ADMITTED_PAUSED; explicit reviewed migration/dispatch marker still required")
if __name__=="__main__": main()

