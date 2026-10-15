"""Seal the local review bytes; does not admit, deploy, connect or play."""
import ast
import gzip
import hashlib
import io
import pathlib
import tarfile
from common import *

ROOT=pathlib.Path(__file__).resolve().parent
def main():
    if (ROOT/"APPROVAL.json").exists() or (ROOT/"DISPATCH_ENABLED").exists(): raise Blocked("preparation cannot supply approval/dispatch")
    validate_plan(read(ROOT/"PLAN.json"))
    old=read(ROOT/"prior-evidence"/"legacy"/"PLAN.json")
    for name in ("arc3_env.py","arc3cell.py","structcode.py"):
        if filehash(ROOT/name)!=old["code_sha256"][name]: raise Blocked("frozen evaluator/sandbox differs")
    cross=read(ROOT/"CROSS_VERSION_SERVER.json")
    if not cross["all_fingerprints_equal"] or not cross["all_evaluator_bindings_equal"] or not cross["legacy_identities_equal"]:
        raise Blocked("cross-version proof failed")
    if cross["plan_canonical_sha256"]!=digest(read(ROOT/"PLAN.json")):
        raise Blocked("cross-version proof belongs to another plan")
    if cross["fingerprints"]!=[c["fingerprint"] for c in read(ROOT/"PLAN.json")["conditions"]]:
        raise Blocked("cross-version condition binding differs")
    linux=read(ROOT/"LINUX_V2_PROBES.json"); cleanup=read(ROOT/"LINUX_PROBE_CLEANUP.json")
    if not linux["passed"] or not linux["fixture_files_cleaned"] or not cleanup["all_probe_units_absent"]:
        raise Blocked("Linux probe/cleanup evidence incomplete")
    collection=read(ROOT/"LINUX_COLLECTION_PROBES.json")
    if not collection["passed"] or not collection["fixture_files_cleaned"] or collection["manual_reset_failed_calls"]!=0:
        raise Blocked("focused Linux collection proof incomplete")
    if collection["supervisor_sha256"]!=filehash(ROOT/"supervisor.py") or collection["common_sha256"]!=filehash(ROOT/"common.py"):
        raise Blocked("focused Linux proof belongs to different controller bytes")
    if not all(c.get("absent") and c.get("empty") for c in collection["cleanup"]):
        raise Blocked("focused collection units remain")
    checkpoint=read(ROOT/"CHECKPOINT.json")
    if checkpoint["state"]["status"]!="paused_exhausted" or checkpoint["job"]["MainPID"]!="0" or checkpoint["code_mismatches"] or checkpoint["protected_stop_mismatches"]:
        raise Blocked("checkpoint is not a preserved paused state")
    legacy=ROOT/"prior-evidence"/"legacy"
    if filehash(legacy/"PLAN.json")!=checkpoint["plan_sha256"]: raise Blocked("first-study plan changed")
    for receipt in checkpoint["receipts"]:
        run=legacy/"runs"/receipt["fingerprint"]
        if filehash(run/"TERMINAL.json")!=receipt["receipt_sha256"] or filehash(run/"RESULTS.json")!=receipt["results_sha256"]:
            raise Blocked("first-study receipt/result changed")
    preview=read(ROOT/"LEGACY_REGISTRY_PREVIEW.json")
    if preview["active"] is not None or preview["studies"] or len(preview["conditions"])!=9:
        raise Blocked("unexpected admission/legacy preview")
    for path in ROOT.glob("*.py"): ast.parse(path.read_text())
    paths=sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name!="MANIFEST.json")
    if any(p.is_symlink() for p in paths): raise Blocked("review cannot package symlinks")
    manifest={"schema":"cellforge-review-bundle/2","status":"PREPARED_ONLY","plan_sha256":digest(read(ROOT/"PLAN.json")),
              "qualified":False,"admitted_plans":0,"new_game_plays":0,"remote_model_calls":0,
              "files":{p.relative_to(ROOT).as_posix():filehash(p) for p in paths}}
    atomic(ROOT/"MANIFEST.json",manifest)
    archive=ROOT.parent/(ROOT.name+".tgz")
    with archive.open("wb") as out:
        with gzip.GzipFile(filename="",mode="wb",fileobj=out,mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode="w") as tar:
                for path in sorted(paths+[ROOT/"MANIFEST.json"]):
                    data=path.read_bytes()
                    ti=tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                    ti.size=len(data); ti.mode=0o644; ti.mtime=0
                    tar.addfile(ti,io.BytesIO(data))
    with tarfile.open(archive,"r:gz") as tar:
        for name,h in manifest["files"].items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest()!=h: raise Blocked("archive bytes changed")
        if read(ROOT/"MANIFEST.json")!=json.loads(tar.extractfile("MANIFEST.json").read()): raise Blocked("archive manifest differs")
    h=filehash(archive)
    archive.with_suffix(archive.suffix+".sha256").write_text(h+"  "+archive.name+"\n",encoding="utf-8")
    print(canonical({"archive":str(archive),"archive_sha256":h,"manifest_sha256":filehash(ROOT/"MANIFEST.json"),
                     "plan_sha256":manifest["plan_sha256"],"files":len(manifest["files"])}).decode())
if __name__=="__main__": main()

