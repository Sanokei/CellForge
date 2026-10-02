"""Fail-closed experiment data preflight; never supplies answers to a predictor."""
import json,hashlib,re,pathlib
def load_json_strict(path):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError("duplicate JSON key:"+key)
            out[key]=value
        return out
    def constant(value):raise ValueError("nonfinite JSON constant:"+value)
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"),object_pairs_hook=pairs,parse_constant=constant)
def validate_action_manifest(manifest):
    seen=set()
    for section in ("fresh","archive","safety","reporting"):
        rows=manifest.get(section)
        if not isinstance(rows,list) or not 1<=len(rows)<=128:raise ValueError("nonempty bounded "+section+" case list required")
        for row in rows:
            if set(row)!={"id","request","initial","expected","action"}:raise ValueError("exact action fixture fields")
            if not isinstance(row["id"],str) or row["id"] in seen:raise ValueError("unique IDs across development and reporting required")
            seen.add(row["id"])
            request=row["request"]
            if not isinstance(request,str) or not request.isascii() or len(request)>256:raise ValueError("bounded ASCII request")
            if row["action"] not in ("write_file","edit_in_place","edit_replace","read_file","none"):raise ValueError("declared action label")
            tokens=[m.group()[1:-1] if m.group()[0] in "'\"" else m.group().strip("?!,;:").rstrip(".") for m in re.finditer(r"""'[^']*'|"[^"]*"|\S+""",request)]
            for field in ("initial","expected"):
                if not isinstance(row[field],dict):raise ValueError("file-state mapping required")
                for name,text in row[field].items():
                    if not isinstance(name,str) or pathlib.Path(name).name!=name or any(c in name for c in ("/","\\","..",":")):raise ValueError("isolated leaf fixture filenames")
                    if not isinstance(text,str) or len(text.encode("utf-8"))>4096:raise ValueError("bounded literal fixture text")
            for name,text in row["expected"].items():
                if row["initial"].get(name)!=text and name not in tokens:raise ValueError("expected changed filename absent from request:"+row["id"])
    return {"kind":"copied-span-actions","cases":len(seen),"valid":True}
def validate_logic_cases(manifest):
    rows=manifest.get("cases");seen=set();partitions=set()
    if not isinstance(rows,list) or not 1<=len(rows)<=128:raise ValueError("bounded logic case list required")
    for row in rows:
        if set(row)!={"id","bits","partition","expected"}:raise ValueError("exact bit fixture fields")
        if row["id"] in seen:raise ValueError("duplicate logic case ID")
        seen.add(row["id"]);partitions.add(row["partition"])
        if row["partition"] not in ("development","frozen-report"):raise ValueError("declared logic partition")
        if not isinstance(row["bits"],list) or len(row["bits"])!=4 or any(type(x) is not int or x not in (0,1) for x in row["bits"]):raise ValueError("ordered four input bits")
        if not isinstance(row["expected"],list) or len(row["expected"])!=2 or any(type(x) is not int or x not in (0,1) for x in row["expected"]):raise ValueError("ordered two target bits")
        # Named controlled max task reference is evaluator-only; no predictor sees it.
        x=row["bits"];target=max(x[0]+2*x[1],x[2]+2*x[3])
        if row["expected"]!=[target&1,target>>1]:raise ValueError("inconsistent two-bit max label")
    if partitions!={"development","frozen-report"}:raise ValueError("distinct development/report partitions required")
    return {"kind":"two-bit-max","cases":len(rows),"valid":True}
def freeze_manifest(path,kind):
    manifest=load_json_strict(path)
    receipt=validate_action_manifest(manifest) if kind=="action" else validate_logic_cases(manifest) if kind=="logic" else None
    if receipt is None:raise ValueError("explicit supported data kind required")
    return dict(receipt,sha256=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest())
def assert_manifest_unchanged(path,expected_sha):
    load_json_strict(path)
    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=expected_sha:raise ValueError("frozen manifest changed")
