"""Externally anchored INPUT preflight, not a scientific or lineage acceptance gate."""
import hashlib, pathlib, re
from .experiment_manifest import load_json_strict, validate_action_manifest
from .joint_weight_population import digest
from .cell_population import validate_literal

REQUIRED = frozenset(("MANIFEST.json", "parent-result.json", "baseline.json", "run.py",
                     "AUTHORING_CONTRACT.json", "INTERFACES.json", "RUNTIME_SOURCES.json"))

def file_hash(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def validate_snapshot(snapshot):
    """Content identities, ancestry graph and selected live membership, not score replay."""
    cells=snapshot["cells"]; live=snapshot["live"]
    if not isinstance(cells,dict) or not cells:raise ValueError("nonempty parent cells required")
    selected=set()
    for family,ids in live.items():
        if not isinstance(ids,list) or len(ids)!=len(set(ids)):raise ValueError("distinct live IDs")
        for cid in ids:
            if cid in selected or cid not in cells:raise ValueError("unknown or duplicate live cell")
            cell=cells[cid]
            if cell["state"]!="live" or cell["structure"]!=family:raise ValueError("retired or mistyped live cell")
            selected.add(cid)
    for cid,cell in cells.items():
        validate_literal(cell["params"])
        if cell["id"]!=cid or digest({"structure":cell["structure"],"params":cell["params"]})!=cid:
            raise ValueError("parent content identity mismatch")
        if (cell["state"]=="live") != (cid in selected):raise ValueError("live state membership mismatch")
        if any(parent not in cells for parent in cell["parents"]):raise ValueError("unknown ancestor")
    visited=set();visiting=set()
    def visit(cid):
        if cid in visiting:raise ValueError("cyclic ancestry")
        if cid in visited:return
        visiting.add(cid)
        for parent in cells[cid]["parents"]:visit(parent)
        visiting.remove(cid);visited.add(cid)
    for cid in cells:visit(cid)
    for family,history in snapshot["histories"].items():
        for record in history:
            cid=record["graph_hash"]
            if cid not in cells or cells[cid]["structure"]!=family:raise ValueError("unknown historical opponent")
            if record["scores"]!=cells[cid]["scores"]:raise ValueError("historical scores differ")
    return selected

def resolve_live_router(snapshot, expected_id):
    selected=validate_snapshot(snapshot)
    if expected_id not in selected:raise ValueError("router is not selected live")
    cell=snapshot["cells"][expected_id]
    if "router" not in cell["structure"]:raise ValueError("selected cell is not router")
    # Actual selected tensors, never a separate router-author file.
    return cell["params"]

def verify_input_anchor(root, expected_declaration_sha256):
    """expected SHA must be supplied independently; this never reads it from run artifacts."""
    root=pathlib.Path(root).resolve()
    if not isinstance(expected_declaration_sha256,str) or not re.fullmatch("[0-9a-f]{64}",expected_declaration_sha256):
        raise ValueError("independent declaration SHA256 required")
    declaration_path=root/"DECLARATION.json"
    if file_hash(declaration_path)!=expected_declaration_sha256:raise ValueError("external declaration anchor mismatch")
    declaration=load_json_strict(declaration_path)
    if declaration.get("schema")!="anchored-action-inputs/1":raise ValueError("explicit anchored input schema required")
    dependencies=declaration.get("dependencies")
    if not isinstance(dependencies,dict) or not REQUIRED.issubset(dependencies):
        raise ValueError("minimum input artifact coverage missing")
    for name,expected in dependencies.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or path==declaration_path or not path.is_file():
            raise ValueError("invalid dependency path")
        if not isinstance(expected,str) or not re.fullmatch("[0-9a-f]{64}",expected) or file_hash(path)!=expected:
            raise ValueError("anchored dependency changed:"+name)
    manifest=load_json_strict(root/"MANIFEST.json")
    validate_action_manifest(manifest)
    # Closure requires exact declared runtime paths, not an artifact Boolean.
    sources=load_json_strict(root/"RUNTIME_SOURCES.json")
    if not isinstance(sources,list) or not sources or len(sources)!=len(set(sources)):
        raise ValueError("explicit nonempty runtime source closure required")
    if any(name not in dependencies for name in sources):raise ValueError("unbound runtime source")
    for contract in ("AUTHORING_CONTRACT.json","INTERFACES.json"):
        if not isinstance(load_json_strict(root/contract),dict):raise ValueError("typed contract mapping required")
    parent=load_json_strict(root/"parent-result.json")
    for context in parent["contexts"]:validate_snapshot(context["population"])
    return {"input_anchor_verified":True,"declaration_sha256":expected_declaration_sha256,
            "dependencies":sorted(dependencies),"runtime_acceptance":False,
            "scope":"Input commitment only; semantic replay and authoring/charge/timing gates remain mandatory"}
