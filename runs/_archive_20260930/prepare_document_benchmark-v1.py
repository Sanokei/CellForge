"""Private operator dataset staging. Public receipts contain aggregates only.

Run as evaluator inside its private directory; never run on the author host.
No inference, answer extraction, or paid API is performed.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

COMMIT = "d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384"
REPO = "mayubo2333/MMLongBench-Doc"


def split_document(doc_id):
    raw = hashlib.sha256(("cellforge-mmlongbench-doc-v1:" + doc_id).encode()).digest()
    return "heldout" if int.from_bytes(raw[:8], "big") % 5 == 0 else "development"


def audit_inventory(documents):
    groups = {}
    for doc_id, digest in documents.items():
        groups.setdefault(digest, []).append(doc_id)
    conflicts = sum(len({split_document(i) for i in ids}) > 1 for ids in groups.values())
    canonical = {i: min(ids) for ids in groups.values() for i in ids}
    return {"duplicate_groups": sum(len(ids) > 1 for ids in groups.values()),
            "duplicate_split_conflicts": conflicts, "canonical_ids": canonical,
            "freeze_allowed": conflicts == 0}


def download(url, destination, expected_size, remaining):
    if expected_size > remaining:
        raise ValueError("download budget")
    if destination.exists():
        raise ValueError("refuse existing file")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(url, timeout=45) as response, temporary.open("xb") as output:
        while True:
            block = response.read(min(1024 * 1024, remaining - count + 1))
            if not block:
                break
            count += len(block)
            if count > remaining or count > expected_size:
                raise ValueError("source size or budget exceeded")
            output.write(block)
            digest.update(block)
    if count != expected_size:
        raise ValueError("source size mismatch")
    temporary.replace(destination)
    return count, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--public-receipt", required=True)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    root = Path(args.private_root).resolve()
    if os.name != "posix" or root.parent != Path("/srv/cellforge-private") or not 0 < args.max_bytes <= 500 * 1024 * 1024:
        raise SystemExit("private evaluator path and <=500MiB budget required")
    if root.exists():
        raise SystemExit("private run already exists")
    os.umask(0o077)
    root.mkdir(mode=0o700)
    (root / "documents").mkdir(mode=0o700)
    public = Path(args.public_receipt)
    if public.exists():
        raise SystemExit("public receipt already exists")
    started = time.monotonic()
    state = {"protocol": "private-document-staging/1", "source_commit": COMMIT,
             "status": "staging", "bytes_downloaded": 0, "documents_staged": 0,
             "byte_limit": args.max_bytes, "benchmark_evaluations": 0, "score": None}
    def publish():
        temporary = public.with_suffix(".partial")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.chmod(0o644)
        temporary.replace(public)
    publish()
    try:
        url = f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"
        with urllib.request.urlopen(url, timeout=45) as response:
            metadata = response.read(2 * 1024 * 1024 + 1)
        if len(metadata) > 2 * 1024 * 1024:
            raise ValueError("metadata cap")
        tree = json.loads(metadata)
        if tree.get("truncated"):
            raise ValueError("incomplete source tree")
        (root / "source-tree.json").write_bytes(metadata)
        state["source_tree_sha256"] = hashlib.sha256(metadata).hexdigest()
        entries = {x["path"]: x for x in tree["tree"] if x.get("type") == "blob"}
        annotation = entries["data/samples.json"]
        base = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
        size, annotation_hash = download(base + "data/samples.json", root / "samples.json", annotation["size"], args.max_bytes)
        state["bytes_downloaded"] += size
        samples = json.loads((root / "samples.json").read_bytes())
        if not isinstance(samples, list) or any(not isinstance(s.get("doc_id"), str) for s in samples):
            raise ValueError("annotation schema")
        state["annotation_sha256"] = annotation_hash
        state["questions_total"] = len(samples)
        pdfs = sorted((p, x) for p, x in entries.items() if p.startswith("data/documents/") and p.endswith(".pdf"))
        state["documents_total"] = len(pdfs)
        hashes = {}
        for path, entry in pdfs:
            if entry["size"] > args.max_bytes - state["bytes_downloaded"]:
                continue
            destination = root / "documents" / Path(path).name
            size, digest = download(base + urllib.parse.quote(path), destination,
                                    entry["size"], args.max_bytes - state["bytes_downloaded"])
            hashes[destination.name] = digest
            state["bytes_downloaded"] += size
            state["documents_staged"] += 1
            publish()
        audit = audit_inventory(hashes)
        inventory = {"source_commit": COMMIT, "documents": hashes, "audit": audit,
                     "split_by_doc_id": {i: split_document(i) for i in hashes}}
        inventory_bytes = json.dumps(inventory, sort_keys=True).encode()
        (root / "inventory.json").write_bytes(inventory_bytes)
        state.update({"inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                      "duplicate_groups": audit["duplicate_groups"],
                      "duplicate_split_conflicts": audit["duplicate_split_conflicts"],
                      "development_documents": sum(split_document(i) == "development" for i in hashes),
                      "heldout_documents": sum(split_document(i) == "heldout" for i in hashes),
                      "questions_in_staged_documents": sum(s["doc_id"] in hashes for s in samples),
                      "complete_dataset_staged": len(hashes) == len(pdfs),
                      "status": "partial_staged_unscored" if audit["freeze_allowed"] else "split_conflict_unscored",
                      "full_split_frozen": False,
                      "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    except Exception as error:
        # Do not leak document IDs, annotation content, URL paths or targets.
        state.update({"status": "staging_failed_unscored", "error_type": type(error).__name__})
    state["elapsed_seconds"] = time.monotonic() - started
    publish()


if __name__ == "__main__":
    main()
