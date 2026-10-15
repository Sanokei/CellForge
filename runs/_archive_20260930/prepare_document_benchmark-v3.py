"""Private operator dataset staging. Public receipts contain aggregates only.

Run as evaluator inside its private directory; never run on the author host.
No inference, answer extraction, or paid API is performed.
"""
import argparse
import hashlib
import json
import os
import time
import re
import subprocess
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


def revise_group_split(root, public):
    """Private duplicate grouping and bounded related-edition screening."""
    prior = json.loads((root / "inventory.json").read_bytes())
    if prior["source_commit"] != COMMIT:
        raise ValueError("source mismatch")
    hashes = prior["documents"]
    for name, digest in hashes.items():
        path = root / "documents" / name
        if path.parent != root / "documents" or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("inventory integrity mismatch")
    audit = audit_inventory(hashes)
    canonical = audit["canonical_ids"]
    grouped = {name: split_document(canonical[name]) for name in hashes}
    sketches = {}
    failures = 0
    for name in sorted(set(canonical.values())):
        try:
            result = subprocess.run(["pdftotext", "-f", "1", "-l", "2", str(root / "documents" / name), "-"],
                                    capture_output=True, timeout=10)
            if result.returncode or len(result.stdout) > 1024 * 1024:
                raise ValueError("text screening failed")
            words = re.findall(r"[a-z0-9]+", result.stdout.decode("utf-8", errors="replace").lower())
            shingles = {hashlib.sha256(" ".join(words[i:i+4]).encode()).hexdigest() for i in range(max(0, len(words)-3))}
            if len(shingles) < 50:
                raise ValueError("insufficient screening text")
            sketches[name] = set(sorted(shingles)[:128])
        except Exception:
            failures += 1
    names = sorted(sketches)
    related = []
    for i, left in enumerate(names):
        for right in names[i+1:]:
            a, b = sketches[left], sketches[right]
            similarity = len(a & b) / min(len(a), len(b))
            if similarity >= .6:
                related.append({"left": left, "right": right, "sketch_overlap": similarity,
                                "cross_split": grouped[left] != grouped[right]})
    revised = {"protocol": "document-group-inventory/2", "source_commit": COMMIT,
               "previous_inventory_sha256": hashlib.sha256((root / "inventory.json").read_bytes()).hexdigest(),
               "documents": hashes, "canonical_ids": canonical, "split_by_canonical_group": grouped,
               "related_edition_candidates": related, "screening_failures": failures,
               "screening": "first two PDF pages text, bottom128 SHA256 four-word shingles, overlap>=0.6; audit only, never inference"}
    raw = json.dumps(revised, sort_keys=True).encode()
    destination = root / "inventory-v2.json"
    if destination.exists() or public.exists():
        raise ValueError("preserve existing revised audit")
    destination.write_bytes(raw)
    receipt = {"protocol": "document-group-inventory/2", "source_commit": COMMIT,
               "status": "canonical_groups_assigned_related_edition_review_unresolved",
               "documents": len(hashes), "unique_pdf_groups": len(set(hashes.values())),
               "duplicate_groups": audit["duplicate_groups"], "resolved_duplicate_split_conflicts": audit["duplicate_split_conflicts"],
               "remaining_identical_pdf_split_conflicts": 0,
               "development_documents": sum(s == "development" for s in grouped.values()),
               "heldout_documents": sum(s == "heldout" for s in grouped.values()),
               "related_edition_screened_groups": len(sketches), "screening_failures": failures,
               "related_edition_candidate_pairs": len(related),
               "cross_split_related_candidate_pairs": sum(p["cross_split"] for p in related),
               "manual_related_edition_review": "not completed", "full_split_frozen": False,
               "inventory_sha256": hashlib.sha256(raw).hexdigest(),
               "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "benchmark_evaluations": 0, "score": None}
    public.write_text(json.dumps(receipt, indent=2) + "\n")
    public.chmod(0o644)


def blob_hash(path):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_size, remaining, expected_blob=None):
    if expected_size > remaining:
        raise ValueError("download budget")
    if destination.exists():
        if destination.stat().st_size != expected_size or expected_blob is None or blob_hash(destination) != expected_blob:
            raise ValueError("existing file integrity mismatch")
        return expected_size, hashlib.sha256(destination.read_bytes()).hexdigest()
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
    if expected_blob is not None and blob_hash(destination) != expected_blob:
        raise ValueError("downloaded git blob integrity mismatch")
    return count, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--public-receipt", required=True)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.private_root).resolve()
    if os.name != "posix" or root.parent != Path("/srv/cellforge-private") or not 0 < args.max_bytes <= 800 * 1024 * 1024:
        raise SystemExit("private evaluator path and <=800MiB budget required")
    if args.audit_only:
        os.umask(0o077)
        revise_group_split(root, Path(args.public_receipt))
        return
    if root.exists() and not args.resume:
        raise SystemExit("private run already exists")
    if args.resume:
        previous = json.loads((root / "inventory.json").read_bytes())
        if previous.get("source_commit") != COMMIT:
            raise SystemExit("resume source version mismatch")
        for name, digest in previous["documents"].items():
            path = root / "documents" / name
            if path.parent != root / "documents" or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise SystemExit("resume private inventory integrity failure")
    os.umask(0o077)
    root.mkdir(mode=0o700, exist_ok=args.resume)
    (root / "documents").mkdir(mode=0o700, exist_ok=args.resume)
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
        size, annotation_hash = download(base + "data/samples.json", root / "samples.json", annotation["size"], args.max_bytes, annotation["sha"])
        state["bytes_downloaded"] += size
        samples = json.loads((root / "samples.json").read_bytes())
        if not isinstance(samples, list) or any(not isinstance(s.get("doc_id"), str) for s in samples):
            raise ValueError("annotation schema")
        state["annotation_sha256"] = annotation_hash
        state["questions_total"] = len(samples)
        pdfs = sorted((p, x) for p, x in entries.items() if p.startswith("data/documents/") and p.endswith(".pdf"))
        names = [Path(p).name for p, _ in pdfs]
        if len(set(names)) != len(names) or not {s["doc_id"] for s in samples}.issubset(set(names)):
            raise ValueError("duplicate flattened filenames or missing annotation document coverage")
        state["documents_total"] = len(pdfs)
        hashes = {}
        for path, entry in pdfs:
            if entry["size"] > args.max_bytes - state["bytes_downloaded"]:
                continue
            destination = root / "documents" / Path(path).name
            size, digest = download(base + urllib.parse.quote(path), destination,
                                    entry["size"], args.max_bytes - state["bytes_downloaded"], entry["sha"])
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
                      "status": ("complete_staged_unscored" if len(hashes) == len(pdfs) else "partial_staged_unscored") if audit["freeze_allowed"] else "split_conflict_unscored",
                      "full_split_frozen": False,
                      "related_edition_audit": "not completed",
                      "annotation_document_coverage_verified": True,
                      "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    except Exception as error:
        # Do not leak document IDs, annotation content, URL paths or targets.
        state.update({"status": "staging_failed_unscored", "error_type": type(error).__name__})
    state["elapsed_seconds"] = time.monotonic() - started
    publish()
    if state["status"] == "staging_failed_unscored":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
