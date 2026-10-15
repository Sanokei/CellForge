"""Same-item actual-arm diagnostic; timing observation changes no model inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from PIL import Image
from cellforge.ir.graph import Graph
from cellforge.runtime.document import DocumentBudget, DocumentPredictor
from document_phase_observer import PhaseObserver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--public-receipt")
    parser.add_argument("--worker-input")
    parser.add_argument("--worker-output")
    parser.add_argument("--worker-progress")
    args = parser.parse_args()
    root = Path(args.private_root).resolve()
    if root.parent != Path("/srv/cellforge-private"):
        raise SystemExit("private operator required")
    language, ocr = Graph.load(args.language), Graph.load(args.ocr)
    budget = DocumentBudget(max_answer_tokens=48, ocr_batch_size=16)
    if args.worker_input:
        request = json.loads(Path(args.worker_input).read_bytes())
        observer = PhaseObserver(args.worker_progress)
        observer.checkpoint("load_rasters")
        pages = []
        for path in request["page_paths"]:
            with Image.open(path) as image:
                pages.append(np.asarray(image.convert("L"), dtype=np.uint8))
        with observer.observe_inference():
            result = DocumentPredictor(language, ocr, budget)({"pages": pages, "question": request["question"]}, observation_mode="actual", shuffle_seed=90221)
        Path(args.worker_output).write_text(json.dumps({"status": result["status"], "answer": result["answer"],
            "language_hash": language.hash(), "ocr_hash": ocr.hash()}))
        return
    public = Path(args.public_receipt)
    if public.exists() or (root / "development-phase-policy-v3.json").exists():
        raise SystemExit("preserve diagnostic evidence")
    old = json.loads((root / "development-probe-policy-v2.json").read_bytes())
    selection = json.loads((root / "development-probe-selection-v2.json").read_bytes())
    if language.hash() != old["language_hash"] or ocr.hash() != old["ocr_hash"] or DocumentPredictor(language, ocr, budget).contract() != old["contract"]:
        raise SystemExit("same-model/contract invariant failed")
    source_root = Path(__file__).resolve().parents[1]
    for name, digest in old["source_sha256"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise SystemExit("original inference source changed")
    samples = json.loads((root / "samples.json").read_bytes())
    record = samples[selection["annotation_index"]]
    if record["doc_id"] != selection["doc_id"] or hashlib.sha256(record["question"].encode()).hexdigest() != selection["question_sha256"]:
        raise SystemExit("same-item invariant failed")
    progress = root / "development-phase-progress-v3.json"
    policy = {"protocol": "document-phase-diagnostic/3", "parent_policy_sha256": hashlib.sha256((root / "development-probe-policy-v2.json").read_bytes()).hexdigest(),
              "same_item": True, "same_model_contract": True, "arm": "actual", "worker_seconds": 120,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("document_phase_observer.py")]},
              "scope": "bottleneck diagnosis only; no model selection or score improvement claim"}
    frozen = root / "development-phase-policy-v3.json"
    frozen.write_text(json.dumps(policy, sort_keys=True))
    with tempfile.TemporaryDirectory(dir=root, prefix="phase-raster-") as temp:
        render_started = time.monotonic()
        subprocess.run(["pdftoppm", "-r", "72", "-gray", "-png", str(root / "documents" / selection["doc_id"]), str(Path(temp) / "page")], capture_output=True, check=True, timeout=120)
        render_seconds = time.monotonic() - render_started
        paths = sorted(Path(temp).glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        rendered_hash = hashlib.sha256("".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()).hexdigest()
        parent = json.loads(Path("/srv/cellforge-public/document-development-probe-v2.json").read_bytes())
        if rendered_hash != parent["rendered_pages_combined_sha256"]:
            raise SystemExit("same rendered pages invariant failed")
        request, result = Path(temp) / "input.json", root / "development-phase-result-v3.json"
        request.write_text(json.dumps({"page_paths": [str(p) for p in paths], "question": record["question"]}))
        started = time.monotonic()
        try:
            subprocess.run([sys.executable, __file__, "--private-root", str(root), "--language", args.language, "--ocr", args.ocr,
                            "--worker-input", str(request), "--worker-output", str(result), "--worker-progress", str(progress)], capture_output=True, check=True, timeout=120)
            status = json.loads(result.read_bytes())["status"]
        except subprocess.TimeoutExpired:
            status = "worker_timeout"
        except Exception as error:
            status = "operator_failure_" + type(error).__name__
        worker_seconds = time.monotonic() - started
    numeric = json.loads(progress.read_bytes()) if progress.exists() else {"phase": "no_checkpoint"}
    receipt = {"protocol": policy["protocol"], "policy": policy, "policy_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
               "status": status, "render_seconds": render_seconds, "worker_seconds": worker_seconds,
               "numeric_checkpoint": numeric, "same_rendered_pages_sha256": rendered_hash,
               "heldout_inference": False, "benchmark_score": None, "custom_score": None,
               "checkpoint_contains": "phase name and numeric counts/timing only; no document ID, question, image content or answer"}
    public.write_text(json.dumps(receipt, indent=2) + "\n")
    public.chmod(0o644)


if __name__ == "__main__":
    os.umask(0o077)
    main()
