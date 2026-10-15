"""One bounded complete-development-document probe, never a full benchmark."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from PIL import Image
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.document import DocumentBudget, DocumentPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--public-receipt")
    parser.add_argument("--worker-input")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    root = Path(args.private_root).resolve()
    if root.parent != Path("/srv/cellforge-private"):
        raise SystemExit("private operator required")
    language, ocr = Graph.load(args.language), Graph.load(args.ocr)
    budget = DocumentBudget(max_answer_tokens=48, ocr_batch_size=16)
    if args.worker_input:
        request = json.loads(Path(args.worker_input).read_bytes())
        pages = []
        for path in request["page_paths"]:
            with Image.open(path) as image:
                pages.append(np.asarray(image.convert("L"), dtype=np.uint8))
        result = DocumentPredictor(language, ocr, budget)({"pages": pages, "question": request["question"]}, observation_mode=request["mode"], shuffle_seed=90221)
        Path(args.worker_output).write_text(json.dumps({"status": result["status"], "answer": result["answer"],
            "crop_count": result["crop_count"], "generated_tokens": result.get("generated_tokens", 0),
            "ocr_batches": result.get("ocr_batches", 0), "ocr_failures": result.get("ocr_failures", 0),
            "context_bytes": result.get("context_bytes", 0),
            "language_hash": language.hash(), "ocr_hash": ocr.hash()}))
        return
    output = Path(args.public_receipt)
    if output.exists() or (root / "development-probe-policy-v2.json").exists():
        raise SystemExit("preserve probe evidence")
    inventory = json.loads((root / "inventory-v3.json").read_bytes())
    feasible = json.loads((root / "development-render-budget-v1.json").read_bytes())
    # Selection by metadata and lexicographic ID only, without question/answer inspection.
    ids = sorted(r["doc_id"] for r in feasible if r["default_input_budget_feasible"] and inventory["split_by_family"][r["doc_id"]] == "development")
    if not ids:
        raise SystemExit("no complete development document fits frozen budget")
    doc_id = ids[0]
    samples = json.loads((root / "samples.json").read_bytes())
    record = next(s for s in samples if s["doc_id"] == doc_id)
    (root / "development-probe-selection-v2.json").write_text(json.dumps({"doc_id": doc_id,
        "annotation_index": samples.index(record), "question_sha256": hashlib.sha256(record["question"].encode()).hexdigest()}))
    renderer_policy = json.loads((root / "render-policy-v1.json").read_bytes())
    renderer_version = subprocess.run(["pdftoppm", "-v"], capture_output=True, check=True).stderr.decode().splitlines()[0]
    if renderer_version != renderer_policy["renderer"]:
        raise SystemExit("renderer version changed")
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((source_root / "src").rglob("*.py"))}
    policy = {"protocol": "bounded-document-development-probe/2", "renderer_policy_sha256": hashlib.sha256((root / "render-policy-v1.json").read_bytes()).hexdigest(),
        "renderer_policy": renderer_policy, "source_sha256": source_hashes,
        "language_hash": language.hash(), "ocr_hash": ocr.hash(), "contract": DocumentPredictor(language, ocr, budget).contract(),
        "selection": "first lexicographic feasible canonical development PDF; first annotation question; all pages",
        "limits": {"documents": 1, "questions": 1, "arms": ["actual", "withheld", "shuffled"], "worker_seconds_per_arm": 120},
        "metric": "CUSTOM case-sensitive exact short answer after outer whitespace, not official F1",
        "scope": "bounded development subset, not full benchmark; family guard not yet final-confirmation certified"}
    frozen = root / "development-probe-policy-v2.json"
    frozen.write_text(json.dumps(policy, sort_keys=True))
    with tempfile.TemporaryDirectory(dir=root, prefix="development-raster-") as temp:
        prefix = str(Path(temp) / "page")
        subprocess.run(["pdftoppm", "-r", "72", "-gray", "-png", str(root / "documents" / doc_id), prefix], capture_output=True, check=True, timeout=120)
        paths = sorted(Path(temp).glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        rendered_hash = hashlib.sha256("".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()).hexdigest()
        results = {}
        for mode in policy["limits"]["arms"]:
            request_path, result_path = Path(temp) / f"{mode}-input.json", Path(temp) / f"{mode}-result.json"
            # Worker receives exactly rendered full document/question/mode, never labels/evidence.
            request_path.write_text(json.dumps({"page_paths": [str(p) for p in paths], "question": record["question"], "mode": mode}))
            try:
                subprocess.run([sys.executable, __file__, "--private-root", str(root), "--language", args.language, "--ocr", args.ocr,
                                "--worker-input", str(request_path), "--worker-output", str(result_path)], capture_output=True, check=True, timeout=120)
                result = json.loads(result_path.read_bytes())
                if result.pop("language_hash") != language.hash() or result.pop("ocr_hash") != ocr.hash():
                    raise ValueError("worker artifact changed")
                answer = result.pop("answer")
                result["custom_exact_correct"] = int(isinstance(answer, str) and answer.strip() == str(record["answer"]).strip())
            except subprocess.TimeoutExpired:
                result = {"status": "worker_timeout", "custom_exact_correct": 0}
            except Exception as error:
                result = {"status": "operator_failure", "error_type": type(error).__name__, "custom_exact_correct": 0}
            results[mode] = result
    receipt = {"protocol": policy["protocol"], "policy": policy, "policy_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
               "aggregate": results, "documents": 1, "questions": 1, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "rendered_pages_combined_sha256": rendered_hash,
               "heldout_inference": False, "official_metric": None, "frontier_comparison": None}
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    output.chmod(0o644)


if __name__ == "__main__":
    os.umask(0o077)
    main()
