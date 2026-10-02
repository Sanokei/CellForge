"""Static interface audit only: no benchmark items or answers are loaded."""
import argparse
import hashlib
import json
from pathlib import Path
from cellforge.ir.graph import Graph

SOURCE_COMMIT = "d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384"


def document_split(doc_id):
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("document ID must be a nonempty string")
    # Split is by identity of whole PDF, never by individual question.
    digest = hashlib.sha256(("cellforge-mmlongbench-doc-v1:" + doc_id).encode()).digest()
    return "heldout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "development"


def inference_payload(record, page_images):
    """Explicit whitelist prevents target/evidence metadata reaching inference."""
    if not isinstance(record.get("question"), str) or not isinstance(record.get("doc_id"), str):
        raise ValueError("document question schema invalid")
    return {"doc_id": record["doc_id"], "question": record["question"], "page_images": page_images}


def local_exact_short_answer(prediction, reference):
    """CUSTOM case-sensitive exact match after outer whitespace; not official F1."""
    if not isinstance(prediction, str) or not isinstance(reference, str):
        return False
    return prediction.strip() == reference.strip()


def inspect_graph(path):
    graph = Graph.load(path)
    graph.validate()
    declared = graph.meta.get("document_interface")
    compatible = declared == "document-question-answer/1" and "document_answer" in graph.entries
    return {"path": str(path).replace("\\", "/"), "hash": graph.hash(),
            "task": graph.meta.get("task"), "vocab": graph.meta.get("vocab"),
            "pixel_contract": graph.meta.get("pixel_contract"),
            "document_interface_declared": declared,
            "page_question_answer_entry_present": "document_answer" in graph.entries,
            "static_interface_compatible": compatible,
            "behavioral_readiness": "unverified" if compatible else "incompatible",
            "score": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/document-crossover/readiness.json")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise SystemExit("preserve existing readiness receipt")
    population = Path("runs/parallel-vision/breeding/population")
    graphs = [Path("runs/unified-byte/artifact")] + sorted(p.parent for p in population.glob("*/HASH"))
    sources = Path("runs/document-crossover/sources")
    report = {"protocol": "mmlongbench-doc-crossover-readiness/1",
        "benchmark": "MMLongBench-Doc", "official_source": "https://github.com/mayubo2333/MMLongBench-Doc",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources.glob("*"))},
        "status": "incompatible_unscored", "score": None, "benchmark_evaluations": 0,
        "dataset_inventory": "not loaded; document split membership not instantiated",
        "split_rule": "sha256('cellforge-mmlongbench-doc-v1:' + doc_id) first8bytes big-endian modulo5: 0 heldout, otherwise development",
        "candidates": [inspect_graph(p) for p in graphs],
        "accepted_interfaces": ["document-question-answer/1 integrated graph", "composed-document-system/1 frozen modular OCR plus actual language model"],
        "composed_system": {"status": "not audited in this receipt", "score": None},
        "blockers": ["No audited page-image/question-conditioned document answer system",
                     "OCR candidates accept only 32x128 digit-line rasters",
                     "Byte language graph actual tool probe yielded 3 invalid actions and zero calls/responses",
                     "No independently run matched frontier baseline",
                     "Official GPT-4o extraction evaluator unavailable without a paid API; no API called"],
        "scoring": {"official": "answer extraction and type-aware scoring; not executed",
                    "local_custom": "case-sensitive exact short answer with outer whitespace stripped; not official F1"},
        "final_data_isolation": "only official README/evaluator code inspected; no annotation file or PDF downloaded",
        "readiness_checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "score": None, "candidates": len(graphs)}))


if __name__ == "__main__":
    main()
