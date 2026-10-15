import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("document_check", Path(__file__).parents[1] / "scripts/check_document_readiness.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_document_split_is_deterministic_and_question_independent():
    assert module.document_split("synthetic-document.pdf") == module.document_split("synthetic-document.pdf")
    splits = {module.document_split(f"synthetic-{i}.pdf") for i in range(100)}
    assert splits == {"development", "heldout"}


def test_inference_whitelist_excludes_answers_and_evidence_pages():
    record = {"doc_id": "synthetic.pdf", "question": "Which label?", "answer": "secret",
              "evidence_pages": [5], "evidence_sources": ["table"], "answer_format": "String"}
    payload = module.inference_payload(record, ["page-raster"])
    assert set(payload) == {"doc_id", "question", "page_images"}
    assert "secret" not in repr(payload)


def test_custom_scorer_does_not_claim_official_normalization():
    assert module.local_exact_short_answer(" 17 ", "17")
    assert not module.local_exact_short_answer("The answer is 17", "17")
    assert not module.local_exact_short_answer("ABC", "abc")
    assert not module.local_exact_short_answer(None, "17")
