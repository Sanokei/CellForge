import importlib.util
import hashlib
import pytest
from pathlib import Path

spec = importlib.util.spec_from_file_location("stage_document", Path(__file__).parents[1] / "scripts/prepare_document_benchmark.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_pdf_split_conflict_blocks_freeze():
    ids = {module.split_document(f"synthetic-{i}.pdf"): f"synthetic-{i}.pdf" for i in range(100)}
    documents = {ids["development"]: "same-pdf-hash", ids["heldout"]: "same-pdf-hash"}
    result = module.audit_inventory(documents)
    assert result["duplicate_groups"] == 1
    assert result["duplicate_split_conflicts"] == 1
    assert not result["freeze_allowed"]


def test_distinct_pdf_hashes_do_not_merge():
    result = module.audit_inventory({"a.pdf": "a-hash", "b.pdf": "b-hash"})
    assert result["freeze_allowed"]
    assert result["canonical_ids"] == {"a.pdf": "a.pdf", "b.pdf": "b.pdf"}


def test_cached_download_requires_official_blob_integrity(tmp_path):
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"synthetic PDF bytes")
    blob = hashlib.sha1(b"blob 19\0synthetic PDF bytes").hexdigest()
    size, digest = module.download("https://unused.invalid", path, 19, 100, blob)
    assert size == 19 and digest == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"corrupted PDF bytes")
    with pytest.raises(ValueError):
        module.download("https://unused.invalid", path, 19, 100, blob)
