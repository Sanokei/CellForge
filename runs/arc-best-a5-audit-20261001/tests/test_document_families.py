import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("family_audit", Path(__file__).parents[1] / "scripts/audit_document_families.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_conservative_family_union_keeps_transitive_relations():
    grouped = module.components(["a.pdf", "b.pdf", "c.pdf", "d.pdf"],
                                [("a.pdf", "b.pdf"), ("b.pdf", "c.pdf")])
    assert grouped == {"a.pdf": "a.pdf", "b.pdf": "a.pdf", "c.pdf": "a.pdf", "d.pdf": "d.pdf"}
