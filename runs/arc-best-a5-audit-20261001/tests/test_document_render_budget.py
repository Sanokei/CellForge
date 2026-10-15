import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("render_budget", Path(__file__).parents[1] / "scripts/audit_document_render_budget.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_page_crop_budget_counts_edges():
    assert module.raster_size(612, 792) == (612, 792)
    assert module.crop_count(612, 792) == 125
    assert module.crop_count(129, 33) == 4
