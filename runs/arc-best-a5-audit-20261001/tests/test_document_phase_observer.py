import importlib.util
import json
from pathlib import Path
import numpy as np
from cellforge.runtime import document

spec = importlib.util.spec_from_file_location("phase_observer", Path(__file__).parents[1] / "scripts/document_phase_observer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_observer_forwards_values_restores_and_records_no_contents(monkeypatch, tmp_path):
    outputs = [{"text": "private recognized text"}]
    fake_batch = lambda self, images: outputs
    fake_generate = lambda *args, **kwargs: (["private answer"], {"failure": [None]})
    monkeypatch.setattr(document.OCRPredictor, "predict_batch", fake_batch)
    monkeypatch.setattr(document, "generate", fake_generate)
    path = tmp_path / "progress.json"
    observer = module.PhaseObserver(path)
    with observer.observe_inference():
        assert document.OCRPredictor.predict_batch(None, [np.zeros((32,128)), np.zeros((32,128))]) is outputs
        assert document.generate(None, 1, prompts=["private question"])[0] == ["private answer"]
    assert document.OCRPredictor.predict_batch is fake_batch
    assert document.generate is fake_generate
    result = json.loads(path.read_text())
    assert result["ocr_batches"] == 1 and result["ocr_crops"] == 2
    assert result["phase"] == "completed"
    assert "private" not in path.read_text()
