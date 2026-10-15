"""Opt-in future isolated-worker timing observer; never records input contents.

Not used by the frozen document development probe v2. Wrappers forward the
same values to authored inference and restore originals on context exit.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
from cellforge.runtime import document


class PhaseObserver:
    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()
        self.state = {"protocol": "document-phase-observer/1", "phase": "initialized",
                      "ocr_batches": 0, "ocr_crops": 0, "ocr_seconds": 0.,
                      "language_seconds": 0., "context_bytes": 0}
        self.checkpoint("initialized")

    def checkpoint(self, phase, **numeric_fields):
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in numeric_fields.values()):
            raise ValueError("phase metadata must be numeric")
        self.state.update(numeric_fields)
        self.state["phase"] = phase
        self.state["elapsed_seconds"] = time.monotonic() - self.started
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        temporary.write_text(json.dumps(self.state, sort_keys=True) + "\n")
        temporary.replace(self.path)

    @contextmanager
    def observe_inference(self):
        original_batch = document.OCRPredictor.predict_batch
        original_generate = document.generate
        observer = self
        def batch(predictor, images, *args, **kwargs):
            observer.checkpoint("ocr")
            started = time.monotonic()
            result = original_batch(predictor, images, *args, **kwargs)
            observer.checkpoint("ocr", ocr_batches=observer.state["ocr_batches"] + 1,
                                ocr_crops=observer.state["ocr_crops"] + len(images),
                                ocr_seconds=observer.state["ocr_seconds"] + time.monotonic() - started)
            return result
        def generate(*args, **kwargs):
            prompts = kwargs.get("prompts", [])
            observer.checkpoint("language", context_bytes=sum(len(p.encode("utf-8")) for p in prompts))
            started = time.monotonic()
            result = original_generate(*args, **kwargs)
            observer.checkpoint("completed", language_seconds=time.monotonic() - started)
            return result
        document.OCRPredictor.predict_batch = batch
        document.generate = generate
        try:
            yield self
        finally:
            document.OCRPredictor.predict_batch = original_batch
            document.generate = original_generate
