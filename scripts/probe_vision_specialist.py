"""Development-only frozen OCR observation probe; no final data or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import ImageFont
from cellforge.data.render import render_one
from cellforge.ir.graph import Graph
from cellforge.runtime.vision import OCRPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--font", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictor = OCRPredictor(Graph.load(args.artifact))
    font = ImageFont.truetype(args.font, 20)
    cases = []
    for source in ["0", "7", "123", "90817", "42", "hello"]:
        image = render_one(source, font, 6, 0)
        observation = predictor(image)
        cases.append({"source_text": source, "prediction": observation,
                      "exact": observation["text"] == source,
                      "scope": "in-vocabulary-digits" if source.isdigit() else "out-of-vocabulary-negative-control"})
    blank = predictor(np.zeros((32, 128), dtype=np.uint8))
    result = {"protocol": "ocr-observation-development-smoke/1", "final_data_accessed": False,
              "graph_hash": predictor.graph_hash,
              "font_sha256": hashlib.sha256(Path(args.font).read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256(Path(__import__('cellforge.runtime.vision', fromlist=['x']).__file__).read_bytes()).hexdigest(),
              "cases": cases, "blank_negative_control": blank,
              "limitation": "Frozen digit OCR; text observations are not learned image embeddings or evidence of general vision/LLM reasoning."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_hash": predictor.graph_hash, "digit_exact": sum(c['exact'] for c in cases if c['source_text'].isdigit()), "digit_cases": 5, "output": args.output}))


if __name__ == "__main__":
    main()
