"""Private development-only PDF page/raster feasibility audit, no inference."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def raster_size(width_points, height_points, dpi=72):
    if dpi != 72 or width_points <= 0 or height_points <= 0:
        raise ValueError("frozen raster contract requires positive dimensions and 72 DPI")
    return math.ceil(width_points), math.ceil(height_points)


def crop_count(width, height):
    return math.ceil(width / 128) * math.ceil(height / 32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--public-receipt", required=True)
    args = parser.parse_args()
    root = Path(args.private_root).resolve()
    output = Path(args.public_receipt)
    if root.parent != Path("/srv/cellforge-private") or output.exists():
        raise SystemExit("private operator root and fresh aggregate output required")
    inventory = json.loads((root / "inventory-v2.json").read_bytes())
    dev = sorted({inventory["canonical_ids"][i] for i, split in inventory["split_by_canonical_group"].items() if split == "development"})
    version = subprocess.run(["pdftoppm", "-v"], capture_output=True, check=True).stderr.decode().splitlines()[0]
    config = {"protocol": "document-raster-policy/1", "renderer": version,
              "dpi": 72, "colors": "grayscale uint8; black ink white background", "pages": "all ordered pages",
              "command": "pdftoppm -r 72 -gray -png INPUT OUTPUT_PREFIX",
              "crop_policy": "complete nonoverlapping32x128 row-major, white edge pad, invert255-x",
              "question_serialization": "exact UTF8 question string; DocumentPredictor canonical compact JSON with ordered observations",
              "scope": "development-only metadata feasibility; heldout PDFs not opened; no inference"}
    frozen = root / "render-policy-v1.json"
    if frozen.exists():
        raise SystemExit("preserve frozen renderer policy")
    frozen.write_text(json.dumps(config, sort_keys=True))
    private_results = []
    failures = 0
    for name in dev:
        try:
            info = subprocess.run(["pdfinfo", "-f", "1", "-l", "10000", "-box", str(root / "documents" / name)],
                                  capture_output=True, timeout=15, check=True).stdout.decode()
            pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
            sizes = re.findall(r"^Page\s+\d+ size:\s+([\d.]+) x ([\d.]+) pts", info, re.M)
            if len(sizes) != pages:
                raise ValueError("all-page dimension metadata unavailable")
            shapes = [raster_size(float(w), float(h)) for w, h in sizes]
            crops = sum(crop_count(w, h) for w, h in shapes)
            private_results.append({"doc_id": name, "pages": pages, "crops": crops,
                                    "pixels": sum(w*h for w,h in shapes),
                                    "default_input_budget_feasible": pages <= 64 and crops <= 4096 and all(w*h <= 16000000 for w,h in shapes)})
        except Exception:
            failures += 1
    private_path = root / "development-render-budget-v1.json"
    private_path.write_text(json.dumps(private_results, sort_keys=True))
    receipt = {"protocol": "document-render-budget-audit/1", "renderer_policy": config,
               "policy_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
               "private_results_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
               "development_unique_pdf_groups": len(dev), "metadata_audited": len(private_results), "metadata_failures": failures,
               "pages_total": sum(r["pages"] for r in private_results),
               "max_pages": max((r["pages"] for r in private_results), default=0),
               "crops_total": sum(r["crops"] for r in private_results),
               "max_crops": max((r["crops"] for r in private_results), default=0),
               "default_input_budget_feasible_documents": sum(r["default_input_budget_feasible"] for r in private_results),
               "documents_with_at_most_256_crops": sum(r["crops"] <= 256 for r in private_results),
               "related_edition_split_status": "unresolved; no heldout confirmation permitted",
               "benchmark_evaluations": 0, "score": None,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    output.chmod(0o644)


if __name__ == "__main__":
    main()
