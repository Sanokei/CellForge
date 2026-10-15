"""Private all-page document-family screening; publishes aggregates only."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from PIL import Image


def components(names, pairs):
    parent = {n: n for n in names}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for left, right in pairs:
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    return {n: find(n) for n in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--public-receipt", required=True)
    args = parser.parse_args()
    root, output = Path(args.private_root).resolve(), Path(args.public_receipt)
    if root.parent != Path("/srv/cellforge-private") or output.exists() or (root / "inventory-v3.json").exists():
        raise SystemExit("fresh private audit required")
    old = json.loads((root / "inventory-v2.json").read_bytes())
    names = sorted(set(old["canonical_ids"].values()))
    pairs = [(p["left"], p["right"]) for p in old["related_edition_candidates"]]
    sketches, rasters = {}, {}
    failures, text_pages_groups, raster_groups = 0, 0, 0
    for name in names:
        path = root / "documents" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != old["documents"][name]:
            raise SystemExit("private document integrity failure")
        try:
            run = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, timeout=30, check=True)
            if len(run.stdout) > 8 * 1024 * 1024:
                raise ValueError("all-page text cap")
            words = re.findall(r"[a-z0-9]+", run.stdout.decode("utf-8", errors="replace").lower())
            if len(words) >= 100:
                shingles = {hashlib.sha256(" ".join(words[i:i+4]).encode()).hexdigest() for i in range(len(words)-3)}
                sketches[name] = set(sorted(shingles)[:128])
                text_pages_groups += 1
            else:
                with tempfile.TemporaryDirectory(dir=root, prefix="raster-audit-") as temp:
                    prefix = str(Path(temp) / "page")
                    subprocess.run(["pdftoppm", "-scale-to", "64", "-gray", "-png", str(path), prefix],
                                   capture_output=True, timeout=90, check=True)
                    fingerprints = set()
                    for image_path in Path(temp).glob("page-*.png"):
                        with Image.open(image_path) as image:
                            image = image.convert("L").resize((16, 16))
                            bits = bytes(int(v < 192) for v in image.getdata())
                        if sum(bits) > 3:
                            fingerprints.add(hashlib.sha256(bits).hexdigest())
                    if not fingerprints:
                        raise ValueError("no raster audit signal")
                    rasters[name] = fingerprints
                    raster_groups += 1
        except Exception:
            failures += 1
    for index, left in enumerate(names):
        for right in names[index+1:]:
            related = False
            if left in sketches and right in sketches:
                a, b = sketches[left], sketches[right]
                related = len(a & b) / min(len(a), len(b)) >= .6
            if left in rasters and right in rasters:
                related |= len(rasters[left] & rasters[right]) >= min(2, len(rasters[left]), len(rasters[right]))
            if related:
                pairs.append((left, right))
    grouped = components(names, pairs)
    splits = {n: ("heldout" if int.from_bytes(hashlib.sha256(("cellforge-mmlongbench-doc-v1:" + grouped[old["canonical_ids"][n]]).encode()).digest()[:8], "big") % 5 == 0 else "development") for n in old["documents"]}
    revised = {"protocol": "document-family-inventory/3", "source_commit": old["source_commit"],
               "documents": old["documents"], "byte_canonical_ids": old["canonical_ids"],
               "family_canonical_ids": grouped, "split_by_family": splits, "related_pairs": pairs,
               "screening_failures": failures, "full_split_frozen": False}
    raw = json.dumps(revised, sort_keys=True).encode()
    (root / "inventory-v3.json").write_bytes(raw)
    receipt = {"protocol": revised["protocol"], "source_commit": old["source_commit"],
               "unique_byte_groups": len(names), "conservative_family_components": len(set(grouped.values())),
               "all_page_text_screened_groups": text_pages_groups, "all_page_raster_screened_groups": raster_groups,
               "screening_failures": failures, "unioned_related_pairs": len(pairs),
               "development_documents": sum(s == "development" for s in splits.values()),
               "heldout_documents": sum(s == "heldout" for s in splits.values()),
               "remaining_flagged_pair_cross_split_conflicts": sum(splits[a] != splits[b] for a,b in pairs),
               "inventory_sha256": hashlib.sha256(raw).hexdigest(), "full_split_frozen": False,
               "limitations": "Automated similarity grouping is conservative but cannot prove absence of all related editions; no final confirmation permitted until guard reviewed",
               "benchmark_evaluations": 0, "score": None,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    output.chmod(0o644)


if __name__ == "__main__":
    main()
