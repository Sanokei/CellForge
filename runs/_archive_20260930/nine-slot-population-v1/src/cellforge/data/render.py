"""Synthetic text-line rendering with full manifests.

Ink is bright: pixel = uint8 ink coverage, normalized by 255 before the model
sees it.  Font files, seeds and renderer settings are recorded in manifests
but never enter model inputs.  Fonts are not shipped with the repository;
point ``CELLFORGE_FONT_DIR`` at a directory of licensed TTF files.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

H, W = 32, 128


@dataclass
class RenderSpec:
    charset: str = "0123456789"
    min_len: int = 1
    max_len: int = 8
    fonts: list = field(default_factory=lambda: ["DejaVuSansMono.ttf"])
    size_range: tuple = (20, 20)
    x_range: tuple = (2, 14)
    y_jitter: int = 2
    noise_sigma: tuple = (0.0, 0.0)
    blur: tuple = (0.0, 0.0)
    exclude_strings: list = field(default_factory=list)  # e.g. dev strings excluded from final


def font_dir() -> Path:
    return Path(os.environ.get("CELLFORGE_FONT_DIR", "fonts"))


def font_path(name: str) -> Path:
    p = Path(name)
    if not p.is_absolute():
        p = font_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"font {p} not found; set CELLFORGE_FONT_DIR")
    return p


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def render_one(text: str, font: ImageFont.FreeTypeFont, x: int, y_off: int):
    """Render text; return uint8 array or None if any ink would be clipped."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    y = (H - (b - t)) // 2 - t + y_off
    if x + l < 0 or x + r > W or y + t < 0 or y + b > H:
        return None
    d.text((x, y), text, fill=255, font=font)
    return np.asarray(img, dtype=np.uint8)


def generate(spec: RenderSpec, n: int, seed: int, split: str):
    """Generate n examples. Returns (images uint8 [n,H,W], labels, manifest)."""
    rng = np.random.default_rng(seed)
    fonts = {}
    font_hashes = {}
    for f in spec.fonts:
        p = font_path(f)
        font_hashes[f] = file_sha256(p)
    exclude = set(spec.exclude_strings)
    images, labels, records = [], [], []
    attempts = 0
    while len(labels) < n:
        attempts += 1
        if attempts > n * 50:
            raise RuntimeError("renderer cannot satisfy spec without clipping")
        L = int(rng.integers(spec.min_len, spec.max_len + 1))
        text = "".join(rng.choice(list(spec.charset), size=L))
        if text in exclude:
            continue
        fname = spec.fonts[int(rng.integers(len(spec.fonts)))]
        size = int(rng.integers(spec.size_range[0], spec.size_range[1] + 1))
        key = (fname, size)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(font_path(fname)), size)
        x = int(rng.integers(spec.x_range[0], spec.x_range[1] + 1))
        yj = int(rng.integers(-spec.y_jitter, spec.y_jitter + 1)) if spec.y_jitter else 0
        arr = render_one(text, fonts[key], x, yj)
        if arr is None:
            continue
        sigma = float(rng.uniform(*spec.noise_sigma))
        if sigma > 0:
            noisy = arr.astype(np.float64) + rng.normal(0, sigma * 255, arr.shape)
            arr = np.clip(np.round(noisy), 0, 255).astype(np.uint8)
        images.append(arr)
        labels.append(text)
        records.append({"font": fname, "size": size, "x": x, "y_jitter": yj, "noise_sigma": round(sigma, 5)})
    imgs = np.stack(images)
    manifest = {
        "split": split,
        "seed": seed,
        "n": n,
        "spec": asdict(spec) | {"exclude_strings": len(spec.exclude_strings)},
        "font_sha256": font_hashes,
        "pixel_contract": {"height": H, "width": W, "dtype": "uint8", "ink": "bright", "normalize": "x/255"},
        "images_sha256": hashlib.sha256(imgs.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest(),
        "records": records,
    }
    return imgs, labels, manifest


def save_dataset(directory: str | Path, imgs, labels, manifest, png_count: int = 0):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "images.npz", images=imgs)
    (d / "labels.json").write_text(json.dumps(labels))
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if png_count:
        (d / "png").mkdir(exist_ok=True)
        for i in range(min(png_count, len(labels))):
            Image.fromarray(imgs[i]).save(d / "png" / f"{i:05d}.png")
    return hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()


def load_dataset(directory: str | Path):
    d = Path(directory)
    with np.load(d / "images.npz", allow_pickle=False) as z:
        imgs = z["images"]
    labels = json.loads((d / "labels.json").read_text())
    manifest = json.loads((d / "manifest.json").read_text())
    if hashlib.sha256(imgs.tobytes()).hexdigest() != manifest["images_sha256"]:
        raise ValueError(f"dataset {d} images fail integrity check")
    return imgs, labels, manifest
