#!/usr/bin/env python3
"""Read-only pre-publication scan.

Walks a directory and flags paths that must not be published because they may
hold sealed, held-out, private or secret material. Nothing is modified.

Classes:
  HOLD  - must not be published until the owner clears it (exit code 1).
  WARN  - needs a human look (large file, token-like string, FINAL*.json).

Exit code: 0 no HOLD findings, 1 at least one HOLD finding, 2 usage error.
"""
import argparse
import os
import re
import sys

HOLD_DIR_NAMES = re.compile(
    r"^(anchor|_accept|dev|final.*|sealed.*|heldout.*|held-out.*|private.*)$", re.I)
HOLD_ANY_NAME = re.compile(r"(server-aggregate|id_rsa|id_ed25519|\.pem$|\.key$)", re.I)
HOLD_EXACT_NAMES = {".env", ".netrc"}
HOLD_SUFFIXES = (".tar.gz", ".tgz", ".pkl", ".pickle", ".zip")
WARN_FILE_NAMES = re.compile(r"^final.*\.json$", re.I)
TEXT_SUFFIXES = (".txt", ".md", ".json", ".jsonl", ".py", ".sh", ".cfg", ".ini",
                 ".yaml", ".yml", ".toml", ".env", ".tex", ".csv", ".log")
SECRET_PATTERNS = [
    ("api_key", re.compile(r"api[_-]?key", re.I)),
    ("bearer", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}")),
    ("sk_token", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("long_hex_64plus", re.compile(r"(?<![0-9A-Fa-f])[0-9a-fA-F]{64,}(?![0-9A-Fa-f])")),
    ("long_base64", re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])")),
]
# sha256 hashes are normal provenance in receipts: report a long hex string only
# when the same line also mentions a key-like word.
HASH_CONTEXT = re.compile(r"(key|secret|token|passw)", re.I)
SKIP_DIRS = {".git", "__pycache__", "node_modules"}


def classify_path(parts, is_dir):
    """Return a list of (level, reason) for one path given its components."""
    out = []
    name = parts[-1]
    check = parts if is_dir else parts[:-1]
    for part in check:
        if HOLD_DIR_NAMES.match(part):
            out.append(("HOLD", "directory name '%s' matches sealed/eval-side pattern" % part))
            break
    if not is_dir:
        low = name.lower()
        if low in HOLD_EXACT_NAMES:
            out.append(("HOLD", "credential file name"))
        if HOLD_ANY_NAME.search(name):
            out.append(("HOLD", "name matches server-aggregate or key file"))
        if low.endswith(HOLD_SUFFIXES):
            out.append(("HOLD", "archive/pickle may hide contents (%s)" % low.rsplit(".", 1)[-1]))
        if WARN_FILE_NAMES.match(name):
            out.append(("WARN", "name starts with 'final' (owner confirm it is not final-set data)"))
    return out


def scan_text(path):
    hits = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for lineno, line in enumerate(fh, 1):
                for label, rx in SECRET_PATTERNS:
                    m = rx.search(line)
                    if not m:
                        continue
                    if label == "long_hex_64plus" and not HASH_CONTEXT.search(line):
                        continue
                    if label == "long_base64":
                        tok = m.group(0)
                        if tok.isalpha() or not re.search(r"\d", tok):
                            continue
                        if re.fullmatch(r"[0-9a-fA-F]+", tok):
                            continue
                    hits.append((lineno, label))
                    break
                if len(hits) >= 5:
                    break
    except OSError:
        pass
    return hits


def scan(root, max_mb, max_text_kb):
    findings = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        rel_dir = os.path.relpath(dirpath, root)
        base = [] if rel_dir == "." else rel_dir.split(os.sep)
        for d in list(dirnames):
            hits = classify_path(base + [d], True)
            if HOLD_DIR_NAMES.match(d):
                # flagged as a unit; never descend into or read it
                findings.append(("HOLD", "/".join(base + [d]), hits[0][1]))
                dirnames.remove(d)
        for f in sorted(filenames):
            parts = base + [f]
            rel = "/".join(parts)
            for lvl, why in classify_path(parts, False):
                findings.append((lvl, rel, why))
            full = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size > max_mb * 1024 * 1024:
                findings.append(("WARN", rel, "file is %.1f MB (> %s MB)" % (size / 1048576.0, max_mb)))
            if f.lower().endswith(TEXT_SUFFIXES) and size <= max_text_kb * 1024:
                for lineno, label in scan_text(full):
                    findings.append(("WARN", rel, "line %d looks like a secret (%s)" % (lineno, label)))
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Read-only pre-publication scan. Flags sealed/held-out/private paths "
                    "(anchor, _accept, dev, final*, sealed*, heldout*, private*, server-aggregate), "
                    "archives (*.tar.gz), pickles, .env, id_rsa*, secret-looking strings in small text "
                    "files, and large files. Exits 1 if any HOLD-class path is found.")
    ap.add_argument("directory", help="directory to scan")
    ap.add_argument("--max-mb", type=float, default=20.0, help="warn on files larger than this (default 20)")
    ap.add_argument("--max-text-kb", type=int, default=512,
                    help="only scan text files up to this size for secrets (default 512)")
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.directory):
        print("not a directory: %s" % args.directory, file=sys.stderr)
        return 2
    findings = scan(args.directory, args.max_mb, args.max_text_kb)
    findings.sort(key=lambda x: (x[0] != "HOLD", x[1]))
    if not args.quiet:
        for lvl, rel, why in findings:
            print("%-4s %s : %s" % (lvl, rel, why))
    holds = sum(1 for f in findings if f[0] == "HOLD")
    print("SUMMARY: %d HOLD, %d WARN in %s" % (holds, len(findings) - holds, args.directory))
    return 1 if holds else 0


if __name__ == "__main__":
    sys.exit(main())
