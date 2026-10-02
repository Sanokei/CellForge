#!/usr/bin/env bash
# Build paper/cellforge.tex on the server (LaTeX is installed there) and fetch the PDF + log.
set -euo pipefail
HOST=${HOST:-cellforge-1}
cd "$(dirname "$0")/../paper"
ssh "$HOST" 'rm -rf /tmp/paper && mkdir -p /tmp/paper'
tar czf - cellforge.tex $(ls *.bib *.png *.pdf 2>/dev/null | grep -v '^cellforge.pdf$' || true) 2>/dev/null | ssh "$HOST" 'tar xzmf - -C /tmp/paper'
ssh "$HOST" 'cd /tmp/paper && { latexmk -pdf -interaction=nonstopmode -halt-on-error cellforge.tex > build.log 2>&1; status=$?; echo exit=$status; tail -3 build.log; if [ "$status" -ne 0 ]; then tail -40 build.log; exit "$status"; fi; }'
scp -q "$HOST:/tmp/paper/cellforge.pdf" cellforge.pdf
scp -q "$HOST:/tmp/paper/build.log" build.log
ls -l cellforge.pdf
