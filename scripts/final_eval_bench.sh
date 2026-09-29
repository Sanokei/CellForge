#!/usr/bin/env bash
# Usage: scripts/final_eval_bench.sh <local artifact dir> <label> [n] [benches]
set -euo pipefail
ART=$1; LABEL=$2; N=${3:-1000}; BENCH=${4:-mmlu,arc_challenge,hellaswag}; HOST=${HOST:-cellforge-1}
NAME=bench-$(cut -c1-16 "$ART/HASH")
tar czf - -C "$ART" . 2>/dev/null | ssh "$HOST" "mkdir -p /srv/cellforge-inbox/$NAME && tar xzmf - -C /srv/cellforge-inbox/$NAME && chmod -R a+rX /srv/cellforge-inbox/$NAME"
ssh "$HOST" "cd /tmp && sudo -u evaluator /opt/cellforge/venv/bin/python -m cellforge.reference.bench_operator \
  --artifact /srv/cellforge-inbox/$NAME --private /srv/cellforge-private --public /srv/cellforge-public \
  --label "$LABEL" --n $N --benches $BENCH --split ${SPLIT:-all}"
