#!/usr/bin/env bash
set -uo pipefail
cd /srv/cellforge-inbox/arc-best-a5-audit-20261001
export PYTHONPATH="$PWD/src"
export ARC_ROOT=/srv/cellforge-private/arcagi2/data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
run=runs/weights-arc-best-a5-v1
printf '%s\n' "$ARC_ROOT" > "$run/ARC_ROOT.txt"
nice -n 15 timeout 1200 /opt/cellforge/venv/bin/python scripts/score_arc_weights.py "$run" --arc-root "$ARC_ROOT"
score_exit=$?
printf '%s\n' "$score_exit" > "$run/score-exit.txt"
if [ "$score_exit" -eq 0 ]; then
  /opt/cellforge/venv/bin/python -m unittest tests.test_arc_goal -v > "$run/arc-goal-tests.log" 2>&1
  printf '%s\n' "$?" > "$run/test-exit.txt"
fi
