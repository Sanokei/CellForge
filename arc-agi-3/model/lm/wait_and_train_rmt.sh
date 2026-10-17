#!/bin/bash
# Track-3 RMT gate (v2, 2026-10-16 ~01:35 EDT).
# Sequencing: we train THIRD. Track 2 is currently running:
#   lm/train_world.py --init lm/router_v3.pt --corpus traces_w2.jsonl,traces_fc.jsonl
#   --out router_v3world --steps 500
# (Track 2 re-based on v3 after the v5 dev regression.)
#
# Fire condition A (preferred): Track 2 done -> router_v3world.pt+json
#   present, mtime stable 120s, no train_*.py running, traces_fc.jsonl stable.
# Fire condition B (fallback): past 05:45 EDT, no train_*.py running,
#   traces_fc.jsonl stable. Guarantees a window before the 07:00 results
#   deadline; the no-training-running guard keeps us third in practice.
# Then: full RMT training, warm-start router_v3.pt (coordinator: v5
# regressed on dev; v3 is the known-good base).
LM=/srv/arc3-claude/v6run/model/lm
export OMP_NUM_THREADS=1
PY=/srv/arc3-claude/venv/bin/python
MARKER=$LM/rmt_waiter.status
echo "rmt waiter v2 started $(date -u)" > "$MARKER"
deadline=$((SECONDS + 16200))   # hard stop ~06:05 EDT
fallback_epoch=$(date -d "2026-10-16 05:45:00 EDT" +%s)

stable120() { # $1 = file -> 0 if mtime unchanged over 120s
  [ -f "$1" ] || return 1
  m1=$(stat -c %Y "$1"); sleep 120
  m2=$(stat -c %Y "$1"); [ "$m1" = "$m2" ]
}
no_training() { ! pgrep -f "train_.*\.py" >/dev/null 2>&1; }

fire() {
  echo "$1 starting RMT training (warm router_v3.pt) $(date -u)" >> "$MARKER"
  cd "$LM" || exit 3
  $PY train_rmt.py --epochs 4 --warm router_v3.pt --out router_rmt \
      > train_rmt.log 2>&1
  rc=$?
  echo "TRAINING_EXIT $rc $(date -u)" >> "$MARKER"
  exit $rc
}

while [ $SECONDS -lt $deadline ]; do
  # Condition A: Track 2 complete
  if [ -f "$LM/router_v3world.pt" ] && [ -f "$LM/router_v3world.json" ] \
     && no_training; then
    echo "A: v3world present, stability check $(date -u)" >> "$MARKER"
    if stable120 "$LM/router_v3world.json" && no_training \
       && stable120 "$LM/traces_fc.jsonl"; then
      fire "TRACK2_DONE"
    fi
  fi
  # Condition B: fallback window
  if [ "$(date +%s)" -ge "$fallback_epoch" ] && no_training; then
    echo "B: fallback window, stability check $(date -u)" >> "$MARKER"
    if stable120 "$LM/traces_fc.jsonl"; then
      fire "FALLBACK"
    fi
  fi
  sleep 180
done
echo "TIMEOUT prerequisites never completed $(date -u)" >> "$MARKER"
exit 2
