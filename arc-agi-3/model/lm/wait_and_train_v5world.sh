#!/bin/bash
# Track-2 world-model gate: wait for Track 1 to finish (router_v5.pt +
# router_v5.json + traces_fc.jsonl, mtime stable, no train_lm.py running),
# then launch the full v5world training run. CPU courtesy: never co-runs
# with a track-1 training process.
LM=/srv/arc3-claude/v6run/model/lm
export OMP_NUM_THREADS=1
PY=/srv/arc3-claude/venv/bin/python
MARKER=$LM/v5world_waiter.status
echo "waiter started $(date -u)" > "$MARKER"
deadline=$((SECONDS + 16200))   # ~4.5h budget from launch
while [ $SECONDS -lt $deadline ]; do
  if [ -f "$LM/router_v5.json" ] && [ -f "$LM/router_v5.pt" ] && [ -f "$LM/traces_fc.jsonl" ]; then
    if ! pgrep -f "train_lm.py" >/dev/null 2>&1; then
      m1=$(stat -c %Y "$LM/router_v5.json")
      echo "files present, stability check $(date -u)" >> "$MARKER"
      sleep 120
      m2=$(stat -c %Y "$LM/router_v5.json")
      if [ "$m1" = "$m2" ] && ! pgrep -f "train_lm.py" >/dev/null 2>&1; then
        echo "TRACK1_DONE starting training $(date -u)" >> "$MARKER"
        cd /srv/arc3-claude/v6run/model || exit 3
        $PY lm/train_world.py --init lm/router_v5.pt --corpus traces_fc.jsonl \
            --out router_v5world --steps 500 > lm/train_v5world.log 2>&1
        rc=$?
        echo "TRAINING_EXIT $rc $(date -u)" >> "$MARKER"
        exit $rc
      fi
    fi
  fi
  sleep 180
done
echo "TIMEOUT track1 files never completed $(date -u)" >> "$MARKER"
exit 2
