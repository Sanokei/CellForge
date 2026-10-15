base=/srv/cellforge-inbox/two-loop-20261001
nohup /srv/arc3-claude/venv/bin/python "$base/moe_snapshot_watch.py" > "$base/moe-scheduler.log" 2>&1 < /dev/null &
echo $! > "$base/moe-scheduler.pid"
echo MOE_SCHEDULER_PID=$(cat "$base/moe-scheduler.pid")
