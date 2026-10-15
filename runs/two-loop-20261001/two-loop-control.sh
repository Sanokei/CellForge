#!/usr/bin/env bash
set -euo pipefail
base=/srv/cellforge-inbox/two-loop-20261001
case "${1:-}" in cells) module=cellforge.construction.action_cell_service; root="$base/runs/weights-action-continuation"; controller="$base/control/cell-anchors.json"; cadence=(--poll-seconds 2);;
moe) module=cellforge.construction.arc_moe_service; root="$base/runs/weights-arc-moe-continuation"; controller="$base/control/moe-anchors.json"; cadence=(--cadence 60);;
*) echo "usage: $0 cells|moe start|status|stop INITIAL_DECLARATION_SHA256" >&2; exit 2;; esac
operation="${2:-}";pidfile="$root/SERVICE.pid"
owned_pid() {
 test -f "$pidfile" || return 1
 pid=$(cat "$pidfile")
 [[ "$pid" =~ ^[0-9]+$ ]] || return 1
 kill -0 "$pid" 2>/dev/null || return 1
 tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -- "$module" >/dev/null || return 1
 tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -- "$root" >/dev/null || return 1
}
case "$operation" in
start)
 anchor="${3:?independently recorded initial anchor required}"
 [[ "$anchor" =~ ^[0-9a-f]{64}$ ]] || exit 2
 test -d "$root"
 actual=$(sha256sum "$root/DECLARATION.json" | cut -d' ' -f1)
 test "$actual" = "$anchor" || { echo "anchor differs";exit 3; }
 if owned_pid;then echo "already running PID=$pid";exit 0;fi
 test ! -e "$root/SERVICE.lock" && test ! -e "$root/EXECUTION.lock" || { echo "owner/recovery lock present";exit 4; }
 mkdir -p "$base/control"
 cd "$root"
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src" \
 nohup nice -n 10 /srv/arc3-claude/venv/bin/python -m "$module" "$root" --anchor "$anchor" \
 --external-controller "$controller" --max-seconds 1800 "${cadence[@]}" >> "$root/service.log" 2>&1 < /dev/null &
 pid=$!;printf '%s\n' "$pid" > "$pidfile"
 printf 'START_PID=%s ROOT=%s LOG=%s\n' "$pid" "$root" "$root/service.log";;
status)
 if owned_pid;then echo "PROCESS_ACTIVE_PID=$pid";else echo "NO_OWNED_ACTIVE_PROCESS";fi
 test ! -f "$root/SERVICE_STATUS.json" || cat "$root/SERVICE_STATUS.json"
 test ! -f "$root/service.log" || tail -n 4 "$root/service.log";;
stop)
 if owned_pid;then
  printf '{"requested_pid":%s,"operation":"operator_stop","utc":"%s"}\n' "$pid" "$(date -u +%FT%TZ)" > "$root/STOP_RECEIPT.json"
  kill -TERM "$pid"
  for step in $(seq 1 20);do kill -0 "$pid" 2>/dev/null || break;sleep .1;done
  if kill -0 "$pid" 2>/dev/null;then echo "stop pending; preserve locks";exit 5;fi
  test ! -f "$root/SERVICE.lock" || { test "$(cat "$root/SERVICE.lock")" = "$pid" && rm -- "$root/SERVICE.lock"; }
  echo "STOPPED_PID=$pid"
 else echo "NO_OWNED_ACTIVE_PROCESS";fi;;
*) echo "expected start|status|stop" >&2;exit 2;;
esac
