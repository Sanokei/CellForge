set -eu
base=/srv/cellforge-inbox/two-loop-20261001
root=$base/runs/weights-arc-moe-smallskill-v2
anchor=99364caef0ae6e8e9283ed435cb4d398bbee80e455a1b3ea8503f2ed9f283f08
control=$base/control/moe-smallskill-v2.json
test "$(sha256sum "$root/DECLARATION.json" | cut -d' ' -f1)" = "$anchor"
test ! -e "$root/SERVICE.lock"
cd "$root"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src" nohup nice -n 10 /srv/arc3-claude/venv/bin/python -m cellforge.construction.arc_moe_service "$root" --anchor "$anchor" --external-controller "$control" --max-seconds 1800 --cadence 30 > "$root/service.log" 2>&1 < /dev/null &
echo $! > "$root/SERVICE.pid"
for step in $(seq 1 15);do test ! -f "$root/STATE-0.json" || break;sleep 1;done
test -f "$root/STATE-0.json"
PYTHONPATH="$base:$root/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nohup nice -n 10 /srv/arc3-claude/venv/bin/python -m server_author_controller drive "$root" --kind moe --anchor "$anchor" --control "$control" --workspace "$base/author-moe-v2" --max-seconds 1200 > "$base/author-moe-v2.log" 2>&1 < /dev/null &
echo $! > "$base/author-moe-v2.pid"
echo MOE_SERVICE_PID=$(cat "$root/SERVICE.pid") AUTHOR_CONTROLLER_PID=$(cat "$base/author-moe-v2.pid")
