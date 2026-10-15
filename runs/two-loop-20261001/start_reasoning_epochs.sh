set -eu
base=/srv/cellforge-inbox/two-loop-20261001
root=$base/runs/weights-reasoning-cell-epochs
anchor=18bff7985ec2306bea9b816bbcd4359c2c57235586129f56dd12465ce5ccf164
/srv/arc3-claude/venv/bin/python "$base/record_reasoning_initial.py"
PYTHONPATH="$root/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nohup nice -n 10 /srv/arc3-claude/venv/bin/python -m cellforge.construction.reasoning_cell_epochs run "$root" --anchor "$anchor" --control "$base/control/reasoning-cell-epochs" --max-seconds 1500 > "$root/service.log" 2>&1 < /dev/null &
echo $! > "$root/SERVICE.pid"
echo REASONING_CELL_PID=$(cat "$root/SERVICE.pid")

