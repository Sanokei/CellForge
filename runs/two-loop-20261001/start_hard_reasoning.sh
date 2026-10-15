set -eu
base=/srv/cellforge-inbox/two-loop-20261001
runtime=$base/hard-reasoning-runtime
root=$base/runs/weights-hard-signed-deduction
if test -e "$runtime" || test -e "$root"; then echo 'refusing duplicate'; exit 1; fi
mkdir "$runtime"
tar -xzf "$base/hard-reasoning-runtime.tar.gz" -C "$runtime"
export PYTHONPATH="$runtime/src" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
anchor=$(/srv/arc3-claude/venv/bin/python -m cellforge.construction.hard_reasoning_service init "$root" --seeds 801 802 803)
nohup /srv/arc3-claude/venv/bin/python -m cellforge.construction.hard_reasoning_service run "$root" --control "$base/control/hard-signed-deduction" --anchor "$anchor" > "$root/service.log" 2>&1 < /dev/null &
echo "hard_reasoning_pid=$! anchor=$anchor"
