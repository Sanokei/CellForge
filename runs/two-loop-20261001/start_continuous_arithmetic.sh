set -eu
base=/srv/cellforge-inbox/two-loop-20261001
runtime=$base/continuous-arithmetic-runtime
root=$base/runs/weights-arithmetic-subtraction-continuous
control=$base/control/arithmetic-subtraction-continuous
test ! -e "$runtime";mkdir "$runtime"
tar -xzf "$base/continuous-arithmetic-runtime.tar.gz" -C "$runtime"
export PYTHONPATH="$runtime/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
anchor=$(/srv/arc3-claude/venv/bin/python -m cellforge.construction.arithmetic_subtraction_service init "$root" --seeds 401 402 403)
mkdir -p "$control"
printf '{"anchor":"%s","recorded_before_author_and_selection":true,"route":"trusted_nonmodel_server_recorder"}\n' "$anchor" > "$control/INITIAL-COMMITMENT.json"
nohup nice -n 10 /srv/arc3-claude/venv/bin/python -m cellforge.construction.arithmetic_subtraction_service run "$root" --control "$control" --anchor "$anchor" > "$root/service.log" 2>&1 < /dev/null &
echo $! > "$root/SERVICE.pid"
echo ARITHMETIC_PID=$(cat "$root/SERVICE.pid") INITIAL_ANCHOR=$anchor
