set -eu
base=/srv/cellforge-inbox/two-loop-20261001
root=$base/runs/weights-spatial-hardlane
if test -e "$root"; then echo 'refusing duplicate root'; exit 1; fi
mkdir "$root"
tar -xzf "$base/spatial-hardlane.tar.gz" -C "$root"
export PYTHONPATH="$root/src" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
nohup /srv/arc3-claude/venv/bin/python -m cellforge.construction.reasoning_cell_epochs run "$root" --anchor 275bdc5f584b772f693fba7dd0a9037ab791ec31263a4e691cf91ebca2d422b1 --control "$base/control/spatial-hardlane" --max-seconds 0 > "$root/service.log" 2>&1 < /dev/null &
echo "spatial_pid=$!"
