set -eu
base=/srv/cellforge-inbox/two-loop-20261001
runtime=$base/planning-runtime
root=$base/runs/weights-resource-planning
if test -e "$runtime" || test -e "$root"; then echo 'refusing duplicate'; exit 1; fi
mkdir "$runtime"
tar -xzf "$base/planning-runtime.tar.gz" -C "$runtime"
export PYTHONPATH="$runtime/src" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/srv/arc3-claude/venv/bin/python -c 'import pathlib,json,hashlib; r=pathlib.Path("/srv/cellforge-inbox/two-loop-20261001/planning-runtime"); m=json.loads((r/"SOURCE_MANIFEST.json").read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in m.items()); print("all frozen source hashes verified")'
anchor=$(/srv/arc3-claude/venv/bin/python -m cellforge.construction.planning_cell_service init "$root" --seeds 901 902 903)
nohup /srv/arc3-claude/venv/bin/python -m cellforge.construction.planning_cell_service run "$root" --control "$base/control/resource-planning" --anchor "$anchor" > "$root/service.log" 2>&1 < /dev/null &
echo "planning_pid=$! anchor=$anchor"
