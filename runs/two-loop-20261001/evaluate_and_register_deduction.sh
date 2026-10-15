set -eu
base=/srv/cellforge-inbox/two-loop-20261001
export PYTHONPATH="$base/hard-reasoning-runtime/src" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/srv/arc3-claude/venv/bin/python "$base/evaluate_hard_reasoning_reporting.py"
cp "$base/selected_cell_registry.py" "$base/hard_reasoning_adapter.py" "$base/shared-registry-runtime/src/cellforge/construction/"
export PYTHONPATH="$base/shared-registry-runtime/src"
/srv/arc3-claude/venv/bin/python -m cellforge.construction.selected_cell_registry "$base/runs/weights-arithmetic-subtraction-continuous" "$base/runs/weights-spatial-hardlane" "$base/shared-cell-registry" --reasoning-root "$base/runs/weights-hard-signed-deduction" > "$base/shared-cell-registry/refresh-three-families.log"
cat "$base/shared-cell-registry/LATEST.json"
