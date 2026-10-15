set -eu
base=/srv/cellforge-inbox/two-loop-20261001
runtime=$base/shared-registry-runtime
if ! test -d "$runtime"; then mkdir "$runtime"; cp -a "$base/runs/weights-spatial-hardlane/src" "$runtime/src"; fi
cp "$base/selected_cell_registry.py" "$base/cell_population.py" "$runtime/src/cellforge/construction/"
export PYTHONPATH="$runtime/src" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/srv/arc3-claude/venv/bin/python -m cellforge.construction.selected_cell_registry "$base/runs/weights-arithmetic-subtraction-continuous" "$base/runs/weights-spatial-hardlane" "$base/shared-cell-registry"
