base=/srv/cellforge-inbox/two-loop-20261001
root=$base/runs/weights-arc-moe-smallskill-v2
PYTHONPATH="$base:$root/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nohup /srv/arc3-claude/venv/bin/python "$base/server_author_controller.py" drive "$root" --kind moe --anchor 99364caef0ae6e8e9283ed435cb4d398bbee80e455a1b3ea8503f2ed9f283f08 --control "$base/control/moe-smallskill-v2.json" --workspace "$base/author-moe-v2" --max-seconds 1200 > "$base/author-moe-v2-resume.log" 2>&1 < /dev/null &
echo $! > "$base/author-moe-v2.pid"
