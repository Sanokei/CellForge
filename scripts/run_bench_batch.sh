#!/usr/bin/env bash
# Runs the benchmark suite on a list of "artifact_dir:label" specs; writes runs/bench_results/<label>.json
cd /c/Users/wkeif/Documents/GitHub/CellForge
for spec in "$@"; do
  art=${spec%%:*}; lab=${spec##*:}
  bash scripts/final_eval_bench.sh "$art" "$lab" 1000 > runs/bench_results/$lab.raw 2> runs/bench_results/$lab.err
  python -c "t=open('runs/bench_results/$lab.raw').read(); open('runs/bench_results/$lab.json','w').write(t[t.index('{'):])"
done
echo batch-done > runs/bench_results/DONE_$$
