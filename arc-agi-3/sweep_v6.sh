#!/bin/bash
# usage: sweep_v6.sh AGENT OUTFILE game1 game2 ...   (<=2 workers, low priority)
agent=$1; out=$2; shift 2
cd /srv/arc3-claude/v6run
printf '%s\n' "$@" | xargs -P2 -I{} sh -c "nice -n 10 ../venv/bin/python run.py --agent $agent --game {} 2>/dev/null | grep '^RESULT' >> $out"
