#!/bin/bash
# usage: sweep_blitz.sh AGENT OUTFILE   (runs all 15 dev games, 2 workers)
agent=$1; out=$2
cd /srv/arc3-claude/v6run/blitz
for g in bp35 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 re86 s5i5 sc25 sk48 tr87 tu93; do echo $g; done | xargs -P2 -I{} sh -c "nice -n 10 /srv/arc3-claude/venv/bin/python run_blitz.py --agent $agent --game {} 2>/dev/null | grep ^RESULT >> $out"
echo "sweep done -> $out"
