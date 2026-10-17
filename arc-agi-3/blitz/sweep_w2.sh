#!/bin/bash
# usage: sweep_w2.sh AGENT OUTFILE [GAMES...]
agent=$1; out=$2; shift 2
games="$*"
[ -z "$games" ] && games="bp35 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 re86 s5i5 sc25 sk48 tr87 tu93"
cd /srv/arc3-claude/v6run/blitz
rm -f "$out"
for g in $games; do echo $g; done | xargs -P6 -I{} sh -c "nice -n 10 /srv/arc3-claude/venv/bin/python run_blitz.py --agent $agent --game {} 2>/dev/null | grep ^RESULT >> $out"
echo "sweep done -> $out"
