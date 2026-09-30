#!/bin/bash
# usage: sweep.sh AGENT BUDGET OUTFILE [GAMES...]   (2 parallel procs, nice 10)
AGENT=$1; BUDGET=$2; OUT=$3; shift 3
cd /srv/arc3-claude/agent
GAMES=${@:-"lf52 g50t wa30 dc22 su15 bp35 sb26 ft09 r11l cd82 m0r0 cn04 tu93 sc25 sk48 tn36 ka59 sp80 ls20 tr87 re86 lp85 s5i5 ar25 vc33"}
: > $OUT
for g in $GAMES; do echo $g; done | xargs -P 2 -I{} sh -c "nice -n 10 ../venv/bin/python run.py --agent $AGENT --game {} --budget $BUDGET 2>/dev/null | grep '^RESULT' >> $OUT"
echo done >> $OUT.done
