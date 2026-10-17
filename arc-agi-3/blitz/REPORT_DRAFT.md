# Blitz Report (DRAFT)

## Best variant: blitz4
- Code: /srv/arc3-claude/v6run/blitz/blitz_agent.py (BlitzExplorer4)
- Runner: /srv/arc3-claude/v6run/blitz/run_blitz.py --agent blitz4
- Mean dev score (15 games, seed 0, budget 3000): 0.191
- Baseline v6:pr,nm,ck,gv: 0.1867

## What changed vs baseline
blitz4 adds a stall-triggered diverse-restart loop to explorer5:
- When the state graph is fully explored (no frontier), instead of the
  baseline\s lattice-spam + random walk, blitz4 RESETs to the level root,
