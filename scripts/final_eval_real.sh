#!/usr/bin/env bash
# Usage: scripts/final_eval_real.sh <local artifact dir> <label>
set -euo pipefail
ART=$1; LABEL=$2; HOST=${HOST:-cellforge-1}
NAME=real-$(cut -c1-16 "$ART/HASH")
tar czf - -C "$ART" . 2>/dev/null | ssh "$HOST" "mkdir -p /srv/cellforge-inbox/$NAME && tar xzmf - -C /srv/cellforge-inbox/$NAME && chmod -R a+rX /srv/cellforge-inbox/$NAME"
ssh "$HOST" "cd /tmp && sudo -u evaluator /opt/cellforge/venv/bin/python -m cellforge.reference.real_operator \
  --artifact /srv/cellforge-inbox/$NAME --private /srv/cellforge-private --public /srv/cellforge-public \
  --dev-train /srv/cellforge-public/tinyreal/dev_train.json --label '$LABEL'"
