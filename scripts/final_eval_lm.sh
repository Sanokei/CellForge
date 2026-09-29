#!/usr/bin/env bash
# Usage: scripts/final_eval_lm.sh <local artifact dir> <label>
set -euo pipefail
ART=$1; LABEL=$2; HOST=${HOST:-cellforge-1}
NAME=lm-$(cut -c1-16 "$ART/HASH")
tar czf - -C "$ART" . | ssh "$HOST" "mkdir -p /srv/cellforge-inbox/$NAME && tar xzf - -C /srv/cellforge-inbox/$NAME && chmod -R a+rX /srv/cellforge-inbox/$NAME"
ssh "$HOST" "cd /tmp && sudo -u evaluator /opt/cellforge/venv/bin/python -m cellforge.reference.lm_operator \
  --artifact /srv/cellforge-inbox/$NAME --private /srv/cellforge-private --public /srv/cellforge-public \
  --dev-train /srv/cellforge-public/tinylm/dev_train.json --label '$LABEL'"
