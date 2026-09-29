#!/usr/bin/env bash
# Usage: scripts/final_eval.sh <local artifact dir> <protocol> <label> [splits]
set -euo pipefail
ART=$1; PROTO=$2; LABEL=$3; SPLITS=${4:-}; HOST=${HOST:-cellforge-1}
NAME=$(cat "$ART/HASH" | cut -c1-16)
tar czf - -C "$ART" . | ssh "$HOST" "mkdir -p /srv/cellforge-inbox/$NAME && tar xzf - -C /srv/cellforge-inbox/$NAME && chmod -R a+rX /srv/cellforge-inbox/$NAME"
scp -q "$(dirname "$ART")/author_visible_strings.txt" "$HOST:/srv/cellforge-inbox/$NAME.exclude.txt"
ssh "$HOST" "cd /tmp && sudo -u evaluator CELLFORGE_FONT_DIR=/opt/cellforge/fonts /opt/cellforge/venv/bin/python -m cellforge.reference.operator \
  --artifact /srv/cellforge-inbox/$NAME --protocol $PROTO --private /srv/cellforge-private --public /srv/cellforge-public \
  --exclude-strings /srv/cellforge-inbox/$NAME.exclude.txt --label '$LABEL' --splits '$SPLITS'"
