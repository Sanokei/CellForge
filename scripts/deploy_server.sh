#!/usr/bin/env bash
# Deploy CellForge code to the evaluation server (read-only code for evaluator).
set -euo pipefail
HOST=${1:-cellforge-1}
tar czf - --exclude=runs --exclude=fonts --exclude=__pycache__ --exclude=.git --exclude=paper src pyproject.toml \
  | ssh "$HOST" 'set -e
    id evaluator >/dev/null 2>&1 || useradd -r -m -d /home/evaluator -s /usr/sbin/nologin evaluator
    id builder >/dev/null 2>&1 || useradd -m -s /bin/bash builder
    mkdir -p /opt/cellforge/code /opt/cellforge/fonts /srv/cellforge-private /srv/cellforge-public /srv/cellforge-inbox
    rm -rf /opt/cellforge/code/src && tar xzf - -C /opt/cellforge/code
    for f in /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf /usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf \
             /usr/share/fonts/truetype/freefont/FreeMono.ttf /usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf; do
      ln -sf $f /opt/cellforge/fonts/; done
    [ -x /opt/cellforge/venv/bin/python ] || python3 -m venv /opt/cellforge/venv
    /opt/cellforge/venv/bin/pip install -q numpy==1.26.4 pillow==10.2.0 >/dev/null
    /opt/cellforge/venv/bin/pip install -q --no-deps -e /opt/cellforge/code >/dev/null
    chown -R root:root /opt/cellforge/code && chmod -R a+rX,go-w /opt/cellforge
    chown evaluator:evaluator /srv/cellforge-private && chmod 700 /srv/cellforge-private
    chown evaluator:evaluator /srv/cellforge-public && chmod 755 /srv/cellforge-public
    chmod 1777 /srv/cellforge-inbox
    echo deployed'
