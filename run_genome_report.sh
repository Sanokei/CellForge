#!/usr/bin/env bash
# Current literal-weight reports. Legacy allele history stays separate.
set -eu
cd "$(dirname "$0")"
"${CELLFORGE_PYTHON:-python}" scripts/weights_progress.py "$@"
# Versioned outputs are printed above; never overwrite/open old reports.
