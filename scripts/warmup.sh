#!/usr/bin/env bash
# Train the cube warm-up checkpoint every fit starts from (a few seconds).
# Run it once per network size; the default matches `main.py fit`'s defaults.
#
#   scripts/warmup.sh                       # -> warmup/cube_warmup_axis.pt
#   scripts/warmup.sh --layers 4 --neurons 32 --out warmup/cube_warmup_4x32.pt
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...

"$PYTHON" main.py warmup "$@"
