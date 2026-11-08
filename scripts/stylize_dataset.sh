#!/usr/bin/env bash
# Cubic stylization of every point cloud in a folder: the step that makes the
# pipeline's *_cubic_lam3.ply inputs from normalised clouds with normals.
#
#   scripts/stylize_dataset.sh                                  # every Datasets/Genus0/*.ply
#   IN=Datasets/Genus0 SHAPES="horse rocker-arm" scripts/stylize_dataset.sh
#   LAM=6 ITERS=100 scripts/stylize_dataset.sh                  # other settings
#
# Outputs: $OUT/<shape>/<shape>_cubic_lam<LAM>.ply, with metrics.json and
# before_after.png beside it (OUT defaults to log/cubic_lam<LAM>).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...
IN=${IN:-Datasets/Genus0}
LAM=${LAM:-3}
ITERS=${ITERS:-60}
OUT=${OUT:-log/cubic_lam$LAM}

if [ -n "${SHAPES:-}" ]; then
    files=(); for s in $SHAPES; do files+=("$IN/$s.ply"); done
else
    files=("$IN"/*.ply)
fi

for path in "${files[@]}"; do
    shape=$(basename "$path" .ply)
    echo "== $shape"
    "$PYTHON" main.py stylize --points "$path" --out "$OUT/$shape" \
        --lam "$LAM" --iters "$ITERS"
done
