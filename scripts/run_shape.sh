#!/usr/bin/env bash
# The whole per-shape workflow, in order:
#   1. cube warm-up (only if the checkpoint is missing)
#   2. fit: extract the loops and fit them (seam distillation + frozen start)
#   3. seam check of the fitted network
#   4. code clusters of the fitted loops
#   5. seam view and sweep movie of the fitted loops
#   6. propagation into tubes
#
#   scripts/run_shape.sh bunny_points_cubic_lam3
#   SAMPLES=9 EPOCHS=200 FREEZE=20 scripts/run_shape.sh horse_points_cubic_lam3   # quick pass
#   FREEZE=0 scripts/run_shape.sh bunny_points_cubic_lam3                         # no frozen start
#
# The shape is a file name (without .ply) in $DATA. Everything goes into one
# folder per run, $OUT (default log/<shape>_<date>_<time>/, so a rerun never
# overwrites an earlier one):
#   fit/            the fitted run: model, loops, planes/, figures, clusters, seam view, movie
#   propagation/    the tubes
#   <shape>.log     the whole console output
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...
DATA=${DATA:-Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid}
WARMUP=${WARMUP:-warmup/cube_warmup_unit_centroid.pt}
SAMPLES=${SAMPLES:-29}
EPOCHS=${EPOCHS:-1000}
DISTILL=${DISTILL:-10}
FREEZE=${FREEZE:-100}          # epochs the shared MLP stays at the warm-up's map
K=${K:-10}                     # clusters for the code clustering

SHAPE=${1:?usage: scripts/run_shape.sh <shape name in $DATA>}
OUT=${OUT:-log/${SHAPE}_$(date +%Y%m%d_%H%M%S)}
RUN=$OUT/fit
mkdir -p "$OUT"
exec > >(tee "$OUT/$SHAPE.log") 2>&1     # everything below is also kept in the log

echo "== 1. warm-up"
if [ -f "$WARMUP" ]; then
    echo "  $WARMUP exists, reused"
else
    # one unit cube (--cubes 0), centre features, plane axis as input: must match the fit
    "$PYTHON" model/warmup_cube.py --out "$WARMUP" --feature-pool centroid --cubes 0 --axis-condition
fi

echo "== 2. fit"
# --save-plane-files: propagation (step 6) reads the per-plane slice and fit files
"$PYTHON" plc_main.py --train --path "$DATA/$SHAPE.ply" --out "$RUN" \
    --samples "$SAMPLES" --epochs "$EPOCHS" --normal 0.01 --triplane-tv 0.01 \
    --warmup "$WARMUP" --distill "$DISTILL" --freeze-f-epochs "$FREEZE" \
    --save-plane-files --snapshot-every 0 \
    --feature-pool centroid --axis-condition

echo "== 3. seam check"
"$PYTHON" utils/seam_check.py "$RUN/${SHAPE}_model.pt"

echo "== 4. code clusters"
"$PYTHON" utils/code_clusters.py "$RUN/${SHAPE}_triplane.npz" --k "$K"

echo "== 5. seam view and sweep movie"
"$PYTHON" utils/seam_z0_view.py "$RUN"
"$PYTHON" utils/loop_sweep_movie.py --run "$RUN"

echo "== 6. propagation"
"$PYTHON" utils/loop_propagation.py --run "$RUN" --out "$OUT/propagation" --step all
