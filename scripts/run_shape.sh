#!/usr/bin/env bash
# The whole per-shape workflow, in order:
#   1. cube warm-up (only if the checkpoint is missing)
#   2. fit: extract the loops and fit them (seam distillation + frozen start)
#   3. seam check of the fitted network
#   4. code clusters, at the end of the run and at the end of the freeze
#   5. seam view and sweep movie of the fitted loops
#   6. propagation into tubes
#
#   scripts/run_shape.sh bunny_points_cubic_lam3
#   SAMPLES=9 EPOCHS=200 FREEZE=20 scripts/run_shape.sh horse_points_cubic_lam3   # quick pass
#   FREEZE=0 scripts/run_shape.sh bunny_points_cubic_lam3                         # no frozen start
#
# The shape is a file name (without .ply) in $DATA. Everything goes into one
# folder per shape, $OUT (default log/<shape>/):
#   fit/            the fitted run: model, loops, figures, clusters, seam view, movie
#   propagation/    the tubes
#   <shape>.log     the whole console output
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...
DATA=${DATA:-Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid}
WARMUP=${WARMUP:-warmup/cube_warmup_axis.pt}
SAMPLES=${SAMPLES:-29}
EPOCHS=${EPOCHS:-1000}
DISTILL=${DISTILL:-10}
FREEZE=${FREEZE:-100}          # epochs the shared MLP stays at the warm-up's map
K=${K:-10}                     # clusters for the code clustering

SHAPE=${1:?usage: scripts/run_shape.sh <shape name in $DATA>}
OUT=${OUT:-log/$SHAPE}
RUN=$OUT/fit
mkdir -p "$OUT"
exec > >(tee "$OUT/$SHAPE.log") 2>&1     # everything below is also kept in the log

echo "== 1. warm-up"
if [ -f "$WARMUP" ]; then
    echo "  $WARMUP exists, reused"
else
    "$PYTHON" main.py warmup --out "$WARMUP"
fi

echo "== 2. fit"
freeze_args=(--freeze-f-epochs "$FREEZE")
if [ "$FREEZE" -gt 0 ]; then
    freeze_args+=(--save-states "$FREEZE")
fi
"$PYTHON" main.py fit --path "$DATA/$SHAPE.ply" --out "$RUN" \
    --samples "$SAMPLES" --epochs "$EPOCHS" --normal 0.01 --triplane-tv 0.01 \
    --warmup "$WARMUP" --distill "$DISTILL" "${freeze_args[@]}" --snapshot-every 0

echo "== 3. seam check"
"$PYTHON" main.py seam-check "$RUN/${SHAPE}_model.pt"

echo "== 4. code clusters"
"$PYTHON" main.py clusters "$RUN/${SHAPE}_triplane.npz" --k "$K"
if [ "$FREEZE" -gt 0 ]; then
    tag=e$(printf '%04d' "$FREEZE")
    "$PYTHON" main.py clusters "$RUN/${SHAPE}_triplane.npz" --k "$K" \
        --state "$RUN/states/${SHAPE}_state_$tag.pt" --tag "$tag"
fi

echo "== 5. seam view and sweep movie"
"$PYTHON" main.py seam-view "$RUN"
"$PYTHON" main.py sweep-movie --run "$RUN"

echo "== 6. propagation"
"$PYTHON" main.py propagate --run "$RUN" --out "$OUT/propagation" --step all
