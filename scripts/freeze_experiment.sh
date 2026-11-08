#!/usr/bin/env bash
# The frozen-start comparison: on eight shapes, the distillation fit with and
# without --freeze-f-epochs 100 (29 planes per axis, 1000 epochs, --normal 0.01,
# --distill 10), then the triplane codes of both arms clustered at k = 10 -- at
# the end of the run, and for the frozen arm also at epoch 100, the end of the
# freeze.
#
#   scripts/freeze_experiment.sh                      # all eight shapes
#   SHAPES="horse_points_cubic_lam3" scripts/freeze_experiment.sh
#
# Outputs: log/axis_seam10_<shape>/ (control; shared with distill_experiment.sh
# and reused when already there), log/axis_frz100_<shape>/ (frozen, with
# states/<shape>_state_e0100.pt), each with its console log beside it, and
# <stem>_code_clusters*.png/.csv/.ply inside each run.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...
DATA=${DATA:-Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid}
WARMUP=${WARMUP:-warmup/cube_warmup_axis.pt}
SAMPLES=${SAMPLES:-29}
EPOCHS=${EPOCHS:-1000}
FREEZE=${FREEZE:-100}
SHAPES=${SHAPES:-"bunny_points_cubic_lam3 armadillo_normal_cubic_lam3 horse_points_cubic_lam3
                  lucy_points_cubic_lam3 laurent_hand_normal_cubic_lam3 rocker-arm_points_cubic_lam3
                  neptune_normal_cubic_lam3 dancing_children100K_cubic_lam3"}

[ -f "$WARMUP" ] || "$PYTHON" main.py warmup --out "$WARMUP"
mkdir -p log
fit=("$PYTHON" main.py fit --samples "$SAMPLES" --epochs "$EPOCHS" --normal 0.01
     --triplane-tv 0.01 --warmup "$WARMUP" --distill 10 --snapshot-every 0)

for shape in $SHAPES; do
    control=log/axis_seam10_$shape
    frozen=log/axis_frz${FREEZE}_$shape
    if [ ! -f "$control/${shape}_model.pt" ]; then
        echo "== $shape, control -> $control"
        "${fit[@]}" --path "$DATA/$shape.ply" --out "$control" > "$control.log" 2>&1
    fi
    echo "== $shape, frozen for $FREEZE epochs -> $frozen"
    "${fit[@]}" --path "$DATA/$shape.ply" --out "$frozen" \
        --freeze-f-epochs "$FREEZE" --save-states "$FREEZE" > "$frozen.log" 2>&1

    "$PYTHON" main.py clusters "$control/${shape}_triplane.npz" --k 10
    "$PYTHON" main.py clusters "$frozen/${shape}_triplane.npz" --k 10
    "$PYTHON" main.py clusters "$frozen/${shape}_triplane.npz" --k 10 \
        --state "$frozen/states/${shape}_state_e$(printf '%04d' "$FREEZE").pt" \
        --tag "e$(printf '%04d' "$FREEZE")"
done
