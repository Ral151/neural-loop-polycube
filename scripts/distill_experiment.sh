#!/usr/bin/env bash
# The seam-distillation comparison: on six shapes, the same fit with and
# without --distill (29 planes per axis, 1000 epochs, triplane + TV 0.01,
# --normal 0.01). Matched arms share the shape, planes and pair graph.
#
#   scripts/distill_experiment.sh            # all six shapes
#   SHAPES="bunny_points_cubic_lam3" scripts/distill_experiment.sh
#
# Outputs: log/axis_tri_tv01_<shape>/ (baseline), log/axis_seam10_<shape>/
# (distillation), each with its console log beside it, and the seam distance D
# of both arms printed at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...
DATA=${DATA:-Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid}
WARMUP=${WARMUP:-warmup/cube_warmup_random_ranges.pt}
SAMPLES=${SAMPLES:-29}
EPOCHS=${EPOCHS:-1000}
SHAPES=${SHAPES:-"bunny_points_cubic_lam3 armadillo_normal_cubic_lam3 horse_points_cubic_lam3
                  lucy_points_cubic_lam3 laurent_hand_normal_cubic_lam3 rocker-arm_points_cubic_lam3"}

[ -f "$WARMUP" ] || "$PYTHON" model/warmup_cube.py --out "$WARMUP"
mkdir -p log

for shape in $SHAPES; do
    for arm in baseline distill; do
        if [ "$arm" = baseline ]; then run=log/axis_tri_tv01_$shape; extra=(); \
        else run=log/axis_seam10_$shape; extra=(--distill 10); fi
        echo "== $shape, $arm -> $run"
        "$PYTHON" plc_main.py --train --path "$DATA/$shape.ply" --out "$run" \
            --samples "$SAMPLES" --epochs "$EPOCHS" --normal 0.01 --triplane-tv 0.01 \
            --warmup "$WARMUP" --snapshot-every 0 "${extra[@]}" > "$run.log" 2>&1
    done
done

for shape in $SHAPES; do
    for run in "log/axis_tri_tv01_$shape" "log/axis_seam10_$shape"; do
        echo "== $run"
        "$PYTHON" utils/seam_check.py "$run/${shape}_model.pt"
    done
done
