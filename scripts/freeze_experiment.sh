#!/usr/bin/env bash
# The frozen-start comparison: on eight shapes, the distillation fit with and
# without --freeze-f-epochs 100 (29 planes per axis, 1000 epochs, --normal 0.01,
# --distill 10), then the triplane codes of both arms clustered at k = 10 at the
# end of the run. (plc_main.py keeps no per-epoch states, so the epoch-100
# clustering the old main.py version also made is not available.)
#
#   scripts/freeze_experiment.sh                      # all eight shapes
#   SHAPES="horse_points_cubic_lam3" scripts/freeze_experiment.sh
#
# Outputs: log/axis_seam10_<shape>/ (control; shared with distill_experiment.sh
# and reused when already there), log/axis_frz100_<shape>/ (frozen), each with
# its console log beside it, and <stem>_code_clusters.png/.csv/.ply inside each run.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...
DATA=${DATA:-Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid}
WARMUP=${WARMUP:-warmup/cube_warmup_random_ranges.pt}
SAMPLES=${SAMPLES:-29}
EPOCHS=${EPOCHS:-1000}
FREEZE=${FREEZE:-100}
SHAPES=${SHAPES:-"bunny_points_cubic_lam3 armadillo_normal_cubic_lam3 horse_points_cubic_lam3
                  lucy_points_cubic_lam3 laurent_hand_normal_cubic_lam3 rocker-arm_points_cubic_lam3
                  neptune_normal_cubic_lam3 dancing_children100K_cubic_lam3"}

[ -f "$WARMUP" ] || "$PYTHON" model/warmup_cube.py --out "$WARMUP"
mkdir -p log
fit=("$PYTHON" plc_main.py --train --samples "$SAMPLES" --epochs "$EPOCHS" --normal 0.01
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
        --freeze-f-epochs "$FREEZE" > "$frozen.log" 2>&1

    "$PYTHON" utils/code_clusters.py "$control/${shape}_triplane.npz" --k 10
    "$PYTHON" utils/code_clusters.py "$frozen/${shape}_triplane.npz" --k 10
done
