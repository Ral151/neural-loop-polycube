#!/usr/bin/env bash
# Dual-loop polycube of one shape, in order:
#   1. cube warm-up for this network size and harmonics (only if the checkpoint is missing)
#   2. fit: extract the loops and fit them (skipped when $OUT/fit already holds a fit)
#   3. seam check of the fitted network
#   4. dual check: do all the loops form a polycube loop structure?
#   5. dual cause: why the unmatched crossings are unmatched
#   6. dual polycube: select a valid subset of the loops and build the polycube
#
#   scripts/dual_polycube.sh armadillo_normal_cubic_lam3
#   FREQS=4 scripts/dual_polycube.sh armadillo_normal_cubic_lam3                 # fewer harmonics
#   LAYERS=4 NEURONS=256 FREQS=1 scripts/dual_polycube.sh bunny_points_cubic_lam3
#   SAMPLES=9 EPOCHS=200 FREEZE=20 STARTS=50 scripts/dual_polycube.sh horse_points_cubic_lam3   # quick pass
#   OUT=log/<an earlier run> scripts/dual_polycube.sh <its shape>                # steps 3-6 on an existing fit
#
# The shape is a file name (without .ply) in $DATA. Everything goes into $OUT
# (default log/<shape>_dual_<layers>x<neurons>_f<freqs>_<date>_<time>/):
#   fit/                the fitted run, as scripts/run_shape.sh writes it
#   polycube/           dual_primal.png, polycube.obj, layout_on_shape.obj, structure.npz, kept_loops.txt
#   <shape>_dual.log    the whole console output
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}            # the active environment, or set PYTHON=...
DATA=${DATA:-Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid}
SAMPLES=${SAMPLES:-29}
EPOCHS=${EPOCHS:-1000}
DISTILL=${DISTILL:-10}
FREEZE=${FREEZE:-100}          # epochs the shared MLP stays at the warm-up's map
LAYERS=${LAYERS:-2}
NEURONS=${NEURONS:-64}
FREQS=${FREQS:-8}              # harmonics of the loop parameter; 8 gave the largest polycube on armadillo
GATE=${GATE:-0.04}             # two crossings are one when they agree within this distance
STARTS=${STARTS:-400}          # starting cubes the selection is grown from
## the warm-up has to match the fit's network, so its name carries the three settings
WARMUP=${WARMUP:-warmup/cube_warmup_unit_centroid_${LAYERS}x${NEURONS}_f${FREQS}.pt}

SHAPE=${1:?usage: scripts/dual_polycube.sh <shape name in $DATA>}
OUT=${OUT:-log/${SHAPE}_dual_${LAYERS}x${NEURONS}_f${FREQS}_$(date +%Y%m%d_%H%M%S)}
RUN=$OUT/fit
mkdir -p "$OUT"
exec > >(tee "$OUT/${SHAPE}_dual.log") 2>&1     # everything below is also kept in the log

if [ -f "$RUN/${SHAPE}_triplane.npz" ]; then
    echo "== 1-2. $RUN holds a fit, reused (LAYERS, NEURONS, FREQS, SAMPLES, EPOCHS are not read)"
else
    echo "== 1. warm-up (${LAYERS}x${NEURONS}, $FREQS harmonics)"
    if [ -f "$WARMUP" ]; then
        echo "  $WARMUP exists, reused"
    else
        "$PYTHON" model/warmup_cube.py --out "$WARMUP" --feature-pool centroid --cubes 0 \
            --axis-condition --layers "$LAYERS" --neurons "$NEURONS" --freqs "$FREQS"
    fi

    echo "== 2. fit"
    # --save-plane-files: the dual cause step (5) reads the per-plane slice files
    "$PYTHON" plc_main.py --train --path "$DATA/$SHAPE.ply" --out "$RUN" \
        --samples "$SAMPLES" --epochs "$EPOCHS" --normal 0.01 --triplane-tv 0.01 \
        --warmup "$WARMUP" --distill "$DISTILL" --freeze-f-epochs "$FREEZE" \
        --save-plane-files --snapshot-every 0 \
        --feature-pool centroid --axis-condition \
        --layers "$LAYERS" --neurons "$NEURONS" --freqs "$FREQS"
fi

echo "== 3. seam check"
"$PYTHON" utils/seam_check.py "$RUN/${SHAPE}_model.pt"

echo "== 4. dual check: all loops as one loop structure"
"$PYTHON" utils/dual_check.py "$RUN" --gate "$GATE"

echo "== 5. dual cause: why crossings are unmatched"
"$PYTHON" utils/dual_cause.py "$RUN" "$DATA/$SHAPE.ply" --gate "$GATE"

echo "== 6. dual polycube"
"$PYTHON" utils/dual_polycube.py "$RUN" --out "$OUT/polycube" --gate "$GATE" --starts "$STARTS"
