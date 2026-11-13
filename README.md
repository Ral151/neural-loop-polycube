# Neural Polycube Generation from Point Cloud via Cross-section Loops

Cut a point cloud with x, y and z planes, fit every cross-section loop with one
shared neural network, and link neighbouring loops into tubes. The tubes and
loops are the input to a polycube (a coarse shape made of axis-aligned boxes);
**the polycube step itself is not implemented yet**.

A polycube cut by an axis-aligned plane gives an axis-aligned
polygon, so the cross-sections of a shape on the three planes already
describe its outline. The work so far makes those loops clean and
consistent: each loop runs the same way round as its neighbours (orientation)
and starts at the same place as its neighbours (the seam, where z = 0).

## Layout

```
plc_main.py             the entry point: extract the loops and, with --train, fit them
environment.yaml        the conda environment (see Setup)
model/                  the pipeline
  loop_extraction.py    LoopExtractor: planes, slabs, Poisson fields, loops; fit figures
  warmup_cube.py        trains the cube warm-up checkpoint every fit starts from
  mini_nn/              the neural fit
    network.py          the shared MLP, conditioned on a triplane feature (or an embedding row)
    triplane.py         the three 2D feature grids and their pooling
    trainer.py          Trainer: the losses, distillation, seam phases; checkpoints
    warmup.py           cube warm-up training, warm-up checkpoint save/load
    chamfer.py, curves.py, cubes.py, square.py, data.py
utils/                  tools run on a finished fit (python utils/<script>.py --help)
  loop_propagation.py   stable loops -> tracks -> aligned seams -> tubes
  align_loops.py        circular-shift seam solve used by propagation
  seam_check.py         seam distance D of a checkpoint
  code_clusters.py      cluster the loops' triplane codes, show them in 3D
  seam_z0_view.py       mark every loop's z = 0 seam
  loop_sweep_movie.py   plane-by-plane movie of the fitted loops
  loop_shift_probe.py   decode loops at shifted triplane queries
  dual_loops.py         the loops as a dual-loop structure: crossings, regions, selection, polycube
  dual_check.py         do all the loops of a run form a polycube loop structure?
  dual_cause.py         why a loop crossing has no partner on the other loop
  dual_polycube.py      select a valid subset of the loops and build its polycube
  cubic_stylize_points.py  cubic stylization of a point cloud (prepares the inputs)
  ply_io.py             read / write point-cloud PLYs
  ...                   further plots and probes used in the experiments (see their docstrings)
scripts/                workflow and experiment launchers (.sh)
Datasets/               point clouds (not in git, see Data)
```

## Setup

All dependencies are listed in `environment.yaml`. Create the conda
environment once, from the repository root:

```bash
conda env create -f environment.yaml
```

activate it in every new terminal before using the repository:

```bash
conda activate neural-loop-polycube
```

A CUDA GPU is recommended; everything also runs on the CPU, much more slowly.
PyTorch is installed for CUDA 12.8. For another CUDA version, or a CPU-only
machine, change the `--extra-index-url` line in `environment.yaml` to the one
https://pytorch.org gives for your system before creating the environment.

After pulling a change to `environment.yaml`, update the environment with
`conda env update -f environment.yaml --prune`; to remove it,
`conda env remove -n neural-loop-polycube`.

Run every command from the repository root, in the activated environment. The
scripts in `scripts/` use `python` from the active environment; set
`PYTHON=/path/to/python` to use another one.

## Data

The inputs are point clouds stored as PLY **with normals** (`nx ny nz`),
centred and scaled so the shape fits in [-1, 1] (longest side about 2). Several
thresholds are absolute (0.15 for a bad loop, 0.025 RMS for a stable loop), so
an unnormalised cloud will fail them.

The scripts expect this layout (the dataset itself is not part of the
repository; place it next to `plc_main.py`):

```
Datasets/Genus0/<shape>.ply                                                        # original shapes
Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid/<shape>_cubic_lam3.ply  # pipeline inputs
```

The `*_cubic_lam3.ply` inputs are the original shapes after cubic stylization
(step 0 below).

### Using your own point cloud

1. Make sure it has normals, then centre and scale it:

   ```python
   import numpy as np
   from utils.ply_io import load_points, write_cloud

   points, normals = load_points('my_shape.ply')          # normals must not be None
   centre = (points.min(0) + points.max(0)) / 2
   scale = np.ptp(points, axis=0).max() / 2
   write_cloud('Datasets/Genus0/my_shape.ply', (points - centre) / scale, normals)
   ```

2. Stylize it, and run the workflow on the folder holding the result:

   ```bash
   python utils/cubic_stylize_points.py --points Datasets/Genus0/my_shape.ply --out my_inputs --lam 3
   DATA=my_inputs scripts/run_shape.sh my_shape_cubic_lam3
   ```

## Quick start

```bash
# one shape end to end: warm-up (if missing), fit, seam check, clusters,
# seam view, movie, tubes -> log/<shape>_<date>_<time>/
scripts/run_shape.sh bunny_points_cubic_lam3
SAMPLES=9 EPOCHS=200 FREEZE=20 scripts/run_shape.sh horse_points_cubic_lam3    # quick pass
```

`scripts/run_shape.sh` uses the recommended settings: a warm-up on one unit cube,
centre features (`--feature-pool centroid`) and the plane axis as an input
(`--axis-condition`). `plc_main.py` run by hand has different defaults (see
[Defaults](#defaults-plc_mainpy-vs-run_shapesh)), so copy the script's flags
when running it directly. `python plc_main.py --help` lists every option.

## The pipeline

### 0. Input preparation — `utils/cubic_stylize_points.py`

Mesh-free cubic stylization (Liu & Jacobson 2019) of a point cloud with normals.
Each point's k-nearest-neighbour patch stands in for the paper's mesh cell, and
the energy is the paper's two terms: an as-rigid-as-possible term that keeps the
shape and an L1 term on the rotated normals that pulls every normal towards
+-x, +-y or +-z (`--lam`, 3 for the provided inputs; the script's default is 6).
It alternates a per-point rotation step with one sparse linear solve.

```bash
python utils/cubic_stylize_points.py --points Datasets/Genus0/rocker-arm.ply \
    --out log/cubic/rocker-arm --lam 3 --iters 60
```

Writes `<out>/<stem>_cubic_lam3.ply` (normals re-estimated from the stylized
geometry) and `metrics.json` (energy history, share of axis-aligned normals,
displacement). A 200k-point cloud takes about 15 s on a GPU.
`scripts/stylize_dataset.sh` does a whole folder. (The `before_after.png`
preview needs a renderer module that is not in this repository, so it is
skipped with a warning.)

### 1. Cube warm-up — `model/warmup_cube.py`

The network is first trained on cubes: their cross-sections are exact squares
(4 planes per axis, none through the centre), with three terms: fit to the
square, direction (every loop counter-clockwise about its axis) and seam
(`z = 0` at the same corner on neighbouring loops). The checkpoint holds only
the network and the triplane grid, so it is trained once per network setup and
reused by every shape.

```bash
python model/warmup_cube.py --cubes 0 --feature-pool centroid --axis-condition \
    --out warmup/cube_warmup_unit_centroid.pt      # what run_shape.sh trains
```

`--cubes 0` trains on one unit cube; the default `--cubes 8` draws 8 cubes of
random size and place every step, which fits its squares much less well (0.25
against 0.005 after 300 steps, 0.085 with `--axis-condition`). `--neurons`,
`--layers`, `--freqs`, `--resolution`, `--channels`, `--feature-pool` and
`--axis-condition` must match the fit's, or the fit refuses the checkpoint.

### 2. Extraction — `plc_main.py`

For each axis, `--samples` planes are placed inside the bounding box (inset by
`--margin`). Each plane keeps the points within a thin slab (`--band`), minus
points whose surface lies flat in the plane (`--max-align`). The slice is
projected to 2D, a Poisson field is solved from its points and normals, and each
enclosed region becomes one loop, so a plane can hold several loops. Without
`--train` this is all `plc_main.py` does; it writes `<stem>_slices.ply` / `.png`
(and the field figures with `--field`).

### 3. Fit — `plc_main.py --train`

Extraction, then one MLP fits every loop. Its input is
`[sin 2πz, cos 2πz, 48-D triplane feature]`, plus the loop's plane normal with
`--axis-condition`, and it outputs the loop's point at parameter `z` in the
plane's frame. `--freqs F` adds the higher harmonics `sin 2πkz, cos 2πkz` for
`k = 2..F` to the input (default 1, the single pair), so the network can draw
sharp corners. The feature is read from three 32×32 grids with 16 channels each:
at the loop's centre (`--feature-pool centroid`) or averaged over the grid cells
the loop's points occupy (`occupied_cells`). Either way, nearby loops start
from similar codes.

The loss:

| term | flag | role |
|---|---|---|
| chamfer | always | the curve lies on the loop's points; a point is only paired with a target whose normal is within `--match-angle` (150°) |
| normal orientation | `--normal W` | the loop runs the way its point normals say; held at W for the whole run |
| triplane TV | `--triplane-tv W` | smooth feature grids (default 0.01) |
| seam distillation | `--distill W` | hold each neighbour pair's seam distance D under the warm-up teacher's (one-sided) |
| cycle consistency | `--cycle W` | train the inverse map from a curve point back to its z (off by default) |
| repulsion | `--repel W` | keep far-apart parameters from landing on the same point (off by default) |
| seam phase | `--seam-align W` | a separate per-loop seam phase solve (off by default) |

There is no bend term: nothing in the loss opposes a sharp corner, so a loop can
follow a cubic slice. The `bend` printed at the end (a circle reads 39.5) is a
reading of the final curves, not a loss.

A loop is written as **bad** when any fitted point is more than 0.15 from its
target, or its mean error is above `--bad-loop-err` (default 0.015 of the
longest extent); `--max-align-err` adds a third, normal-agreement test (off by
default). `utils/reclassify_run.py` re-applies other limits to a finished run.

Other options: `--freeze-f-epochs N` (the frozen start, below), `--no-warmup`
(blank start), `--refine-every` / `--add-planes` / `--samples-end` (add planes
during the fit), `--feature embedding` (one learned row per loop instead of the
triplane; the warm-up and distillation need the triplane),
`--snapshot-every N` (a fit figure every N epochs).

Outputs in `--out` (default `log/<stem>/`):

| file | content |
|---|---|
| `<stem>_model.pt` | network, triplane grid and replay metadata |
| `<stem>_good_loops.ply`, `<stem>_bad_loops.ply` | fitted loops, split by the bad-loop rules |
| `<stem>_loop_split.txt` | one `g`/`b` per loop, to interleave the two PLYs |
| `<stem>_loop_errors.csv` | per-loop mean / max / alignment error and which rule flagged it |
| `<stem>_fits.png`, `planes.txt`, `bad_loops.txt` | fit figure and per-plane errors (`--max-err` highlights) |
| `planes/` | per-plane slice / loop / fit files, only with `--save-plane-files` (**propagation needs them**) |
| `<stem>_epoch_terms.csv`, `<stem>_loop_winding.npz` | per-epoch, per-axis losses and loop windings |
| `<stem>_triplane.npz`, `<stem>_feature_graph.csv`, `<stem>_feature_smoothness.txt` | field samples and the neighbour graph |

#### The frozen start (`--freeze-f-epochs 100`, recommended)

The warm-up's loop is a square the size of the cube's cross-section, while a
slice's points usually sit in a smaller, off-centre part of the box. The first
updates therefore drag every loop a long way onto its points, and because the
shared MLP `f` trains from the first step, that drag rewrites `f` and turns whole
loop families round (the "reverse normal" problem). With `--freeze-f-epochs N`,
`f` stays at the warm-up's weights for the first N epochs and only the triplane
grid trains: the field places each loop while `f` keeps every loop closed and
running the same way. After N epochs `f` is released and the ordinary fit
continues. The epoch lines carry `[f frozen]`, and the switch prints `f map
released`.

### 4. Propagation — `utils/loop_propagation.py --run <fit run>`

| step | what it does |
|---|---|
| 1 | keep the stable loops (in-plane RMS under `--rms`, default 0.025) |
| 2 | move each stable loop to the adjacent planes and link it where it explains the points, growing tracks |
| 2b | optional (`--step 2b`, after 1): the same, allowing a bounded Fourier deformation |
| 3 | align the seams along each track; split tracks at pairs that disagree |
| 4 | loft every aligned chain of two or more loops into a tube (`step4/tubes.ply`) |

`--step all` runs 1–4; `--good-only` drops the bad loops first. Outputs go to
`--out` (default `log/loop_propagation/`), one `stepN/` folder each plus a
`summary.md`. The fit must have been run with `--save-plane-files`: every step
measures the loops against their own labelled slice points, which only the
per-plane files hold.

### 5. Inspecting a run

```bash
python utils/seam_check.py log/<run>/fit/<stem>_model.pt             # D at z = 0 per neighbour pair
python utils/code_clusters.py log/<run>/fit/<stem>_triplane.npz --k 10  # -> <stem>_code_clusters.png/.csv/.ply
python utils/seam_z0_view.py log/<run>/fit                          # -> <stem>_good_loops_z0.ply / .png
python utils/seam_z0_view.py 'log/*/fit' --sheet log/seams.png      # also one grid over many runs
python utils/loop_sweep_movie.py --run log/<run>/fit [--preview]    # -> loops_axes_sweep.mp4
python utils/loop_shift_probe.py --run log/<run>/fit                # decode at shifted queries
```

`seam_check.py` decodes every loop from its centre feature, so for a run fitted
with `--feature-pool occupied_cells` its D is not that of the exported curves
(it prints a warning); measure those runs on their exported loops instead.

`code_clusters.py` groups the loops by the cosine similarity of their 48-D
triplane codes (average linkage; k by silhouette unless `--k` is given) and
draws them in world space and in the codes' PCA basis; the PLY puts a sphere in
the cluster's colour at every loop centre.

### 6. Polycube from the loops (experimental) — `utils/dual_polycube.py`

The x-, y- and z-loops are read as the X-, Y- and Z-loops of a polycube loop
structure (Snoep, Speckmann, Verbeek, *Robust Construction of Polycube
Segmentations via Dual Loops*, arXiv 2402.00652). No mesh is needed: two loops
of different axes can only cross on the line their planes share, and the
crossings, with the surface normal, give the regions between the loops.

```bash
scripts/dual_polycube.sh armadillo_normal_cubic_lam3     # fit with 8 harmonics, then steps below
python utils/dual_check.py log/<run>/fit                 # how far all the loops are from a valid structure
python utils/dual_cause.py log/<run>/fit                 # why crossings are unmatched (needs planes/)
python utils/dual_polycube.py log/<run>/fit              # -> log/<run>/polycube/
```

`dual_check.py` reports the unmatched crossings, the Euler characteristic (2
for a valid genus-0 structure) and the regions that break the paper's rules.
The full set of loops is not valid, so `dual_polycube.py` selects a jointly
valid subset: it starts from a cube (one loop per axis, each pair crossing
twice), adds loops while the set stays valid, and repeats this from `--starts`
starting cubes (default 400; `--starts 1` is a quick single pass). The best
set becomes a polycube: a region is a corner, a crossing a quad face, and the
coordinates come from the plane offsets. It writes `dual_primal.png` (kept
loops, quad layout on the shape, polycube), `polycube.obj`,
`layout_on_shape.obj`, `structure.npz` and `kept_loops.txt`. All three scripts
only read the run folder.

How much of the shape the polycube covers depends on how well the fitted
loops agree where they cross; `--freqs 8` gave the largest one on armadillo.
The method, the measurements and the open problems are in `Documentation.md`.

## Defaults: `plc_main.py` vs `run_shape.sh`

`plc_main.py`'s own defaults are not the recommended setup:

| setting | `plc_main.py` default | `scripts/run_shape.sh` |
|---|---|---|
| planes per axis / epochs | 5 / 300 | 29 / 1000 |
| warm-up | `warmup/cube_warmup_random_ranges.pt` (random cubes) | `warmup/cube_warmup_unit_centroid.pt` (one unit cube) |
| feature pooling | `occupied_cells` | `centroid` |
| plane axis as input | off | `--axis-condition` |
| normal / distillation / frozen start | 0 / 0 / 0 | 0.01 / 10 / 100 epochs |
| per-plane files | off | `--save-plane-files` |

On armadillo (29 planes, 1000 epochs) the `run_shape.sh` setup gives 1 bad loop
(by the 0.15 point rule), seam D median 0.0007 and 24 tubes from 109 loops; the
`plc_main.py` defaults, with the same planes, epochs and loss weights, give 12
bad loops and seams with a much worse tail (D p90 0.13).

## Scripts

| script | what it runs |
|---|---|
| `scripts/warmup.sh` | `model/warmup_cube.py` (arguments pass through) |
| `scripts/run_shape.sh <shape>` | the whole per-shape workflow: warm-up if missing, fit, seam check, code clusters, seam view, sweep movie, propagation; all in `log/<shape>_<date>_<time>/` as `fit/`, `propagation/` and `<shape>.log` |
| `scripts/dual_polycube.sh <shape>` | the dual-loop polycube of one shape: warm-up if missing, fit, seam check, `dual_check`, `dual_cause`, `dual_polycube`; all in `log/<shape>_dual_<layers>x<neurons>_f<freqs>_<date>_<time>/` as `fit/`, `polycube/` and `<shape>_dual.log`. With `OUT` set to an earlier run folder it skips the fit and reads that run's `fit/` |
| `scripts/distill_experiment.sh` | six shapes, with and without distillation, 29 planes, 1000 epochs, then the seam check of both arms |
| `scripts/freeze_experiment.sh` | eight shapes, the distillation fit with and without the 100-epoch frozen start, then the code clusters at k = 10 |
| `scripts/stylize_dataset.sh` | cubic stylization of every cloud in a folder (default `Datasets/Genus0`, `LAM=3`, `ITERS=60`) into `log/cubic_lam3/<shape>/` |

Settings are passed as environment variables. `run_shape.sh` reads `SAMPLES`,
`EPOCHS`, `DISTILL`, `FREEZE`, `K`, `DATA`, `WARMUP`, `OUT` and `PYTHON`;
`dual_polycube.sh` reads the same except `K`, plus `LAYERS`, `NEURONS`, `FREQS`
(default 8), `GATE` and `STARTS`; the
experiment scripts read `SAMPLES`, `EPOCHS`, `DATA`, `WARMUP`, `SHAPES` and
`PYTHON` (and `FREEZE`); `stylize_dataset.sh` reads `IN`, `OUT`, `SHAPES`, `LAM`,
`ITERS` and `PYTHON`. The two experiment scripts keep `plc_main.py`'s own
defaults for the warm-up, pooling and axis input.

## Notes

- **Open problems:** a section that splits into several loops (fingers, legs)
  cannot be tracked as one closed loop; the polycube stage is still to be built.
- The earlier `main.py` pipeline (`old_model/`) was removed; it is in the git
  history before the `new-pipeline` branch's cleanup commit.
