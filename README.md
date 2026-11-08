# Neural Polycube Generation from Point Cloud via Cross-section Loops

Cut a point cloud with x, y and z planes, fit every cross-section loop with one
shared neural network, and link neighbouring loops into tubes. The tubes and
loops are the input to a polycube (a coarse shape made of axis-aligned boxes);
**the polycube step itself is not implemented yet**.

A polycube cut by an axis-aligned plane gives an axis-aligned
polygon, so the cross-sections of a shape on the three plane already
describe its outline. The work so far makes those loops clean and
consistent: each loop runs the same way round as its neighbours (orientation)
and starts at the exact same starting line (where z = 0).

## Layout

```
main.py                 the single entry point: python main.py <command>
environment.yaml        the conda environment (see Setup)
model/                  the neural network
  network.py            a shared MLP and TriplaneField
  losses.py             chamfer and normal-orientation terms; bend and winding readings
  cube.py               the unit cube and its square cross-sections
  warmup.py             cube warm-up training, warm-up checkpoint save/load
  trainer.py            Trainer (fit, seam phases, distillation), checkpoints
  fit.py                the extract / fit run and its command line
utils/                  geometry, I/O and tools around the network
  ply_io.py             read / write point-cloud PLYs
  cubic_stylize.py      cubic stylization of a point cloud (prepares the inputs)
  slicing.py            planes, slabs, Poisson fields, loop centres
  visualization.py      slice, field and fit figures; per-plane files
  loop_extraction.py    LoopExtractor: runs the extraction stages
  warmup_cube.py        generates the cube warm-up checkpoint
  propagation.py        stable loops -> tracks -> aligned seams -> tubes
  align_loops.py        circular-shift seam solve used by propagation
  seam_check.py         seam distance D of a checkpoint
  shift_probe.py        decode loops at shifted triplane queries
  seam_view.py          mark every loop's z = 0 seam
  sweep_movie.py        plane-by-plane movie of the fitted loops
  code_clusters.py      cluster the loops' triplane codes, show them in 3D
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

Run every command from the repository root, in the activated environment.

## Data

The inputs are point clouds stored as PLY **with normals** (`nx ny nz`),
centred and scaled so the shape fits in [-1, 1] (longest side about 2). Several
thresholds are absolute (0.15 for a bad loop, 0.025 RMS for a stable loop), so
an unnormalised cloud will fail them.

The scripts expect this layout (the dataset itself is not part of the
repository; place it next to `main.py`):

```
Datasets/Genus0/<shape>.ply                                                     # original shapes
Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid/<shape>_cubic_lam3.ply  # pipeline inputs
```

The `*_cubic_lam3.ply` inputs are the original shapes after cubic stylization
(`main.py stylize`, below).

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
   python main.py stylize --points Datasets/Genus0/my_shape.ply --out my_inputs --lam 3
   DATA=my_inputs scripts/run_shape.sh my_shape_cubic_lam3
   ```

## Quick start

```bash
# 1. the cube warm-up, once (a few seconds) -> warmup/cube_warmup_axis.pt
python main.py warmup

# 2. one shape end to end: fit, seam check, clusters, seam view, movie, tubes
scripts/run_shape.sh bunny_points_cubic_lam3
SAMPLES=9 EPOCHS=200 FREEZE=20 scripts/run_shape.sh horse_points_cubic_lam3    # quick pass
```

`python main.py --help` lists every command; `python main.py <command> --help`
lists its options.

## The pipeline

### 0. Input preparation — `main.py stylize`

Mesh-free cubic stylization (Liu & Jacobson 2019) of a point cloud with normals.
Each point's k-nearest-neighbour patch stands in for the paper's mesh cell, and
the energy is the paper's two terms: an as-rigid-as-possible term that keeps the
shape and an L1 term on the rotated normals that pulls every normal towards
+-x, +-y or +-z (`--lam`, 3 for the provided inputs). It alternates a per-point
rotation step with one sparse linear solve.

```bash
python main.py stylize --points Datasets/Genus0/rocker-arm.ply \
    --out log/cubic/rocker-arm --lam 3 --iters 60
```

Writes `<out>/<stem>_cubic_lam3.ply` (double precision, normals re-estimated
from the stylized geometry), `metrics.json` (energy history, share of
axis-aligned normals, displacement) and `before_after.png`. A 200k-point cloud
takes about 15 s on a GPU. `scripts/stylize_dataset.sh` does a whole folder.

### 1. Cube warm-up — `main.py warmup`

The network is first trained on a unit cube: 12 square
cross-sections (4 planes per axis, none through the centre), with three terms:
fit to the square, direction (every loop counter-clockwise about its axis) and
seam (`z = 0` at the same corner on neighbouring loops). The checkpoint holds
only the network and the triplane grid, so it is trained once per network size
and reused by every shape. Its `--neurons`, `--layers`, `--resolution`,
`--channels` and `--no-axis-condition` must match the fit's, or the fit refuses
it.

### 2. Extraction — `main.py extract`

For each axis, `--samples` planes are placed inside the bounding box (inset by
`--margin`). Each plane keeps the points within a thin slab (`--band`), minus
points whose surface lies flat in the plane (`--max-align`). The slice is
projected to 2D, a Poisson field is solved from its points and normals, and each
enclosed region becomes one loop, so a plane can hold several loops. Writes
`<stem>_slices.ply` / `.png` (and `<stem>_field.png` with `--field`).

### 3. Fit — `main.py fit`

Extraction, then one MLP fits every loop. Its input is
`[sin 2πz, cos 2πz, 48-D triplane feature, plane axis]`, and it outputs the
loop's point at parameter `z` in the plane's frame. The triplane feature is
sampled at the loop's centre from three 32×32 grids, so nearby loops start from
similar codes. The loss:

| term | flag | role |
|---|---|---|
| chamfer | always | the curve lies on the loop's points |
| normal orientation | `--normal W` | the loop runs the way its point normals say; annealed to 0 |
| triplane TV | `--triplane-tv W` | smooth feature grids (default 0.01) |
| feature smoothness | `--code-smooth W` | neighbouring loops get similar codes |
| seam distillation | `--distill W` | hold each neighbour pair's seam distance D under the warm-up teacher's (one-sided) |

There is no bend term: nothing in the loss opposes a sharp corner, so a loop can
follow a cubic slice. The `bend` printed at the end (a circle reads 39.5) is a
reading of the final curves, not a loss.

Other options: `--freeze-f-epochs N` (the frozen start, below),
`--save-states E1,E2,...` (keep the network after those epochs, in
`states/<stem>_state_eNNNN.pt`), `--no-warmup` (blank start), `--seam-align` (a
separate per-loop seam phase solve), `--refine-every` / `--add-planes` /
`--samples-end` (add planes during the fit), `--feature embedding` (one learned
row per loop instead of the triplane; the warm-up and distillation need the
triplane).

Outputs in `--out` (default `log/<stem>/`):

| file | content |
|---|---|
| `<stem>_model.pt` | network, triplane grid and replay metadata |
| `<stem>_good_loops.ply`, `<stem>_bad_loops.ply` | fitted loops; bad = any point past 0.15 from its target |
| `<stem>_loop_split.txt` | one `g`/`b` per loop, to interleave the two PLYs |
| `<stem>_fits.png`, `planes/`, `planes.txt`, `bad_loops.txt` | figures and per-plane errors (`--max-err` highlights) |
| `<stem>_epoch_terms.csv`, `<stem>_loop_winding.npz` | per-epoch, per-axis losses and loop windings |
| `<stem>_triplane.npz`, `<stem>_feature_graph.csv`, `<stem>_feature_smoothness.txt` | field samples and the neighbour graph |
| `states/<stem>_state_eNNNN.pt` | the network after the epochs `--save-states` names |

A loop that ends up running against its normals is reversed before it is
written; the console reports how many were.

#### The frozen start (`--freeze-f-epochs 100`, recommended)

The warm-up's loop is a square the size of the cube's cross-section, while a
slice's points usually sit in a smaller, off-centre part of the box. The first
updates therefore drag every loop a long way onto its points, and because the
shared MLP `f` trains from the first step, that drag rewrites `f` and turns whole
loop families round (the "reverse normal" problem). With `--freeze-f-epochs N`,
`f` stays at the warm-up's weights for the first N epochs and only the triplane
grid trains: the field places each loop while `f` keeps every loop closed and
running the same way. After N epochs `f` is released and the ordinary fit
continues. The epoch lines carry `[f frozen]`, the switch prints `f map
released`, and plane-refinement stages share one budget.

### 4. Propagation — `main.py propagate --run <fit run>`

| step | what it does |
|---|---|
| 1 | keep the stable loops (in-plane RMS under `--rms`, default 0.025) |
| 2 | move each stable loop to the adjacent planes and link it where it explains the points, growing tracks |
| 2b | optional (`--step 2b`, after 1): the same, allowing a bounded Fourier deformation |
| 3 | align the seams along each track; split tracks at pairs that disagree |
| 4 | loft every aligned chain of two or more loops into a tube (`step4/tubes.ply`) |

`--step all` runs 1–4; `--good-only` drops the bad loops first. Outputs go to
`--out`, by default `<run>_propagation/`, one `stepN/` folder each plus a
`summary.md`.

### 5. Inspecting a run

```bash
python main.py seam-check log/<run>/<stem>_model.pt           # D at z = 0 per neighbour pair
python main.py shift-probe --run log/<run>                     # -> log/<run>/loop_shift/
python main.py seam-view log/<run>                             # -> <stem>_good_loops_z0.ply / .png
python main.py seam-view 'log/*/fit' --sheet log/seams.png     # also one grid over many runs
python main.py sweep-movie --run log/<run> [--preview]         # -> loops_axes_sweep.mp4
python main.py clusters log/<run>/<stem>_triplane.npz --k 10   # -> <stem>_code_clusters.png/.csv/.ply
python main.py clusters log/<run>/<stem>_triplane.npz --k 10 \
    --state log/<run>/states/<stem>_state_e0100.pt --tag e0100 # codes at epoch 100
```

`clusters` groups the loops by the cosine similarity of their 48-D triplane
codes (average linkage; k by silhouette unless `--k` is given) and draws them in
world space and in the codes' PCA basis; the PLY puts a sphere in the cluster's
colour at every loop centre. With the frozen start, the epoch-100 grouping is
already close to the run's final one: the freeze places the loops, and the rest
of the run refines their shapes.

## Scripts

| script | what it runs |
|---|---|
| `scripts/warmup.sh` | the cube warm-up (arguments pass through) |
| `scripts/run_shape.sh <shape>` | the whole per-shape workflow: warm-up if missing, fit with `--distill 10 --freeze-f-epochs 100`, seam check, code clusters (final and end of freeze), seam view, sweep movie, propagation; all in `log/<shape>/` as `fit/`, `propagation/` and `<shape>.log` |
| `scripts/distill_experiment.sh` | six shapes, with and without distillation, 29 planes, 1000 epochs, then the seam check of both arms |
| `scripts/freeze_experiment.sh` | eight shapes, the distillation fit with and without the 100-epoch frozen start, then the code clusters at k = 10 (final, and epoch 100 for the frozen arm) |
| `scripts/stylize_dataset.sh` | cubic stylization of every cloud in a folder (default `Datasets/Genus0`, `LAM=3`, `ITERS=60`) into `log/cubic_lam3/<shape>/` |

Settings are passed as environment variables. The fitting scripts read
`SAMPLES`, `EPOCHS`, `DATA`, `WARMUP` and `PYTHON` (`run_shape.sh` and
`freeze_experiment.sh` also `FREEZE`; `run_shape.sh` also `K` and `OUT`);
`stylize_dataset.sh` reads `IN`, `OUT`, `SHAPES`, `LAM`, `ITERS` and `PYTHON`.
The experiment scripts' shape lists refer to the provided dataset; set `SHAPES`
to run them on others.

## Notes

- **Open problems:** a section that splits into several loops (fingers, legs)
  cannot be tracked as one closed loop; the polycube stage is still to be built.
