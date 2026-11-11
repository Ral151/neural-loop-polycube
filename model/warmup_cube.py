"""Train the warm-up and save the checkpoint a shape's fit starts from.

The warm-up sees cubes and nothing else. Its loops are their cross-sections --
exact squares on the planes they are cut by -- and the network is trained so that
every loop runs the same way round, puts z = 0 where its neighbouring loops put
theirs, and lies on its cube. No cloud, no bounding box and no shape is read: by
default it draws a fresh batch of smaller, off-centre cubes every step
(`mini_nn.random_cube_loops`); `--cubes 0` trains on the one unit cube instead.

The checkpoint holds the shared MLP and the triplane grid. `plc_main` loads it as
the initialisation of every run (`--warmup`, on by default), so this is run once,
not per shape:

    python warmup_cube.py
    # -> warmup/cube_warmup_random_ranges.pt, the path plc_main loads

Warm with the network and grid the fit uses: `--neurons`, `--layers`,
`--resolution` and `--channels` have to match `plc_main`'s, or the load is
refused rather than applied in part.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

## run as `python model/warmup_cube.py`, the repository root is not on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import mini_nn  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description='Warm a triplane network on random-size, random-place cubes '
                    'and save the checkpoint plc_main loads.')
    parser.add_argument('--steps', type=int, default=300,
                        help='optimisation steps; every cube loop is drawn in every '
                             'step')
    parser.add_argument('--planes', type=int, default=4,
                        help='even planes per axis on the cube, so the seam term '
                             'has 3 * (planes - 1) neighbouring pairs; the exact '
                             'centre is skipped, where the three centre cuts would '
                             'share one query')
    parser.add_argument('--samples', type=int, default=512,
                        help='points round every cube loop, and the parameter grid the '
                             'curve is drawn at')
    parser.add_argument('--cubes', type=int, default=8, metavar='N',
                        help='train on a batch of N cubes of random size and place, '
                             'redrawn every --resample-every steps (default 8); '
                             '0 keeps the one unit cube instead')
    parser.add_argument('--cube-half-min', type=float, default=0.05, metavar='H',
                        help='smallest half-side of a random cube (default 0.05)')
    parser.add_argument('--cube-half-max', type=float, default=0.45, metavar='H',
                        help='largest half-side of a random cube (default 0.45); '
                             'kept below half the grid so each cube leaves a '
                             'border of field around it')
    parser.add_argument('--resample-every', type=int, default=1, metavar='N',
                        help='redraw the random cubes every N steps; ignored '
                             'without --cubes')
    parser.add_argument('--normal', dest='align', type=float, default=1.0,
                        help='weight of the direction term, held against the cube\'s '
                             'outward face normals; 0 trains position and seam only')
    parser.add_argument('--seam', type=float, default=1.0,
                        help='weight of the seam term, the distance D between loops '
                             'adjacent along an axis, differentiated through the '
                             'network; 0 leaves z = 0 free')
    parser.add_argument('--lr', type=float, default=3e-2,
                        help='starting learning rate, cosine-annealed over the run')
    parser.add_argument('--neurons', type=int, default=64,
                        help='width of the shared network; must match the fit')
    parser.add_argument('--layers', type=int, default=2,
                        help='hidden layers; must match the fit')
    parser.add_argument('--resolution', type=int, default=32,
                        help='triplane cells per side; must match the fit')
    parser.add_argument('--channels', type=int, default=16,
                        help='triplane channels per grid; must match the fit')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for the initial weights and the grid')
    parser.add_argument('--feature-pool', choices=('centroid', 'occupied_cells'),
                        default='occupied_cells',
                        help='how a cube loop reads the field: the mean of the grid '
                             'nodes its own points occupy (occupied_cells, default) '
                             'or its centre (centroid). A fit that pools the shape '
                             'loops the same way needs a warm-up trained this way too')
    parser.add_argument('--axis-condition', action='store_true',
                        help='also feed every loop the unit normal of the plane it '
                             'was cut on. A fit with --axis-condition needs a warm-up '
                             'trained with it, and one without needs one without')
    parser.add_argument('--out', default='warmup/cube_warmup_random_ranges.pt',
                        help='checkpoint path (default warmup/'
                             'cube_warmup_random_ranges.pt, the path plc_main '
                             'loads)')
    parser.add_argument('--snapshot-every', type=int, default=0, metavar='N',
                        help='write a fit grid every N steps, so the warm-up can be '
                             'watched arriving rather than only judged at the end; 0 '
                             '(default) writes none')
    parser.add_argument('--snapshot-dir', default=None,
                        help='folder for the step grids; default is the checkpoint '
                             'path without its suffix plus _snapshots')
    return parser


def save_snapshot(path, cube, out, readings, step, steps, planes=None):
    """One panel per cube loop: its target square, the fitted curve, their mean
    distance and the loop's direction cosine.

    `out` and the cube's targets share the plane-local frame the warm-up trains in,
    so a panel is a straight projection onto the loop's own (u, v). The axes are
    fixed at a little beyond the square, so a curve still far off the cube runs out
    of frame instead of shrinking the square to a dot.

    A random-cube batch holds several cubes; the panels show the first one, and
    `planes` says how many cuts per axis it has. A single unit cube needs no
    `planes`: its loops are all there are.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    targets = np.asarray(cube['targets'], dtype=np.float64)
    axes = np.asarray(cube['axes'], dtype=np.float64)
    queries = np.asarray(cube['queries'], dtype=np.float64)
    if planes is None:
        planes = len(targets) // 3
    loops = 3 * planes
    targets, axes, queries, out = (targets[:loops], axes[:loops],
                                   queries[:loops], out[:loops])
    centres = cube.get('centres')
    centre = (np.asarray(centres[0], dtype=np.float64) if centres is not None
              else np.zeros(3))
    halves = cube.get('halves')
    cos = readings['cos']
    if cos is not None:
        cos = np.asarray(cos.detach().cpu() if hasattr(cos, 'detach') else cos)
        cos = cos[:loops]
    colours = ('#1f77b4', '#2ca02c', '#d62728')
    reach = 1.5 * float(np.abs(targets).max())

    fig, axs = plt.subplots(3, planes, figsize=(2.4 * planes, 7.0),
                            dpi=120, squeeze=False, constrained_layout=True)
    for k in range(loops):
        a = k // planes
        axis = axes[k]
        u = mini_nn.in_plane_u(axis)
        v = np.cross(axis, u)
        tgt = np.stack([targets[k] @ u, targets[k] @ v], -1)
        fit = np.stack([out[k] @ u, out[k] @ v], -1)
        ax = axs[a][k % planes]
        ax.plot(np.r_[tgt[:, 0], tgt[0, 0]], np.r_[tgt[:, 1], tgt[0, 1]],
                '--', color='0.55', lw=1.0)
        ax.plot(np.r_[fit[:, 0], fit[0, 0]], np.r_[fit[:, 1], fit[0, 1]],
                '-', color=colours[a], lw=1.4)
        err = float(np.linalg.norm(out[k] - targets[k], axis=1).mean())
        offset = float((queries[k] - centre) @ axis)
        title = f'{"xyz"[a]} = {offset:+.2f} | err {err:.3f}'
        if cos is not None:
            title += f' | cos {cos[k]:+.2f}'
        ## the two things the panels exist to catch: a curve off its square and one
        ## circulating against the outward normals
        bad = err > 0.05 or (cos is not None and cos[k] < 0.0)
        ax.set_title(('! ' if bad else '') + title, fontsize=7,
                     color='#bf1b1b' if bad else 'black')
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)

    direction = ('' if cos is None else
                 f' | direction cos {float(cos.mean()):+.3f}')
    gap = '' if readings['gap'] is None else f' | seam D {readings["gap"]:.4f}'
    where = ('' if halves is None else
             f' | cube half {float(halves[0]):.2f} at '
             f'({centre[0]:+.2f}, {centre[1]:+.2f}, {centre[2]:+.2f})')
    fig.suptitle(f'cube warm-up step {step}/{steps} | fit '
                 f'{readings["fit"]:.4f}{direction}{gap}{where}', fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def main(args=None):
    args = build_parser().parse_args(args)
    if args.steps < 1:
        raise SystemExit(f'--steps must be at least 1, got {args.steps}')
    if args.planes < 2 or args.planes % 2:
        raise SystemExit(f'--planes must be even and at least 2, got {args.planes}')
    if args.samples < 4:
        raise SystemExit(f'--samples must be at least 4, got {args.samples}')
    if args.cubes < 0:
        raise SystemExit(f'--cubes must be 0 or a count, got {args.cubes}')
    if not 0.0 < args.cube_half_min <= args.cube_half_max < mini_nn.UNIT_CUBE_HALF:
        raise SystemExit(f'--cube-half-min/--cube-half-max must satisfy 0 < min <= '
                         f'max < {mini_nn.UNIT_CUBE_HALF}, got '
                         f'{args.cube_half_min} / {args.cube_half_max}')
    if args.resample_every < 1:
        raise SystemExit(f'--resample-every must be at least 1, got '
                         f'{args.resample_every}')
    if args.snapshot_every < 0:
        raise SystemExit(f'--snapshot-every must be 0 or a count of steps, got '
                         f'{args.snapshot_every}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ## the grids live on the unit cube itself: the warm-up's object and the field's
    ## domain are the same object. A fit later lays this cube on its cloud's box.
    frame = dict(centre=np.zeros(3), half=np.full(3, mini_nn.UNIT_CUBE_HALF),
                 resolution=args.resolution, channels=args.channels, tv=0.0)
    model = mini_nn.TriplaneNN(num_neurons=args.neurons,
                               num_layers=args.layers, triplane=frame,
                               axis_condition=args.axis_condition).to(device)
    loops = 3 * args.planes
    layout = (f'{args.cubes} random cubes per step, redrawn every '
              f'{args.resample_every} step(s), half in '
              f'[{args.cube_half_min}, {args.cube_half_max}]'
              if args.cubes else f'{loops} loops ({args.planes} per axis)')
    print(f'cube warm-up: {layout}, {args.samples} samples each, '
          f'{args.layers}x{args.neurons} net, {args.feature_pool} features, '
          f'axis condition {"on" if args.axis_condition else "off"}, '
          f'{args.resolution} x {args.resolution} x 3 grids of {args.channels} '
          f'channels, on {device}')

    snapshot = None
    on_step_every = max(1, args.snapshot_every)
    if args.snapshot_every:
        snapshot_dir = (Path(args.snapshot_dir) if args.snapshot_dir else
                        Path(str(Path(args.out).with_suffix('')) + '_snapshots'))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        name = 'batch' if args.cubes else 'cube'
        print(f'  a fit grid every {args.snapshot_every} steps -> {snapshot_dir}')

        def snapshot(step, out, readings):
            path = snapshot_dir / f'{name}_e{step:04d}.png'
            save_snapshot(path, readings['cube'], out.cpu().numpy(), readings,
                          step, args.steps, planes=args.planes)
            print(f'  snapshot {step}/{args.steps} -> {path}')

    report = mini_nn.warm_on_cube(model, steps=args.steps, lr=args.lr,
                                  align=args.align, seam=args.seam,
                                  planes=args.planes, samples=args.samples,
                                  feature_pool=args.feature_pool, cubes=args.cubes,
                                  cube_half=(args.cube_half_min,
                                             args.cube_half_max),
                                  resample_every=args.resample_every,
                                  seed=args.seed,
                                  on_step=snapshot,
                                  on_step_every=on_step_every)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mini_nn.save_warmup(out, model, feature_pool=args.feature_pool,
                        seed=args.seed, lr=args.lr,
                        align=args.align, seam=args.seam, warmup=report)
    print(f'  checkpoint -> {out}')
    print(f'  plc_main loads {Path("warmup/cube_warmup_random_ranges.pt")} by '
          f'default; fit with --warmup {out} to load this one, or --no-warmup '
          f'to fit from a blank network')
    return report


if __name__ == '__main__':
    main()
