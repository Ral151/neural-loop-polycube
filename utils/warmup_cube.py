"""Train the cube warm-up checkpoint that every shape's fit starts from.

The network is fitted to the 12 square cross-section loops of a unit cube so that
all loops run the same way round and share their z = 0 seam. Run once, not per
shape; the network/grid sizes must match the fit's:

    python main.py warmup                  # -> warmup/cube_warmup_axis.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model.cube import UNIT_CUBE_HALF
from model.network import miniNN
from model.warmup import save_warmup, warm_on_cube


def build_parser():
    """Command-line options of `main.py warmup`."""
    parser = argparse.ArgumentParser(
        prog='main.py warmup',
        description='Warm a triplane network on the unit cube and save the '
                    'checkpoint main.py fit loads.')
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
    parser.add_argument('--no-axis-condition', dest='axis_condition',
                        action='store_false',
                        help='warm a network that does not read the slice axis, for '
                             '`main.py fit --no-axis-condition`')
    parser.add_argument('--out', default='warmup/cube_warmup_axis.pt',
                        help='checkpoint path; main.py fit loads '
                             'warmup/cube_warmup_axis.pt by default')
    return parser


def main(argv=None):
    """Warm a fresh network on the cube, save the checkpoint and return the report."""
    args = build_parser().parse_args(argv)
    if args.steps < 1:
        raise SystemExit(f'--steps must be at least 1, got {args.steps}')
    if args.planes < 2 or args.planes % 2:
        raise SystemExit(f'--planes must be even and at least 2, got {args.planes}')
    if args.samples < 4:
        raise SystemExit(f'--samples must be at least 4, got {args.samples}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # the triplane spans the unit cube; a fit later maps this cube onto its cloud's box
    frame = dict(centre=np.zeros(3), half=np.full(3, UNIT_CUBE_HALF),
                 resolution=args.resolution, channels=args.channels, tv=0.0)
    model = miniNN(1, num_neurons=args.neurons, num_layers=args.layers,
                   feature_mode='triplane', triplane=frame,
                   axis_condition=args.axis_condition).to(device)
    loops = 3 * args.planes
    print(f'cube warm-up: {loops} loops ({args.planes} per axis), {args.samples} '
          f'samples each, {"axis-conditioned " if args.axis_condition else ""}'
          f'{args.layers}x{args.neurons} net, '
          f'{args.resolution} x {args.resolution} x 3 grids of {args.channels} '
          f'channels, on {device}')

    report = warm_on_cube(model, steps=args.steps, lr=args.lr,
                          align=args.align, seam=args.seam,
                          planes=args.planes, samples=args.samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_warmup(out, model, seed=args.seed, lr=args.lr,
                align=args.align, seam=args.seam, warmup=report)
    print(f'  checkpoint -> {out}')
    print(f'  load it in a fit with `main.py fit --warmup {out}`; '
          f'--no-warmup fits from a blank network')
    return report
