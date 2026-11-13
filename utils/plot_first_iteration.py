"""The first iteration against the targets, one panel per loop of an axis.

The fit's initial state is not a checkpoint of its own: the triplane starts at
zero, the phases are zero and the network is the cube warm-up, so epoch 0 is
that warm-up read at the run's own queries, axes and triplane frame.  The saved
model carries all three, which is what makes the first iteration of a finished
run recoverable rather than lost.

Each panel holds the plane's target points, the epoch-0 curve and the final
curve, both drawn in the plane's own (u, v) frame -- the same basis the run's
fit figures use, so counter-clockwise in a panel is counter-clockwise about the
plane normal.  An arrow on each curve shows which way it is walked: that is the
one thing chamfer cannot show and the reason a loop can turn over unseen.

  python utils/plot_first_iteration.py log/axis_seam10_horse_repro horse_points_cubic_lam3 y
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _paths  # noqa: E402
from model import mini_nn
from model.mini_nn import winding_number
from model.loop_extraction import plane_basis

AXES = 'xyz'
PLY_TYPES = {'float': '<f4', 'float32': '<f4', 'double': '<f8',
             'float64': '<f8', 'uchar': 'u1', 'uint8': 'u1',
             'int': '<i4', 'int32': '<i4'}


def read_ply(path):
    """Vertices, vertex normals and colours of a PLY, in file order."""
    raw = Path(path).read_bytes()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    head = raw[:end].decode('ascii')
    names, types, in_vertex, n = [], [], False, 0
    for line in head.splitlines():
        if line.startswith('element vertex '):
            in_vertex, n = True, int(line.split()[-1])
        elif line.startswith('element '):
            in_vertex = False
        elif in_vertex and line.startswith('property ') and \
                len(line.split()) == 3:
            _, kind, name = line.split()
            names.append(name)
            types.append(PLY_TYPES[kind])
    table = np.frombuffer(raw, dtype=np.dtype(list(zip(names, types))),
                          count=n, offset=end)
    xyz = np.stack([table['x'], table['y'], table['z']], 1).astype(np.float64)
    if 'nx' in names:
        nrm = np.stack([table['nx'], table['ny'], table['nz']], 1)
        nrm = nrm.astype(np.float64)
    else:
        nrm = None
    rgb = (np.stack([table['red'], table['green'], table['blue']], 1)
           if 'red' in names else None)
    return xyz, nrm, rgb


def first_iteration(ckpt, warmup):
    """Every loop as the fit draws it before its first update, in the cloud frame."""
    model = mini_nn.TriplaneNN(num_neurons=ckpt['num_neurons'],
                               num_layers=ckpt['num_layers'], pe=ckpt['pe'],
                               triplane=dict(ckpt['triplane']),
                               axis_condition=bool(ckpt.get('axis_condition', False)),
                               num_freqs=int(ckpt.get('num_freqs', 1)))
    mini_nn.load_warmup(warmup, model)
    model.eval()
    n = int(ckpt['latent_points'])
    z = torch.linspace(0, 1, n + 1)[:-1, None]
    queries = torch.as_tensor(ckpt['queries'])
    with torch.no_grad():
        feat = model.triplane(queries)
        out = model(z[None].expand(len(queries), -1, -1), None, feat,
                    torch.as_tensor(ckpt['axes']))[0]
    ## the network draws each loop in its plane's frame, whose origin is the plane's
    ## centre point; the export adds it back, and so must any replay of the fit.
    ## The checkpoint keeps that point per loop, not per plane
    frame = np.asarray(ckpt['plane_point'])
    return out.numpy() + frame[:, None, :]


def plane_file(run, axis, offset, kind):
    """The plane folder's file for this slice, found by its own name."""
    stem = f'plane_*_{axis}{offset:+.2f}_{kind}.ply'
    hits = sorted(glob.glob(str(run / 'planes' / stem)))
    if not hits:
        raise FileNotFoundError(f'no {kind} for {axis}{offset:+.2f} in {run}')
    return hits[0]


def arrow(ax, points, colour, fracs=(0.12, 0.37, 0.62, 0.87), span=90,
          lw=1.8):
    """Filled arrows along the curve at `fracs` of its points."""
    for frac in fracs:
        k = int(len(points) * frac) % len(points)
        a, b = points[k], points[(k + span) % len(points)]
        ax.annotate('', xy=b, xytext=a,
                    arrowprops=dict(arrowstyle='-|>', color=colour, lw=lw,
                                    mutation_scale=14, shrinkA=0,
                                    shrinkB=0), zorder=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('stem')
    ap.add_argument('axis', choices=tuple(AXES))
    ap.add_argument('--warmup', default='warmup/cube_warmup_random_ranges.pt')
    ap.add_argument('--out', default=None)
    ap.add_argument('--zoom', action='store_true',
                    help='frame each panel on the target, not the warm-up square')
    ap.add_argument('--turned-only', action='store_true',
                    help='only the loops whose final fit runs the other way')
    args = ap.parse_args()

    axis_i = AXES.index(args.axis)
    ckpt = torch.load(args.run / f'{args.stem}_model.pt', map_location='cpu',
                      weights_only=False)
    first = first_iteration(ckpt, args.warmup)
    keys = np.asarray(ckpt['keys'])
    axes = np.asarray(ckpt['axes'])
    labels = np.abs(axes).argmax(1)
    loops = np.flatnonzero(labels == axis_i)
    n_points = int(ckpt['latent_points'])

    wind = np.load(args.run / f'{args.stem}_loop_winding.npz')
    W, epoch = wind['winding'], wind['epoch']

    ## every loop's panel content is gathered first, so the grid can be laid out
    ## over a filtered subset as easily as over the whole axis
    report = []
    for i in loops:
        offset, slot = float(keys[i][1]), int(keys[i][2])
        target, _, _ = read_ply(plane_file(args.run, args.axis, offset,
                                           'slice'))
        final, _, _ = read_ply(plane_file(args.run, args.axis, offset, 'fit'))
        final = final.reshape(-1, n_points, 3)[slot]
        u, v = plane_basis(axes[i] / np.linalg.norm(axes[i]))
        w0, fold0 = winding_number(first[i], axes[i])
        w1, fold1 = winding_number(final, axes[i])
        ## the export turns a loop round when its mean cosine says it runs
        ## against the points' normals, so one saved curve can disagree with the
        ## fit's own winding; the panel shows the fit's direction and marks the
        ## loop that was turned for the files
        w_fit = float(W[-1, i])
        turned_export = abs(w_fit) > 0.5 and w1 * w_fit < 0
        if turned_export:
            final = final[::-1]
            w1, fold1 = winding_number(final, axes[i])
        flip = None
        for e in range(1, len(W)):
            if W[e, i] < 0 and W[e - 1, i] >= 0:
                flip = int(epoch[e])
                break
        report.append(dict(loop=int(i), offset=offset, target=target,
                           first=first[i].copy(), final=final, u=u, v=v,
                           w0=float(w0), w1=float(w1), flip=flip,
                           fold0=float(fold0), fold1=float(fold1),
                           turned_export=bool(turned_export)))

    if args.turned_only:
        report = [r for r in report if r['w1'] < -0.5]

    nx = 6
    ny = int(np.ceil(len(report) / nx))
    fig, axs = plt.subplots(ny, nx, figsize=(3.0 * nx, 3.05 * ny), dpi=150)
    axs = np.atleast_2d(axs)
    for k, r in enumerate(report):
        ax = axs[k // nx, k % nx]
        u, v = r['u'], r['v']
        ax.scatter(r['target'] @ u, r['target'] @ v, s=1.2, c='0.55',
                   alpha=.85, linewidths=0, zorder=2.5)
        ax.plot((r['first'] @ u), (r['first'] @ v), '-', c='tab:blue', lw=1.1,
                zorder=3)
        ax.plot((r['final'] @ u), (r['final'] @ v), '-', c='tab:orange',
                lw=1.2, zorder=4)
        arrow(ax, np.stack([r['first'] @ u, r['first'] @ v], 1), 'tab:blue')
        arrow(ax, np.stack([r['final'] @ u, r['final'] @ v], 1), 'tab:orange')

        keep = 'turned' if r['w1'] < -0.5 else 'kept'
        neg = [int(epoch[e]) for e in range(len(W)) if W[e, r['loop']] < -0.5]
        if r['w1'] < -0.5:
            when = (f"turned at e{r['flip']}" if r['flip'] is not None
                    else 'turned')
        elif neg:
            when = f'negative e{neg[0]}-{neg[-1]}'
        else:
            when = 'never negative'
        mark = ', export-flipped' if r['turned_export'] else ''
        ax.set_title(f"{args.axis} {r['offset']:+.2f}   {keep}: "
                     f"w0 {r['w0']:+.2f} -> w1k {r['w1']:+.2f}{mark}\n"
                     f"{when}, fold {r['fold0']:.2f} -> {r['fold1']:.2f}",
                     fontsize=7)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        if args.zoom:
            ## framed on what the fit is asked to reach: the target points plus
            ## a little of the final curve, so the warm-up square leaves the view
            pts = np.stack([np.concatenate([r['target'] @ u, r['final'] @ u]),
                            np.concatenate([r['target'] @ v, r['final'] @ v])],
                           axis=1)
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.12 * (hi - lo).max() + 0.01
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)

    for ax in axs.ravel()[len(report):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], marker='o', ls='', ms=3, color='0.62'),
               plt.Line2D([], [], color='tab:blue'),
               plt.Line2D([], [], color='tab:orange')]
    fig.legend(handles, ['target points', 'first iteration (epoch 0)',
                         'final fit (epoch 1000)'],
               loc='upper right', ncol=3, frameon=False, fontsize=10)
    turned = sum(r['w1'] < -0.5 for r in report)
    which = (f'the {len(report)} loops that end up'
             f'{" reversed" if args.turned_only else ""}' if args.turned_only
             else f'all {len(report)} loops, {turned} of which end up reversed')
    fig.suptitle(f'{args.stem} {args.axis}: the first iteration is the cube '
                 f'warm-up and runs counter-clockwise on {which}\n'
                 f'panels use the plane basis (u, v), so counter-clockwise on '
                 f'screen is counter-clockwise about +{args.axis}; winding read '
                 f'about each curve\'s own centre', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = args.out or str(args.run /
                          (f'first_iteration_{args.axis}'
                           f'{"_turned" if args.turned_only else ""}'
                           f'{"_zoom" if args.zoom else ""}.png'))
    fig.savefig(out)
    print(f'wrote {out}')
    for r in report:
        print(f"  loop {r['loop']:3d} {args.axis}{r['offset']:+.2f}  "
              f"w {r['w0']:+.2f} -> {r['w1']:+.2f}  flip {r['flip']}  "
              f"fold {r['fold0']:.2f} -> {r['fold1']:.2f}")


if __name__ == '__main__':
    main()
