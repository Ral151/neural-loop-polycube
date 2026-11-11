"""What the direction term sees: each fitted point and the target it is paired with.

The fit pairs every predicted point with its nearest target point, and the
direction term reads its normal there.  Neither set carries an order, so the
pairing is blind to which way the loop is walked.  This draws that pairing:
100 points sampled at evenly spaced z on the fitted loop, a line to the target
point each one is paired with, and the cosine the term would take at that pair
-- green where the curve's implied normal agrees with the cloud's, red where it
does not.

  python utils/plot_loop_matching.py log/axis_seam10_horse_repro horse_points_cubic_lam3 y --turned-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _paths  # noqa: E402
from model import mini_nn
from model.loop_extraction import plane_basis
from plot_first_iteration import arrow, plane_file, read_ply

AXES = 'xyz'
GREEN, RED = '#1a9850', '#d73027'


def fitted(model, queries, frame, n):
    """Every loop as the trained network draws it, in the cloud frame, with its
    analytic tangent."""
    z = torch.linspace(0, 1, n + 1)[:-1, None]
    with torch.no_grad():
        feat = model.triplane(queries)
    outs, tans = [], []
    for i in range(len(queries)):
        zz = z.clone().requires_grad_(True)
        out = model(zz[None].expand(1, -1, -1), None, feat[i:i + 1])[0]
        tans.append(mini_nn.curve_tangent(out, zz).detach())
        ## the network predicts in the plane's frame; the export adds the plane's
        ## centre point back, which is what puts the curve on the cloud
        outs.append(out.detach()[0] + frame[i][None, :])
    return outs, tans


def panel(ax, axis, r, head=None):
    """One loop: the curve, 100 evenly spaced z samples, the target each pairs
    with, and the cosine the direction term reads at that pair."""
    u, v = r['u'], r['v']
    tgt, pts = r['target'], r['pts']
    p = pts[r['idx']]
    q = tgt[r['pair']]
    ok = r['cos'] >= 0
    ax.scatter(tgt @ u, tgt @ v, s=1.2, c='0.62', alpha=.8, linewidths=0,
               zorder=1)
    ax.scatter(q @ u, q @ v, s=7, c='0.25', linewidths=0, zorder=2)
    for a, b in zip(p, q):
        ax.plot([a @ u, b @ u], [a @ v, b @ v], '-', lw=0.8, alpha=1.0,
                color='0.45', zorder=3)
    ax.plot(pts @ u, pts @ v, '-', c='tab:orange', lw=1.0, zorder=4)
    ax.scatter(p[ok] @ u, p[ok] @ v, s=13, c=GREEN, edgecolors='w',
               linewidths=.4, zorder=5)
    ax.scatter(p[~ok] @ u, p[~ok] @ v, s=13, c=RED, edgecolors='w',
               linewidths=.4, zorder=5)
    arrow(ax, np.stack([pts @ u, pts @ v], 1), 'tab:orange',
          fracs=(0.2, 0.7), span=60, lw=1.4)

    if head is None:
        head = f"{axis} {r['offset']:+.2f}"
    mean = float(r['cos'].mean())
    ax.set_title(f'{head}\n'
                 f'mean cos {mean:+.2f}, cos<0 on '
                 f'{int((r["cos"] < 0).sum())}/{len(r["cos"])}, '
                 f'cos>0.5 on {int((r["cos"] > 0.5).sum())}/{len(r["cos"])}',
                 fontsize=7)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)
    uv = np.stack([u, v], 1)
    P = np.concatenate([tgt @ uv, pts @ uv])
    lo, hi = P.min(0), P.max(0)
    pad = 0.08 * (hi - lo).max() + 0.005
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)


def targets_for(run, axis, offset, slot, curves):
    """The plane's targets, split by loop when the plane holds more than one."""
    xyz, nrm, rgb = read_ply(plane_file(run, axis, offset, 'loops'))
    if len(curves) == 1 or rgb is None:
        return xyz, nrm
    rgb = np.asarray(rgb)
    code = (rgb[:, 0].astype(np.int64) * 65536
            + rgb[:, 1].astype(np.int64) * 256 + rgb[:, 2])
    groups = {}
    for c in np.unique(code):
        m = code == c
        ## grey is the colour of a point no loop owns
        if np.all(rgb[m][0] == 158):
            continue
        groups[int(c)] = m
    ## a colour names a loop by the curve it lies on, not by the order of the
    ## file, so the split is read off the geometry and checked by size
    owner = {}
    for c, m in groups.items():
        owner[c] = int(np.argmin([np.median(cKDTree(curve).query(xyz[m])[0])
                                  for curve in curves]))
    want = [m for c, m in groups.items() if owner[c] == slot]
    if len(want) != 1:
        raise RuntimeError(f'{axis}{offset:+.2f}: cannot split {len(groups)} '
                           f'colours over {len(curves)} loops')
    return xyz[want[0]], None if nrm is None else nrm[want[0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('stem')
    ap.add_argument('axis', choices=tuple(AXES))
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--turned-only', action='store_true',
                    help='only the loops whose final fit runs the other way')
    ap.add_argument('--warmup', default='warmup/cube_warmup_random_ranges.pt')
    ap.add_argument('--epoch0', action='store_true',
                    help='pair the cube warm-up to the targets instead of the fit')
    ap.add_argument('--state', type=Path,
                    help='a saved state_dict, e.g. an epoch replayed from a run, '
                         'to evaluate instead of the final checkpoint')
    ap.add_argument('--tag', help='output tag (default fit, e0, or the state name)')
    ap.add_argument('--label', help='what the curve is called in the title')
    args = ap.parse_args()

    axis_i = AXES.index(args.axis)
    ckpt = torch.load(args.run / f'{args.stem}_model.pt', map_location='cpu',
                      weights_only=False)
    if args.epoch0 or args.state is not None:
        model = mini_nn.TriplaneNN(num_neurons=ckpt['num_neurons'],
                                   num_layers=ckpt.get('num_layers', 2),
                                   pe=ckpt['pe'],
                                   triplane=dict(ckpt['triplane']))
        if args.state is None:
            mini_nn.load_warmup(args.warmup, model)
        else:
            model.load_state_dict(torch.load(args.state, map_location='cpu',
                                             weights_only=False), strict=False)
        model.eval()
    else:
        model, _ = mini_nn.load_model(args.run / f'{args.stem}_model.pt')
    keys = np.asarray(ckpt['keys'])
    axes = np.asarray(ckpt['axes'])
    labels = np.abs(axes).argmax(1)
    loops = np.flatnonzero(labels == axis_i)
    n = int(ckpt['latent_points'])

    queries = torch.as_tensor(ckpt['queries'])
    axes_t = torch.as_tensor(axes)
    frame = np.asarray(ckpt['plane_point'])
    curves, tangents = fitted(model, queries,
                              torch.as_tensor(frame), n)

    wind = np.load(args.run / f'{args.stem}_loop_winding.npz')
    W, epoch = wind['winding'], wind['epoch']

    ## the plane's loops in slot order, so a colour group and a slot line up
    plane_of = {(int(keys[i][0]), round(float(keys[i][1]), 2)): [] for i in loops}
    for i in loops:
        plane_of[(int(keys[i][0]), round(float(keys[i][1]), 2))].append(int(i))
    for members in plane_of.values():
        members.sort(key=lambda i: int(keys[i][2]))

    ## the fit read back from the saved file, as a check that the network we
    ## evaluate is the one the run exported
    exported = args.state is None and not args.epoch0
    worst = 0.0
    report = []
    for i in loops:
        offset, slot = float(keys[i][1]), int(keys[i][2])
        members = plane_of[(int(keys[i][0]), round(offset, 2))]
        plane_curves = [curves[j].numpy() for j in members]
        target, target_n = targets_for(args.run, args.axis, offset,
                                       members.index(int(i)), plane_curves)
        if exported:
            saved, _, _ = read_ply(plane_file(args.run, args.axis, offset, 'fit'))
            saved = saved.reshape(-1, n, 3)[members.index(int(i))]
            d = cKDTree(saved).query(curves[i].numpy())[0]
            worst = max(worst, float(np.max(d)))

        u, v = plane_basis(axes[i] / np.linalg.norm(axes[i]))
        w1 = float(W[-1, i])
        pts = curves[i].numpy()
        ## the state's own direction, read about the curve's own centre, so the
        ## panels say which way the loop runs as drawn
        w_now = float(mini_nn.winding_number(pts, axes[i])[0])
        ## evenly spaced z: n points sit at z = k/n, so these sample k/m
        idx = (np.arange(args.samples) * n) // args.samples
        pair = cKDTree(target).query(pts[idx])[1]
        cos = mini_nn.normal_cosine(tangents[i][idx][None],
                                    torch.as_tensor(target_n[None, pair]),
                                    axes_t[i][None])[0].numpy()
        report.append(dict(loop=int(i), offset=offset, u=u, v=v,
                           pts=pts, target=target, idx=idx, pair=pair,
                           cos=cos, w1=w1, w_now=w_now,
                           flip=next((int(epoch[e]) for e in range(1, len(W))
                                      if W[e, i] < 0 <= W[e - 1, i]), None)))

    if args.turned_only:
        ## the set is the loops that end up reversed, so an early epoch shows
        ## the same loops as the final fit
        report = [r for r in report if r['w1'] < -0.5]
    if not report:
        raise SystemExit(f'{args.axis}: no loops to draw'
                         + (' with winding < -0.5' if args.turned_only else ''))

    nx = 6
    ny = int(np.ceil(len(report) / nx))
    fig, axs = plt.subplots(ny, nx, figsize=(3.1 * nx, 3.15 * ny), dpi=150)
    axs = np.atleast_2d(axs)
    for k, r in enumerate(report):
        ax = axs[k // nx, k % nx]
        head = (f"{args.axis} {r['offset']:+.2f}  w {r['w_now']:+.2f}"
                f"{'' if r['w1'] >= -0.5 else ' -> reversed'}")
        if not exported:
            head = (f"{args.axis} {r['offset']:+.2f}\nwinding {r['w_now']:+.2f}"
                    f", final {r['w1']:+.2f}")
        panel(ax, args.axis, r, head)

    for ax in axs.ravel()[len(report):]:
        ax.axis('off')
    what = ('the cube warm-up (epoch 0)' if args.epoch0 else
            args.label or (f'epoch {args.tag}' if args.state is not None
                           else 'the fitted loop'))
    handles = [Line2D([], [], color='tab:orange'),
               Line2D([], [], color='0.62', marker='o', ms=3, ls=''),
               Line2D([], [], color='0.45'),
               Line2D([], [], color=GREEN, marker='o', ls='', ms=4),
               Line2D([], [], color=RED, marker='o', ls='', ms=4)]
    fig.legend(handles, [what,
                         'target points',
                         'pairing link',
                         'pair with cos \u2265 0', 'pair with cos < 0'],
               loc='upper right', ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f'{args.stem} {args.axis}: 100 points at evenly spaced z on '
                 f'{what}, each paired to its nearest target'
                 f'\npanels in the plane basis (u, v); cos is that of the '
                 f'direction term, tangent \u00d7 plane normal against the '
                 f'target normal', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    tag = args.tag or ('e0' if args.epoch0 else
                       ('fit' if args.state is None else args.state.stem))
    out = args.run / (f'matching_{args.axis}_{tag}'
                      f'{"_turned" if args.turned_only else ""}.png')
    fig.savefig(out)
    print(f'wrote {out}')
    if exported:
        print(f'  network vs saved fit: worst point distance {worst:.2e}')
    for r in report:
        print(f"  loop {r['loop']:3d} {args.axis}{r['offset']:+.2f}  "
              f"winding {r['w_now']:+.2f}  final {r['w1']:+.2f}  "
              f"mean cos {r['cos'].mean():+.3f}  "
              f"cos<0 {int((r['cos'] < 0).sum())}/{len(r['cos'])}")


if __name__ == '__main__':
    main()
