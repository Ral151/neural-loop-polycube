"""Every loop on a slice axis, at every early epoch, against its targets.

The run saves only its final checkpoint, so epochs 1-20 are replayed by
`replay_epochs.py` and read back as plain state dicts.  For each epoch this
draws one sheet per axis: every loop on that axis, with 100 points at evenly
spaced z, each paired to its nearest target, coloured by the cosine the
direction term reads there.  Panel positions are the same on every sheet, so a
loop can be followed epoch by epoch.  It also writes one summary sheet of the
winding of every loop over the whole epoch range, next to the count that runs
clockwise.

  python utils/plot_axis_epochs.py log/axis_seam10_horse_repro \
      horse_points_cubic_lam3 --first 0 --last 20
"""
from __future__ import annotations

import argparse
import csv
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
from plot_epoch_matching import build
from plot_loop_matching import targets_for, panel, GREEN, RED

AXES = 'xyz'


def all_curves(model, queries, frame, n):
    """Every loop in one batched forward, with its analytic tangent.

    The network conditions each sample on its own feature, so a batch of 139
    loops is as valid as the trainer's batch of a few, and much faster than 139
    separate passes.
    """
    z = torch.linspace(0, 1, n + 1)[:-1, None]
    z = z[None].expand(len(queries), -1, -1).clone().requires_grad_(True)
    with torch.no_grad():
        feat = model.triplane(queries)
    out = model(z, None, feat)[0]
    tan = mini_nn.curve_tangent(out, z).detach()
    return out.detach() + frame[:, None, :], tan


def axis_terms(run, stem):
    """Per-axis fit and the run's own reversed count, by epoch."""
    path = run / f'{stem}_epoch_terms.csv'
    out = {}
    if path.exists():
        for row in csv.DictReader(open(path)):
            out[(int(row['epoch']), row['axis'])] = (
                float(row['fit']), int(row['reversed']))
    return out


def sheet(report, axis, what, note, out, columns=6):
    nr = int(np.ceil(len(report) / columns))
    fig, axs = plt.subplots(nr, columns, figsize=(3.1 * columns, 3.15 * nr),
                            dpi=150, squeeze=False)
    for k, r in enumerate(report):
        panel(axs[k // columns, k % columns], axis, r,
              f"{axis} {r['offset']:+.2f}\nwinding {r['w']:+.2f}")
    for ax in axs.ravel()[len(report):]:
        ax.axis('off')
    handles = [Line2D([], [], color='tab:orange'),
               Line2D([], [], color='0.62', marker='o', ms=3, ls=''),
               Line2D([], [], color='0.45'),
               Line2D([], [], color=GREEN, marker='o', ls='', ms=4),
               Line2D([], [], color=RED, marker='o', ls='', ms=4)]
    fig.legend(handles, [f'loop at {what}', 'target points',
                         'pairing link', 'pair with cos \u2265 0',
                         'pair with cos < 0'],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False, fontsize=10)
    fig.suptitle(note, fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('stem')
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--last', type=int, default=20)
    ap.add_argument('--states', type=Path,
                    help='folder replay_epochs.py wrote (default run/replay)')
    ap.add_argument('--warmup', default='warmup/cube_warmup_random_ranges.pt')
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--columns', type=int, default=6)
    args = ap.parse_args()

    ckpt = torch.load(args.run / f'{args.stem}_model.pt', map_location='cpu',
                      weights_only=False)
    keys = np.asarray(ckpt['keys'])
    axes = np.asarray(ckpt['axes'])
    n = int(ckpt['latent_points'])
    queries = torch.as_tensor(ckpt['queries'])
    axes_t = torch.as_tensor(axes)
    frame = torch.as_tensor(np.asarray(ckpt['plane_point']))

    loops = [np.flatnonzero(np.abs(axes).argmax(1) == k) for k in range(3)]

    ## the final fit fixes which target colour belongs to which loop of a plane
    final, _ = mini_nn.load_model(args.run / f'{args.stem}_model.pt')
    final_curves, _ = all_curves(final, queries, frame, n)

    plane_of = {}
    for i in range(len(keys)):
        plane_of.setdefault((int(keys[i][0]), round(float(keys[i][1]), 2)),
                            []).append(int(i))
    for members in plane_of.values():
        members.sort(key=lambda i: int(keys[i][2]))

    idx = (np.arange(args.samples) * n) // args.samples
    fixed = {}
    for k in range(3):
        for i in loops[k]:
            offset = float(keys[i][1])
            members = plane_of[(int(keys[i][0]), round(offset, 2))]
            target, target_n = targets_for(
                args.run, AXES[k], offset, members.index(int(i)),
                [final_curves[j].numpy() for j in members])
            u, v = plane_basis(axes[i] / np.linalg.norm(axes[i]))
            fixed[i] = dict(loop=int(i), offset=offset, target=target,
                            target_n=target_n, u=u, v=v, idx=idx)

    states = args.states or args.run / 'replay'
    terms = axis_terms(args.run, args.stem)
    epochs = list(range(args.first, args.last + 1))
    wind = {ax: np.zeros((len(loops[k]), len(epochs)))
            for k, ax in enumerate(AXES)}
    cw = {ax: np.zeros(len(epochs), int) for ax in AXES}

    for e_i, e in enumerate(epochs):
        model = (build(ckpt, None, args.warmup) if e == 0
                 else build(ckpt, str(states / f'state_e{e}.pt'), None))
        curves, tangents = all_curves(model, queries, frame, n)

        for k, axis in enumerate(AXES):
            mine = sorted((int(i) for i in loops[k]),
                          key=lambda i: float(keys[i][1]))
            report = []
            for i in mine:
                f = fixed[i]
                pts = curves[i].numpy()
                pair = cKDTree(f['target']).query(pts[idx])[1]
                cos = mini_nn.normal_cosine(
                    tangents[i][idx][None],
                    torch.as_tensor(f['target_n'][None, pair]),
                    axes_t[i][None])[0].numpy()
                w = float(mini_nn.winding_number(pts, axes[i])[0])
                report.append(dict(f, pts=pts, pair=pair, cos=cos, w=w))

            wind[axis][:, e_i] = [r['w'] for r in report]
            cw[axis][e_i] = sum(r['w'] < -0.5 for r in report)
            mean_cos = float(np.mean([r['cos'].mean() for r in report]))
            fit, diag = terms.get((e, axis), (None, None))
            what = 'the cube warm-up (epoch 0)' if e == 0 else f'epoch {e}'
            note = (f'{args.stem} {axis}, {what}: all {len(mine)} loops on '
                    f'this axis, {args.samples} points at evenly spaced z'
                    f'\n{cw[axis][e_i]}/{len(mine)} wound clockwise as drawn, '
                    f'mean cos {mean_cos:+.2f}; '
                    f'fit {"n/a" if fit is None else f"{fit:.4f}"}; '
                    f'run diagnostic '
                    f'{"n/a" if diag is None else f"{diag}/{len(mine)}"} '
                    f'negative; direction weight 0.0100')
            out = args.run / f'matching_{axis}_e{e}.png'
            sheet(report, axis, what, note, out, args.columns)
        print(f'epoch {e:2d}: ' + '  '.join(
            f'{ax} {cw[ax][e_i]:2d}/{len(loops[k])}' for k, ax in
            enumerate(AXES)))

    fig, axs = plt.subplots(2, 3, figsize=(16, 9),
                            gridspec_kw=dict(height_ratios=(4, 1)))
    for k, axis in enumerate(AXES):
        ax = axs[0, k]
        im = ax.imshow(wind[axis], aspect='auto', interpolation='nearest',
                       extent=(epochs[0] - .5, epochs[-1] + .5,
                               len(loops[k]), 0),
                       cmap='RdBu_r', vmin=-1.2, vmax=1.2)
        ax.set_title(f'{axis}: winding of every {axis} loop ({len(loops[k])}) '
                     f'over epochs {epochs[0]}-{epochs[-1]}')
        ax.set_ylabel(f'{axis} loop, by plane offset')
        ax.set_xlabel('epoch')
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label='net winding about +axis')
        ax = axs[1, k]
        ax.step(epochs, cw[axis], where='mid', color='0.2')
        ax.fill_between(epochs, 0, cw[axis], step='mid', color='0.75')
        ax.set_ylim(-0.5, len(loops[k]) + 0.5)
        ax.set_xlabel('epoch')
        ax.set_ylabel('clockwise')
        ax.grid(alpha=0.25)
    fig.suptitle(f'{args.stem}: every loop on every axis, epochs '
                 f'{epochs[0]}-{epochs[-1]} (blue = clockwise about +axis)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = args.run / f'winding_axes_e{epochs[0]}_{epochs[-1]}.png'
    fig.savefig(out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()