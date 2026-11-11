"""One sheet per early epoch: all the loops that end up reversed, at that epoch.

A run saves only its final checkpoint, so these states are replayed by
`replay_epochs.py` and read back as plain state dicts.  Each sheet draws every
cross section that ends up running the other way (26 of the 30 y loops in this
run), with 100 points at evenly spaced z, each paired to its nearest target, and
the cosine the direction term reads at that pair.  Panel positions are the same
on every sheet, so a loop can be followed epoch by epoch.

  python utils/plot_epoch_sheets.py log/axis_seam10_horse_repro \
      horse_points_cubic_lam3 y --first 3 --last 13
"""
from __future__ import annotations

import argparse
import re
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
from plot_loop_matching import fitted, targets_for, panel, GREEN, RED

AXES = 'xyz'


def log_fits(run):
    """Overall fit per epoch, from the run's own epoch lines."""
    log = run.parent / f'{run.name}.log'
    fits = {}
    if log.exists():
        for line in open(log):
            hit = re.match(r'\s*(\d+) epoch: loss: [\d.]+ \(fit ([\d.]+)', line)
            if hit:
                fits[int(hit.group(1))] = float(hit.group(2))
    return fits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('stem')
    ap.add_argument('axis', choices=tuple(AXES))
    ap.add_argument('--first', type=int, default=3)
    ap.add_argument('--last', type=int, default=13)
    ap.add_argument('--states', type=Path,
                    help='folder replay_epochs.py wrote (default run/replay)')
    ap.add_argument('--warmup', default='warmup/cube_warmup_random_ranges.pt',
                    help='the warm-up, drawn as epoch 0')
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--columns', type=int, default=6)
    args = ap.parse_args()

    axis_i = AXES.index(args.axis)
    ckpt = torch.load(args.run / f'{args.stem}_model.pt', map_location='cpu',
                      weights_only=False)
    keys = np.asarray(ckpt['keys'])
    axes = np.asarray(ckpt['axes'])
    loops = np.flatnonzero(np.abs(axes).argmax(1) == axis_i)
    n = int(ckpt['latent_points'])
    queries = torch.as_tensor(ckpt['queries'])
    axes_t = torch.as_tensor(axes)
    frame = torch.as_tensor(np.asarray(ckpt['plane_point']))

    W = np.load(args.run / f'{args.stem}_loop_winding.npz')['winding']
    turned = sorted((int(i) for i in loops if W[-1, i] < -0.5),
                    key=lambda i: float(keys[i][1]))
    slots = {int(i): k for k, i in enumerate(loops)}

    ## the final fit fixes which target colour belongs to which loop of a plane
    final, _ = mini_nn.load_model(args.run / f'{args.stem}_model.pt')
    final_curves, _ = fitted(final, queries[loops],
                             frame[loops], n)

    plane_of = {}
    for i in loops:
        plane_of.setdefault((int(keys[i][0]), round(float(keys[i][1]), 2)),
                            []).append(int(i))
    for members in plane_of.values():
        members.sort(key=lambda i: int(keys[i][2]))

    idx = (np.arange(args.samples) * n) // args.samples
    fixed = {}
    for i in turned:
        offset = float(keys[i][1])
        members = plane_of[(int(keys[i][0]), round(offset, 2))]
        local = [slots[j] for j in members]
        target, target_n = targets_for(
            args.run, args.axis, offset, members.index(i),
            [final_curves[k].numpy() for k in local])
        u, v = plane_basis(axes[i] / np.linalg.norm(axes[i]))
        fixed[i] = dict(loop=i, offset=offset, target=target,
                        target_n=target_n, u=u, v=v, idx=idx)

    states = args.states or args.run / 'replay'
    fits = log_fits(args.run)
    for e in range(args.first, args.last + 1):
        if e == 0:
            model = build(ckpt, None, args.warmup)
        else:
            model = build(ckpt, str(states / f'state_e{e}.pt'), None)
        curves, tangents = fitted(model, queries[loops],
                                  frame[loops], n)

        report = []
        for i in turned:
            f = fixed[i]
            pts = curves[slots[i]].numpy()
            pair = cKDTree(f['target']).query(pts[idx])[1]
            cos = mini_nn.normal_cosine(
                tangents[slots[i]][idx][None],
                torch.as_tensor(f['target_n'][None, pair]),
                axes_t[i][None])[0].numpy()
            report.append(dict(f, pts=pts, pair=pair, cos=cos,
                               w=float(mini_nn.winding_number(
                                   pts, axes[i])[0])))

        clock = sum(r['w'] < -0.5 for r in report)
        mean_cos = float(np.mean([r['cos'].mean() for r in report]))
        nc = args.columns
        nr = int(np.ceil(len(report) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.15 * nr),
                                dpi=150, squeeze=False)
        for k, r in enumerate(report):
            panel(axs[k // nc, k % nc], args.axis, r,
                  f"{args.axis} {r['offset']:+.2f}\nwinding {r['w']:+.2f}")
        for ax in axs.ravel()[len(report):]:
            ax.axis('off')

        handles = [Line2D([], [], color='tab:orange'),
                   Line2D([], [], color='0.62', marker='o', ms=3, ls=''),
                   Line2D([], [], color='0.45'),
                   Line2D([], [], color=GREEN, marker='o', ls='', ms=4),
                   Line2D([], [], color=RED, marker='o', ls='', ms=4)]
        fig.legend(handles, [f'loop at epoch {e}', 'target points',
                             'pairing link', 'pair with cos \u2265 0',
                             'pair with cos < 0'],
                   loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5,
                   frameon=False, fontsize=10)
        diag = ('n/a' if e == 0 else
                f"{int((W[e - 1, loops] < 0).sum())}/{len(loops)}")
        fit = f"{fits[e - 1]:.4f}" if e > 0 and (e - 1) in fits else 'n/a'
        what = 'the cube warm-up (epoch 0)' if e == 0 else f'epoch {e}'
        fig.suptitle(
            f'{args.stem} {args.axis}, {what}: all {len(turned)} loops that '
            f'end up reversed, 100 points at evenly spaced z'
            f'\n{clock}/{len(turned)} wound clockwise as drawn, mean cos '
            f'{mean_cos:+.2f}; the run\'s own diagnostic reads {diag} '
            f'negative at this epoch; fit {fit}, direction weight '
            f'{"0.0100" if e else "n/a"}', fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        out = args.run / f'matching_{args.axis}_e{e}_turned.png'
        fig.savefig(out)
        plt.close(fig)
        print(f'wrote {out}  ({clock}/{len(turned)} clockwise, mean cos '
              f'{mean_cos:+.2f}, run diagnostic {diag})')


if __name__ == '__main__':
    main()