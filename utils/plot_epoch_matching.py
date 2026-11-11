"""The first epochs of a fit, loop by loop, against the targets they are paired to.

A run saves only its final checkpoint, so epochs 1 and 2 are replayed from the
run's arguments and read back as plain state dicts.  Rows are epochs, columns are
the loops that end up running the other way, sampled at the same evenly spaced z
on every row: the same point of the loop can be followed from the warm-up square
to the first updates, and the pairing and the cosine the direction term reads at
that point compared directly.

  python utils/plot_epoch_matching.py log/axis_seam10_horse_repro horse_points_cubic_lam3 y \
      --state e1=log/axis_seam10_horse_repro/replay/state_e1.pt \
      --state e2=log/axis_seam10_horse_repro/replay/state_e2.pt
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
from plot_loop_matching import fitted, targets_for, panel, GREEN, RED

AXES = 'xyz'


def build(ckpt, state, warmup):
    model = mini_nn.TriplaneNN(num_neurons=ckpt['num_neurons'],
                               num_layers=ckpt.get('num_layers', 2),
                               pe=ckpt['pe'],
                               triplane=dict(ckpt['triplane']))
    if state is None:
        mini_nn.load_warmup(warmup, model)
    else:
        ## either a bare state dict from replay_epochs.py or a finished run's
        ## `*_model.pt`, which keeps the same weights under 'state_dict'
        saved = torch.load(state, map_location='cpu', weights_only=False)
        model.load_state_dict(saved.get('state_dict', saved), strict=False)
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('stem')
    ap.add_argument('axis', choices=tuple(AXES))
    ap.add_argument('--state', action='append', default=[], metavar='LABEL=PATH',
                    help='an epoch to replay, in row order')
    ap.add_argument('--warmup', default='warmup/cube_warmup_random_ranges.pt')
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--columns', type=int, default=6)
    ap.add_argument('--out', type=Path)
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
    frame = np.asarray(ckpt['plane_point'])

    rows = [('epoch 0 (warm-up)', None)]
    rows += [tuple(s.split('=', 1)) for s in args.state]

    ## every state draws every loop, because the targets are split by the loop a
    ## colour lies on, and the final fit is the one that sits on them
    drawn = {}
    for label, state in rows:
        curves, tangents = fitted(build(ckpt, state, args.warmup), queries,
                                  torch.as_tensor(frame), n)
        drawn[label] = (curves, tangents)
        print(f'{label}: {len(curves)} loops')

    final_curves = drawn[rows[-1][0]][0]
    W = np.load(args.run / f'{args.stem}_loop_winding.npz')['winding']
    turned = sorted((int(i) for i in loops if W[-1, i] < -0.5),
                    key=lambda i: float(keys[i][1]))
    if len(turned) < args.columns:
        raise SystemExit(f'only {len(turned)} turned loops, asked for '
                         f'{args.columns} columns')
    pick = [turned[int(round(k * (len(turned) - 1) / (args.columns - 1)))]
            for k in range(args.columns)]

    ## the plane's loops in slot order, so a colour group and a slot line up
    plane_of = {}
    for i in loops:
        plane_of.setdefault((int(keys[i][0]), round(float(keys[i][1]), 2)),
                            []).append(int(i))
    for members in plane_of.values():
        members.sort(key=lambda i: int(keys[i][2]))

    idx = (np.arange(args.samples) * n) // args.samples
    report = []
    for i in pick:
        offset = float(keys[i][1])
        members = plane_of[(int(keys[i][0]), round(offset, 2))]
        slot = members.index(i)
        curves = [final_curves[j].numpy() for j in members]
        target, target_n = targets_for(args.run, args.axis, offset, slot, curves)
        u, v = plane_basis(axes[i] / np.linalg.norm(axes[i]))
        row = dict(loop=i, offset=offset, target=target, idx=idx, u=u, v=v)
        for label, _ in rows:
            cur, tan = drawn[label]
            pts = cur[i].numpy()
            pair = cKDTree(target).query(pts[idx])[1]
            cos = mini_nn.normal_cosine(
                tan[i][idx][None],
                torch.as_tensor(target_n[None, pair]),
                axes_t[i][None])[0].numpy()
            ## the loop's own direction, read about its own centre
            w = float(mini_nn.winding_number(pts, axes[i])[0])
            row[label] = dict(pts=pts, pair=pair, cos=cos, w=w)
        report.append(row)

    nr, nc = len(rows), args.columns
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.15 * nr), dpi=150,
                            squeeze=False)
    for r, (label, _) in enumerate(rows):
        for c, row in enumerate(report):
            d = row[label]
            panel(axs[r, c], args.axis,
                  dict(u=row['u'], v=row['v'], target=row['target'],
                       pts=d['pts'], idx=row['idx'], pair=d['pair'],
                       cos=d['cos'], offset=row['offset']),
                  f"{args.axis} {row['offset']:+.2f}\n"
                  f"winding {d['w']:+.2f}")
        axs[r, 0].set_ylabel(label, fontsize=9)

    handles = [Line2D([], [], color='tab:orange'),
               Line2D([], [], color='0.62', marker='o', ms=3, ls=''),
               Line2D([], [], color='0.45'),
               Line2D([], [], color=GREEN, marker='o', ls='', ms=4),
               Line2D([], [], color=RED, marker='o', ls='', ms=4)]
    fig.legend(handles, ['loop at this epoch', 'target points',
                         'pairing link', 'pair with cos \u2265 0',
                         'pair with cos < 0'],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False, fontsize=10)
    fig.suptitle(f'{args.stem} {args.axis}: the {nc} of {len(turned)} '
                 f'eventually reversed loops that span the axis, at the same '
                 f'{args.samples} evenly spaced z on every row'
                 f'\ncolumns are loops, rows are epochs; cos is that of the '
                 f'direction term, tangent \u00d7 plane normal against the '
                 f'target normal', fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = args.out or args.run / f'matching_{args.axis}_epochs.png'
    fig.savefig(out)
    print(f'wrote {out}')
    for row in report:
        print(f"  {args.axis}{row['offset']:+.2f}  " + '  '.join(
            f"{label}: w {row[label]['w']:+.2f} "
            f"cos<0 {int((row[label]['cos'] < 0).sum())}/{len(idx)}"
            for label, _ in rows))


if __name__ == '__main__':
    main()
