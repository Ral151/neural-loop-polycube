"""Every loop of one axis at a replayed state: curve, pairing, cos and winding.

The fit saves one checkpoint, so the epoch a comparison is about is read from a
state dict `replay_epochs.py` wrote.  This draws the same figure
`plot_axis_epochs.py` draws for an early epoch, once per state: every loop on
the axis, 100 points at evenly spaced z, each paired to its nearest target
point, coloured by the cosine the direction term reads there, with the loop's
own-centre winding in the title.

  python utils/plot_reverse_fix.py log/axis_seam10_horse_repro \
      horse_points_cubic_lam3 y \
      --state baseline_e50=log/reverse_fix_horse/replay_baseline/state_e50.pt \
      --state frozen_e50=log/reverse_fix_horse/replay_freeze50/state_e50.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import _paths  # noqa: E402
from model import mini_nn
from model.loop_extraction import plane_basis
from plot_epoch_matching import build
from plot_loop_matching import targets_for
from plot_axis_epochs import all_curves, sheet

AXES = 'xyz'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('stem')
    ap.add_argument('axis', choices=tuple(AXES))
    ap.add_argument('--state', action='append', default=[], metavar='LABEL=PATH')
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--columns', type=int, default=6)
    args = ap.parse_args()

    axis_i = AXES.index(args.axis)
    ckpt = torch.load(args.run / f'{args.stem}_model.pt', map_location='cpu',
                      weights_only=False)
    keys = np.asarray(ckpt['keys'])
    axes = np.asarray(ckpt['axes'])
    n = int(ckpt['latent_points'])
    queries = torch.as_tensor(ckpt['queries'])
    axes_t = torch.as_tensor(axes)
    frame = torch.as_tensor(np.asarray(ckpt['plane_point']))
    loops = np.flatnonzero(np.abs(axes).argmax(1) == axis_i)
    idx = (np.arange(args.samples) * n) // args.samples

    ## the fitted model says which target colour group is which slot: an early
    ## curve can be far from its slice and would otherwise swap two loops
    final, _ = mini_nn.load_model(args.run / f'{args.stem}_model.pt')
    final_curves, _ = all_curves(final, queries, frame, n)

    plane_of = {}
    for i in loops:
        plane_of.setdefault((int(keys[i][0]), round(float(keys[i][1]), 2)),
                            []).append(int(i))
    for members in plane_of.values():
        members.sort(key=lambda i: int(keys[i][2]))

    targets = {}
    for i in loops:
        offset = float(keys[i][1])
        members = plane_of[(int(keys[i][0]), round(offset, 2))]
        targets[int(i)] = targets_for(
            args.run, args.axis, offset, members.index(int(i)),
            [final_curves[j].numpy() for j in members])

    for spec in args.state:
        label, path = spec.split('=', 1)
        model = build(ckpt, path, None)
        curves, tangents = all_curves(model, queries, frame, n)
        report = []
        for i in loops:
            pts = curves[int(i)].numpy()
            target, target_n = targets[int(i)]
            pair = cKDTree(target).query(pts[idx])[1]
            cos = mini_nn.normal_cosine(
                tangents[int(i)][idx][None],
                torch.as_tensor(target_n[None, pair]),
                axes_t[i][None])[0].numpy()
            u, v = plane_basis(axes[i] / np.linalg.norm(axes[i]))
            report.append(dict(
                loop=int(i), offset=float(keys[i][1]), u=u, v=v,
                target=target, pts=pts, idx=idx, pair=pair, cos=cos,
                w=float(mini_nn.winding_number(pts, axes[i])[0])))

        clock = sum(r['w'] < -0.5 for r in report)
        against = sum(r['cos'].mean() < -0.1 for r in report)
        mean_cos = float(np.mean([r['cos'].mean() for r in report]))
        note = (f'{args.stem} {args.axis} at {label}: {len(report)} loops, '
                f'{clock} clockwise, {against} against the points\' normals, '
                f'mean cos {mean_cos:+.3f}'
                f'\neach curve: {args.samples} points at evenly spaced z, each '
                f'paired to its nearest target; colour is the direction '
                f'term\'s cos')
        out = args.run / f'matching_{args.axis}_{label}.png'
        sheet(report, args.axis, label, note, out, columns=args.columns)
        print(f'wrote {out}')
        print(f'  {args.axis} {label}: {clock}/{len(report)} clockwise, '
              f'{against} against normals, mean cos {mean_cos:+.3f}, '
              f'median winding {np.median([r["w"] for r in report]):+.3f}')


if __name__ == '__main__':
    main()
