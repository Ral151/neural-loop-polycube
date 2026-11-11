"""Count reversed loops and measure the fit of replayed states.

The runs keep one checkpoint at the end, so a comparison at an early epoch is
made from state dicts `replay_epochs.py` wrote.  For each state this draws every
loop, pairs it to its own target slice, and reports, per slice axis and in
total:

  reversed     loops whose own-centre winding is negative, i.e. the loop as
               drawn runs clockwise seen from +axis.  This is the measure that
               does not depend on where a sampled target's centroid falls, which
               is what the trainer's per-epoch diagnostic uses and why it
               undercounts a folded or off-centre slice
  winding      mean signed winding over the axis (+1 counter-clockwise)
  fold         mean of sum|dtheta|/2pi; 1 is a clean single turn, above that the
               curve retraces part of itself
  against      loops whose mean direction cosine is below -0.1, the vote the
               export uses.  Undercounts a folded family, so read it next to the
               winding rather than instead of it
  mean cos     mean over loops of the direction term's cosine
  fit          squared chamfer over the full target slice, predicted -> target
               plus target -> predicted, averaged over loops.  Full targets make
               the number independent of which points a training epoch sampled

  python utils/reverse_fix_report.py log/axis_seam10_horse_repro \
      horse_points_cubic_lam3 \
      --state baseline=log/reverse_fix_horse/replay_baseline/state_e50.pt \
      --state frozen50=log/reverse_fix_horse/replay_freeze50/state_e50.pt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import _paths  # noqa: E402
from model import mini_nn
from plot_epoch_matching import build
from plot_loop_matching import targets_for
from plot_axis_epochs import all_curves

AXES = 'xyz'


def plane_members(keys, loops):
    """Each plane's loops, in slot order, keyed by (axis index, offset)."""
    groups = {}
    for i in loops:
        groups.setdefault((int(keys[i][0]), round(float(keys[i][1]), 2)),
                          []).append(int(i))
    for members in groups.values():
        members.sort(key=lambda i: int(keys[i][2]))
    return groups


def evaluate(ckpt, state, run, stem):
    """One state: per-loop winding, direction and full-target chamfer."""
    model = build(ckpt, str(state), None)
    keys = np.asarray(ckpt['keys'])
    axes = np.asarray(ckpt['axes'])
    n = int(ckpt['latent_points'])
    queries = torch.as_tensor(ckpt['queries'])
    axes_t = torch.as_tensor(axes)
    frame = torch.as_tensor(np.asarray(ckpt['plane_point']))

    ## the assignment of target colour groups to slots uses the fitted model:
    ## an early curve can still be far from its slice, and a panel that swaps two
    ## loops of the same plane would report the wrong target for both
    final, _ = mini_nn.load_model(run / f'{stem}_model.pt')
    final_curves, _ = all_curves(final, queries, frame, n)
    curves, tangents = all_curves(model, queries, frame, n)

    loops = list(range(len(keys)))
    groups = plane_members(keys, loops)
    rows = []
    for i in loops:
        axis_i = int(np.abs(axes[i]).argmax())
        offset = float(keys[i][1])
        members = groups[(axis_i, round(offset, 2))]
        target, target_n = targets_for(
            run, AXES[axis_i], offset, members.index(i),
            [final_curves[j].numpy() for j in members])
        pts = curves[i].numpy()
        near = cKDTree(target).query(pts)[1]
        d_out = np.linalg.norm(pts - target[near], axis=1)
        d_in = cKDTree(pts).query(target)[0]
        cos = mini_nn.normal_cosine(
            tangents[i][None],
            torch.as_tensor(target_n[None, near]),
            axes_t[i][None])[0].numpy()
        w, fold = mini_nn.winding_number(pts, axes[i])
        rows.append(dict(
            loop=i, axis=AXES[axis_i], offset=offset,
            winding=float(w), fold=float(fold),
            mean_cos=float(cos.mean()), cos_neg=float((cos < 0).mean()),
            against=bool(cos.mean() < -0.1),
            fit_cd=float((d_out ** 2).mean() + (d_in ** 2).mean()),
            fit_out=float((d_out ** 2).mean()),
            dist_out=float(d_out.mean())))
    return rows


def summarise(rows, label):
    out = []
    for name in AXES + 'all':
        group = rows if name == 'all' else [r for r in rows if r['axis'] == name]
        if not group:
            continue
        out.append(dict(
            state=label, axis=name, loops=len(group),
            reversed=sum(r['winding'] < 0 for r in group),
            against=sum(r['against'] for r in group),
            winding=np.mean([r['winding'] for r in group]),
            fold=np.mean([r['fold'] for r in group]),
            mean_cos=np.mean([r['mean_cos'] for r in group]),
            cos_neg=np.mean([r['cos_neg'] for r in group]),
            fit_cd=np.mean([r['fit_cd'] for r in group]),
            fit_out=np.mean([r['fit_out'] for r in group]),
            dist_out=np.mean([r['dist_out'] for r in group])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path,
                    help='a finished run: supplies the network configuration, '
                         'the loop keys and the target slices')
    ap.add_argument('stem')
    ap.add_argument('--state', action='append', default=[], metavar='LABEL=PATH',
                    help='a replayed state dict, in table order')
    ap.add_argument('--out', type=Path,
                    help='CSV of the summary (default <run>/reverse_fix_report.csv)')
    args = ap.parse_args()

    ckpt = torch.load(args.run / f'{args.stem}_model.pt', map_location='cpu',
                      weights_only=False)
    summary, per_loop = [], []
    for spec in args.state:
        label, path = spec.split('=', 1)
        rows = evaluate(ckpt, Path(path), args.run, args.stem)
        per_loop += [dict(state=label, **r) for r in rows]
        summary += summarise(rows, label)

    header = ['state', 'axis', 'loops', 'reversed', 'against', 'winding',
              'fold', 'mean_cos', 'cos_neg', 'fit_cd', 'fit_out', 'dist_out']
    print(f"{'state':<16} {'axis':<4} {'loops':>5} {'rev':>4} {'against':>7} "
          f"{'winding':>8} {'fold':>6} {'cos':>7} {'fit_cd':>10} "
          f"{'fit_out':>10} {'dist_out':>9}")
    for row in summary:
        print(f"{row['state']:<16} {row['axis']:<4} {row['loops']:>5} "
              f"{row['reversed']:>4} {row['against']:>7} "
              f"{row['winding']:>8.3f} {row['fold']:>6.2f} "
              f"{row['mean_cos']:>+7.3f} {row['fit_cd']:>10.6f} "
              f"{row['fit_out']:>10.6f} {row['dist_out']:>9.5f}")

    out = args.out or args.run / 'reverse_fix_report.csv'
    with open(out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(summary)
    loops_path = out.with_name(out.stem + '_loops.csv')
    with open(loops_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_loop[0]))
        writer.writeheader()
        writer.writerows(per_loop)
    print(f'wrote {out} and {loops_path}')


if __name__ == '__main__':
    main()
