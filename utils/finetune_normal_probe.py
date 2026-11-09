#!/usr/bin/env python3
"""Track loop normals from the warmed state through fine-tuning.

The warm-up probe shows the cube warm-up hands over a consistent circulation.  This
script follows that state into the fit, where the normal term is annealed to
zero, and records how many loops run against their target normals as the epochs
pass.

Usage:
    conda run -n seawall python plc_utils/finetune_normal_probe.py \
        --path normalized_shapes_cuboid/horse_points_cubic_lam3.ply \
        --samples 29 --warmup 30 --epochs 1000 --every 100
"""
import argparse
import csv
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: E402
from warmup_normal_probe import build, extract, measure


def summarise(rows, epoch):
    out = []
    for axis in 'xyz':
        sub = [r for r in rows if r['axis'] == axis]
        if sub:
            out.append(dict(
                epoch=epoch, axis=axis, loops=len(sub),
                reversed=sum(r['reversed'] for r in sub),
                median_cos=float(np.median([r['cos'] for r in sub])),
                area_negative=sum(r['area'] < 0 for r in sub),
            ))
    out.append(dict(epoch=epoch, axis='all', loops=len(rows),
                    reversed=sum(r['reversed'] for r in rows),
                    median_cos=float(np.median([r['cos'] for r in rows])),
                    area_negative=sum(r['area'] < 0 for r in rows)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default='normalized_shapes_cuboid/'
                                     'horse_points_cubic_lam3.ply')
    ap.add_argument('--samples', type=int, default=29)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--normal', type=float, default=0.01,
                    help='fine-tune direction weight, annealed to zero over the run; '
                         '0.01 is the pipeline default')
    ap.add_argument('--every', type=int, default=100)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--neurons', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pretrain-normal', type=float, default=1.0,
                    help='weight of the direction term during the warm-up, held '
                         'against the cube\'s outward face normals; 0 trains '
                         'position and seam only')
    ap.add_argument('--out', default='log/finetune_normals')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'extracting {args.samples} planes per axis from {args.path}')
    ex, loops, targets, normals, frame = extract(args.path, args.samples)
    print(f'  {len(loops)} loops')

    torch.manual_seed(args.seed)
    trainer = build(ex, targets, normals, frame, 'triplane', args.seed,
                    align=args.normal,
                    layers=args.layers, neurons=args.neurons)
    with (out / 'warmup.log').open('w') as fh, redirect_stdout(fh):
        report = mini_nn.warm_on_cube(trainer.model, steps=args.warmup,
                                      align=args.pretrain_normal)
    print(f'  warm-up {args.warmup} steps, fit {report["fit"]:.4f}, direction '
          f'{report["cos"]:+.3f} cos')

    trace = []
    rows, _ = measure(trainer, ex, targets, normals)
    trace += summarise(rows, 0)
    print(f'  epoch 0: {sum(r["reversed"] for r in rows)}/{len(rows)} reversed, '
          f'median cos {np.median([r["cos"] for r in rows]):+.3f}')

    def watch(n):
        rr, _ = measure(trainer, ex, targets, normals)
        trace.extend(summarise(rr, n))
        print(f'  epoch {n:>4}: {sum(r["reversed"] for r in rr)}/{len(rr)} '
              f'reversed, median cos {np.median([r["cos"] for r in rr]):+.3f}, '
              f'area<0 {sum(r["area"] < 0 for r in rr)}', file=sys.__stdout__)

    with (out / 'train.log').open('w') as fh, redirect_stdout(fh):
        trainer.train(epochs=args.epochs, every=args.every, on_epoch=watch)

    with (out / 'epochs.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'axis', 'loops', 'reversed',
                                           'median_cos', 'area_negative'])
        w.writeheader()
        w.writerows(trace)
    ## the last state per loop, so the loops that end reversed can be named and
    ## checked against their size rather than only counted
    final_rows, _ = measure(trainer, ex, targets, normals)
    with (out / 'final_loops.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(final_rows[0]))
        w.writeheader()
        w.writerows(final_rows)
    print(f'wrote {out / "epochs.csv"} and {out / "final_loops.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
