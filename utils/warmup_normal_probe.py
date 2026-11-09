#!/usr/bin/env python3
"""What the cube warm-up does to a loop's normals.

This probe uses the unit-cube mode of the warm-up (`mini_nn.warm_on_cube` with
`cubes=0`): every cube loop is
walked counter-clockwise seen from its plane normal, so the cube fixes a
circulation. This probe asks whether the circulation that transfers to a real
shape agrees with the normals the cloud's points carry, and whether it depends on
where the loop sits. The model is warmed with `mini_nn.warm_on_cube`, which sets
the triplane frame to the cube for the run and restores the shape's afterwards, so
the measurement below reads the field the way the fit will.

For each requested step count a fresh trainer is built, warmed, and measured:

  cos        mean cosine between the curve's implied normal (tangent x plane
             normal) and the normal of the nearest target point, over the loop
  reversed   cos < -0.1, the same rule rosa_main uses to turn a loop
  area       signed area of the decoded curve in its plane; positive means
             counter-clockwise seen from +axis, i.e. the square's own sense

In triplane mode the probe also decodes at random query points and compares the
circulation there with the nearest loop's, which shows whether a centroid away
from a warmed query can come back reversed.

Usage:
    conda run -n seawall python plc_utils/warmup_normal_probe.py \
        --path normalized_shapes_cuboid/horse_points_cubic_lam3.ply \
        --samples 9 --steps 0,30,300 --feature both
"""
import argparse
import csv
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: E402
from plc_model import mini_nn
from plc_model.loop_extraction import LoopExtractor


def unit(a, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)


def plane_basis(axis):
    """An in-plane (u, v) basis, u along the world axis least aligned with `axis`."""
    axis = unit(axis)
    u = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = u - (u @ axis) * axis
    u = unit(u)
    v = np.cross(axis, u)
    return u, v


def signed_area(points, axis):
    """Signed area in the plane; positive is counter-clockwise seen from +axis."""
    u, v = plane_basis(axis)
    p = np.asarray(points, dtype=np.float64)
    x, y = p @ u, p @ v
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def extract(path, samples):
    ex = LoopExtractor(path, samples=samples)
    ex.load().make_planes().cut().find_centres().group_loops()
    loops = ex.loops
    targets = [pts - ex.p_point[s] for s, _, pts, _ in loops]
    normals = [n for *_, n in loops]
    frame = [ex.p_point[s] for s, *_ in loops]
    return ex, loops, targets, normals, frame


def build(ex, targets, normals, frame, feature, seed, align=0.01,
          layers=2, neurons=64):
    if feature == 'triplane':
        extent = np.ptp(ex.points, axis=0)
        pad = 0.02 * float(extent.max())
        lo, hi = ex.points.min(0) - pad, ex.points.max(0) + pad
        tri = dict(centre=0.5 * (lo + hi), half=0.5 * (hi - lo),
                   resolution=32, channels=16, tv=0.01)
        queries = np.stack([frame[i] + targets[i].mean(0)
                            for i in range(len(targets))])
        return mini_nn.Trainer(
            targets, num_models=len(targets), num_neurons=neurons,
            num_layers=layers, normals=normals, align_normal=align,
            seed=seed, feature_mode='triplane', triplane=tri,
            query_points=queries, feature_tv=0.01)
    return mini_nn.Trainer(
        targets, num_models=len(targets), codes=np.arange(len(targets)),
        num_neurons=neurons, num_layers=layers, normals=normals,
        align_normal=align, seed=seed, feature_mode='embedding')


def measure(trainer, ex, targets, normals):
    """One row of normal statistics per loop, in the order the extractor found them."""
    rows = []
    pred_normals = []
    for i, (s, _, pts, _) in enumerate(ex.loops):
        axis = unit(ex.p_normal[s])
        pred = trainer.predict(i).cpu().numpy()
        tangent = trainer.predict_tangent(i)
        n_pred = np.cross(tangent, axis[None, :])
        pred_normals.append(n_pred)

        tgt = np.asarray(targets[i], dtype=np.float64)
        tn = np.asarray(normals[i], dtype=np.float64)
        d = torch.cdist(torch.as_tensor(pred, dtype=torch.float32),
                        torch.as_tensor(tgt, dtype=torch.float32))
        near = d.argmin(1).cpu().numpy()
        paired = tn[near]

        p_hat = unit(n_pred)
        t_hat = unit(paired)
        cos = float(np.einsum('ij,ij->i', p_hat, t_hat).mean())

        tgt_hat = unit(tn)
        axis_dot = tgt_hat @ axis
        in_plane = unit(tgt_hat - axis_dot[:, None] * axis[None, :])
        rows.append(dict(
            loop=i,
            axis='xyz'[int(np.argmax(np.abs(axis)))],
            offset=float(ex.p_point[s][int(np.argmax(np.abs(axis)))]),
            centre=[float(v) for v in pts.mean(0)],
            n_target=int(len(tgt)),
            n_pred=int(len(pred)),
            fit_err=float(d.min(1).values.mean()),
            tgt_axis_cos=float(axis_dot.mean()),
            tgt_inplane=float(np.linalg.norm(
                tgt_hat - axis_dot[:, None] * axis[None, :], axis=1).mean()),
            cos=cos,
            reversed=bool(cos < -0.1),
            area=signed_area(pred, axis),
        ))
    return rows, pred_normals


def decode_at(trainer, xyz, axis):
    """Decode the curve at an arbitrary world query and return points and normals."""
    q = torch.as_tensor(np.asarray(xyz, dtype=np.float32).reshape(1, 3),
                        device=trainer.device)
    feat = trainer.model.triplane.sample(q)
    z = trainer.z[None].clone().requires_grad_(True)
    out = trainer.model(z, None, feat)[0]
    tangent = mini_nn.curve_tangent(out, z)[0].detach().cpu().numpy()
    points = out[0].detach().cpu().numpy()
    return points, np.cross(tangent, unit(axis)[None, :])


def perturb_check(trainer, ex, rows, rng, extent, k=3,
                  scales=(0.01, 0.03, 0.06, 0.10)):
    """Decode at centroids displaced from each loop and compare circulation."""
    centres = np.asarray([r['centre'] for r in rows])
    axes = [unit(ex.p_normal[s]) for s, *_ in ex.loops]
    out = []
    for scale in scales:
        for i, c in enumerate(centres):
            for _ in range(k):
                d = rng.normal(size=3)
                d = d / (np.linalg.norm(d) + 1e-12) * scale * extent
                pts, _ = decode_at(trainer, c + d, axes[i])
                a = signed_area(pts, axes[i])
                out.append(dict(scale=scale, loop=i,
                                distance=float(np.linalg.norm(d)),
                                area=a, same_sign=bool(a * rows[i]['area'] > 0)))
    return out


def random_query_check(trainer, ex, rows, rng, k=300):
    """Decode at random xyz queries and compare circulation with the nearest loop."""
    centres = np.asarray([r['centre'] for r in rows])
    axes = [unit(ex.p_normal[s]) for s, *_ in ex.loops]
    lo, hi = ex.points.min(0), ex.points.max(0)
    queries = rng.uniform(lo, hi, size=(k, 3))
    dist = np.linalg.norm(queries[:, None, :] - centres[None, :, :], axis=-1)
    near = dist.argmin(1)
    out = []
    for q, j in zip(queries, near):
        pts, _ = decode_at(trainer, q, axes[j])
        out.append(dict(
            query=[float(v) for v in q],
            nearest=j,
            distance=float(np.linalg.norm(q - centres[j])),
            area=signed_area(pts, axes[j]),
            same_sign=bool(signed_area(pts, axes[j]) * rows[j]['area'] > 0),
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', default='normalized_shapes_cuboid/'
                                     'horse_points_cubic_lam3.ply')
    ap.add_argument('--samples', type=int, default=9,
                    help='planes per axis for the probe (fewer is faster)')
    ap.add_argument('--steps', default='0,30,300',
                    help='comma-separated warm-up lengths to compare')
    ap.add_argument('--feature', choices=('embedding', 'triplane', 'both'),
                    default='both')
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--neurons', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pretrain-normal', type=float, default=1.0,
                    help='weight of the direction term during the warm-up, held '
                         'against the cube\'s outward face normals; 0 trains '
                         'position and seam only')
    ap.add_argument('--perturb', type=int, default=200,
                    help='random triplane queries per run; 0 to skip')
    ap.add_argument('--out', default='log/warmup_normals')
    args = ap.parse_args()

    steps_list = [int(s) for s in args.steps.split(',') if s.strip()]
    features = ['embedding', 'triplane'] if args.feature == 'both' else [args.feature]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f'extracting {args.samples} planes per axis from {args.path}')
    ex, loops, targets, normals, frame = extract(args.path, args.samples)
    print(f'  {len(loops)} loops on {len(ex.slices)} planes')

    all_rows, random_rows, perturbed_rows = [], [], []
    extent = float(np.ptp(ex.points, axis=0).max())
    for feature in features:
        for steps in steps_list:
            print(f'\n== {feature}, {steps} warm-up steps ==')
            torch.manual_seed(args.seed)
            trainer = build(ex, targets, normals, frame, feature, args.seed,
                            layers=args.layers, neurons=args.neurons)
            log = out / f'pretrain_{feature}_{steps}.log'
            with log.open('w') as fh, redirect_stdout(fh):
                ## the cube warm-up is only defined for the triplane field; in
                ## embedding mode there is nothing to transfer and steps are 0
                err = (mini_nn.warm_on_cube(trainer.model, steps=steps,
                                            align=args.pretrain_normal)['fit']
                       if steps and feature == 'triplane' else float('nan'))
            rows, pred_normals = measure(trainer, ex, targets, normals)
            for r in rows:
                r.update(feature=feature, steps=steps, pretrain_err=err)
            all_rows += rows
            print(f'  mapping error {err:.4f}' if steps else '  no warm-up')
            print(f'  reversed {sum(r["reversed"] for r in rows)}/{len(rows)}, '
                  f'median cos {np.median([r["cos"] for r in rows]):+.3f}, '
                  f'area<0 {sum(r["area"] < 0 for r in rows)}/{len(rows)}')
            for axis in 'xyz':
                sub = [r for r in rows if r['axis'] == axis]
                if sub:
                    print(f'    {axis}: {sum(r["reversed"] for r in sub)}/{len(sub)} '
                          f'reversed, median cos '
                          f'{np.median([r["cos"] for r in sub]):+.3f}, '
                          f'area<0 {sum(r["area"] < 0 for r in sub)}/{len(sub)}')

            if feature == 'triplane' and args.perturb:
                check = random_query_check(trainer, ex, rows, rng, args.perturb)
                for c in check:
                    c.update(steps=steps)
                random_rows += check
                same = sum(c['same_sign'] for c in check) / len(check)
                print(f'  random queries: {same:.1%} keep the nearest loop\'s '
                      f'circulation sign ({len(check)} draws)')

                near = perturb_check(trainer, ex, rows, rng, extent)
                for p in near:
                    p['steps'] = steps
                perturbed_rows += near
                for scale in sorted({p['scale'] for p in near}):
                    sub = [p for p in near if p['scale'] == scale]
                    frac = sum(p['same_sign'] for p in sub) / len(sub)
                    print(f'  centroid + {scale * extent:.3f} world units '
                          f'({scale:.0%} extent): {frac:.1%} keep the sign '
                          f'({len(sub)} draws)')

    csv_path = out / 'warmup_normals.csv'
    fields = ['feature', 'steps', 'loop', 'axis', 'offset', 'centre', 'n_target',
              'n_pred', 'fit_err', 'tgt_axis_cos', 'tgt_inplane', 'cos',
              'reversed', 'area', 'pretrain_err']
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            row = dict(r)
            row['centre'] = ' '.join(f'{v:.6f}' for v in r['centre'])
            w.writerow(row)

    if random_rows:
        with (out / 'random_queries.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['steps', 'query', 'nearest',
                                               'distance', 'area', 'same_sign'])
            w.writeheader()
            for c in random_rows:
                row = dict(c)
                row['query'] = ' '.join(f'{v:.6f}' for v in c['query'])
                w.writerow(row)

    if perturbed_rows:
        with (out / 'perturbed_queries.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['steps', 'scale', 'loop',
                                               'distance', 'area', 'same_sign'])
            w.writeheader()
            w.writerows(perturbed_rows)

    (out / 'summary.json').write_text(json.dumps(dict(
        path=args.path, samples=args.samples, steps=steps_list,
        features=features, rows=all_rows), indent=1))

    print(f'\nwrote {csv_path}')
    print('A reversed loop is cos < -0.1, the turn rule rosa_main applies.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
