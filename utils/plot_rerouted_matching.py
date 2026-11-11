"""The chamfer pairing the fit uses, and the pairs the normal test rerouted.

`CD_Loss` pairs every predicted point with its nearest target, except that a
pair is forbidden when the curve's implied normal and the target's normal are
more than `--match-angle` apart. A forbidden nearest is replaced by the closest
allowed target -- that replacement is a reroute -- and a point with no allowed
partner is left out of the loss.

This draws both pairings on the final fit: the plain nearest as a hollow dot,
the pair the loss actually used as a filled dot, a grey link where they agree,
a red link where the normal test rerouted it, and a dashed link with no filled
dot where the point was left unmatched.  It also prints the same matched and
rerouted fractions `match_meta` reports, plus the chamfer loss with and without
the normal test, so the number in the run log can be audited loop by loop.

  python utils/plot_rerouted_matching.py log/20261109_armadillo_1000ep \
      armadillo_normal_cubic_lam3 --top 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _paths  # noqa: E402
from model import mini_nn
from model.loop_extraction import LoopExtractor, plane_basis, plane_keys
from plot_first_iteration import arrow

AXES = 'xyz'
GREY, RED, DARK = '0.45', '#d73027', '0.15'


def curves_and_tangents(ckpt, model, frame, n, queries, found):
    """Every loop as the run drew it, in the cloud frame, with its analytic tangent.

    The feature has to be read the way the trainer read it: an
    `occupied_cells` run pools the unique grid nodes the loop's target points
    occupy, so `model.triplane(queries)` -- the centroid sample -- draws a
    different curve. The occupied-node counts are checked against the ones the
    checkpoint stored, which is the same structure the fit was trained on.
    """
    pool = ckpt.get('feature_pool', 'centroid')
    if pool == 'occupied_cells':
        sets = [p for (_, _, p, _) in found.loops]
        cells, masks = mini_nn.cell_pool_structure(model.triplane, sets)
        counts = torch.stack([m.sum(-1).long() for m in masks], dim=-1).numpy()
        saved = ckpt.get('cell_counts')
        if saved is not None and not np.array_equal(counts, np.asarray(saved)):
            raise SystemExit('the re-extracted occupied cells do not match the '
                             'checkpoint cell_counts; pass the run\'s --samples')
        with torch.no_grad():
            feat = model.triplane.pool_cells(cells, masks)
    else:
        with torch.no_grad():
            feat = model.triplane(queries)
        if ckpt.get('cell_counts') is not None:
            raise SystemExit('the checkpoint was trained with occupied cells')
    z = torch.linspace(0, 1, n + 1)[:-1, None]
    phases = ckpt.get('phases')
    outs, tans = [], []
    for i in range(len(ckpt['keys'])):
        zi = z if phases is None else torch.remainder(z + float(phases[i]), 1.0)
        with torch.no_grad():
            curve = model(zi[None], None, feat[i:i + 1])[0][0]
        zz = zi.clone().requires_grad_(True)
        out = model(zz[None], None, feat[i:i + 1])[0]
        ## a bare N x 1 coord drops curve_tangent's batch axis, so this is N x 3
        tans.append(mini_nn.curve_tangent(out, zz).detach())
        outs.append(curve + frame[i][None, :])
    return outs, tans


def pairing(curve, target, n_pred, n_plane, max_angle):
    """Plain nearest and normal-gated nearest, plus the masks the loss uses."""
    curve, target = curve.float(), target.float()
    dm = torch.cdist(curve, target)
    near_plain = dm.argmin(1)
    if max_angle >= 180.0 or n_pred is None or n_plane is None:
        return (near_plain, near_plain, torch.ones_like(near_plain, dtype=torch.bool),
                torch.zeros_like(near_plain, dtype=torch.bool))
    cos = (n_pred / n_pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)) @ (
        n_plane / n_plane.norm(dim=-1, keepdim=True).clamp_min(1e-12)).T
    ok = cos >= np.cos(np.radians(max_angle))
    near_ok = ok.any(1)
    near = torch.where(near_ok, dm.masked_fill(~ok, float('inf')).argmin(1),
                       near_plain)
    return near, near_plain, near_ok, (near != near_plain) & near_ok


def panel(ax, axis, r):
    """One loop: targets, curve, plain pair, gated pair, rerouted links."""
    u, v = r['u'], r['v']
    tgt, pts, idx = r['target'], r['pts'], r['idx']
    p = pts[idx]
    base = r['target'].numpy() @ np.stack([u, v], 1)
    ax.scatter(base[:, 0], base[:, 1], s=1.2, c='0.62', alpha=.8,
               linewidths=0, zorder=1)
    ax.plot(pts @ u, pts @ v, '-', c='tab:orange', lw=1.0, zorder=2)

    pair = r['pair'][idx]
    plain = r['plain'][idx]
    ok = r['ok'][idx]
    moved = r['moved'][idx]
    q = tgt[pair].numpy() @ np.stack([u, v], 1)
    qp = tgt[plain].numpy() @ np.stack([u, v], 1)

    for k in range(len(idx)):
        a, b = p[k] @ np.stack([u, v], 1), q[k]
        colour, ls = ((DARK, '--') if not ok[k] else
                      (RED, '-') if moved[k] else (GREY, '-'))
        ax.plot([a[0], b[0]], [a[1], b[1]], ls, lw=0.75, color=colour,
                alpha=.9, zorder=3)
    ax.scatter(qp[:, 0], qp[:, 1], s=10, facecolors='none', edgecolors=GREY,
               linewidths=.6, zorder=4)
    ax.scatter(q[ok, 0], q[ok, 1], s=7, c='0.2', linewidths=0, zorder=4)
    ax.scatter(q[~ok, 0], q[~ok, 1], s=9, marker='x', c=DARK,
               linewidths=.8, zorder=4)
    ax.scatter(p[:, 0], p[:, 1], s=13, facecolors='none', edgecolors=GREY,
               linewidths=.6, zorder=5)

    arrow(ax, np.stack([pts @ u, pts @ v], 1), 'tab:orange',
          fracs=(0.2, 0.7), span=60, lw=1.2)

    n = len(idx)
    ax.set_title(f"{r['axis']} {r['offset']:+.2f}  "
                 f"rerouted {int(moved.sum())}/{n}, unmatched {int((~ok).sum())}\n"
                 f"CD {r['cd_plain']:.4f} -> {r['cd_gated']:.4f}", fontsize=7)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)
    uv = np.stack([tgt.numpy() @ u, tgt.numpy() @ v], 1)
    P = np.concatenate([uv, pts @ np.stack([u, v], 1)])
    lo, hi = P.min(0), P.max(0)
    pad = 0.08 * (hi - lo).max() + 0.005
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('stem')
    ap.add_argument('--path', type=Path,
                    help='input cloud (default normalized_shapes_cuboid/<stem>.ply)')
    ap.add_argument('--samples', type=int,
                    help='input samples per axis (default: the checkpoint\'s)')
    ap.add_argument('--match-angle', type=float, default=150.0)
    ap.add_argument('--axis', choices=('x', 'y', 'z', 'all'), default='all')
    ap.add_argument('--top', type=int, default=12,
                    help='draw the loops with the most rerouted points')
    ap.add_argument('--draw-samples', type=int, default=100)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    ckpt = torch.load(args.run / f'{args.stem}_model.pt', map_location='cpu',
                      weights_only=False)
    model, _ = mini_nn.load_model(args.run / f'{args.stem}_model.pt')
    keys = np.asarray(ckpt['keys'])
    axes = np.asarray(ckpt['axes'])
    queries = torch.as_tensor(ckpt['queries'])
    axes_t = torch.as_tensor(axes)
    frame = torch.as_tensor(np.asarray(ckpt['plane_point']))
    n = int(ckpt['latent_points'])

    path = args.path or Path('normalized_shapes_cuboid') / f'{args.stem}.ply'
    found = LoopExtractor(path, samples=args.samples or int(ckpt['samples']),
                          out='/tmp/plc_reroute_check')
    found.load().make_planes().cut().find_centres(False).group_loops()
    ext_keys = np.asarray(plane_keys(found.p_normal, found.p_point, found.loops),
                          dtype=np.float64)
    if len(ext_keys) != len(keys) or not np.allclose(ext_keys, keys, atol=1e-6):
        raise SystemExit('the re-extracted plane set does not match the '
                         'checkpoint keys; pass the run\'s --samples/--margin')

    curves, tangents = curves_and_tangents(ckpt, model, frame, n, queries, found)

    ## the run's own report, loop by loop, so the curves drawn here are shown to
    ## be the ones that were fitted before any pairing is read off them
    run_report = None
    csv = args.run / f'{args.stem}_loop_errors.csv'
    if csv.exists():
        import csv as _csv
        with open(csv) as fh:
            run_report = list(_csv.DictReader(fh))
        worst = 0.0
        for i, row in enumerate(run_report):
            target = torch.as_tensor(found.loops[i][2], dtype=torch.float32)
            e = float(torch.cdist(curves[i].float(), target).min(1).values.mean())
            worst = max(worst, abs(e - float(row['mean_err'])))
        print(f'  curves vs {csv.name}: worst per-loop mean error '
              f'difference {worst:.2e}')

    stats = []
    with torch.no_grad():
        for i in range(len(keys)):
            target = torch.as_tensor(found.loops[i][2], dtype=torch.float32)
            tgt_n = found.loops[i][3]
            if tgt_n is None:
                raise SystemExit(f'loop {i}: the input cloud has no normals')
            tgt_n = torch.as_tensor(tgt_n, dtype=torch.float32)
            axis = axes_t[i][None]
            n_plane = mini_nn.project_normals(tgt_n[None], axis)[0]
            pred_n = mini_nn.curve_normal(tangents[i][None], axis)[0]
            curve = curves[i].float()
            near, plain, ok, moved = pairing(curve, target, pred_n, n_plane,
                                             args.match_angle)
            loss_fn = mini_nn.CD_Loss(max_angle=args.match_angle)
            cd_plain, _, _ = loss_fn(curve[None], target[None], None, None)
            cd_gated, near_mod, meta = loss_fn(
                curve[None], target[None], n_plane[None], pred_n[None],
                args.match_angle)
            ## the module and the local pairing must be the same decision
            if args.match_angle < 180.0:
                if not torch.equal(near_mod[0], near):
                    raise SystemExit(f'loop {i}: reroute read differs from CD_Loss')
                if bool((meta['near_ok'] != ok).any()):
                    raise SystemExit(f'loop {i}: match mask differs from CD_Loss')
            stats.append(dict(
                loop=i, axis=AXES[axes[i].argmax()], offset=float(keys[i][1]),
                target=target, n_target=len(target), n_curve=len(curves[i]),
                pts=curves[i].numpy(), pair=near.numpy(), plain=plain.numpy(),
                ok=ok.numpy(), moved=moved.numpy(),
                u=plane_basis(axes[i] / np.linalg.norm(axes[i]))[0],
                v=plane_basis(axes[i] / np.linalg.norm(axes[i]))[1],
                cd_plain=float(cd_plain), cd_gated=float(cd_gated),
                align=(float(run_report[i]['align_err'])
                       if run_report is not None else float('nan')),
                matched=meta['matched'] if meta else 1.0,
                rerouted=meta['rerouted'] if meta else 0.0,
                matched_back=meta['matched_back'] if meta else 1.0,
                rerouted_back=meta['rerouted_back'] if meta else 0.0,
                n_moved=int(moved.sum()), n_unmatched=int((~ok).sum())))

    total = sum(s['n_curve'] for s in stats)
    print(f'{len(stats)} loops, {total} predicted points, '
          f'match angle {args.match_angle:g} deg')
    print(f'  matched {np.mean([s["matched"] for s in stats]):.1%} '
          f'(reverse {np.mean([s["matched_back"] for s in stats]):.1%}), '
          f'rerouted {np.mean([s["rerouted"] for s in stats]):.1%} '
          f'(reverse {np.mean([s["rerouted_back"] for s in stats]):.1%})')
    print(f'  CD per-loop mean  plain {np.mean([s["cd_plain"] for s in stats]):.6f}'
          f'  gated {np.mean([s["cd_gated"] for s in stats]):.6f}'
          f'  ratio {np.mean([s["cd_gated"] / s["cd_plain"] for s in stats]):.3f}')

    picked = stats
    if args.axis != 'all':
        picked = [s for s in picked if s['axis'] == args.axis]
    picked = sorted(picked, key=lambda s: (-s['n_moved'], -s['n_unmatched'],
                                           s['axis'], s['offset']))
    picked = picked[:args.top]
    for s in picked:
        print(f"  loop {s['loop']:3d} {s['axis']}{s['offset']:+.2f}  "
              f"M {s['n_target']}  matched {s['matched']:.1%}  "
              f"rerouted {s['n_moved']}/{s['n_curve']}  "
              f"unmatched {s['n_unmatched']}  "
              f"CD {s['cd_plain']:.6f} -> {s['cd_gated']:.6f}")

    idx = (np.arange(args.draw_samples) * n) // args.draw_samples
    for s in picked:
        s['idx'] = idx
    nx = 6
    ny = max(1, int(np.ceil(len(picked) / nx)))
    fig, axs = plt.subplots(ny, nx, figsize=(3.1 * nx, 3.3 * ny), dpi=150,
                            squeeze=False)
    for k, s in enumerate(picked):
        panel(axs[k // nx, k % nx], s['axis'], s)
    for ax in axs.ravel()[len(picked):]:
        ax.axis('off')

    handles = [Line2D([], [], color='tab:orange'),
               Line2D([], [], color='0.62', marker='o', ms=3, ls=''),
               Line2D([], [], color=GREY, marker='o', mfc='none', ms=4, ls=''),
               Line2D([], [], color=GREY),
               Line2D([], [], color=RED),
               Line2D([], [], color=DARK, ls='--')]
    fig.legend(handles, ['fitted loop', 'target points',
                         'plain nearest (not used)',
                         'pair unchanged by the normal test',
                         'pair rerouted by the normal test',
                         'no allowed target (left out of CD)'],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=6,
               frameon=False, fontsize=9)
    fig.suptitle(
        f"{args.stem}: the {len(picked)} of {len(stats)} loops with the most "
        f"rerouted pairs, {int(args.draw_samples)} evenly spaced z per panel\n"
        f"matched {np.mean([s['matched'] for s in stats]):.1%}, "
        f"rerouted {np.mean([s['rerouted'] for s in stats]):.1%} of all points at "
        f"{args.match_angle:g} deg; CD plain "
        f"{np.mean([s['cd_plain'] for s in stats]):.6f}"
        f" -> gated {np.mean([s['cd_gated'] for s in stats]):.6f}", fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    out = args.out or args.run / 'matching_rerouted.png'
    fig.savefig(out)
    print(f'wrote {out}')

    ## the worst loop on its own, every point linked: one reversed loop can carry
    ## a third of a run's reroutes, and at panel size its links are unreadable
    worst = picked[0]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)
    worst['idx'] = np.arange(len(worst['pts']))
    panel(ax, worst['axis'], worst)
    fig.legend(handles, ['fitted loop', 'target points',
                         'plain nearest (not used)',
                         'pair unchanged by the normal test',
                         'pair rerouted by the normal test',
                         'no allowed target (left out of CD)'],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=False, fontsize=9)
    fig.suptitle(f"{args.stem}: the worst loop, every point linked\n"
                 f"{worst['axis']} {worst['offset']:+.2f}, rerouted "
                 f"{worst['n_moved']}/{worst['n_curve']} "
                 f"({worst['n_moved'] / worst['n_curve']:.1%}) at "
                 f"{args.match_angle:g} deg; align {worst['align']:.2f} "
                 f"(mean cos {1 - worst['align']:+.2f}, 0 is aligned)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out = args.run / 'matching_rerouted_worst.png'
    fig.savefig(out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
