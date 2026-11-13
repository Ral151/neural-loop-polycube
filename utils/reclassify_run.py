"""Reclassify a finished run against the current export criteria, without fitting.

The split PLYs carry the verdict the run reached at its end. When the criteria
change afterwards, the curves do not have to be learned again: the checkpoint
keeps the network, the field and every loop's key and plane frame, and the input
cloud still holds the targets. This rebuilds the same loops, replays the saved
weights, recomputes every per-loop error and rewrites

  <stem>_good_loops.ply, <stem>_bad_loops.ply, <stem>_loop_split.txt,
  <stem>_loop_errors.csv

The replay is checked against the old split PLYs first, so a mismatch in the
extraction or the checkpoint stops the run instead of overwriting the files.

  python utils/reclassify_run.py log/matchangle_noaxis_150_dancing_children100K
  python utils/reclassify_run.py log/my_run --bad-loop-err 0.02 --max-align-err 0.5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import _paths  # noqa: E402
from model import mini_nn
from model.loop_extraction import (AXIS_COLORS, LoopExtractor, plane_keys,
                                       write_points)

BAD_POINT_ERR = 0.15
BAD_LOOP_ERR = 0.015
BAD_ALIGN_ERR = 0.0


def find_stem(run: Path, stem: str | None) -> str:
    if stem:
        return stem
    found = sorted(run.glob('*_model.pt'))
    if len(found) != 1:
        raise SystemExit(f'expected one *_model.pt in {run}, found {len(found)}; '
                         f'name one with --stem')
    return found[0].name[:-len('_model.pt')]


def input_path(run: Path, stem: str, override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        ## runs with a header name the cloud on the second line; older ones open
        ## with it, so read the header lines rather than a fixed index
        lines = (run / 'run.txt').read_text().splitlines()
        names = [line.strip() for line in lines[:5]
                 if line.strip().lower().endswith(('.ply', '.obj', '.stl', '.off'))]
        if not names:
            raise SystemExit(f'{run}/run.txt holds no input path; use --path')
        path = Path(names[0])
    if path.exists():
        return path
    raise SystemExit(f'input cloud {path} is missing; pass --path')


def rebuild_loops(path: Path, ckpt, samples: int | None, margin: float,
                  band: float, max_align: float):
    """The run's loops and targets, in the same order as the checkpoint keys."""
    found = LoopExtractor(path, samples=samples or int(ckpt['samples']),
                          margin=margin, band=band, max_align=max_align)
    found.load().make_planes().cut().find_centres(False).group_loops()
    loops = found.loops
    targets = [pts - found.p_point[s] for s, _, pts, _ in loops]
    normals = [n for *_, n in loops]
    frame = [found.p_point[s] for s, _, _, _ in loops]
    keys = plane_keys(found.p_normal, found.p_point, loops)

    saved = np.asarray(ckpt['keys'], dtype=np.float64)
    if len(keys) != len(saved):
        raise SystemExit(f'rebuilt {len(keys)} loops, checkpoint holds {len(saved)}; '
                         f'the extraction settings do not match the run')
    ## offsets are stored unrounded, but a rebuild can move the last bit; the key
    ## is (axis, offset, slot), so compare at a tolerance far below a plane spacing
    same = (np.array([k[0] for k in keys]) == saved[:, 0].astype(int)).all()
    same = same and np.allclose([k[1] for k in keys], saved[:, 1], atol=1e-9)
    same = same and (np.array([k[2] for k in keys]) == saved[:, 2].astype(int)).all()
    if not same:
        raise SystemExit('rebuilt loop keys do not match the checkpoint; the run '
                         'was cut differently, so the replay cannot be trusted')
    return found, loops, targets, normals, frame


def build_trainer(ckpt, targets, normals, frame):
    """A Trainer in the run's own configuration, with the saved weights loaded."""
    query_sets = [frame[i] + targets[i] for i in range(len(targets))]
    queries = np.stack([frame[i] + targets[i].mean(0)
                        for i in range(len(targets))])
    ## checkpoints written before the pooled feature existed hold no key; they
    ## trained on the centroid
    feature_pool = ckpt.get('feature_pool', 'centroid')
    trainer = mini_nn.Trainer(
        targets, num_models=int(ckpt['num_models']),
        latent_points=int(ckpt['latent_points']),
        num_neurons=int(ckpt['num_neurons']),
        num_layers=int(ckpt.get('num_layers', 2)),
        num_freqs=int(ckpt.get('num_freqs', 1)),
        normals=normals,
        feature_mode='triplane', triplane=dict(ckpt['triplane']),
        query_points=queries,
        query_sets=query_sets if feature_pool == 'occupied_cells' else None,
        feature_pool=feature_pool)
    trainer.model.load_state_dict(ckpt['state_dict'], strict=False)
    trainer.model.eval()
    return trainer


def predict(trainer, loops, frame, p_normal):
    """The run's fitted curves and normals, exactly as `fit_loops.fitted` reads them."""
    local = [trainer.predict(i).cpu().numpy() for i in range(len(loops))]
    tangents = [trainer.predict_tangent(i) for i in range(len(loops))]
    normals = []
    for t, (s, *_) in zip(tangents, loops):
        axis = p_normal[s] / (np.linalg.norm(p_normal[s]) + 1e-12)
        normals.append(np.cross(t, axis[None, :]))
    preds = [p + frame[i] for i, p in enumerate(local)]
    return local, preds, normals, tangents


def measure(local, targets, normals, tangents, trainer, p_normal, loops):
    """Per-loop point error and the paired normals, as `fit_loops.fitted` does."""
    errors, paired = [], []
    for p, t, n in zip(local, targets, normals):
        d, near = cKDTree(t).query(p)
        errors.append(d)
        paired.append(n[near])

    axes = trainer.axes
    if axes is None:
        axes = torch.from_numpy(
            np.stack([p_normal[s] for s, *_ in loops])).float().to(trainer.device)
    d1 = torch.from_numpy(np.stack(tangents)).float().to(trainer.device)
    pair = torch.from_numpy(np.stack(paired)).float().to(trainer.device)
    align = mini_nn.normal_penalty(d1, pair, axes, per_curve=True).cpu().numpy()
    return errors, align


def check_against_old(run: Path, stem: str, old_split, replay_blocks):
    """Compare the replay to the PLYs the run wrote, before replacing them.

    Older runs reversed some loops before export; a loop that matches only when
    read backwards is that flip, not a different fit. Returns the worst per-loop
    difference and how many loops the old file had reversed.
    """
    good_path = run / f'{stem}_good_loops.ply'
    bad_path = run / f'{stem}_bad_loops.ply'
    if not good_path.exists() or not bad_path.exists() or not old_split:
        return None, 0
    good = np.asarray(trimesh.load(good_path, process=False).vertices)
    bad = np.asarray(trimesh.load(bad_path, process=False).vertices)
    at = {'g': 0, 'b': 0}
    src = {'g': good, 'b': bad}
    worst, turned = 0.0, 0
    for flag, block in zip(old_split, replay_blocks):
        n = len(block)
        old = src[flag][at[flag]:at[flag] + n]
        at[flag] += n
        if old.shape != block.shape:
            return float('nan'), turned
        forward = float(np.abs(old - block).max())
        backward = float(np.abs(old[::-1] - block).max())
        worst = max(worst, min(forward, backward))
        turned += backward < forward
    return worst, turned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='finished run folder')
    ap.add_argument('--stem', help='output stem (default: the one *_model.pt)')
    ap.add_argument('--path', help='input cloud; default: second line of run.txt')
    ap.add_argument('--samples', type=int,
                    help='planes per axis; default: the checkpoint\'s own')
    ap.add_argument('--margin', type=float, default=0.1)
    ap.add_argument('--band', type=float, default=0.0075)
    ap.add_argument('--max-align', type=float, default=0.95)
    ap.add_argument('--point-err', type=float, default=BAD_POINT_ERR)
    ap.add_argument('--bad-loop-err', type=float, default=BAD_LOOP_ERR)
    ap.add_argument('--max-align-err', type=float, default=BAD_ALIGN_ERR)
    ap.add_argument('--out', type=Path,
                    help='where to write; default: the run folder')
    args = ap.parse_args()

    run = args.run
    stem = find_stem(run, args.stem)
    ckpt = torch.load(run / f'{stem}_model.pt', map_location='cpu',
                      weights_only=False)
    path = input_path(run, stem, args.path)
    found, loops, targets, normals, frame = rebuild_loops(
        path, ckpt, args.samples, args.margin, args.band, args.max_align)
    trainer = build_trainer(ckpt, targets, normals, frame)
    local, preds, pred_normals, tangents = predict(
        trainer, loops, frame, found.p_normal)
    errors, align = measure(local, targets, normals, tangents, trainer,
                            found.p_normal, loops)

    old_split = ''
    split_path = run / f'{stem}_loop_split.txt'
    if split_path.exists():
        old_split = split_path.read_text().strip()
    diff, turned = check_against_old(run, stem, old_split, preds)
    if diff is not None:
        if not np.isfinite(diff) or diff > 1e-3:
            raise SystemExit(f'replay differs from the saved PLYs by {diff:g}; '
                             f'refusing to reclassify')
        note = (f', {turned} of them reversed in the old file' if turned else '')
        print(f'  replay matches the saved PLYs (max difference {diff:.2e}{note})')

    extent = np.ptp(found.points, axis=0).max()
    loop_limit = args.bad_loop_err * extent
    mean = np.array([e.mean() for e in errors])
    bad_point = np.array([e.max() > args.point_err for e in errors])
    bad_mean = (mean > loop_limit if args.bad_loop_err > 0.0
                else np.zeros(len(errors), dtype=bool))
    bad_align = (align > args.max_align_err if args.max_align_err > 0.0
                 else np.zeros(len(errors), dtype=bool))
    bad_loops = bad_point | bad_mean | bad_align
    bad = np.repeat(bad_loops, [len(e) for e in errors])

    keys = np.asarray(ckpt['keys'])
    axis = np.repeat(keys[:, 0].astype(int), [len(e) for e in errors])
    points = np.concatenate(preds)
    loop_normals = np.concatenate(pred_normals)
    good_colors = AXIS_COLORS[axis]
    bad_colors = np.tile(np.array([255, 0, 0, 255], np.uint8), (len(bad), 1))

    out = args.out or run
    out.mkdir(parents=True, exist_ok=True)
    for name, keep, cols in (('good', ~bad, good_colors),
                             ('bad', bad, bad_colors)):
        dest = out / f'{stem}_{name}_loops.ply'
        if keep.any():
            write_points(dest, points[keep], cols[keep], loop_normals[keep])
        else:
            print(f'  no {name} loops: {dest} not written')
    (out / f'{stem}_loop_split.txt').write_text(
        ''.join('b' if b else 'g' for b in bad_loops) + '\n')

    with open(out / f'{stem}_loop_errors.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['axis', 'offset', 'slot', 'points', 'mean_err', 'max_err',
                         'align_err', 'bad_point', 'bad_mean', 'bad_align', 'bad'])
        for key, e, m, a, bp, bm, ba, b in zip(
                keys, errors, mean, align, bad_point, bad_mean, bad_align,
                bad_loops):
            writer.writerow([int(key[0]), f'{key[1]:.4f}', int(key[2]), len(e),
                             f'{m:.6f}', f'{float(e.max()):.6f}', f'{float(a):.6f}',
                             int(bp), int(bm), int(ba), int(b)])

    print(f'  {len(errors)} loops, mean error {mean.min():.3f}-{mean.max():.3f}, '
          f'bad past {loop_limit:.3f}' if args.bad_loop_err > 0.0 else
          f'  {len(errors)} loops, mean error {mean.min():.3f}-{mean.max():.3f}, '
          f'mean limit off')
    print(f'  {int(bad_loops.sum())} bad loops: '
          f'{int(bad_point.sum())} point, {int(bad_mean.sum())} mean, '
          f'{int(bad_align.sum())} alignment (a loop can raise more than one)')
    print(f'  -> {out / f"{stem}_good_loops.ply"}, '
          f'{out / f"{stem}_bad_loops.ply"}, '
          f'{out / f"{stem}_loop_split.txt"}, '
          f'{out / f"{stem}_loop_errors.csv"}')


if __name__ == '__main__':
    main()
