"""Seam agreement of a fitted checkpoint: the seam distance D of neighbouring loops.

D (the quantity `--seam-align` and `--distill` act on) is measured on the run's
own *_feature_graph.csv pairs, at the checkpoint's seam phases when it has them
and at z = 0. `--against` compares pair by pair with another run on the same planes.

    python main.py seam-check log/my_run/<stem>_model.pt
    python main.py seam-check --against log/other_run/<stem>_model.pt log/my_run/<stem>_model.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model.trainer import load_model


def read_graph(path, keys):
    """Pairs of loop indices, matched to this checkpoint by (axis, offset, slot)."""
    index = {tuple(np.round(np.asarray(k, dtype=np.float64), 6)): i
             for i, k in enumerate(keys)}
    pairs = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            axis = int(row['axis'])
            src = index.get((axis, round(float(row['src_offset']), 6),
                             int(row['src_slot'])))
            tgt = index.get((axis, round(float(row['tgt_offset']), 6),
                             int(row['tgt_slot'])))
            if src is None or tgt is None:
                continue
            pairs.append((src, tgt))
    return np.asarray(pairs, dtype=np.int64)


def curves_of(model, meta):
    """The network on the checkpoint's own parameter grid, one curve per loop."""
    n = int(meta['latent_points'])
    count = int(meta['num_models'])
    z = torch.linspace(0, 1, n + 1)[:-1, None]
    with torch.no_grad():
        if meta['feature_mode'] == 'triplane':
            if meta.get('queries') is None:
                raise SystemExit('this checkpoint has no queries; re-run main.py fit to '
                                 'save them')
            query = torch.as_tensor(np.asarray(meta['queries']), dtype=torch.float32)
            feat = model.triplane(query)
            axis = None
            if meta.get('axis_condition'):
                axes = meta.get('axes')
                if axes is None:
                    raise SystemExit('this checkpoint is axis-conditioned but carries '
                                     'no axes; re-run main.py fit to save them')
                axis = torch.as_tensor(np.asarray(axes), dtype=torch.float32)
            return model(z[None].expand(count, -1, -1), None, feat, axis).detach()
        codes = torch.as_tensor(np.asarray(meta['codes']), dtype=torch.int64)
        return model(z[None].expand(count, -1, -1), codes).detach()


def seam_distance(curves, pairs, phase=None):
    """D per pair at the given phase, exactly as Trainer.align_phases reads it."""
    n = curves.shape[1]
    centred = curves.mean(1, keepdim=True)
    if phase is None:
        shift = torch.zeros(curves.shape[0], dtype=curves.dtype)
    else:
        shift = torch.as_tensor(np.asarray(phase), dtype=curves.dtype)
    u = torch.remainder(torch.linspace(0, 1, n + 1)[:-1][None, :]
                        + shift[:, None], 1.0) * n
    lo = torch.floor(u)
    k0 = lo.long() % n
    w = (u - lo)[..., None]
    drawn = (curves.gather(1, k0[..., None].expand(-1, -1, 3)) * (1.0 - w)
             + curves.gather(1, ((k0 + 1) % n)[..., None].expand(-1, -1, 3)) * w
             - centred)
    vi, vj = drawn[pairs[:, 0]], drawn[pairs[:, 1]]
    s2 = 0.5 * ((vi ** 2).sum(-1).mean(1) + (vj ** 2).sum(-1).mean(1))
    return (((vi - vj) ** 2).sum(-1).mean(1) / s2.clamp_min(1e-12)).numpy()


def summary(d):
    """One-line statistics of a D array."""
    return (f'median {np.median(d):.4f}  mean {d.mean():.4f}  '
            f'p90 {np.percentile(d, 90):.4f}  max {d.max():.4f}')


def describe(path):
    """Load a checkpoint and its graph, measure D and return the report lines."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model, meta = load_model(path)
    graph = Path(path).with_name(Path(path).name.replace('_model.pt', '_feature_graph.csv'))
    if not graph.is_file():
        raise SystemExit(f'{path}: no graph beside the checkpoint; expected {graph}')
    keys = np.asarray(ckpt['keys'], dtype=np.float64)
    pairs = read_graph(graph, keys)
    if not len(pairs):
        raise SystemExit(f'{path}: the graph matched none of the checkpoint keys')
    curves = curves_of(model, meta)
    lines = [f'{path}', f'  {len(pairs)} pairs from {graph.name}']
    at_zero = seam_distance(curves, pairs, None)
    if ckpt.get('phases') is None:
        lines.append(f'  D at z = 0 (exported): {summary(at_zero)}')
        exported = at_zero
    else:
        at_phase = seam_distance(curves, pairs, ckpt['phases'])
        lines.append(f'  D at the checkpoint phase (exported): {summary(at_phase)}')
        lines.append(f'  D at z = 0 (phase-free copy):         {summary(at_zero)}')
        exported = at_phase
    return dict(path=path, pairs=pairs, exported=exported,
                phases=ckpt.get('phases'), keys=keys, lines=lines)


def main(argv=None):
    """Print the seam report of a checkpoint, optionally compared with another."""
    parser = argparse.ArgumentParser(prog='main.py seam-check',
                                     description=__doc__.splitlines()[0])
    parser.add_argument('model', help='a *_model.pt written by main.py fit')
    parser.add_argument('--against', default=None, metavar='MODEL',
                        help='another *_model.pt on the same plane set; the per-pair '
                             'comparison is on the shared keys')
    args = parser.parse_args(argv)

    refit = describe(args.model)
    print('\n'.join(refit['lines']))

    if args.against:
        other = describe(args.against)
        print('\n'.join(other['lines']))
        # match pairs by loop key, not index, so a reordered graph still lines up
        other_at = {tuple(np.round(k, 6)): m for m, k in enumerate(other['keys'])}
        worse, compared = [], 0
        for (i, j), dij in zip(refit['pairs'], refit['exported']):
            ki = tuple(np.round(refit['keys'][i], 6))
            kj = tuple(np.round(refit['keys'][j], 6))
            ti, tj = other_at.get(ki), other_at.get(kj)
            if ti is None or tj is None:
                continue
            hit = np.where((other['pairs'][:, 0] == ti) & (other['pairs'][:, 1] == tj))[0]
            if not len(hit):
                hit = np.where((other['pairs'][:, 0] == tj) & (other['pairs'][:, 1] == ti))[0]
            if not len(hit):
                continue
            other_d = float(other['exported'][hit[0]])
            compared += 1
            if dij > other_d + 0.02:
                worse.append((ki, kj, float(dij), other_d))
        print(f'  compared {compared} pairs against {args.against}: '
              f'{len(worse)} worse by more than 0.02')
        for (ki, kj, dij, td) in sorted(worse, key=lambda r: -r[2])[:10]:
            print(f'    {ki} - {kj}: D {dij:.4f} against other {td:.4f}')
