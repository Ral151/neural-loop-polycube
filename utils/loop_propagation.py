"""Loop propagation test: stable loops -> propagated tracks -> aligned tubes.

Standalone implementation of `loop_propagation.md`.  It reads a finished loop
run's per-plane PLYs and `planes.txt` and imports only `align_loops.py`; nothing
here is imported by `loop_extraction.py` or `plc_main.py`.

Steps
-----
1. per-loop in-plane RMS against its own labelled slice points, with the stable
   cut and the diagnostic best switch to an adjacent plane;
2. two-sided translation transfer from every stable loop to the components on
   adjacent planes, null-calibrated, grown into tracks and sorted into kept,
   adopted, discarded and orphan loops;
2b. optional: the same propagation where the seed may also deform along its
   in-plane normal (low-order Fourier modes) within a fixed budget; the
   target's own fitted loop still continues the track;
3. `align_loops.synchronize` on every track, split at switched-off pairs;
4. loft every aligned sub-chain of two or more loops and export the tubes.

Usage
-----
    python utils/loop_propagation.py --step 1
    python utils/loop_propagation.py --step 2
    python utils/loop_propagation.py --step 2b
    python utils/loop_propagation.py --step 3
    python utils/loop_propagation.py --step 4
    python utils/loop_propagation.py --all
    python utils/loop_propagation.py --step all --good-only   # good loops only

Each step reads the previous step's JSON and writes its own directory under
`log/loop_propagation/stepN/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgb

from scipy.spatial import cKDTree

import _paths  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
from align_loops import (circular_shift, loop_coordinates, pair_residual,
                         switch, synchronize)

AX = {'x': 0, 'y': 1, 'z': 2}
OTHER = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}
REF = {'x': 1, 'y': 0, 'z': 0}          # world axis used for in-plane e1
TAU = 0.02
K = 256

PALETTE = ['#1f77b4', '#d95f02', '#2ca02c', '#9467bd', '#8c564b',
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e',
           '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']


# --------------------------------------------------------------------------- #
# run loading and geometry helpers
# --------------------------------------------------------------------------- #
def parse_planes(run: Path):
    rows = []
    for line in (run / 'planes.txt').read_text().splitlines():
        m = re.match(r'plane_(\d+) ([xyz]) ([+-][\d.]+) \| (\d+) pts \| '
                     r'(\d+) loops', line)
        if m:
            n = int(m.group(5))
            ## a slice too thin to hold a loop writes no _fit.ply, so it is not
            ## a section the propagation can read or continue a track across
            if n == 0:
                continue
            rows.append(dict(idx=int(m.group(1)), axis=m.group(2),
                             off=float(m.group(3)), n=n))
    return rows


def plane_files(run: Path, p):
    ## The writer rounds the offset to two decimals from the plane's own normal
    ## fit, while planes.txt carries three, so rebuilding the name by formatting
    ## p['off'] can land one hundredth away (y -0.415 formats as -0.41 here but
    ## the file is -0.42).  Match the plane's prefix instead.
    stem = f"plane_{p['idx']:02d}_{p['axis']}"
    hits = sorted((run / 'planes').glob(f'{stem}*_fit.ply'))
    if len(hits) != 1:
        raise SystemExit(f'{stem}: expected one fit file, found {len(hits)}')
    base = Path(str(hits[0])[:-len('_fit.ply')])
    return hits[0], Path(str(base) + '_loops.ply'), base


def slices_file(run: Path):
    ## The run folder is named for the experiment, the cloud for the input shape,
    ## so `run.name + '_slices.ply'` only holds when the two coincide.  There is
    ## exactly one slice cloud per run; find it by suffix.
    hits = sorted(run.glob('*_slices.ply'))
    if len(hits) != 1:
        raise SystemExit(f'{run}: expected one *_slices.ply, found {len(hits)}')
    return hits[0]


def resample2(v, m=K):
    v = np.asarray(v, float)
    vv = np.vstack([v, v[:1]])
    s = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(vv, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], m, endpoint=False)
    return np.column_stack([np.interp(t, s, vv[:, j]) for j in range(v.shape[1])])


def resample3(v, m=K):
    return resample2(v, m)


def densify(p, m=2048):
    """Uniform arclength samples of an already closed polyline."""
    return resample2(p, m)


def canonical(b, axis):
    """Plane centre, signed normal and basis of a planar loop.

    The normal is signed along the cut axis so e2 = n x e1 never flips, and e1
    is a fixed world axis projected into the plane, so adjacent planes share an
    in-plane orientation.
    """
    c = np.asarray(b, float).mean(0)
    w, V = np.linalg.eigh(np.cov((np.asarray(b, float) - c).T))
    n = V[:, 0]
    if n[AX[axis]] < 0:
        n = -n
    ref = np.eye(3)[REF[axis]]
    e1 = ref - n * (n @ ref)
    e1 /= np.linalg.norm(e1) + 1e-18
    e2 = np.cross(n, e1)
    return c, n, e1, e2


def signed_area2(q):
    q = np.asarray(q, float)
    return float(0.5 * (np.dot(q[:, 0], np.roll(q[:, 1], -1)) -
                        np.dot(q[:, 1], np.roll(q[:, 0], -1))))


def load_split_loops(run: Path, good_only: bool = False):
    """Rebuild the loop blocks from the run-level split PLYs.

    A run that does not keep `planes/` still writes `<stem>_good_loops.ply` and
    `<stem>_bad_loops.ply`, whose vertices are the fitted curves in checkpoint
    order, plus `<stem>_loop_split.txt` with one flag per loop.  The triplane
    npz carries the same loops' axis, offset and slot in that order, so the two
    together give back what `load_run` reads from the per-plane files.  Only the
    keys, block geometry and a fitted plane normal are needed for colouring and
    drawing; the labelled slice points that the per-plane PLYs also hold are not.
    """
    npz_hits = sorted(run.glob('*_triplane.npz'))
    if len(npz_hits) != 1:
        raise SystemExit(f'{run}: expected one *_triplane.npz, found '
                         f'{len(npz_hits)}')
    keys = np.load(npz_hits[0])['keys']
    split_hits = sorted(run.glob('*_loop_split.txt'))
    if len(split_hits) != 1:
        raise SystemExit(f'{run}: expected one *_loop_split.txt, found '
                         f'{len(split_hits)}')
    flags = split_hits[0].read_text().strip()
    if len(flags) != len(keys):
        raise SystemExit(f'{split_hits[0].name}: {len(flags)} flags for '
                         f'{len(keys)} loops')

    blocks = [None] * len(keys)
    for flag, name in (('g', 'good_loops'), ('b', 'bad_loops')):
        idx = [i for i, c in enumerate(flags) if c == flag]
        if not idx:
            continue
        hits = sorted(run.glob(f'*_{name}.ply'))
        if len(hits) != 1:
            raise SystemExit(f'{run}: expected one *_{name}.ply, found '
                             f'{len(hits)}')
        v = np.asarray(trimesh.load(hits[0], process=False).vertices, float)
        if len(v) % len(idx):
            raise SystemExit(f'{hits[0].name}: {len(v)} vertices do not split '
                             f'into {len(idx)} blocks')
        per = len(v) // len(idx)
        for j, i in enumerate(idx):
            blocks[i] = v[j * per:(j + 1) * per]

    loops = []
    for i, (key, block) in enumerate(zip(keys, blocks)):
        axis = 'xyz'[int(key[0])]
        c, n, e1, e2 = canonical(block, axis)
        b2 = np.stack([(block - c) @ e1, (block - c) @ e2], 1)
        if signed_area2(b2) < 0:
            block = block[::-1]
            b2 = b2[::-1]
        loops.append(dict(
            key=f'{i}:{int(key[2])}', plane=i, axis=axis,
            off=float(key[1]), k=int(key[2]), n=int(len(block)), block=block,
            pts=block, centre=c, normal=n, e1=e1, e2=e2,
            b2=b2, q2=b2, loop2=resample2(b2),
            radius=float(np.sqrt((b2 ** 2).sum(1).mean()))))
    if good_only:
        keep = [c == 'g' for c in flags]
        loops = [L for L, k in zip(loops, keep) if k]
        print(f'  good loops only: kept {len(loops)} of {len(keep)} loops')
    return loops


def load_run(run: Path, good_only: bool = False):
    ## a run that keeps no per-plane files is still readable from its split PLYs
    if not (run / 'planes').is_dir():
        return [], load_split_loops(run, good_only=good_only)
    planes = parse_planes(run)
    loops = []
    nblock = None
    for p in planes:
        fit, lab, _ = plane_files(run, p)
        fm = trimesh.load(fit, process=False)
        lm = trimesh.load(lab, process=False)
        fv = np.asarray(fm.vertices, float)
        fc = np.asarray(fm.visual.vertex_colors[:, :3])
        lv = np.asarray(lm.vertices, float)
        lc = np.asarray(lm.visual.vertex_colors[:, :3])
        colours, counts = np.unique(fc, axis=0, return_counts=True)
        if len(colours) != p['n']:
            raise SystemExit(f'{fit.name}: {len(colours)} colours for '
                             f"{p['n']} loops")
        ## every loop is drawn at the trainer's latent resolution, which is the
        ## median slice size of the whole run.  That is not a constant: a denser
        ## plane set changes it (2383 at 45 planes, 2354 at 87), so read it here
        ## instead of assuming the run this test was written on.
        if len(np.unique(counts)) != 1:
            raise SystemExit(f'{fit.name}: uneven fit blocks {counts.tolist()}')
        nblock = int(counts[0])
        for k, col in enumerate(colours):
            block = fv[np.all(fc == col, axis=1)]
            pts = lv[np.all(lc == col, axis=1)]
            if len(block) != nblock:
                raise SystemExit(f'{fit.name}: block {k} has {len(block)} points')
            c, n, e1, e2 = canonical(block, p['axis'])
            b2 = np.stack([(block - c) @ e1, (block - c) @ e2], 1)
            if signed_area2(b2) < 0:
                # one traversal direction for every loop, so cyclic shifts and
                # linear lofts never collapse through an orientation flip
                block = block[::-1]
                b2 = b2[::-1]
            q2 = np.stack([(pts - c) @ e1, (pts - c) @ e2], 1)
            loops.append(dict(
                key=f"{p['idx']}:{k}", plane=p['idx'], axis=p['axis'],
                off=p['off'], k=k, n=int(len(pts)), block=block, pts=pts,
                colour=col, centre=c, normal=n, e1=e1, e2=e2,
                b2=b2, q2=q2, loop2=resample2(b2),
                radius=float(np.sqrt((b2 ** 2).sum(1).mean()))))
    if good_only:
        keep = good_loop_mask(run, loops, nblock)
        loops = [L for L, k in zip(loops, keep) if k]
        print(f'  good loops only: kept {len(loops)} of '
              f'{len(keep)} loops')
    return planes, loops


def good_loop_mask(run: Path, loops, nblock: int):
    """Which loaded loops are on the good side of the run's good/bad split.

    `plc_main` writes the two sides of `<stem>_good_loops.ply` as whole M-point
    blocks copied out of the same prediction array that the per-plane
    `_fit.ply` files hold, so a loop is good when the first vertex of a good
    block is one of its own vertices -- the per-plane block may be reversed,
    which is why both ends are tried rather than index 0 alone.
    """
    hits = sorted(run.glob('*_good_loops.ply'))
    if len(hits) != 1:
        raise SystemExit(f'{run}: expected one *_good_loops.ply for --good-only, '
                         f'found {len(hits)}')
    v = np.asarray(trimesh.load(hits[0], process=False).vertices, float)
    if len(v) % nblock:
        raise SystemExit(f'{hits[0].name}: {len(v)} vertices do not split into '
                         f'blocks of {nblock}')
    good_first = v.reshape(-1, nblock, 3)[:, 0, :]
    tree = cKDTree(good_first)
    keep = np.zeros(len(loops), bool)
    for i, L in enumerate(loops):
        d = tree.query(np.vstack([L['block'][:1], L['block'][-1:]]))[0]
        keep[i] = bool(d.min() < 1e-5)
    return keep


def by_axis_planes(planes):
    out = {}
    for a in 'xyz':
        seq = sorted([p for p in planes if p['axis'] == a],
                     key=lambda p: p['off'])
        out[a] = seq
    return out


def axis_offsets(planes):
    out = {}
    for a, seq in by_axis_planes(planes).items():
        out[a] = [p['off'] for p in seq]
    return out


def rgb_uint(c):
    return np.round(np.array(to_rgb(c)) * 255).astype(np.uint8)


def colored_mesh(verts, faces, rgb):
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=False)
    c = np.tile(np.asarray(rgb, np.uint8), (len(m.vertices), 1))
    m.visual.vertex_colors = np.c_[c, np.full(len(c), 255, np.uint8)]
    if m.volume < 0:
        m.invert()
    return m


def ring_tube(path, radius, nring=10, up=(0.0, 1.0, 0.0)):
    """Closed tube around a closed polyline, frame built from `up`."""
    path = np.asarray(path, float)
    n = len(path)
    tng = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    tng /= np.linalg.norm(tng, axis=1, keepdims=True) + 1e-12
    B = np.asarray(up, float)
    B /= np.linalg.norm(B) + 1e-12
    N = np.cross(np.broadcast_to(B, tng.shape), tng)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    B = np.cross(tng, N)
    th = np.linspace(0, 2 * np.pi, nring, endpoint=False)
    ring = (np.cos(th)[None, :, None] * N[:, None, :] +
            np.sin(th)[None, :, None] * B[:, None, :])
    verts = (path[:, None, :] + radius * ring).reshape(-1, 3)
    faces = []
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(len(th)):
            j2 = (j + 1) % len(th)
            a, b = i * len(th) + j, i * len(th) + j2
            c, d = i2 * len(th) + j2, i2 * len(th) + j
            faces += [[a, b, c], [a, c, d]]
    return verts, np.array(faces)


def segment_tube(a, b, radius, nring=8, nseg=8):
    """Open cylinder from a to b with a frame that never degenerates."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = b - a
    t /= np.linalg.norm(t) + 1e-12
    e = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(t, e)
    u /= np.linalg.norm(u) + 1e-12
    w = np.cross(t, u)
    ts = np.linspace(0, 1, nseg + 1)
    cent = a[None, :] + ts[:, None] * (b - a)[None, :]
    th = np.linspace(0, 2 * np.pi, nring, endpoint=False)
    verts = (cent[:, None, :] +
             radius * (np.cos(th)[None, :, None] * u[None, None, :] +
                       np.sin(th)[None, :, None] * w[None, None, :])).reshape(-1, 3)
    faces = []
    for i in range(nseg):
        for j in range(nring):
            j2 = (j + 1) % nring
            p, q = i * nring + j, i * nring + j2
            r, s = (i + 1) * nring + j, (i + 1) * nring + j2
            faces += [[p, q, s], [p, s, r]]
    return verts, np.array(faces)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def tracks_json(out: Path, args):
    """The propagation result the later steps should consume.

    `auto` takes the most recently written one, so running `--step 2b` and
    then step 3 continues that variant, while a plain `--step all` keeps using
    the rigid step 2 it just rewrote.
    """
    choice = getattr(args, 'tracks', 'auto')
    if choice != 'auto':
        return out / choice / 'tracks.json'
    cand = [p for p in (out / 'step2' / 'tracks.json',
                        out / 'step2b' / 'tracks.json') if p.exists()]
    if not cand:
        raise SystemExit('no step2/step2b tracks.json; run the propagation first')
    return max(cand, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------- #
# step 1
# --------------------------------------------------------------------------- #
def step1(run: Path, out: Path, args):
    print('\n== step 1: stable loops ==')
    planes, loops = load_run(run, good_only=getattr(args, 'good_only', False))
    od = axis_offsets(planes)
    by_key = {L['key']: L for L in loops}

    for L in loops:
        d = cKDTree(densify(L['loop2'])).query(L['q2'])[0]
        L['rms'] = float(np.sqrt((d ** 2).mean()))
        L['q90'] = float(np.quantile(d, 0.9))
        L['cov'] = float((d <= args.tau).mean())
        L['rms_rad'] = L['rms'] / max(L['radius'], 1e-12)
        L['stable'] = bool(L['rms'] <= args.rms)
        ## diagnostic only: nearest component on each adjacent plane
        a, off = L['axis'], L['off']
        k = od[a].index(off)
        best = dict(s=0.0, D=float('inf'), partner=None, partner_off=None)
        for kn in (k - 1, k + 1):
            if not (0 <= kn < len(od[a])):
                continue
            group = [M for M in loops if M['axis'] == a
                     and abs(M['off'] - od[a][kn]) < 1e-9]
            if not group:
                continue
            M = min(group, key=lambda m: np.linalg.norm(m['centre'] - L['centre']))
            try:
                delta, D, _ = pair_residual(L['loop2'], M['loop2'])
            except ValueError:
                continue
            s, _ = switch(D, args.lam)
            if s > best['s']:
                best = dict(s=float(s), D=float(D), partner=M['key'],
                            partner_off=M['off'], delta=float(delta))
        L.update(s=best['s'], s_D=best['D'], partner=best['partner'],
                 partner_off=best['partner_off'])

    stable = [L for L in loops if L['stable']]
    rejected = [L for L in loops if not L['stable']]
    rms = np.array([L['rms'] for L in loops])
    ax_counts = {a: (sum(L['stable'] for L in loops if L['axis'] == a),
                     sum((not L['stable']) for L in loops if L['axis'] == a))
                 for a in 'xyz'}
    d1 = out / 'step1'
    d1.mkdir(parents=True, exist_ok=True)

    fields = ['key', 'plane', 'axis', 'off', 'k', 'n', 'rms', 'q90', 'cov',
              'radius', 'rms_rad', 's', 's_D', 'partner', 'partner_off',
              'stable']
    write_csv(d1 / 'loops.csv', loops, fields)
    save_json(d1 / 'loops.json', dict(
        run=str(run), tau=args.tau, rms_cut=args.rms, lam=args.lam, k=K,
        loops=[{f: L[f] for f in fields if f in L} for L in loops]))

    # distribution and per-axis strip
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.6), dpi=150,
                            gridspec_kw={'width_ratios': [1.15, 1, 1, 1]})
    ax = axs[0]
    ax.hist(rms, bins=np.linspace(0, max(0.10, rms.max()), 30),
            color='#b9c6d6', edgecolor='white')
    ax.axvline(args.rms, color='#bf1b1b', lw=1.4, label=f'cut {args.rms:.3f}')
    ax.set_xlabel('in-plane RMS'); ax.set_ylabel('loops')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f'{len(stable)} stable / {len(rejected)} rejected', fontsize=9)

    for j, a in enumerate('xyz'):
        ax = axs[j + 1]
        sub = sorted([L for L in loops if L['axis'] == a], key=lambda L: L['off'])
        ax.plot([L['off'] for L in sub], [L['rms'] for L in sub], 'o',
                ms=3.6, color='#4575b4', label='loop')
        ax.plot([L['off'] for L in [L for L in sub if L['stable']]],
                [L['rms'] for L in [L for L in sub if L['stable']]], 'o',
                ms=4.2, mfc='#2a9d8f', mec='#2a9d8f', label='stable')
        ax.axhline(args.rms, color='#bf1b1b', lw=1)
        ax.set_xlabel(f'{a} offset'); ax.set_ylabel('RMS')
        ax.set_title(f'{a}: {ax_counts[a][0]} stable, {ax_counts[a][1]} rejected',
                     fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Step 1 — per-loop in-plane RMS against its own slice points',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(d1 / 'rms_distribution.png')
    plt.close(fig)

    # threshold sensitivity
    cuts = np.linspace(0.005, 0.045, 81)
    counts = [(int((rms <= c).sum()), int((rms > c).sum())) for c in cuts]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    ax.plot(cuts, [c[0] for c in counts], color='#2a9d8f', label='stable')
    ax.plot(cuts, [c[1] for c in counts], color='#bf1b1b', label='rejected')
    ax.axvline(args.rms, color='k', lw=1, ls=':')
    ax.set_xlabel('RMS cut'); ax.set_ylabel('loops')
    ax.legend(frameon=False)
    ax.set_title('Step 1 — stable count against the cut', fontsize=10)
    fig.tight_layout(); fig.savefig(d1 / 'thresholds.png'); plt.close(fig)

    # threshold counts in the summary
    counts_txt = ', '.join(f'<={t:.3f}: {int((rms <= t).sum())}'
                           for t in (0.012, 0.015, 0.020, 0.022, 0.025, 0.030))

    # montages
    for name, group in (('stable_montage.png', stable),
                        ('rejected_montage.png', rejected)):
        cols = 8
        rows = int(np.ceil(len(group) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.62 * rows),
                                dpi=150)
        axs = np.atleast_1d(axs).ravel()
        for ax, L in zip(axs, group):
            ax.plot(L['q2'][:, 0], L['q2'][:, 1], '.', ms=0.35,
                    color='0.78', rasterized=True)
            ax.plot(L['loop2'][:, 0], L['loop2'][:, 1],
                    color=tuple(L['colour'] / 255), lw=0.9)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{L['axis']}{L['off']:+.2f} c{L['k']}\n"
                         f"rms {L['rms']:.3f}", fontsize=6.4)
        for ax in axs[len(group):]:
            ax.axis('off')
        fig.suptitle(('Step 1 — stable loops (RMS <= '
                      f'{args.rms:.3f})' if group is stable else
                      'Step 1 — rejected loops (RMS above the cut)'),
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(d1 / name)
        plt.close(fig)

    # RMS vs diagnostic switch
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    colours = {'x': '#4575b4', 'y': '#2a9d8f', 'z': '#d1495b'}
    for a in 'xyz':
        sub = [L for L in loops if L['axis'] == a]
        ax.scatter([L['rms'] for L in sub], [L['s'] for L in sub], s=26,
                   c=colours[a], alpha=0.8, label=a, edgecolors='white', lw=0.4)
    ax.axvline(args.rms, color='#bf1b1b', lw=1.2, ls='--',
               label=f'cut {args.rms:.3f}')
    ax.axhline(0.3, color='0.4', lw=1, ls=':', label='s = 0.3')
    ax.set_xlabel('in-plane RMS'); ax.set_ylabel('best diagnostic switch s')
    ax.set_title('Step 1 — fit quality and neighbour agreement are different '
                 'populations', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(d1 / 'rms_vs_s.png'); plt.close(fig)

    txt = [
        f"run: {run}",
        f"planes: {len(planes)}, loops: {len(loops)}",
        f"RMS cut: {args.rms:.4f} (point-to-loop distance in each loop's own "
        f"PCA plane)",
        f"stable: {len(stable)} (x {ax_counts['x'][0]}, y {ax_counts['y'][0]}, "
        f"z {ax_counts['z'][0]})",
        f"rejected: {len(rejected)} (x {ax_counts['x'][1]}, y {ax_counts['y'][1]}, "
        f"z {ax_counts['z'][1]})",
        f"RMS quantiles: min {rms.min():.4f}, 25% {np.quantile(rms, .25):.4f}, "
        f"median {np.median(rms):.4f}, 75% {np.quantile(rms, .75):.4f}, "
        f"90% {np.quantile(rms, .9):.4f}, max {rms.max():.4f}",
        f"counts by cut: {counts_txt}",
        f"diagnostic s >= 0.3: {sum(1 for L in loops if L['s'] >= .3)} loops; "
        f"{sum(1 for L in stable if L['s'] >= .3)} among stable",
        "stable keys: " + ', '.join(f"{L['key']}" for L in stable),
    ]
    (d1 / 'summary.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))
    print(f'  wrote {d1}')
    return dict(stable=[L['key'] for L in stable],
                counts=ax_counts, n=len(loops),
                stable_n=len(stable), rejected_n=len(rejected))


# --------------------------------------------------------------------------- #
# step 2
# --------------------------------------------------------------------------- #
def seed_in_plane(src, tgt):
    """The seed loop resampled in the target's plane frame (no re-centring)."""
    c, e1, e2 = tgt['centre'], tgt['e1'], tgt['e2']
    return resample2(np.stack([(src['block'] - c) @ e1,
                               (src['block'] - c) @ e2], 1), K)


def shift_search(pd, q, cap, tau):
    """Best in-plane translation of the densified seed, within `cap`."""
    tree_p = cKDTree(pd)
    best_s = np.zeros(2)
    for step, rad, subsample in ((cap / 3, cap, 2), (cap / 9, cap / 3, 1)):
        base = best_s
        qq = q[::subsample]
        m = int(round(rad / step))
        stage_cov, stage_s = -1.0, base
        for dx in np.arange(-m, m + 1) * step:
            for dz in np.arange(-m, m + 1) * step:
                s = base + np.array([dx, dz])
                if np.linalg.norm(s) > cap + 1e-12:
                    continue
                cov = float((tree_p.query(qq - s, workers=-1)[0] <= tau).mean())
                # Tie-break on the smaller shift: a flat coverage plateau
                # (any shift below tau scores 1.0) must not move the seed.
                if (cov > stage_cov or
                        (cov == stage_cov and
                         np.linalg.norm(s) < np.linalg.norm(stage_s))):
                    stage_cov, stage_s = cov, s
        best_s = stage_s
    return best_s


def two_sided(p, q, shift, tau):
    """Point coverage and loop support of a shifted closed polyline."""
    tree_p = cKDTree(densify(p, 2048))
    tree_q = cKDTree(q)
    cov_pt = float((tree_p.query(q - shift, workers=-1)[0] <= tau).mean())
    cov_loop = float((tree_q.query(p + shift, workers=-1)[0] <= tau).mean())
    return cov_pt, cov_loop


def transfer(src, tgt, tau):
    """How well src's loop explains tgt's points at its physical placement.

    The source loop is projected into the target plane at the position it
    actually occupies in 3D; only a small local correction is searched.  Free
    sliding is forbidden because the hand's cross-sections repeat, so an
    arbitrary loop can otherwise be translated onto an unrelated component.
    """
    p = seed_in_plane(src, tgt)
    q = tgt['q2']
    cap = float(np.clip(0.18 * min(src['radius'], tgt['radius']), 0.01, 0.06))
    best_s = shift_search(densify(p, 2048), q, cap, tau)
    cov_pt, cov_loop = two_sided(p, q, best_s, tau)
    return dict(shift=best_s.tolist(), shift_norm=float(np.linalg.norm(best_s)),
                cov_pt=cov_pt, cov_loop=cov_loop,
                score=float(min(cov_pt, cov_loop)))


def loop_normal(p):
    """In-plane unit normal of a closed polyline, one row per sample."""
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return np.stack([-t[:, 1], t[:, 0]], 1)


def fourier_basis(m, modes):
    """Constant plus cos/sin columns for modes 1..M, at m parameters.

    The constant column is the uniform normal offset (a taper or dilation);
    without it the fit cannot shrink or grow a section, only bend it.
    """
    u = np.arange(m) / m
    cols = [np.ones(m)]
    for k in range(1, modes + 1):
        cols.append(np.cos(2 * np.pi * k * u))
        cols.append(np.sin(2 * np.pi * k * u))
    return np.stack(cols, 1)


def deformable_transfer(src, tgt, tau, modes=8, budget=0.15, accept=0.5,
                        iters=30, damping=0.6):
    """Two-sided transfer after a bounded in-plane deformation of the seed.

    The seed is first shifted as in `transfer`, then its offset along the
    in-plane normal is expanded in a low-order Fourier series and fitted to the
    target's points by robust ICP.  The deformation is not copied into the
    track: the target's own fitted loop continues it, and the deformed seed is
    only the evidence that the edge is a continuation.

    `deform_rel` is the RMS normal offset as a fraction of the seed radius.
    An edge is admissible only when `deform_rel <= budget`, so a pair that
    needs a free re-fit is still a cut.  The rigid solution is always a
    candidate, so enabling the deformation can only add edges.
    """
    p = seed_in_plane(src, tgt)
    q = np.asarray(tgt['q2'], float)
    if len(q) < 8 or len(p) < 4:
        return dict(shift=[0.0, 0.0], shift_norm=0.0, cov_pt=0.0,
                    cov_loop=0.0, score=0.0, rigid_cov_pt=0.0,
                    rigid_cov_loop=0.0, deform_used='rigid', deform_rms=0.0,
                    deform_max=0.0, deform_rel=0.0, used_rel=0.0,
                    admissible=False, n_match=0, curve=p.tolist())
    cap = float(np.clip(0.18 * min(src['radius'], tgt['radius']), 0.01, 0.06))
    shift0 = shift_search(densify(p, 2048), q, cap, tau)
    rigid_pt, rigid_loop = two_sided(p, q, shift0, tau)

    nrm = loop_normal(p)
    B = fourier_basis(len(p), modes)
    ncoef = B.shape[1]
    x = np.zeros(ncoef + 2)
    x[-2:] = shift0
    tree_q = cKDTree(q)
    scale = max(min(src['radius'], tgt['radius']), 1e-9)
    # Discovery radius: how far the seed may move in the first iteration.  It
    # comes from the seed's current distance to the target, capped below the
    # section size, and shrinks to the measurement tolerance so a taper or a
    # shifted section can be found without letting an unrelated loop pull.
    d0, _ = tree_q.query(p + shift0)
    rad0 = float(np.clip(1.5 * np.quantile(d0, 0.9), 2.0 * tau, 0.5 * scale))
    # Light smoothness prior: higher modes cost more, so unsupported modes
    # cannot run away when the target covers only part of the seed.
    ridge = np.zeros(x.size)
    if ncoef:
        ridge[0] = 0.01          # the uniform offset trades with the shift
        kmode = np.repeat(np.arange(1, modes + 1), 2) / max(modes, 1)
        ridge[1:ncoef] = kmode ** 2
    n_match = 0
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        match_rad = rad0 + (2.0 * tau - rad0) * frac
        disp = (B @ x[:ncoef])[:, None] * nrm if ncoef else np.zeros_like(p)
        dist, idx = tree_q.query(p + disp + x[-2:])
        m = dist <= match_rad
        n_match = int(m.sum())
        if n_match < 8:
            break
        u = dist[m] / match_rad
        w = (1.0 - u ** 2) ** 2          # Tukey: far matches lose their pull
        rows = 2 * n_match
        A = np.zeros((rows, x.size))
        b = np.zeros(rows)
        if ncoef:
            A[0::2, :ncoef] = nrm[m, 0:1] * B[m]
            A[1::2, :ncoef] = nrm[m, 1:2] * B[m]
        A[0::2, -2] = 1.0
        A[1::2, -1] = 1.0
        b[0::2] = q[idx[m], 0] - p[m, 0]
        b[1::2] = q[idx[m], 1] - p[m, 1]
        sw = np.repeat(np.sqrt(w), 2)
        A *= sw[:, None]
        b *= sw
        lam = np.sqrt(ridge * rows * 0.02)
        xn, *_ = np.linalg.lstsq(np.vstack([A, np.diag(lam)]),
                                 np.concatenate([b, np.zeros(x.size)]),
                                 rcond=None)
        step = damping * (xn - x)
        x = x + step
        ns = float(np.linalg.norm(x[-2:]))
        if ns > cap:
            x[-2:] *= cap / ns
        if np.linalg.norm(step) < 1e-7:
            break

    disp = (B @ x[:ncoef])[:, None] * nrm if ncoef else np.zeros_like(p)
    shift = x[-2:]
    cov_pt, cov_loop = two_sided(p + disp, q, shift, tau)
    deform_rms = float(np.sqrt((disp ** 2).sum(1).mean()))
    deform_max = float(np.linalg.norm(disp, axis=1).max())
    deform_rel = deform_rms / scale

    # Never let the deformation lower the score: the rigid solution is the
    # zero-deformation fallback.
    # Two admissible explanations, not one winner: the rigid solution needs no
    # deformation, the deformed one is admissible only within the budget.  A
    # rigid pass must never be lost to an over-budget deformation.
    rigid_ok = min(rigid_pt, rigid_loop) >= accept
    deform_ok = min(cov_pt, cov_loop) >= accept and deform_rel <= budget
    if rigid_ok:
        # Deformation is a rescue, not a refinement: if the seed already
        # explains the neighbour, report the zero-deformation solution.
        used = 'rigid'
        cov_pt, cov_loop = rigid_pt, rigid_loop
        curve = p + shift0
        shift = shift0
    else:
        used = 'deformed'
        curve = p + disp + shift
    return dict(shift=shift.tolist(), shift_norm=float(np.linalg.norm(shift)),
                cov_pt=cov_pt, cov_loop=cov_loop,
                score=float(min(cov_pt, cov_loop)),
                rigid_cov_pt=rigid_pt, rigid_cov_loop=rigid_loop,
                deform_used=used, deform_rms=deform_rms,
                deform_max=deform_max, deform_rel=deform_rel,
                used_rel=(0.0 if used == 'rigid' else deform_rel),
                admissible=bool(rigid_ok or deform_ok),
                n_match=n_match, curve=curve.tolist())


def step2(run: Path, out: Path, args, deform: bool = False):
    tag = '2b' if deform else '2'
    print(f'\n== step {tag}: propagate'
          f"{' with a bounded deformation' if deform else ''} ==")
    planes, loops = load_run(run, good_only=getattr(args, 'good_only', False))
    s1 = json.loads((out / 'step1' / 'loops.json').read_text())
    stable_keys = {L['key'] for L in s1['loops'] if L['stable']}
    by_key = {L['key']: L for L in loops}
    for L in loops:
        L['stable'] = L['key'] in stable_keys
    od = axis_offsets(planes)
    d2 = out / f'step{tag}'
    d2.mkdir(parents=True, exist_ok=True)

    def score_fn(src, tgt):
        if not deform:
            return transfer(src, tgt, args.tau)
        return deformable_transfer(src, tgt, args.tau,
                                   modes=args.deform_modes,
                                   budget=args.deform_budget,
                                   accept=args.accept)

    def adjacent(L, side):
        a = L['axis']
        k = od[a].index(L['off'])
        kn = k - 1 if side == 'prev' else k + 1
        if not (0 <= kn < len(od[a])):
            return None
        return [M for M in loops if M['axis'] == a
                and abs(M['off'] - od[a][kn]) < 1e-9]

    # --- null calibration on non-neighbour planes ---------------------------
    null_pt = {a: [] for a in 'xyz'}
    null_loop = {a: [] for a in 'xyz'}
    null_tests = 0
    for key in sorted(stable_keys):
        L = by_key[key]
        a = L['axis']
        k = od[a].index(L['off'])
        away = [j for j in range(len(od[a])) if abs(j - k) >= 3]
        if not away:
            continue
        pick = [away[int(round(q * (len(away) - 1)))]
                for q in (0.1, 0.5, 0.9)]
        pick = sorted(set(pick))
        for j in pick:
            group = [M for M in loops if M['axis'] == a
                     and abs(M['off'] - od[a][j]) < 1e-9]
            best_pt, best_loop = 0.0, 0.0
            for M in group:
                r = score_fn(L, M)
                best_pt = max(best_pt, r['cov_pt'])
                best_loop = max(best_loop, r['cov_loop'])
            null_pt[a].append(best_pt)
            null_loop[a].append(best_loop)
            null_tests += 1
    thr_pt, thr_loop, inconclusive = {}, {}, {}
    null_q = {}
    for a in 'xyz':
        if len(null_pt[a]) >= 4:
            qp = float(np.quantile(null_pt[a], 0.95))
            ql = float(np.quantile(null_loop[a], 0.95))
        else:
            qp = ql = 0.0
        inconclusive[a] = (qp >= 0.45 or ql >= 0.45)
        # Far planes on the same axis are not negatives for a tubular region:
        # the palm repeats along y and each digit repeats along z, so a seed
        # placed in its physical position covers them legitimately.  The
        # calibrated null therefore fails; acceptance uses the fixed physical
        # placement threshold and a bounded local shift instead.
        thr_pt[a] = args.accept
        thr_loop[a] = args.accept
        null_q[a] = dict(pt=qp, loop=ql, n=len(null_pt[a]))
        print(f'  null {a}: n={len(null_pt[a])} q95 pt {qp:.3f} loop {ql:.3f}; '
              f"thresholds {thr_pt[a]:.3f}/{thr_loop[a]:.3f}"
              f"{' (self-similar axis, null unused)' if inconclusive[a] else ''}")

    # --- grow the graph from the stable loops ------------------------------
    attempts, passing, results = [], {}, {}
    tested = set()
    examples = dict(accepted=[], rejected_pt=[], rejected_loop=[],
                    superseded=[])
    active = set(stable_keys)
    round_no = 0
    while True:
        round_no += 1
        fresh = False
        for src_key in sorted(active):
            src = by_key[src_key]
            for side in ('prev', 'next'):
                group = adjacent(src, side)
                if group is None:
                    continue
                for tgt in group:
                    pair = tuple(sorted((src_key, tgt['key'])))
                    if pair in tested:
                        continue
                    tested.add(pair)
                    fresh = True
                    r = score_fn(src, tgt)
                    results[pair] = r
                    a = src['axis']
                    ok = (r['cov_pt'] >= thr_pt[a] and
                          r['cov_loop'] >= thr_loop[a])
                    if deform:
                        ok = ok and r['admissible']
                    reason = ('ok' if ok else
                              'point coverage' if r['cov_pt'] < thr_pt[a] else
                              'loop support' if r['cov_loop'] < thr_loop[a] else
                              'deformation budget')
                    row = dict(src=src_key, tgt=tgt['key'], axis=a,
                               src_off=src['off'], tgt_off=tgt['off'],
                               shift_x=r['shift'][0], shift_z=r['shift'][1],
                               shift_norm=r['shift_norm'],
                               cov_pt=r['cov_pt'], cov_loop=r['cov_loop'],
                               score=r['score'], thr_pt=thr_pt[a],
                               thr_loop=thr_loop[a], passed=bool(ok),
                               reason=reason)
                    if deform:
                        row.update(
                            rigid_cov_pt=r['rigid_cov_pt'],
                            rigid_cov_loop=r['rigid_cov_loop'],
                            deform_used=r['deform_used'],
                            deform_rms=r['deform_rms'],
                            deform_max=r['deform_max'],
                            deform_rel=r['deform_rel'],
                            used_rel=r['used_rel'],
                            admissible=r['admissible'], n_match=r['n_match'])
                    attempts.append(row)
                    if ok:
                        passing[pair] = dict(row)
                    elif reason == 'point coverage' and len(examples['rejected_pt']) < 2:
                        examples['rejected_pt'].append((src, tgt, r, reason))
                    elif reason == 'loop support' and len(examples['rejected_loop']) < 2:
                        examples['rejected_loop'].append((src, tgt, r, reason))
        # greedy keep: highest min-score first, one edge per node per side
        kept, used = {}, {}
        for pair, info in sorted(passing.items(),
                                 key=lambda kv: -kv[1]['score']):
            s, t = pair
            S, T = by_key[s], by_key[t]
            if abs(S['off'] - T['off']) < 1e-12:
                continue
            side_s = 'next' if T['off'] > S['off'] else 'prev'
            side_t = 'prev' if T['off'] > S['off'] else 'next'
            if (s, side_s) in used or (t, side_t) in used:
                info = dict(info)
                info['reason'] = 'superseded'
                if len(examples['superseded']) < 2:
                    examples['superseded'].append(
                        (by_key[info['src']], by_key[info['tgt']], info, 'superseded'))
                continue
            kept[pair] = info
            used[(s, side_s)] = pair
            used[(t, side_t)] = pair
        new_active = set(stable_keys) | {n for p in kept for n in p}
        if new_active == active and not fresh:
            break
        if new_active == active:
            # no new nodes, but all their edges are tested: the matching is final
            break
        active = new_active
        if round_no > 40:
            raise SystemExit('propagation did not settle')

    # tracks = connected components of the kept graph
    parent = {L['key']: L['key'] for L in loops}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for s, t in kept:
        union(s, t)
    groups = {}
    for L in loops:
        groups.setdefault(find(L['key']), []).append(L['key'])
    tracks = []
    for root, keys in groups.items():
        nodes = sorted(keys, key=lambda k: (by_key[k]['axis'], by_key[k]['off']))
        if len(nodes) >= 2 and any((i, j) in kept for i in nodes for j in nodes):
            tracks.append(nodes)
    tracks.sort(key=lambda nodes: (by_key[nodes[0]]['axis'],
                                   by_key[nodes[0]]['off']))

    in_track = {k for t in tracks for k in t}
    cat = dict(stable_kept=[k for k in in_track if by_key[k]['stable']],
               stable_discarded=[k for k in stable_keys if k not in in_track],
               adopted=[k for k in in_track if not by_key[k]['stable']],
               orphan=[k for k in by_key if k not in in_track
                       and not by_key[k]['stable']])
    for k in cat:
        cat[k].sort(key=lambda x: (by_key[x]['axis'], by_key[x]['off'], by_key[x]['k']))

    # --- artifacts ----------------------------------------------------------
    fields = ['src', 'tgt', 'axis', 'src_off', 'tgt_off', 'shift_x', 'shift_z',
              'shift_norm', 'cov_pt', 'cov_loop', 'score', 'thr_pt',
              'thr_loop', 'passed', 'reason']
    if deform:
        fields += ['rigid_cov_pt', 'rigid_cov_loop', 'deform_used',
                   'deform_rms', 'deform_max', 'deform_rel', 'used_rel',
                   'admissible', 'n_match']
    write_csv(d2 / 'attempts.csv', attempts, fields)
    track_rows = []
    for ti, nodes in enumerate(tracks, 1):
        for n in nodes:
            L = by_key[n]
            score = next((kept[p]['score'] for p in kept
                          if n in p), None)
            track_rows.append(dict(track=ti, key=n, plane=L['plane'],
                                   axis=L['axis'], off=L['off'], k=L['k'],
                                   stable=bool(L['stable']),
                                   edge_score=score if score is not None else ''))
    save_json(d2 / 'tracks.json', dict(
        tau=args.tau, accept=args.accept, lam=args.lam,
        deform=(dict(modes=args.deform_modes, budget=args.deform_budget)
                if deform else None),
        null=dict(pt={a: null_pt[a] for a in 'xyz'},
                  loop={a: null_loop[a] for a in 'xyz'},
                  q95=null_q,
                  thr_pt=thr_pt, thr_loop=thr_loop,
                  inconclusive=inconclusive),
        tracks=[dict(id=i, axis=by_key[n[0]]['axis'],
                     keys=n, offsets=[by_key[k]['off'] for k in n],
                     planes=[by_key[k]['plane'] for k in n],
                     stable=[bool(by_key[k]['stable']) for k in n])
                for i, n in enumerate(tracks, 1)],
        edges=[dict(src=s, tgt=t, score=kept[(s, t)]['score'],
                    cov_pt=kept[(s, t)]['cov_pt'],
                    cov_loop=kept[(s, t)]['cov_loop']) for s, t in kept],
        categories={k: v for k, v in cat.items()},
        counts=dict(attempts=len(attempts),
                    passed=sum(1 for a in attempts if a['passed']),
                    kept_edges=len(kept), tracks=len(tracks),
                    stable_kept=len(cat['stable_kept']),
                    stable_discarded=len(cat['stable_discarded']),
                    adopted=len(cat['adopted']), orphan=len(cat['orphan']))))
    write_csv(d2 / 'track_loops.csv', track_rows,
              ['track', 'key', 'plane', 'axis', 'off', 'k', 'stable',
               'edge_score'])

    # null vs accepted figure
    colours = {'x': '#4575b4', 'y': '#2a9d8f', 'z': '#d1495b'}
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.8), dpi=150)
    for ax, key, lab in ((axs[0], 'pt', 'point coverage'),
                         (axs[1], 'loop', 'loop support')):
        for j, a in enumerate('xyz'):
            vals = (null_pt if key == 'pt' else null_loop)[a]
            if vals:
                ax.hist(vals, bins=np.linspace(0, 1, 21), alpha=0.55,
                        color=colours[a], label=f'{a} null')
        acc = [x['cov_pt'] if key == 'pt' else x['cov_loop']
               for x in attempts if x['passed']]
        if acc:
            ax.hist(acc, bins=np.linspace(0, 1, 21), histtype='step',
                    lw=1.8, color='k', label='accepted')
        ax.axvline(args.accept, color='#bf1b1b', lw=1.2, ls='--',
                   label=f'accept {args.accept}')
        ax.set_xlabel(lab); ax.set_ylabel('count')
        ax.set_title(f'{lab}: null vs accepted', fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f'Step {tag} — calibrated two-sided propagation test',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(d2 / 'null.png'); plt.close(fig)

    # decision montage
    picks = ([('accepted', s) for s in examples['accepted'][:2]] +
             [('rejected: point coverage', e) for e in examples['rejected_pt'][:1]] +
             [('rejected: loop support', e) for e in examples['rejected_loop'][:1]] +
             [('superseded', e) for e in examples['superseded'][:1]])
    # fill accepted examples from kept edges if the pool is empty
    if not examples['accepted']:
        for pair, info in list(kept.items())[:2]:
            s, t = pair
            examples['accepted'].append(
                (by_key[s], by_key[t], results.get(pair, info), 'accepted'))
        picks = ([('accepted', e) for e in examples['accepted']] +
                 [('rejected: point coverage', e) for e in examples['rejected_pt'][:1]] +
                 [('rejected: loop support', e) for e in examples['rejected_loop'][:1]] +
                 [('superseded', e) for e in examples['superseded'][:1]])
    if picks:
        ncol = len(picks)
        fig, axs = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.2), dpi=150)
        axs = np.atleast_1d(axs)
        for ax, (label, item) in zip(axs, picks):
            src, tgt, r, reason = item
            if 'shift' not in r:
                r = dict(shift=[r['shift_x'], r['shift_z']],
                         cov_pt=r['cov_pt'], cov_loop=r['cov_loop'])
            p = resample2(np.stack(
                [(src['block'] - tgt['centre']) @ tgt['e1'],
                 (src['block'] - tgt['centre']) @ tgt['e2']], 1), K)
            q = tgt['q2']
            curve = np.asarray(r.get('curve', p + np.asarray(r['shift'])))
            ax.plot(q[:, 0], q[:, 1], '.', ms=0.4, color='0.78', rasterized=True)
            ax.plot(curve[:, 0], curve[:, 1], color='#d1495b',
                    lw=1.2, label=f"seed {r.get('deform_used', 'shifted')}")
            ax.plot(tgt['loop2'][:, 0], tgt['loop2'][:, 1], color='#4575b4',
                    lw=0.9, label='target loop')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            extra = (f"\ndeform used {r.get('used_rel', 0.0):.3f} of radius"
                     if 'deform_rel' in r else '')
            ax.set_title(f"{label}\n{src['key']} -> {tgt['key']}\n"
                         f"pt {r['cov_pt']:.2f}, loop {r['cov_loop']:.2f}{extra}",
                         fontsize=7.2)
            ax.legend(frameon=False, fontsize=5.8, loc='lower right')
        fig.suptitle(f'Step {tag} — propagation decisions', fontsize=11)
        fig.tight_layout(); fig.savefig(d2 / 'decisions.png'); plt.close(fig)

    # 3D track map
    slices = trimesh.load(slices_file(run), process=False).vertices
    fig = plt.figure(figsize=(13, 7), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    sub = slices[::12]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.25, c='0.86', lw=0)
    for ti, nodes in enumerate(tracks, 1):
        col = PALETTE[(ti - 1) % len(PALETTE)]
        for n in nodes:
            L = by_key[n]
            b = L['block']
            ax.plot(b[:, 0], b[:, 1], b[:, 2], color=col, lw=1.5)
            if L['stable']:
                ax.scatter([L['centre'][0]], [L['centre'][1]], [L['centre'][2]],
                           s=14, color=col, edgecolor='k', lw=0.3, depthshade=False)
        ax.plot([], [], color=col, lw=2, label=f'track {ti} ({len(nodes)} loops)')
    for n in cat['orphan']:
        L = by_key[n]
        b = L['block']
        ax.plot(b[:, 0], b[:, 1], b[:, 2], color='0.55', lw=0.7)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(f'Step {tag} — propagated tracks (filled dot = stable seed)',
                 fontsize=11)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    try:
        ax.set_box_aspect(np.ptp(slices, axis=0))
    except Exception:
        pass
    fig.tight_layout(); fig.savefig(d2 / 'tracks.png'); plt.close(fig)

    # strip by axis
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    for ax, a in zip(axs, 'xyz'):
        tr = [(ti, t) for ti, t in enumerate(tracks, 1)
              if by_key[t[0]]['axis'] == a]
        for row, (ti, nodes) in enumerate(tr, 1):
            col = PALETTE[(ti - 1) % len(PALETTE)]
            offs = [by_key[n]['off'] for n in nodes]
            ax.plot(offs, [row] * len(offs), color=col, lw=0.8, zorder=1)
            for n in nodes:
                L = by_key[n]
                ax.scatter([L['off']], [row], s=24, facecolors=col,
                           edgecolors='k' if L['stable'] else 'none',
                           lw=0.5, zorder=3)
        ax.set_yticks(range(1, len(tr) + 1))
        ax.set_yticklabels([f't{ti}' for ti, _ in tr], fontsize=7)
        ax.set_xlabel(f'{a} offset')
        ax.set_title(f'{a} tracks', fontsize=9)
        ax.grid(axis='x', color='0.9')
    fig.suptitle(f'Step {tag} — track extent along each axis '
                 f'(outline = stable seed)', fontsize=11)
    fig.tight_layout(); fig.savefig(d2 / 'tracks_strip.png'); plt.close(fig)

    txt = [
        f"stable seeds: {len(stable_keys)}",
        f"edges tested: {len(attempts)}, passed: {sum(1 for a in attempts if a['passed'])}, "
        f"kept after matching: {len(kept)}",
        f"tracks: {len(tracks)}",
        f"stable kept: {len(cat['stable_kept'])}, "
        f"stable discarded: {len(cat['stable_discarded'])}, "
        f"adopted: {len(cat['adopted'])}, orphan: {len(cat['orphan'])}",
        ("deformation: " +
         (f"modes {args.deform_modes}, budget {args.deform_budget:.2f} of "
          f"seed radius, "
          f"{sum(1 for a in attempts if a['passed'] and a.get('deform_used') == 'deformed')} "
          f"accepted edges use a deformed seed, "
          f"{sum(1 for a in attempts if a.get('reason') == 'deformation budget')} "
          f"rejected by the budget" if deform else "off")),
        "thresholds: " + ', '.join(
            f"{a} pt {thr_pt[a]:.2f} loop {thr_loop[a]:.2f}" for a in 'xyz'),
        "far-plane null q95 (diagnostic only): " + ', '.join(
            f"{a} pt {null_q[a]['pt']:.2f} loop {null_q[a]['loop']:.2f}"
            f"{' self-similar' if inconclusive[a] else ''}" for a in 'xyz'),
        "tracks: " + '; '.join(
            f"t{i} {by_key[n[0]]['axis']} {by_key[n[0]]['off']:+.2f}.."
            f"{by_key[n[-1]]['off']:+.2f} ({len(n)} loops)" for i, n in
            enumerate(tracks, 1)),
    ]
    (d2 / 'summary.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))
    print(f'  wrote {d2}')
    return dict(tracks=len(tracks), stable_kept=len(cat['stable_kept']),
                stable_discarded=len(cat['stable_discarded']),
                adopted=len(cat['adopted']), orphan=len(cat['orphan']))


def step2b(run: Path, out: Path, args):
    return step2(run, out, args, deform=True)


# --------------------------------------------------------------------------- #
# step 3
# --------------------------------------------------------------------------- #
def step3(run: Path, out: Path, args):
    print('\n== step 3: align ==')
    planes, loops = load_run(run, good_only=getattr(args, 'good_only', False))
    by_key = {L['key']: L for L in loops}
    tp = tracks_json(out, args)
    s2 = json.loads(tp.read_text())
    d3 = out / 'step3'
    d3.mkdir(parents=True, exist_ok=True)

    pair_rows, chain_rows, align = [], [], []
    figures = []
    for t in s2['tracks']:
        nodes = t['keys']
        axis = t['axis']
        ls = [by_key[n] for n in nodes]
        loops3, centers, offs = [], [], []
        for L in ls:
            r = resample3(L['block'], K)
            c = r.mean(0)
            x = r - c
            x[:, AX[axis]] = 0.0
            loops3.append(x)
            centers.append(c)
            offs.append(L['off'])
        centers = np.asarray(centers)
        v = loop_coordinates(centers)
        res = synchronize(loops3, centers, offs, lam=args.lam,
                          neighbors=1, min_switch=args.min_switch)
        for p in res.pairs:
            pair_rows.append(dict(track=t['id'], i=p.i, j=p.j,
                                  off_i=offs[p.i], off_j=offs[p.j],
                                  delta=p.delta, D=p.D, s=p.s, keep=p.keep))
        # split at switched-off pairs
        kept = [p.keep for p in res.pairs]
        segs, cur = [], [0]
        for k, kp in enumerate(kept):
            if kp:
                cur.append(k + 1)
            else:
                if len(cur) >= 2:
                    segs.append(cur)
                cur = [k + 1]
        if len(cur) >= 2:
            segs.append(cur)
        for si, seg in enumerate(segs, 1):
            H = [circular_shift(loops3[i], res.alpha[i]) for i in seg]
            seams = np.stack([h[0][list(OTHER[axis])] for h in H])
            spread = max(float(np.linalg.norm(seams[i] - seams[j]))
                         for i in range(len(seams)) for j in range(i))
            ss = [res.pairs[k].s for k in range(len(kept))
                  if seg[0] <= k < seg[-1]]
            dd = [res.pairs[k].D for k in range(len(kept))
                  if seg[0] <= k < seg[-1]]
            csteps = [np.linalg.norm(centers[seg[k + 1]] - centers[seg[k]]) /
                      max(abs(offs[seg[k + 1]] - offs[seg[k]]), 1e-9)
                      for k in range(len(seg) - 1)]
            # Does the section centre advance along the section normal?
            # A generalized cylinder needs most of each centre step on the
            # cut axis; a sliver whose centre slides in-plane is a sheared
            # sheet, not a tube.  drift = in-plane step / axial step.
            drift = []
            for k in range(len(seg) - 1):
                d = centers[seg[k + 1]] - centers[seg[k]]
                a = abs(float(d[AX[axis]]))
                l = float(np.linalg.norm(d - np.eye(3)[AX[axis]] * d[AX[axis]]))
                drift.append(l / max(a, 1e-9))
            chain = dict(
                id=len(chain_rows) + 1, track=t['id'], axis=axis,
                keys=[nodes[i] for i in seg], offsets=[offs[i] for i in seg],
                planes=[by_key[nodes[i]]['plane'] for i in seg],
                n=len(seg), v_span=float(v[seg[-1]] - v[seg[0]]),
                s_min=float(min(ss)) if ss else 1.0,
                s_mean=float(np.mean(ss)) if ss else 1.0,
                D_max=float(max(dd)) if dd else 0.0,
                seam_spread=float(spread),
                seam_rms=float(res.diagnostics.get('residual_rms', 0.0)),
                centre_step_ratio=float(max(csteps)) if csteps else 0.0,
                axis_drift=float(max(drift)) if drift else 0.0,
                alpha=[float(res.alpha[i]) for i in seg],
                cuts=int(len(seg) < len(nodes)))
            chain_rows.append(chain)
            align.append(chain)
            figures.append((t['id'], si, axis, seg, loops3, res.alpha,
                            [nodes[i] for i in seg]))
        # a track that was entirely cut still has no subchain; nothing to add

    fields = ['track', 'i', 'j', 'off_i', 'off_j', 'delta', 'D', 's', 'keep']
    write_csv(d3 / 'pairs.csv', pair_rows, fields)
    write_csv(d3 / 'tracks.csv', chain_rows,
              ['id', 'track', 'axis', 'keys', 'offsets', 'planes', 'n',
               'v_span', 's_min', 's_mean', 'D_max', 'seam_spread', 'seam_rms',
               'centre_step_ratio', 'axis_drift', 'cuts'])
    save_json(d3 / 'align.json', dict(chains=align, params=dict(
        lam=args.lam, min_switch=args.min_switch, k=K)))

    # pair figure
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), dpi=150)
    ax = axs[0]
    for k, p in enumerate(pair_rows):
        ax.scatter(p['D'], p['s'], s=26,
                   c='#2a9d8f' if p['keep'] else '#bf1b1b', zorder=3)
    ax.axhline(args.min_switch, color='0.3', lw=1, ls=':')
    ax.set_xscale('symlog', linthresh=1e-4)
    ax.set_xlabel('normalized pair residual D'); ax.set_ylabel('switch s')
    ax.set_title('every consecutive pair', fontsize=9)
    ax = axs[1]
    for t in s2['tracks']:
        sub = [p for p in pair_rows if p['track'] == t['id']]
        if not sub:
            continue
        ax.plot([p['off_j'] for p in sub], [p['s'] for p in sub], 'o-',
                ms=3.2, lw=0.9, label=f"t{t['id']}")
    ax.axhline(args.min_switch, color='0.3', lw=1, ls=':')
    ax.set_xlabel('plane offset'); ax.set_ylabel('switch s')
    ax.set_title('along each track', fontsize=9)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle('Step 3 — pair residuals and switches', fontsize=11)
    fig.tight_layout(); fig.savefig(d3 / 'pairs.png'); plt.close(fig)

    # seams
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), dpi=150)
    for ch in chain_rows:
        axs[0].plot(ch['offsets'], ch['alpha'], 'o-', ms=3.4, lw=1.0,
                    label=f"c{ch['id']}")
    axs[0].set_xlabel('plane offset'); axs[0].set_ylabel('seam offset alpha (turns)')
    axs[0].set_title('solved seam per loop', fontsize=9)
    axs[0].legend(frameon=False, fontsize=6, ncol=3)
    vals = [p['s'] for p in pair_rows]
    axs[1].hist([p['s'] for p in pair_rows if p['keep']], bins=np.linspace(0, 1, 21),
                color='#2a9d8f', alpha=0.75, label='kept')
    axs[1].hist([p['s'] for p in pair_rows if not p['keep']],
                bins=np.linspace(0, 1, 21), color='#bf1b1b', alpha=0.7,
                label='cut')
    axs[1].axvline(args.min_switch, color='0.3', lw=1, ls=':')
    axs[1].set_xlabel('switch s'); axs[1].set_ylabel('pairs')
    axs[1].legend(frameon=False, fontsize=8)
    axs[1].set_title('kept and cut pairs', fontsize=9)
    fig.suptitle('Step 3 — seams after the cyclic shift', fontsize=11)
    fig.tight_layout(); fig.savefig(d3 / 'seams.png'); plt.close(fig)

    # aligned overlays, pages of up to 6 subchains
    for page in range(0, len(figures), 6):
        chunk = figures[page:page + 6]
        fig, axs = plt.subplots(len(chunk), 3, figsize=(10.5, 2.5 * len(chunk)),
                                dpi=150, squeeze=False)
        for row, (tid, si, axis, seg, loops3, alpha, keys) in enumerate(chunk):
            ij = list(OTHER[axis])
            for col, shifted in ((0, False), (1, True)):
                ax = axs[row, col]
                for k in seg:
                    x = loops3[k]
                    if shifted:
                        x = circular_shift(x, alpha[k])
                    col_k = colormaps['viridis'](k / max(len(seg) - 1, 1))
                    ax.plot(x[:, ij[0]], x[:, ij[1]], lw=0.8, color=col_k)
                    ax.scatter([x[0, ij[0]]], [x[0, ij[1]]], s=12,
                               color=col_k, zorder=3, edgecolor='k',
                               linewidth=0.3)
                ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(('after shift (seam dots)' if shifted else
                              'before shift (seam dots)') +
                             f" — track {tid}.{si}", fontsize=7)
            ax = axs[row, 2]
            allh = []
            for k in seg:
                allh.append(circular_shift(loops3[k], alpha[k])[:, ij])
            allh = np.stack(allh)
            mean = allh.mean(0)
            ax.plot(mean[:, 0], mean[:, 1], color='#d1495b', lw=1.2)
            for h in allh:
                ax.plot(h[:, 0], h[:, 1], color='0.7', lw=0.4)
                ax.scatter([h[0, 0]], [h[0, 1]], s=10, color='0.45',
                           zorder=3, lw=0)
            ax.scatter([mean[0, 0]], [mean[0, 1]], s=16, color='#d1495b',
                       zorder=4, edgecolor='k', linewidth=0.3)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'mean section ({len(seg)} loops)', fontsize=7)
        fig.suptitle(f'Step 3 — synchronized loops, page {page // 6 + 1}',
                     fontsize=11)
        fig.tight_layout()
        name = f'aligned_{page // 6 + 1}.png'
        fig.savefig(d3 / name); plt.close(fig)

    txt = [
        f"tracks source: {tp.parent.name}"
        f"{' (deformed)' if s2.get('deform') else ''}",
        f"tracks in: {len(s2['tracks'])}, aligned sub-chains: {len(chain_rows)}",
        f"pairs: {len(pair_rows)}, kept: {sum(p['keep'] for p in pair_rows)}, "
        f"cuts: {sum(not p['keep'] for p in pair_rows)}",
        "sub-chains: " + '; '.join(
            f"c{ch['id']} t{ch['track']} {ch['axis']} {ch['n']} loops "
            f"v {ch['v_span']:.3f} s {ch['s_min']:.2f} seam {ch['seam_spread']:.4f} "
            f"drift {ch['axis_drift']:.2f}"
            for ch in chain_rows),
        "sheared (axis drift > 2): " + (', '.join(
            f"c{ch['id']} {ch['axis']} drift {ch['axis_drift']:.2f}"
            for ch in chain_rows if ch['axis_drift'] > 2.0) or 'none'),
    ]
    (d3 / 'summary.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))
    print(f'  wrote {d3}')
    return dict(chains=len(chain_rows), pairs=len(pair_rows),
                cuts=sum(not p['keep'] for p in pair_rows))


# --------------------------------------------------------------------------- #
# step 4
# --------------------------------------------------------------------------- #
def build_tube(axis, loops3, centers, v, ids, alpha, nv=32, nz=192):
    H = [circular_shift(loops3[i], alpha[i]) for i in ids]
    ij = list(OTHER[axis])
    H2 = [h[:, ij] for h in H]
    cpath = np.stack([centers[i] for i in ids])
    vv = v[ids]
    vs = np.linspace(vv[0], vv[-1], nv)
    ks = (np.arange(nz) * K // nz).astype(int)
    grid = np.empty((nv, nz, 3))
    for a, q in enumerate(vs):
        k = int(np.clip(np.searchsorted(vv, q) - 1, 0, len(vv) - 2))
        tt = (q - vv[k]) / (vv[k + 1] - vv[k])
        sec = (1 - tt) * H2[k] + tt * H2[k + 1]
        cz = cpath[k] + tt * (cpath[k + 1] - cpath[k])
        grid[a, :, :] = cz[None, :]
        grid[a, :, ij[0]] = cz[ij[0]] + sec[ks, 0]
        grid[a, :, ij[1]] = cz[ij[1]] + sec[ks, 1]
    verts = grid.reshape(-1, 3)
    faces = []
    for a in range(nv - 1):
        for b in range(nz):
            b2 = (b + 1) % nz
            p, q = a * nz + b, a * nz + b2
            r, s = (a + 1) * nz + b, (a + 1) * nz + b2
            faces += [[p, q, s], [p, s, r]]
    return verts, np.array(faces), H


def render_sheets(d4: Path, slices, mesh):
    """Lit six-view sheet and two context views of the exported tube mesh."""
    try:
        import open3d as o3d
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        print(f'  (views skipped: {exc})')
        return
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, float)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))
    om.vertex_colors = o3d.utility.Vector3dVector(
        np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0)
    om.compute_vertex_normals()
    lo = np.minimum(slices.min(0), mesh.vertices.min(0))
    hi = np.maximum(slices.max(0), mesh.vertices.max(0))
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)
    W, H, fov = 760, 620, 36.0
    dist = radius / np.tan(np.deg2rad(fov / 2)) * 1.05
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    r.scene.add_geometry('tubes', om, mat)

    def shot(d, name):
        d = np.asarray(d, float)
        up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else \
            np.array([0.0, 1.0, 0.0])
        r.setup_camera(fov, center, center + d * dist, up)
        im = Image.fromarray(np.asarray(r.render_to_image()))
        ImageDraw.Draw(im).text((12, 10), name, fill=(20, 20, 20))
        return im

    dirs = [((1, 0, 0), '+x'), ((-1, 0, 0), '-x'), ((0, 1, 0), '+y'),
            ((0, -1, 0), '-y'), ((0, 0, 1), '+z'), ((0, 0, -1), '-z')]
    imgs = [shot(d, n) for d, n in dirs]
    sheet = Image.new('RGB', (W * 3, H * 2), 'white')
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % 3) * W, (i // 3) * H))
    sheet.save(d4 / 'tubes_views.png')

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(slices[::2])
    pc.colors = o3d.utility.Vector3dVector(
        np.tile([0.72, 0.72, 0.72], (len(pc.points), 1)))
    mat_pc = o3d.visualization.rendering.MaterialRecord()
    mat_pc.shader = 'defaultUnlit'
    mat_pc.point_size = 1.0
    r.scene.add_geometry('slices', pc, mat_pc)
    views = [shot((0, 0, 1), '+z: slices + tubes'),
             shot((1, 0, 0), '+x: slices + tubes')]
    ctx = Image.new('RGB', (W * 2, H), 'white')
    for i, im in enumerate(views):
        ctx.paste(im, (i * W, 0))
    ctx.save(d4 / 'tubes_context.png')
    print(f'  wrote {d4 / "tubes_views.png"} and {d4 / "tubes_context.png"}')


def step4(run: Path, out: Path, args):
    print('\n== step 4: tubes ==')
    planes, loops = load_run(run, good_only=getattr(args, 'good_only', False))
    by_key = {L['key']: L for L in loops}
    a3 = json.loads((out / 'step3' / 'align.json').read_text())
    a2 = json.loads(tracks_json(out, args).read_text())
    d4 = out / 'step4'
    d4.mkdir(parents=True, exist_ok=True)
    slices = trimesh.load(slices_file(run), process=False).vertices

    # A chain is tube-eligible only when its section centres advance along the
    # section normal.  axis_drift = lateral centre step / axial step; a value
    # above 2 means the "tube" is a sheared sheet following a sliver loop.
    DRIFT_MAX = 2.0
    chains, excluded = [], []
    for ch in a3['chains']:
        if ch.get('axis_drift', 0.0) > DRIFT_MAX:
            excluded.append(dict(
                id=ch['id'], track=ch['track'], axis=ch['axis'],
                keys=ch['keys'], n=ch['n'],
                axis_drift=ch['axis_drift'],
                reason=f'section centre drifts laterally '
                       f'({ch["axis_drift"]:.2f}x the axial step > {DRIFT_MAX})'))
        else:
            chains.append(ch)

    tubes, parts, section_figs = [], [], []
    used = []
    for ci, ch in enumerate(chains):
        axis = ch['axis']
        nodes = ch['keys']
        ls = [by_key[n] for n in nodes]
        loops3, centers = [], []
        for L in ls:
            r = resample3(L['block'], K)
            c = r.mean(0)
            x = r - c
            x[:, AX[axis]] = 0.0
            loops3.append(x)
            centers.append(c)
        centers = np.asarray(centers)
        v = loop_coordinates(centers)
        ids = list(range(len(nodes)))
        nv = max(8, 6 * len(ids))
        verts, faces, H = build_tube(axis, loops3, centers, v, ids,
                                     ch['alpha'], nv=nv)
        c = PALETTE[ci % len(PALETTE)]
        part = colored_mesh(verts, faces, rgb_uint(c))
        parts.append(part)
        # loops and seam connectors as separate components
        up = np.eye(3)[AX[axis]]
        world = [circular_shift(loops3[i], ch['alpha'][i]) + centers[i]
                 for i in range(len(nodes))]
        loop_parts = []
        for h in world:
            lv, lf = ring_tube(h, 0.0035, up=up)
            loop_parts.append(colored_mesh(lv, lf, rgb_uint(c)))
        conn_parts = []
        for k in range(len(world) - 1):
            lv, lf = segment_tube(world[k][0], world[k + 1][0], 0.0025,
                                  nring=6, nseg=8)
            conn_parts.append(colored_mesh(lv, lf, rgb_uint(c)))
        # sections for the figure and metrics
        section_areas = []
        for frac in np.linspace(0, 1, 5):
            q = v[0] + frac * (v[-1] - v[0])
            k = int(np.clip(np.searchsorted(v, q) - 1, 0, len(v) - 2))
            tt = (q - v[k]) / (v[k + 1] - v[k])
            sec = (1 - tt) * H[k][:, list(OTHER[axis])] + \
                tt * H[k + 1][:, list(OTHER[axis])]
            section_areas.append(float(0.5 * abs(
                np.dot(sec[:, 0], np.roll(sec[:, 1], -1)) -
                np.dot(sec[:, 1], np.roll(sec[:, 0], -1)))))
            section_figs.append((ci, frac, sec, c))
        # metrics
        bbox = np.stack([verts.min(0), verts.max(0)])
        s_min = ch['s_min']
        tubes.append(dict(
            id=ci + 1, chain=ch['id'], track=ch['track'], axis=axis,
            keys=nodes, planes=ch['planes'], offsets=ch['offsets'],
            n=len(nodes), v_span=ch['v_span'], s_min=ch['s_min'],
            s_mean=ch['s_mean'], seam_spread=ch['seam_spread'],
            seam_rms=ch['seam_rms'], centre_step_ratio=ch['centre_step_ratio'],
            axis_drift=ch['axis_drift'],
            area_min=min(section_areas), area_max=max(section_areas),
            area_mean=float(np.mean(section_areas)),
            bbox=bbox.tolist(), short=bool(ch['v_span'] < 0.05)))
        used.extend(nodes)
        # loop + connector components for the combined export
        parts.extend(loop_parts)
        parts.extend(conn_parts)

    combined = trimesh.util.concatenate(parts)
    combined.export(d4 / 'tubes.ply')

    # Every loop the export leaves out needs a reason, not just the sheared ones.
    # A loop can be missing because its chain was sheared, because alignment cut
    # the track, because its stable seed found no edge, or because it was never
    # reached at all; these are the four ways the run can discard one.
    used_set = set(used)
    track_of = {k: t['id'] for t in a2['tracks'] for k in t['keys']}
    reason = {}
    for e in excluded:
        for k in e['keys']:
            reason[k] = ('sheared chain', e['reason'])
    for k in track_of:
        if k not in used_set:
            reason.setdefault(
                k, ('seam cut',
                    f'dropped where the seam alignment cut track t{track_of[k]}'))
    for k in a2['categories']['stable_discarded']:
        reason[k] = ('no accepted edge',
                     'stable seed with no accepted propagation edge')
    for k in a2['categories']['orphan']:
        reason[k] = ('not reached',
                     'not a stable seed and never reached by an accepted edge')
    discard_rows = []
    for L in loops:
        if L['key'] in used_set:
            continue
        kind, why = reason.get(L['key'], ('unassigned', 'no reason recorded'))
        discard_rows.append(dict(key=L['key'], axis=L['axis'],
                                 plane=L['plane'], offset=L['off'],
                                 kind=kind, why=why))
    write_csv(d4 / 'discarded.csv', discard_rows,
              ['key', 'axis', 'plane', 'offset', 'kind', 'why'])
    kinds = {}
    for r in discard_rows:
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1

    # cloud + tubes in one file, tube vertices referenced, cloud vertices free
    allv = np.vstack([slices, combined.vertices])
    faces = combined.faces + len(slices)
    cloud = trimesh.Trimesh(allv, faces, process=False)
    ccol = np.vstack([
        np.c_[np.full((len(slices), 3), 200, np.uint8),
              np.full((len(slices), 1), 255, np.uint8)],
        np.asarray(combined.visual.vertex_colors)])
    cloud.visual.vertex_colors = ccol
    cloud.export(d4 / 'tubes_cloud.ply')
    cloud_re = trimesh.load(d4 / 'tubes_cloud.ply', process=False)
    cloud_ok = len(cloud_re.vertices) == len(slices) + len(combined.vertices)

    write_csv(d4 / 'tubes.csv', tubes,
              ['id', 'chain', 'track', 'axis', 'keys', 'planes', 'offsets',
               'n', 'v_span', 's_min', 's_mean', 'seam_spread', 'seam_rms',
               'centre_step_ratio', 'axis_drift', 'area_min', 'area_max',
               'area_mean', 'short'])
    save_json(d4 / 'tubes.json', dict(
        params=dict(tau=args.tau, accept=args.accept, lam=args.lam,
                    min_switch=args.min_switch, k=K, drift_max=DRIFT_MAX),
        tubes=tubes,
        excluded=excluded,
        used=used,
        discarded=sorted([L['key'] for L in loops if L['key'] not in set(used)]),
        discard_reasons=discard_rows,
        n_tubes=len(tubes), n_used=len(used)))

    # 3D figure
    fig = plt.figure(figsize=(13, 7), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    sub = slices[::10]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.22, c='0.86', lw=0)
    # draw each tube with its own colour, sampled vertices (fast and clear)
    ci = 0
    for ch in chains:
        axis = ch['axis']
        nodes = ch['keys']
        ls = [by_key[n] for n in nodes]
        loops3, centers = [], []
        for L in ls:
            r = resample3(L['block'], K); cc = r.mean(0); x = r - cc
            x[:, AX[axis]] = 0.0
            loops3.append(x); centers.append(cc)
        centers = np.asarray(centers)
        v = loop_coordinates(centers)
        nv = max(8, 6 * len(nodes))
        verts, faces, H = build_tube(axis, loops3, centers, v,
                                     list(range(len(nodes))), ch['alpha'],
                                     nv=nv)
        col = PALETTE[ci % len(PALETTE)]
        ax.scatter(verts[::2, 0], verts[::2, 1], verts[::2, 2], s=0.5,
                   c=col, lw=0, alpha=0.85)
        for i in range(len(H)):
            hw = circular_shift(loops3[i], ch['alpha'][i]) + centers[i]
            ax.plot(hw[:, 0], hw[:, 1], hw[:, 2], color=col, lw=1.0)
        ax.plot([], [], color=col, lw=2,
                label=f"tube {ci + 1}: {axis} {len(nodes)} loops")
        ci += 1
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('Step 4 — tubes from the aligned tracks', fontsize=11)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    try:
        ax.set_box_aspect(np.ptp(slices, axis=0))
    except Exception:
        pass
    fig.tight_layout(); fig.savefig(d4 / 'tubes.png'); plt.close(fig)

    # sections figure
    if section_figs:
        cols = 5
        rows = len(tubes)
        fig, axs = plt.subplots(rows, cols, figsize=(1.9 * cols, 1.9 * rows),
                                dpi=150, squeeze=False)
        for ti in range(rows):
            for j in range(cols):
                ax = axs[ti, j]
                item = next((x for x in section_figs
                             if x[0] == ti and abs(x[1] - j / (cols - 1)) < 1e-9),
                            None)
                if item is None:
                    ax.axis('off'); continue
                _, frac, sec, col = item
                ax.plot(sec[:, 0], sec[:, 1], color=col, lw=1.1)
                ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(f'tube {ti + 1}', fontsize=7)
                ax.set_title(f'v {frac:.0%}', fontsize=6.5)
        fig.suptitle('Step 4 — cross-sections along each tube', fontsize=11)
        fig.tight_layout(); fig.savefig(d4 / 'sections.png'); plt.close(fig)

    # reload check
    reloaded = trimesh.load(d4 / 'tubes.ply', process=False)
    comps = reloaded.split(only_watertight=False)
    n_conn = sum(max(ch['n'] - 1, 0) for ch in chains)
    n_exp = len(tubes) + len(used) + n_conn
    bbox = np.stack([reloaded.vertices.min(0), reloaded.vertices.max(0)])
    lo = slices.min(0); hi = slices.max(0)
    overflow = np.maximum(lo - bbox[0], bbox[1] - hi)
    margin = float(np.max(overflow))
    all_keys = [k for ch in chains for k in ch['keys']]
    acc_ok = (len(all_keys) == len(set(all_keys)))
    used_ok = set(used) == set(all_keys)

    fig = plt.figure(figsize=(10, 4.6), dpi=150)
    for j, (elev, azim) in enumerate(((20, -60), (20, 120))):
        ax = fig.add_subplot(1, 2, j + 1, projection='3d')
        vv = np.asarray(reloaded.vertices)
        ax.scatter(vv[::6, 0], vv[::6, 1], vv[::6, 2], s=0.4, c='#4575b4', lw=0)
        ax.set_title(f'reloaded {len(comps)} components', fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect(np.ptp(slices, axis=0))
        except Exception:
            pass
    fig.suptitle(f'Step 4 — reload check: {len(comps)} components '
                 f'(expected {n_exp}), bbox margin {margin:.4f}', fontsize=10)
    fig.tight_layout(); fig.savefig(d4 / 'tubes_check.png'); plt.close(fig)

    txt = [
        f"tubes: {len(tubes)}, loops used: {len(used)}, "
        f"discarded loops: {len(loops) - len(used)} "
        f"({sum(e['n'] for e in excluded)} in sheared chains)",
        "discarded reasons: " + '; '.join(
            f"{k} {kinds.get(k, 0)}" for k in
            ('sheared chain', 'seam cut', 'no accepted edge', 'not reached',
             'unassigned')),
        "tubes: " + '; '.join(
            f"tube {t['id']} {t['axis']} {t['n']} loops v {t['v_span']:.3f} "
            f"s {t['s_min']:.2f} drift {t['axis_drift']:.2f} "
            f"area {t['area_min']:.4f}..{t['area_max']:.4f}"
            f"{' short' if t['short'] else ''}" for t in tubes),
        "excluded as sheared (axis drift > 2): " + ('; '.join(
            f"c{e['id']} {e['axis']} {e['n']} loops drift "
            f"{e['axis_drift']:.2f}" for e in excluded) or 'none'),
        f"reload components: {len(comps)} (expected {n_exp}: "
        f"{len(tubes)} tubes + {len(used)} loops + {n_conn} connectors)",
        f"reload bbox margin past the slice cloud: {margin:.4f}",
        f"loop accounting: unique {acc_ok}, used set equals aligned set {used_ok}",
        f"tubes_cloud.ply keeps all slice vertices: {cloud_ok} "
        f"({len(slices)} slices + {len(combined.vertices)} tube/loop vertices)",
    ]
    (d4 / 'summary.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))
    render_sheets(d4, slices, combined)
    print(f'  wrote {d4}')
    return dict(tubes=len(tubes), used=len(used), comps=len(comps),
                expected=n_exp, margin=margin, acc_ok=bool(acc_ok and used_ok))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def write_summary(out: Path, params, chunks):
    lines = ['# Loop propagation results', '',
             f"Run: `{params['run']}`", '',
             f"Parameters: RMS cut {params['rms']}, tau {params['tau']}, "
             f"accept {params['accept']}, lambda {params['lam']}, "
             f"min switch {params['min_switch']}, K {params['k']}.", '']
    if params.get('deform_modes'):
        lines += [f"Deformation: {params['deform_modes']} Fourier modes, "
                  f"budget {params['deform_budget']:.2f} of the seed radius.",
                  '']
    for title, body in chunks:
        lines += [f'## {title}', '', body, '']
    (out / 'summary.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='log/laurent_hand_normal_cubic_lam3')
    ap.add_argument('--out', default='log/loop_propagation')
    ap.add_argument('--step', default='all',
                    choices=['1', '2', '2b', '3', '4', 'all'])
    ap.add_argument('--tracks', default='auto',
                    choices=['auto', 'step2', 'step2b'])
    ap.add_argument('--rms', type=float, default=0.025)
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--accept', type=float, default=0.5)
    ap.add_argument('--lam', type=float, default=0.02)
    ap.add_argument('--min-switch', type=float, default=0.3)
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--deform-modes', type=int, default=8)
    ap.add_argument('--deform-budget', type=float, default=0.15)
    ap.add_argument('--good-only', action='store_true',
                    help='drop every loop on the bad side of the run\'s '
                         '<stem>_good_loops.ply split before propagation, so the '
                         'tubes are built from good loops alone')
    args = ap.parse_args()
    run = (ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'run.json', dict(
        run=str(run), rms=args.rms, tau=args.tau, accept=args.accept,
        lam=args.lam, min_switch=args.min_switch, k=args.k,
        deform_modes=args.deform_modes, deform_budget=args.deform_budget,
        good_only=args.good_only))
    if args.step in ('1', 'all'):
        r1 = step1(run, out, args)
    if args.step in ('2', 'all'):
        r2 = step2(run, out, args)
    if args.step == '2b':
        r2b = step2b(run, out, args)
    if args.step in ('3', 'all'):
        r3 = step3(run, out, args)
    if args.step in ('4', 'all'):
        r4 = step4(run, out, args)
    # Always assemble the top-level index from every step summary present, so a
    # single-step rerun does not truncate summary.md to that step.
    labels = {'1': 'Step 1 — stable loops',
              '2': 'Step 2 — propagation',
              '2b': 'Step 2b — deformation-budgeted propagation',
              '3': 'Step 3 — alignment',
              '4': 'Step 4 — tubes'}
    chunks = []
    for st in ('1', '2', '2b', '3', '4'):
        f = out / f'step{st}' / 'summary.txt'
        if f.exists():
            chunks.append((labels[st], f.read_text()))
    if chunks:
        write_summary(out, dict(run=run, rms=args.rms, tau=args.tau,
                                accept=args.accept, lam=args.lam,
                                min_switch=args.min_switch, k=args.k,
                                deform_modes=args.deform_modes,
                                deform_budget=args.deform_budget), chunks)


if __name__ == '__main__':
    main()
