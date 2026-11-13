"""Why does a loop crossing have no partner on the other loop?

`dual_check.py` counts the crossings of two loops that the other loop does not
confirm.  Here each of them is walked back through the pipeline, at the point
where loop A passes through the plane of loop B:

    1   A's curve is off the surface there (the fit of A is wrong)
    2a  B's plane dropped that surface: a flat face parallel to it (--max-align)
    2b  B's plane has no points there
    3   B's plane has the points but gave them to no loop
    4   a loop of B's plane owns the points, but its curve runs elsewhere
    5   both curves are there but do not meet (near miss, or running alongside)

1, 4 and 5 are the fit; 2 and 3 are the extraction.

    python utils/dual_cause.py log/<run>/fit
    python utils/dual_cause.py log/<run>/fit path/to/cloud.ply

The run has to hold its per-plane files (`planes/`, written by
`--save-plane-files`).  Without a cloud path, the shape is looked up by the
run's name in the stylized dataset folder.  Only the good loops are read.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import _paths  # noqa: E402
from dual_loops import AXES, LoopStructure, load_loops
from model.loop_extraction import load_points

DATA = _paths._ROOT / 'Datasets/normalized_shapes_cuboid/normalized_shapes_cuboid'


def read_planes(run):
    """Per plane (axis, offset): the sliced points and which of them a loop owns."""
    planes = {}
    for line in (run / 'planes.txt').read_text().splitlines():
        hit = re.match(r'plane_(\d+) ([xyz]) ([+-][\d.]+)', line)
        if not hit:
            continue
        i, ax, off = int(hit.group(1)), AXES.index(hit.group(2)), float(hit.group(3))
        files = sorted((run / 'planes').glob(f'plane_{i:02d}_*_loops.ply'))
        if not files:
            planes[(ax, off)] = (np.zeros((0, 3)), np.zeros(0, bool))
            continue
        mesh = trimesh.load(files[0], process=False)
        v = np.asarray(mesh.vertices, float)
        c = np.asarray(mesh.visual.vertex_colors[:, :3])
        grey = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2])      # points of no loop are grey
        planes[(ax, off)] = (v, ~grey)
    return planes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path, help='folder of a finished fit, with planes/')
    ap.add_argument('cloud', type=Path, nargs='?', default=None,
                    help='the point cloud the run was fitted to (default: looked up '
                         'by the run\'s shape name in the dataset folder)')
    ap.add_argument('--gate', type=float, default=0.04,
                    help='matching distance of two crossings (default 0.04)')
    ap.add_argument('--radius', type=float, default=0.02,
                    help='"there" for a point against points (default 0.02)')
    args = ap.parse_args()
    run, R = args.run, args.radius
    if not (run / 'planes.txt').exists() or not (run / 'planes').is_dir():
        raise SystemExit(f'{run}: no planes.txt / planes/ here; refit with --save-plane-files')
    cloud = args.cloud
    if cloud is None:
        hits = sorted(run.glob('*_triplane.npz'))
        shape = hits[0].name[:-len('_triplane.npz')] if hits else ''
        cloud = DATA / f'{shape}.ply'
        if not cloud.exists():
            raise SystemExit(f'{cloud} not found; pass the cloud as the second argument')

    loops = load_loops(run, good_only=True)
    st = LoopStructure(loops, gate=args.gate)
    print(f'{run}: {len(loops)} good loops, {len(st.verts)} matched crossings, '
          f'{len(st.loose)} unmatched')
    pc, pcn = load_points(cloud)
    pcn = pcn / np.linalg.norm(pcn, axis=1, keepdims=True)
    tree = cKDTree(pc)
    spacing = float(np.median(tree.query(pc[::50], k=2)[0][:, 1]))
    print(f'  cloud {len(pc)} points, median spacing {spacing:.4f}')

    planes = read_planes(run)
    ptree = {k: (cKDTree(v) if len(v) else None, cKDTree(v[own]) if own.any() else None)
             for k, (v, own) in planes.items()}

    def trees_of(ax, off):
        return ptree[min((k for k in planes if k[0] == ax), key=lambda k: abs(k[1] - off))]

    why, where = Counter(), []
    for c, B, b in st.loose:
        p = c['pos']
        d, k = tree.query(p)
        A = st.L_of[c['lid']]['axis']
        if d > R:
            r = '1 this curve is off the surface'
        else:
            t_all, t_own = trees_of(B, b)
            d_all = t_all.query(p)[0] if t_all is not None else 9
            d_own = t_own.query(p)[0] if t_own is not None else 9
            if d_all > R:
                r = ('2a other plane dropped the surface there: flat face parallel to it'
                     if abs(pcn[k][B]) > 0.9 else '2b other plane has no points there')
            elif d_own > R:
                r = '3 other plane kept the points but gave them to no loop'
            else:
                dc = min((st.trees[M['id']].query(p)[0] for M in loops
                          if M['axis'] == B and abs(M['off'] - b) < 2e-3), default=9)
                r = ('4 other loop owns the points but its curve is elsewhere'
                     if dc > 2 * R else
                     '5 both curves are there but do not meet (near miss / running alongside)')
        why[r] += 1
        where.append((r, abs(pcn[k][B]), abs(pcn[k][A])))
    n = max(sum(why.values()), 1)
    for r, c in sorted(why.items()):
        print(f'  {c:5d} {c / n:6.1%}  {r}')
    nm = np.array([(nb, na) for r, nb, na in where if r.startswith('5')])
    if len(nm):
        print(f'  near misses: surface normal along the crossed plane\'s axis > 0.8 in '
              f'{np.mean(nm[:, 0] > 0.8):.1%}, along the loop\'s own axis > 0.8 in '
              f'{np.mean(nm[:, 1] > 0.8):.1%}')
    print(f'  points with a normal within 18 deg of an axis: '
          f'{np.mean(np.abs(pcn).max(1) > 0.95):.1%}')

    ## the fit itself, both ways: curve to cloud, and owned slice points to curve
    dd = np.concatenate([tree.query(L['P'])[0] for L in loops])
    print(f'  fitted curve to cloud: median {np.median(dd):.4f}, p90 {np.quantile(dd, .9):.4f}, '
          f'share over {R}: {np.mean(dd > R):.1%}, over {2 * R}: {np.mean(dd > 2 * R):.1%}')
    miss = []
    for (ax, off), (v, own) in planes.items():
        cur = [L['P'] for L in loops if L['axis'] == ax and abs(L['off'] - off) < 2e-3]
        if own.any() and cur:
            miss.append(cKDTree(np.vstack(cur)).query(v[own])[0])
    if miss:
        miss = np.concatenate(miss)
        print(f'  owned slice points to their plane\'s curves: median {np.median(miss):.4f}, '
              f'p90 {np.quantile(miss, .9):.4f}, share over {R}: {np.mean(miss > R):.1%}, '
              f'over {2 * R}: {np.mean(miss > 2 * R):.1%}')
    tot = sum(len(v) for v, _ in planes.values())
    un = sum(int((~o).sum()) for _, o in planes.values())
    print(f'  slice points given to no loop: {un} of {tot} ({un / max(tot, 1):.1%})')


if __name__ == '__main__':
    main()
