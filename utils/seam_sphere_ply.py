"""Write a loop PLY with a red sphere on every loop's seam start point.

The run's `*_good_loops.ply` stores every loop as one M-vertex block, and the
block's first vertex is the loop's seam sample (z = 0 in its plane frame).  This
script leaves the loop vertices and colours untouched and appends one red
icosphere centred on that first vertex, so a viewer shows the seam start
directly in 3D.

Usage
-----
    python plc_utils/seam_sphere_ply.py                       # lucy triplane+TV run
    python plc_utils/seam_sphere_ply.py --run log/cmp_tri_tv01_lucy
    python plc_utils/seam_sphere_ply.py --radius-frac 0.014 --preview
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import trimesh

RED = np.array([255, 0, 0, 255], np.uint8)


def loop_blocks(run: Path):
    good = sorted(run.glob('*_good_loops.ply'))
    if not good:
        raise SystemExit(f'{run}: no *_good_loops.ply')
    mesh = trimesh.load(good[0], process=False)
    verts = np.asarray(mesh.vertices, float)
    colors = np.asarray(mesh.visual.vertex_colors, np.uint8).copy()

    log = Path(str(run) + '.log')
    found = re.findall(r'M = (\d+) points each', log.read_text())
    if not found:
        raise SystemExit(f'{log}: no "M = ... points each" line')
    M = int(found[-1])
    if len(verts) % M:
        raise SystemExit(f'{good[0]}: {len(verts)} vertices do not split into '
                         f'blocks of M = {M}')
    return good[0], verts.reshape(-1, M, 3), colors.reshape(-1, M, 4)


def write_ply(run: Path, radius_frac: float, subdiv: int):
    src, loops, colors = loop_blocks(run)
    seams = loops[:, 0, :]
    extent = float(np.ptp(loops.reshape(-1, 3), axis=0).max())
    radius = radius_frac * extent

    proto = trimesh.creation.icosphere(subdivisions=subdiv, radius=radius)
    pv = np.asarray(proto.vertices)
    pf = np.asarray(proto.faces, np.int64)
    sv = (seams[:, None, :] + pv[None]).reshape(-1, 3)
    sf = (pf[None] + len(pv) * np.arange(len(seams))[:, None, None]
          ).reshape(-1, 3)

    loop_v = loops.reshape(-1, 3)
    out = trimesh.Trimesh(
        vertices=np.vstack([loop_v, sv]),
        faces=sf + len(loop_v),
        vertex_colors=np.vstack([colors.reshape(-1, 4),
                                 np.tile(RED, (len(sv), 1))]),
        process=False)
    dst = src.with_name(src.name.replace('_good_loops.ply',
                                         '_good_loops_seam_red.ply'))
    out.export(dst)

    back = trimesh.load(dst, process=False)
    centres = np.asarray(back.vertices[len(loop_v):], float).reshape(
        len(seams), -1, 3).mean(1)
    err = float(np.abs(centres - seams).max())
    print(f'{src.name}: {len(seams)} loops, M = {loops.shape[1]}, '
          f'radius = {radius:.4f}')
    print(f'  wrote {dst} ({len(back.vertices)} verts, {len(back.faces)} faces)')
    print(f'  sphere centres vs seam vertices: max error {err:.2e}')
    return dst


def preview(dst: Path, run: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, loops, _ = loop_blocks(run)
    v = loops.reshape(-1, 3)
    red = np.asarray(trimesh.load(dst, process=False).visual.vertex_colors)
    fig = plt.figure(figsize=(11, 9), dpi=150)
    for k, (name, el, az) in enumerate(
            (('isometric', 22, -55), ('front', 8, -90),
             ('side', 8, 0), ('top', 90, -90)), 1):
        ax = fig.add_subplot(2, 2, k, projection='3d')
        ax.scatter(v[::6, 0], v[::6, 1], v[::6, 2], s=0.2, c='#9ecae1', lw=0)
        ax.scatter(loops[:, 0, 0], loops[:, 0, 1], loops[:, 0, 2],
                   s=18, c='#d62728', depthshade=False, zorder=5)
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect(np.ptp(v, axis=0))
        ax.set_axis_off()
        ax.set_title(name, fontsize=9)
    fig.suptitle(f'{run.name} — {len(loops)} loops, red spheres at seam starts',
                 fontsize=11)
    fig.tight_layout()
    png = dst.with_suffix('.png')
    fig.savefig(png, facecolor='white')
    plt.close(fig)
    print(f'  preview {png}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', default='log/cmp_tri_tv01_lucy')
    ap.add_argument('--radius-frac', type=float, default=0.010,
                    help='sphere radius as a fraction of the largest extent '
                         '(default 0.010)')
    ap.add_argument('--subdiv', type=int, default=2)
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    run = Path(a.run)
    dst = write_ply(run, a.radius_frac, a.subdiv)
    if a.preview:
        preview(dst, run)


if __name__ == '__main__':
    main()
