"""Write the good loops with vertices coloured by their slicing axis.

Every loop belongs to an x-, y- or z-plane; colouring whole loop blocks by axis
makes the three families visible in any PLY viewer.  `plc_main` now writes this
colouring into `*_good_loops.ply` directly, so this script is for runs made
before that and leaves the main file untouched.

Usage
-----
    python utils/axis_color_loops.py --run log/cmp_tri_tv01_lucy
    python utils/axis_color_loops.py --run log/cmp_tri_tv01_bunny --preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

import _paths  # noqa: E402
import loop_propagation as lp  # noqa: E402
from model.loop_extraction import AXIS_COLORS, AXIS_HEX  # noqa: E402

AXCOL = dict(zip('xyz', AXIS_HEX))
AXRGB = dict(zip('xyz', AXIS_COLORS))


def write_ply(run: Path, name_suffix: str):
    _, loops = lp.load_run(run, good_only=True)
    blocks = [L['block'] for L in loops]
    colors = [np.tile(AXRGB[L['axis']], (len(L['block']), 1))
              for L in loops]
    mesh = trimesh.Trimesh(vertices=np.vstack(blocks),
                           vertex_colors=np.vstack(colors), process=False)
    src = sorted(run.glob('*_good_loops.ply'))[0]
    dst = src.with_name(src.name.replace('_good_loops.ply', name_suffix))
    mesh.export(dst)
    counts = {a: sum(L['axis'] == a for L in loops) for a in 'xyz'}
    print(f'{src.name}: {len(loops)} loops '
          f'(x {counts["x"]}, y {counts["y"]}, z {counts["z"]}) -> {dst.name}')
    return dst, loops


def preview(dst: Path, loops, run: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    allv = np.vstack([L['block'] for L in loops])
    fig = plt.figure(figsize=(10, 9), dpi=150)
    for k, (name, el, az) in enumerate(
            (('isometric', 22, -55), ('front', 8, -90),
             ('side', 8, 0), ('top', 90, -90)), 1):
        ax = fig.add_subplot(2, 2, k, projection='3d')
        for a in 'xyz':
            pts = np.vstack([L['block'] for L in loops if L['axis'] == a])
            ax.scatter(pts[::5, 0], pts[::5, 1], pts[::5, 2], s=0.25,
                       c=AXCOL[a], lw=0)
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect(np.ptp(allv, axis=0))
        ax.set_axis_off()
        ax.set_title(name, fontsize=9)
    handles = [Line2D([], [], marker='o', ls='', ms=8, color=AXCOL[a],
                      label=f'{a} loops') for a in 'xyz']
    fig.legend(handles=handles, loc='lower center', ncol=3,
               frameon=False, fontsize=10)
    fig.suptitle(f'{run.name} — good loops coloured by plane axis', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    png = dst.with_suffix('.png')
    fig.savefig(png, facecolor='white')
    plt.close(fig)
    print(f'  preview {png}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', default='log/cmp_tri_tv01_lucy')
    ap.add_argument('--suffix', default='_good_loops_xyz.ply')
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    run = Path(a.run)
    dst, loops = write_ply(run, a.suffix)
    if a.preview:
        preview(dst, loops, run)


if __name__ == '__main__':
    main()
