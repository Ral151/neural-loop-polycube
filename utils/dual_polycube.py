"""Polycube from the fitted loops of a run, by the dual-loop construction.

The loops of a run are not a valid polycube loop structure as a whole (see
`dual_check.py`), so a jointly valid subset is selected: start from a cube,
one x-, one y- and one z-loop whose pairs cross exactly twice, and keep adding
loops while the set stays valid.  This is repeated from many starting cubes and
the best structure is turned into a polycube.  No loop is created or changed.

    python utils/dual_polycube.py log/<run>/fit
    python utils/dual_polycube.py log/<run>/fit --out log/<run>/polycube --starts 400
    python utils/dual_polycube.py log/<run>/fit --starts 1        # one start, quick

Written to `--out` (default `<run>/../polycube`):

    dual_primal.png        kept loops, quad layout on the shape, polycube; two views
    polycube.obj           the polycube (quads)
    layout_on_shape.obj    the same quads with each corner at its region on the shape
    structure.npz          loops (kept loop ids), quads, labels (axis, sign),
                           cube_v, shape_v
    kept_loops.txt         the kept loops: id, axis, plane offset, slot

Read-only on the run folder.  `dual_loops.py` describes the construction; the
polycube geometry is this repository's reading of the zones, not the paper's.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import _paths  # noqa: E402,F401
from dual_loops import (AXES, LoopStructure, load_loops, plot_polycube, polycube,
                        save_polycube)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path, help='folder of a finished fit')
    ap.add_argument('--out', type=Path, default=None,
                    help='output folder (default: polycube/ next to the run folder)')
    ap.add_argument('--gate', type=float, default=0.04,
                    help='two crossings on a shared line are one when they agree '
                         'within this distance (default 0.04)')
    ap.add_argument('--starts', type=int, default=400,
                    help='starting cubes to grow from; 1 uses only the cube with '
                         'the longest loops (default 400)')
    ap.add_argument('--seconds', type=float, default=0.0,
                    help='stop trying new starting cubes after this long (default 0: no limit)')
    ap.add_argument('--seed', type=int, default=0,
                    help='order in which the starting cubes are tried (default 0)')
    ap.add_argument('--min-gap', type=float, default=0.0,
                    help='leave out a loop whose centre is closer than this to a kept '
                         'loop of its own axis: a coarser polycube (default 0)')
    ap.add_argument('--near', type=float, default=0.0,
                    help='also forbid a pair of loops when one has an unmatched crossing '
                         'within this distance of the other (default 0: off)')
    ap.add_argument('--pad', type=float, default=0.05,
                    help='how far the end faces of the polycube sit beyond the '
                         'outermost loop (default 0.05)')
    ap.add_argument('--with-bad', action='store_true',
                    help='also offer the loops the fit marked bad (default: good loops only)')
    args = ap.parse_args()
    out = args.out or args.run.parent / 'polycube'

    loops = load_loops(args.run, good_only=not args.with_bad)
    print(f'{args.run}: {len(loops)} loops offered')
    st = LoopStructure(loops, gate=args.gate, near=args.near)
    print(f'  {len(st.verts)} matched crossings, {len(st.loose)} unmatched, '
          f'{len(st.conflict)} conflicting loop pairs')

    S = st.search(starts=args.starts, seconds=args.seconds, seed=args.seed,
                  min_gap=args.min_gap)
    pc = polycube(st, S, pad=args.pad)
    m = pc['map']
    V, E, F = len(m['vs']), len(m['seg_loop']), len(m['faces'])
    per_axis = Counter(AXES[st.L_of[l]['axis']] for l in S)
    print(f'  valid loop structure: {len(S)} loops {dict(sorted(per_axis.items()))}, '
          f'{V} crossings, {E} segments, {F} regions, chi {V - E + F}')
    print('  region sizes:', dict(sorted(Counter(len(f) for f in m['faces']).items())))
    print('  left out:', dict(st.left_out(S)))
    print('  zones per axis:', dict(zip(AXES, pc['zones'])),
          '; every zone lies between its two loops:', pc['ordered'])

    cube_v, quads, labels = pc['cube_v'], pc['quads'], pc['labels']
    flat = sum(len(set(np.round(cube_v[q, lab[0]], 9))) == 1 for q, lab in zip(quads, labels))
    degen = sum(len({tuple(np.round(cube_v[i], 9)) for i in q}) < 4 for q in quads)
    count = Counter(('+' if s > 0 else '-') + AXES[c].upper() for c, s in labels)
    print(f'  polycube: {F} corners, {len(quads)} quad faces, {flat} of them flat on '
          f'their own axis, {degen} degenerate; labels {dict(sorted(count.items()))}')

    save_polycube(pc, out)
    with open(out / 'kept_loops.txt', 'w') as fh:
        fh.write('# id axis offset slot\n')
        for l in sorted(S, key=lambda l: (st.L_of[l]['axis'], st.L_of[l]['off'])):
            L = st.L_of[l]
            fh.write(f'{l} {AXES[L["axis"]]} {L["off"]:+.4f} {L["slot"]}\n')
    plot_polycube(st, pc, out / 'dual_primal.png')
    print(f'  wrote {out}/: dual_primal.png, polycube.obj, layout_on_shape.obj, '
          f'structure.npz, kept_loops.txt')


if __name__ == '__main__':
    main()
