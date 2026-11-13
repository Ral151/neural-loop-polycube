"""Do the fitted loops of a run form a polycube loop structure?

Every x-, y- and z-loop pair is intersected on the line the two planes share
and the two sides are matched; the matched crossings become the vertices of a
map whose faces are the regions between the loops.  The report says how far
the whole set is from a valid structure (Snoep et al., Definition 3): a valid
genus-0 structure has every crossing matched, one component, Euler
characteristic 2, and no region breaking rule 2 or 3.

    python utils/dual_check.py log/<run>/fit
    python utils/dual_check.py log/<run>/fit --good-only --gate 0.03

Read-only on the run folder.  `dual_loops.py` describes the construction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import _paths  # noqa: E402,F401
from dual_loops import AXES, LoopStructure, load_loops


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path, help='folder of a finished fit')
    ap.add_argument('--gate', type=float, default=0.04,
                    help='two crossings on a shared line are one when they agree '
                         'within this distance (default 0.04)')
    ap.add_argument('--good-only', action='store_true',
                    help='leave the loops marked bad out (default: all loops)')
    args = ap.parse_args()

    loops = load_loops(args.run, good_only=args.good_only)
    st = LoopStructure(loops, gate=args.gate)
    stat, dz = st.stat, st.dz
    print(f'{args.run}: {len(loops)} loops, gate {args.gate}')
    print('  planes holding a loop per axis:',
          [len({L['off'] for L in loops if L['axis'] == a}) for a in range(3)])
    if not len(st.verts):
        raise SystemExit('  no matched crossing at all')
    unmatched = {c['lid'] for c, _, _ in st.loose}
    print(f"  lines (plane-plane) carrying a crossing: {stat['lines']}, same count on "
          f"both sides {stat['lines_equal_count']}, every crossing matched "
          f"{stat['lines_clean']}")
    print(f"  crossings {stat['cross']}, unmatched {stat['cross_unmatched']} "
          f"({stat['cross_unmatched'] / max(stat['cross'], 1):.1%}); "
          f"loops with an unmatched crossing {len(unmatched)} of {len(loops)}")
    print(f'  matched pairs (vertices) {len(st.verts)}: mismatch along the line median '
          f'{np.median(dz):.4f}, p90 {np.quantile(dz, .9):.4f}, max {dz.max():.4f}')
    odd = sum(k % 2 for k in st.pair_count.values())
    print(f'  loop pairs that cross: {len(st.pair_count)}, an odd number of times: {odd}')

    m = st.build([L['id'] for L in loops], drop_isolated=True)
    V, E, F = len(m['vs']), len(m['seg_loop']), len(m['faces'])
    comps = st.components(m)
    print(f'  normal sign disagreement between the two loops at a vertex: '
          f'{st.sign_disagree} of {V}')
    print(f'  loops with no intersection at all (isolated): {len(loops) - len(m["S"])}')
    print(f'  map: V {V}, E {E}, F {F}, components {len(comps)} '
          f'(largest {comps[0]} loops), chi {V - E + F} '
          f'(a valid genus-0 structure reads 2 with 1 component)')
    print('  region sizes (segments on the boundary):',
          dict(sorted(Counter(len(f) for f in m['faces']).items())))
    few = sum(len(f) < 3 for f in m['faces'])
    dup = sum(max(st.region_labels(m, f).values()) > 1 for f in m['faces'])
    print(f'  regions with < 3 segments (rule 2): {few}; regions with a repeated '
          f'axis+side label (rule 3): {dup}; of {F}')
    per_axis = Counter(AXES[st.L_of[l]['axis']] for l in m['S'])
    print('  loops in the map per axis:', dict(sorted(per_axis.items())))


if __name__ == '__main__':
    main()
