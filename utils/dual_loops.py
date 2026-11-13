"""Dual-loop reading of a fitted run: shared code of the `dual_*.py` scripts.

The fitted cross-section loops of the x-, y- and z-planes are treated as the
X-, Y- and Z-loops of a polycube loop structure (Snoep, Speckmann, Verbeek,
"Robust Construction of Polycube Segmentations via Dual Loops", arXiv
2402.00652).  Nothing here needs a mesh:

* an x-loop lies in a plane x = a and a y-loop in a plane y = b, so the two can
  only cross on the line {x = a, y = b}.  The crossings of both loops with that
  line are matched along it (`match_crossings`);
* the matched crossings are the vertices of a combinatorial map.  The two loop
  tangents and the surface normal at a crossing give the order of the four
  loop segments round it, and walking that order traces the regions
  (`LoopStructure.build`);
* a set of loops is a polycube loop structure when the map is one sphere and no
  region breaks the paper's Definition 3 (`LoopStructure.problems`);
* a valid subset is grown from a starting cube, one x-, one y- and one z-loop
  (`LoopStructure.search`), and turned into a polycube: a region is a corner, a
  crossing is a quad face, and the coordinates come from the zones
  (`polycube`).

Everything is read-only on the run folder.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import _paths  # noqa: E402,F401
from model.loop_extraction import load_points

AXES = 'xyz'


def _one(run, pattern):
    hits = sorted(Path(run).glob(pattern))
    if not hits:
        raise SystemExit(f'{run}: no {pattern} here; pass the folder of a finished fit '
                         '(the --out of plc_main.py --train)')
    return hits[0]


def load_loops(run, good_only=True):
    """The fitted loops of a run, as dicts (id, axis, off, slot, good, P, N, centre, length).

    `id` is the loop's index in the run's `*_triplane.npz` keys, so it stays the
    same whether the bad loops are read or not.
    """
    run = Path(run)
    keys = np.load(_one(run, '*_triplane.npz'))['keys']
    flags = _one(run, '*_loop_split.txt').read_text().strip()
    blocks = [None] * len(keys)
    for flag, name in (('g', 'good_loops'), ('b', 'bad_loops')):
        idx = [i for i, c in enumerate(flags) if c == flag]
        hits = sorted(run.glob(f'*_{name}.ply'))
        if not idx or not hits:
            continue
        P, N = load_points(hits[0])
        per = len(P) // len(idx)
        for j, i in enumerate(idx):
            blocks[i] = (P[j * per:(j + 1) * per], N[j * per:(j + 1) * per])
    loops = []
    for i, (key, blk) in enumerate(zip(keys, blocks)):
        if blk is None or (good_only and flags[i] != 'g'):
            continue
        P, N = blk
        loops.append(dict(
            id=i, axis=int(key[0]), off=float(key[1]), slot=int(key[2]),
            good=flags[i] == 'g', P=P, N=N, centre=P.mean(0),
            length=float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())))
    return loops


def loop_crossings(L, B, b):
    """Where loop L passes through the plane `coordinate B = b`."""
    P, N = L['P'], L['N']
    f = P[:, B] - b
    g = np.roll(f, -1)
    res = []
    for k in np.flatnonzero((f * g < 0) | ((f == 0) & (g != 0))):
        k2 = (k + 1) % len(P)
        t = f[k] / (f[k] - g[k])
        res.append(dict(lid=L['id'], param=(k + t) / len(P),
                        pos=P[k] + t * (P[k2] - P[k]), tan=P[k2] - P[k], nrm=N[k]))
    return res


def match_crossings(loops, gate=0.04):
    """Every crossing of two loops of different axes, matched on the line they share.

    Returns `verts`, the matched pairs `(crossing of the first axis, crossing of
    the second)`; `loose`, the unmatched ones as `(crossing, crossed axis,
    crossed offset)`; `dz`, the disagreement of each matched pair along its
    line; and `stat`, counts per line.
    """
    offsets = [sorted({L['off'] for L in loops if L['axis'] == a}) for a in range(3)]
    verts, loose, dz, stat = [], [], [], Counter()
    for A in range(3):
        for B in range(A + 1, 3):
            C = 3 - A - B
            for a in offsets[A]:
                for b in offsets[B]:
                    ca = [c for L in loops if L['axis'] == A and L['off'] == a
                          for c in loop_crossings(L, B, b)]
                    cb = [c for L in loops if L['axis'] == B and L['off'] == b
                          for c in loop_crossings(L, A, a)]
                    if not ca and not cb:
                        continue
                    ma, mb = set(), set()
                    if ca and cb:
                        cost = np.abs(np.array([c['pos'][C] for c in ca])[:, None]
                                      - np.array([c['pos'][C] for c in cb])[None, :])
                        for i, j in zip(*linear_sum_assignment(cost)):
                            if cost[i, j] <= gate:
                                ma.add(i), mb.add(j)
                                dz.append(cost[i, j])
                                verts.append((ca[i], cb[j]))
                    loose += [(c, B, b) for i, c in enumerate(ca) if i not in ma]
                    loose += [(c, A, a) for j, c in enumerate(cb) if j not in mb]
                    stat['lines'] += 1
                    stat['lines_equal_count'] += len(ca) == len(cb)
                    stat['lines_clean'] += len(ma) == len(ca) and len(mb) == len(cb)
                    stat['cross'] += len(ca) + len(cb)
    stat['cross_unmatched'] = len(loose)
    return verts, loose, np.array(dz), stat


class LoopStructure:
    """The loops of a run with their matched crossings, and the subsets that are valid.

    `gate`: two crossings on a shared line are one vertex when they agree
    within this distance.  `near`: an unmatched crossing this close to a loop
    of the crossed plane makes the two loops a conflicting pair (0 turns this
    off; an odd crossing count always is a conflict, because two closed curves
    on a sphere cross an even number of times).
    """

    def __init__(self, loops, gate=0.04, near=0.0):
        self.loops = loops
        self.L_of = {L['id']: L for L in loops}
        self.gate = gate
        self.verts, self.loose, self.dz, self.stat = match_crossings(loops, gate)
        n = len(self.verts)
        self.va = np.array([ca['lid'] for ca, _ in self.verts], dtype=int)
        self.vb = np.array([cb['lid'] for _, cb in self.verts], dtype=int)
        self.vpos = np.array([0.5 * (ca['pos'] + cb['pos'])
                              for ca, cb in self.verts]).reshape(n, 3)
        ## the surface normal says which way round the four segments go; the two
        ## loops each carry one, and `sign_disagree` counts where they differ
        self.vsign, self.vnrm = np.ones(n), np.zeros((n, 3))
        self.sign_disagree = 0
        for v, (ca, cb) in enumerate(self.verts):
            x = np.cross(ca['tan'], cb['tan'])
            s = 1.0 if x @ ca['nrm'] > 0 else -1.0
            self.sign_disagree += np.sign(x @ ca['nrm']) != np.sign(x @ cb['nrm'])
            self.vsign[v] = s
            self.vnrm[v] = s * x / (np.linalg.norm(x) + 1e-12)

        self.trees = {L['id']: cKDTree(L['P']) for L in loops}
        self.conflict = set()
        for c, B, b in self.loose:
            for M in loops:
                if (M['axis'] == B and M['off'] == b
                        and self.trees[M['id']].query(c['pos'])[0] < near):
                    self.conflict.add(frozenset((c['lid'], M['id'])))
        self.pair_count = Counter(zip(self.va.tolist(), self.vb.tolist()))
        for (i, j), k in self.pair_count.items():
            if k % 2:
                self.conflict.add(frozenset((i, j)))
        self.by_loop = defaultdict(list)
        for v in range(n):
            self.by_loop[int(self.va[v])].append(v)
            self.by_loop[int(self.vb[v])].append(v)
        self.param_on = [{int(self.va[v]): self.verts[v][0]['param'],
                          int(self.vb[v]): self.verts[v][1]['param']} for v in range(n)]
        ## pairs that cross exactly twice: any three of them, one per axis, are a cube
        self.two = {p for p, k in self.pair_count.items()
                    if k == 2 and frozenset(p) not in self.conflict}

    # ---- the combinatorial map of a subset ---------------------------------
    def build(self, S, drop_isolated=False):
        """The map of the loops in S: segments, regions, and the ring round each crossing.

        A loop that crosses no other loop of S has no place in the map: the
        result is None, or with `drop_isolated` the map of the remaining loops.
        """
        S = set(S)
        va, vb = self.va, self.vb
        vs = {v for l in S for v in self.by_loop[l] if int(va[v]) in S and int(vb[v]) in S}
        on_loop = defaultdict(list)
        for v in vs:
            on_loop[int(va[v])].append((self.param_on[v][int(va[v])], v))
            on_loop[int(vb[v])].append((self.param_on[v][int(vb[v])], v))
        if len(on_loop) < len(S):
            if not drop_isolated:
                return None
            S = set(on_loop)
        ## half-edge 2s runs forwards along segment s and has its region on the
        ## + side of the loop's plane; 2s + 1 runs backwards, region on the - side
        seg_loop, out_at = [], defaultdict(dict)
        for lid, items in on_loop.items():
            items.sort()
            k = len(items)
            for j, (_, v) in enumerate(items):
                seg = len(seg_loop)
                seg_loop.append(lid)
                out_at[v][(lid, 'f')] = 2 * seg
                out_at[items[(j + 1) % k][1]][(lid, 'b')] = 2 * seg + 1
        cw_next, ring_at = {}, {}
        for v in vs:
            a, b = int(va[v]), int(vb[v])
            ring = ([(a, 'f'), (b, 'f'), (a, 'b'), (b, 'b')] if self.vsign[v] > 0 else
                    [(a, 'f'), (b, 'b'), (a, 'b'), (b, 'f')])       # counter-clockwise
            hs = [out_at[v][r] for r in ring]
            ring_at[v] = hs
            for i, h in enumerate(hs):
                cw_next[h] = hs[i - 1]
        H = 2 * len(seg_loop)
        face_of = np.full(H, -1, int)
        faces = []
        for h0 in range(H):
            if face_of[h0] >= 0:
                continue
            h, face = h0, []
            while face_of[h] < 0:
                face_of[h] = len(faces)
                face.append(h)
                h = cw_next[h ^ 1]
            faces.append(face)
        return dict(S=S, vs=sorted(vs), seg_loop=seg_loop, faces=faces, face_of=face_of,
                    ring_at=ring_at)

    def components(self, m):
        """Sizes of the connected groups of loops in a map."""
        parent = {l: l for l in m['S']}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for v in m['vs']:
            parent[find(int(self.va[v]))] = find(int(self.vb[v]))
        return sorted(Counter(find(l) for l in m['S']).values(), reverse=True)

    def region_labels(self, m, face):
        """(axis, side) of every segment on a region's boundary, with its count."""
        return Counter((self.L_of[m['seg_loop'][h >> 1]]['axis'], h & 1) for h in face)

    def problems(self, m):
        """Why the map is not a polycube loop structure; empty when it is one."""
        if m is None:
            return ['isolated loop']
        bad = []
        if len(self.components(m)) > 1:
            bad.append('disconnected')
        chi = len(m['vs']) - len(m['seg_loop']) + len(m['faces'])
        if chi != 2:
            bad.append(f'chi {chi}')
        if any(len(f) < 3 for f in m['faces']):
            bad.append('region with < 3 segments')
        if any(max(self.region_labels(m, f).values()) > 1 for f in m['faces']):
            bad.append('repeated axis+side in a region')
        return bad

    def valid(self, S):
        return not self.problems(self.build(S))

    def allowed(self, S, l):
        return all(frozenset((l, s)) not in self.conflict for s in S)

    # ---- selection -----------------------------------------------------------
    def starting_cubes(self):
        """Every (x-loop, y-loop, z-loop) whose three pairs cross exactly twice."""
        ids = [[L['id'] for L in self.loops if L['axis'] == a] for a in range(3)]
        two = self.two
        return [[lx, ly, lz]
                for lx in ids[0] for ly in ids[1] if (lx, ly) in two
                for lz in ids[2] if (lx, lz) in two and (ly, lz) in two]

    def grow(self, S, min_gap=0.0):
        """Add loops one at a time while the set stays valid.

        The axis with the fewest kept loops goes first, then the loop farthest
        from the kept loops of its own axis; `min_gap` leaves out loops closer
        than that to a kept loop of their axis.
        """
        S = list(S)
        L_of = self.L_of
        while True:
            cand = []
            for L in self.loops:
                if L['id'] in S or not self.allowed(S, L['id']):
                    continue
                same = [np.linalg.norm(L['centre'] - L_of[s]['centre'])
                        for s in S if L_of[s]['axis'] == L['axis']]
                gap = min(same) if same else 9.0
                if gap >= min_gap:
                    cand.append((-len(same), gap, L['id']))
            cand.sort(reverse=True)
            for _, _, lid in cand:
                if self.valid(S + [lid]):
                    S.append(lid)
                    break
            else:
                return S

    def search(self, starts=400, seconds=0.0, seed=0, min_gap=0.0, log=print):
        """Grow from many starting cubes and keep the best structure.

        `starts` 1 grows only from the cube with the longest loops.  Otherwise
        up to `starts` cubes are tried in a random order (`seed`), for at most
        `seconds` (0: no limit).  Best is the structure whose smallest axis
        family is largest, then the one with most crossings.
        """
        cubes = self.starting_cubes()
        log(f'  {len(cubes)} candidate starting cubes')
        if not cubes:
            raise SystemExit('no starting cube: no x-, y- and z-loop cross each other '
                             'pairwise exactly twice')
        L_of = self.L_of
        if starts <= 1:
            cubes.sort(key=lambda c: -sum(L_of[l]['length'] for l in c))
            S0 = next((c for c in cubes if self.valid(c)), None)
            if S0 is None:
                raise SystemExit('no starting cube is a valid structure on its own')
            S = self.grow(S0, min_gap)
        else:
            order = np.random.default_rng(seed).permutation(len(cubes))[:starts]
            best, hist, t0 = None, [], time.time()
            for k in order:
                if seconds and time.time() - t0 > seconds:
                    break
                if not self.valid(cubes[k]):
                    continue
                S = self.grow(cubes[k], min_gap)
                c = Counter(L_of[l]['axis'] for l in S)
                quads = len(self.build(S)['vs'])
                score = (min(c[0], c[1], c[2]), quads)
                hist.append((len(S), c[0], c[1], c[2], quads))
                if best is None or score > best[0]:
                    best = (score, S)
            if best is None:
                raise SystemExit('no starting cube is a valid structure on its own')
            hist = np.array(hist)
            log(f'  {len(hist)} starts grown in {time.time() - t0:.0f}s: loops kept median '
                f'{np.median(hist[:, 0]):.0f}, max {hist[:, 0].max()}; quads median '
                f'{np.median(hist[:, 4]):.0f}, max {hist[:, 4].max()}; smallest family '
                f'median {np.median(hist[:, 1:4].min(1)):.0f}, max {hist[:, 1:4].min(1).max()}')
            S = best[1]
        log('  starting cube: ' + ', '.join(
            f'{AXES[L_of[l]["axis"]]} = {L_of[l]["off"]:+.3f}' for l in S[:3]))
        return S

    def left_out(self, S):
        """Why each loop outside S is outside, as counts per reason."""
        why = Counter()
        for L in self.loops:
            if L['id'] in S:
                continue
            if not self.allowed(S, L['id']):
                why['conflicts with a kept loop'] += 1
            else:
                p = self.problems(self.build(list(S) + [L['id']]))
                why[p[0] if p else 'closer than min_gap'] += 1
        return why


def polycube(st, S, pad=0.05):
    """The polycube a valid loop set S is dual to.

    A region is a corner and a crossing is a quad face (its four regions).  An
    x-loop crossing a y-loop gives a face along z, signed by the surface
    normal.  Taking the x-loops away leaves the X-zones, and all corners of a
    zone share one x: the midpoint of the plane offsets of the loops under and
    over the zone, or the one loop's offset -+ `pad` for an end zone.

    Returns the corner positions on the polycube (`cube_v`) and on the shape
    (`shape_v`: the mean of the crossings round the region, which need not lie
    on the surface), the quads, their (axis, sign) labels, the zone count per
    axis and `ordered`: whether every zone's lower loop lies under its upper
    loop, which the offset-based coordinates rely on.
    """
    m = st.build(S)
    bad = st.problems(m)
    if bad:
        raise ValueError('not a polycube loop structure: ' + ', '.join(bad))
    L_of = st.L_of
    E, F = len(m['seg_loop']), len(m['faces'])
    face_of = m['face_of']
    seg_axis = np.array([L_of[l]['axis'] for l in m['seg_loop']])
    zone, coord, ordered = [], [], True
    for A in range(3):
        parent = list(range(F))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for seg in range(E):
            if seg_axis[seg] != A:
                parent[find(face_of[2 * seg])] = find(face_of[2 * seg + 1])
        z = np.array([find(f) for f in range(F)])
        below, above = defaultdict(set), defaultdict(set)    # loops under / over each zone
        for seg in range(E):
            if seg_axis[seg] == A:
                lid = m['seg_loop'][seg]
                above[z[face_of[2 * seg + 1]]].add(lid)
                below[z[face_of[2 * seg]]].add(lid)
        zc = {}
        for q in set(z.tolist()):
            lo = [L_of[l]['off'] for l in below[q]]
            hi = [L_of[l]['off'] for l in above[q]]
            if lo and hi:
                zc[q] = 0.5 * (max(lo) + min(hi))
                ordered &= max(lo) < min(hi)
            elif lo:
                zc[q] = max(lo) + pad
            else:
                zc[q] = min(hi) - pad
        for l in S:                       # a loop separates one zone from one other
            if L_of[l]['axis'] == A:
                zb = [q for q in above if l in above[q]]
                zt = [q for q in below if l in below[q]]
                if not (len(zb) == 1 and len(zt) == 1 and zb != zt):
                    raise ValueError(f'{AXES[A]}-loop {l} does not separate two zones')
        zone.append(z)
        coord.append(zc)
    cube_v = np.array([[coord[A][zone[A][f]] for A in range(3)] for f in range(F)])
    shape_v, cnt = np.zeros((F, 3)), np.zeros(F)
    for v in m['vs']:
        for h in m['ring_at'][v]:
            shape_v[face_of[h]] += st.vpos[v]
            cnt[face_of[h]] += 1
    shape_v /= cnt[:, None]
    quads, labels = [], []
    for v in m['vs']:
        quads.append([face_of[h] for h in m['ring_at'][v]])
        C = 3 - L_of[int(st.va[v])]['axis'] - L_of[int(st.vb[v])]['axis']
        labels.append((C, 1 if st.vnrm[v][C] > 0 else -1))
    return dict(map=m, loops=list(S), cube_v=cube_v, shape_v=shape_v,
                quads=np.array(quads), labels=labels,
                zones=[len(c) for c in coord], ordered=bool(ordered))


## face colours by label: +x -x +y -y +z -z
LABEL_RGB = {(0, 1): (238, 118, 118), (0, -1): (190, 80, 80), (1, 1): (120, 120, 238),
             (1, -1): (80, 80, 180), (2, 1): (247, 247, 180), (2, -1): (200, 200, 120)}
AXIS_COLOR = ['#d62728', '#1f77b4', '#bcbd22']


def save_polycube(pc, out):
    """polycube.obj, layout_on_shape.obj (same quads, two sets of corners) and structure.npz."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, vv in (('polycube', pc['cube_v']), ('layout_on_shape', pc['shape_v'])):
        with open(out / f'{name}.obj', 'w') as fh:
            for p in vv:
                fh.write(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
            for q in pc['quads']:
                fh.write('f ' + ' '.join(str(i + 1) for i in q) + '\n')
    np.savez(out / 'structure.npz', loops=np.array(pc['loops']), quads=pc['quads'],
             labels=np.array(pc['labels']), cube_v=pc['cube_v'], shape_v=pc['shape_v'])


def plot_polycube(st, pc, path):
    """Kept loops, the quad layout on the shape and the polycube, from two sides."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    S, quads = set(pc['loops']), pc['quads']
    colors = [np.array(LABEL_RGB[l]) / 255 for l in pc['labels']]
    fig = plt.figure(figsize=(21, 14))
    for row, (elev, azim) in enumerate(((15, -60), (15, 120))):
        for col, title in enumerate(('kept loops', 'quad layout on the shape', 'polycube')):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection='3d')
            if col == 0:
                for L in st.loops:
                    P = np.vstack([L['P'], L['P'][:1]])
                    keep = L['id'] in S
                    ax.plot(P[:, 0], P[:, 2], P[:, 1], lw=1.6 if keep else 0.4,
                            color=AXIS_COLOR[L['axis']] if keep else '#cccccc')
            else:
                vv = pc['shape_v'] if col == 1 else pc['cube_v']
                ax.add_collection3d(Poly3DCollection(
                    [vv[q][:, [0, 2, 1]] for q in quads], facecolors=colors,
                    edgecolors='k', linewidths=0.4))
            ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev, azim)
            ax.set_axis_off()
            if not row:
                ax.set_title(f'{title} ({len(S)} loops, {len(quads)} quads)')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
