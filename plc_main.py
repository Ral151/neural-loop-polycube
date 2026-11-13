"""Run the cross-section loop pipeline end to end.

`model.loop_extraction` owns the geometry: slicing, Poisson fields, loop
centres and loop assignment, behind one `LoopExtractor` object.  This module
owns the run: it exposes the only command line, extracts the loops through that
object and, with ``--train``, fits every loop with the shared mini network and
writes the fit figures and coloured PLY exports.
"""

import argparse
import csv
import os

import numpy as np
import torch

from model import mini_nn
from model.loop_extraction import (AXIS_COLORS, DEFAULT_PLY,
                                       LoopExtractor, bisect_offsets_trace,
                                       plane_keys, save_fit_grid,
                                       save_plane_files, write_points)


## a fitted point past this distance from its target makes the whole loop bad.
## The value is in the plane frame, where the outputs are compared, and is not
## the mean loop error `--max-err` uses to highlight a loop in the fit figures.
BAD_POINT_ERR = 0.15

## a loop whose mean 1 - cos between its predicted normal and the paired target
## normals is past this is exported as bad even when its points fit. 0 (default)
## turns the alignment side of the split off and leaves classification to the
## point limit; a positive value is a mean 1 - cos, where 1 is side-on, 2 reversed,
## and 0.5 a mean 60-degree disagreement.
BAD_ALIGN_ERR = 0.0

## a loop whose mean error is past this fraction of the longest extent is exported
## as bad. The point limit above only sees the worst fitted point, so a curve that
## is off along a whole arc but never far at one point stays good; the mean is the
## size-independent reading of that. 0 turns this side of the split off.
BAD_LOOP_ERR = 0.015


def refine_plan(samples, epochs, every=0, add=2, end=None):
    """
    The plane count and the epochs of every stage a fitting run goes through.

    samples: planes per axis for the first stage, which is what the extractor cut
    epochs: epochs over the whole run, split between the stages
    every: epochs a stage lasts; 0 or less runs one stage and refines nothing
    add: planes added per axis at each stage, by bisecting the widest gaps
    end: planes per axis at the last stage, or None to grow by `add` at every stage,
             which ends the run on samples + add * (stages - 1)

    Refinement is by bisection, so a stage holds every plane the stage before had,
    at the same offsets: the loops on those planes are the same loops, and their
    fits are handed on rather than relearned.

    Returns [(planes per axis, epochs in that stage)], the epochs summing to
    `epochs`.  Once `end` is reached the remaining stages keep its plane count, so
    the tail of a long run spends its epochs on a fit at full density, and they are
    one stage rather than one per `every` epochs: nothing is re-cut at a count that
    has stopped growing, and the anneals then span the whole tail instead of
    restarting inside it.
    """
    if epochs < 1:
        raise ValueError(f'epochs must be at least 1, got {epochs}')
    if samples < 1:
        raise ValueError(f'samples must be at least 1, got {samples}')
    if every <= 0 or add <= 0:
        return [(samples, epochs)]
    if end is None:
        end = samples + add * (-(-epochs // every) - 1)
    if end < samples:
        raise ValueError(f'the last plane set cannot hold fewer planes than the first '
                         f'({end} < {samples})')
    stages = -(-epochs // every)
    plan = []
    for s in range(stages):
        planes = min(samples + add * s, end)
        ## a stage at a count the one before it already had would cut the cloud again
        ## for the same loops, and restart the learning rate and direction anneal
        ## inside a fit that is still the same problem: it is the same stage held longer
        if plan and plan[-1][0] == planes:
            continue
        plan.append([planes, every])
    ## the epochs always add up to what was asked for, whatever the stage length
    plan[-1][1] = epochs - every * (len(plan) - 1)
    return [tuple(stage) for stage in plan]


def build_parser():
    """The command line every run starts from, extraction alone or with training."""
    parser = argparse.ArgumentParser(
        description='Extract cross-section loops from a point cloud and optionally '
                    'fit them with the shared mini network.')
    parser.add_argument('--path', default=DEFAULT_PLY)
    parser.add_argument('--samples', type=int, default=5, help='planes sampled per axis')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='inset from each bounding box face, as a fraction of the axis extent')
    parser.add_argument('--no-image', action='store_true',
                        help='skip the projected slice grid png')
    parser.add_argument('--arrows', default='80',
                        help='normal arrows per panel (0 for none, all for every normal)')
    parser.add_argument('--normal-arrows', default='20',
                        help='arrows drawn from each fitted curve along its '
                             'predicted normal, per loop (0 for none, all for every point)')
    parser.add_argument('--save-plane-files', dest='plane_files',
                        action='store_true', default=True,
                        help='write the per-plane slice/loops/fit PLYs and fit PNGs '
                             'under <out>/planes/. Off by default: planes.txt and '
                             'bad_loops.txt are always written, and the combined fit '
                             'grid and split loop PLYs carry the same verdict')
    parser.add_argument('--field', action='store_true',
                        help='draw the implicit field of each loop behind the slice '
                             'points, and on its own in a second png')
    parser.add_argument('--assign', action='store_true',
                        help='colour each slice point by the loop centre it belongs to')
    parser.add_argument('--band', type=float, default=0.0075,
                        help='half-width of the slab that counts as on a plane, as a '
                             'fraction of the longest extent; narrower is a crisper '
                             'contour with fewer points in it')
    parser.add_argument('--max-align', type=float, default=0.95,
                        help='a point whose normal is within this cosine of a plane '
                             'normal is left out of that plane: its surface lies flat in '
                             'the slice and has no in-plane direction to solve with. '
                             '0.95 is about 18 deg, 0.999 about 2.6 deg which takes out '
                             'a fifth of the hand and a tenth of the cuboid, and 1.0 '
                             'keeps every point')
    parser.add_argument('--train', action='store_true',
                        help='fit the loops with mini nn, one target per loop, and '
                             'export the predictions coloured by their error')
    parser.add_argument('--epochs', type=int, default=300,
                        help='training epochs, one batch of every loop each, so it is '
                             'also the number of steps; a few dozen leaves the curve '
                             'folded back over itself')
    parser.add_argument('--freeze-f-epochs', type=int, default=0, metavar='N',
                        help='freeze the shared MLP -- the map from (sin z, cos z, '
                             'feature, axis) to the curve -- for the first N epochs '
                             'and train only the conditioning field. The warm-up map '
                             'is what keeps every loop closed, consistently oriented '
                             'and starting from one square; the first epochs then say '
                             'where each shape loop is without rewriting that map, so '
                             'the curves are moved onto their slices by the field '
                             'rather than dragged there by changing f. 0 (default) '
                             'trains both from the first step')
    parser.add_argument('--warmup', default='warmup/cube_warmup_random_ranges.pt',
                        metavar='CKPT',
                        help='cube warm-up checkpoint loaded before the first fit, on '
                             'by default. `model/warmup_cube.py` trains it on cubes and '
                             'nothing else: their cross-sections are exact squares, '
                             'and the network learns to draw them on the cubes, all '
                             'circulating the same way and all putting z = 0 where '
                             'their neighbouring loops do. The default is warmed on '
                             'cubes of random size and place with occupied-cell '
                             'features, so a fit that uses it must use '
                             '--feature-pool occupied_cells (the default; or name a '
                             'checkpoint warmed for this run\'s pooling). The checkpoint '
                             'holds the shared network and the triplane field and '
                             'nothing shape-specific -- no rows, no phases, no plane '
                             'set -- so the shape\'s loops, planes and targets are '
                             'its own. It is loaded into the trainer as it is, with '
                             'this run\'s triplane frame kept: the grid is a function '
                             'of unit-cube position, and the frame is where that cube '
                             'is laid on the cloud\'s box. The load is refused if the '
                             'checkpoint\'s network, grid shape or pooling '
                             'differs from this run\'s. A missing path is an error: run '
                             'model/warmup_cube.py first')
    parser.add_argument('--no-warmup', dest='warmup', action='store_const', const=None,
                        help='fit the loops from a blank network, which is the '
                             'ablation of the cube warm-up')
    parser.add_argument('--feature-pool', choices=('centroid', 'occupied_cells'),
                        default='occupied_cells',
                        help='how a loop reads the triplane field. occupied_cells '
                             '(default) snaps each target point to its nearest grid '
                             'node in every plane and averages the unique nodes, so '
                             'the code describes the loop\'s footprint and two loops '
                             'whose centres share a cell still read different codes. '
                             'centroid samples the one query at the loop\'s world '
                             'centre, the 48-D code every earlier run used. Needs '
                             '--feature triplane and a warm-up trained the same way')
    parser.add_argument('--axis-condition', action='store_true',
                        help='also feed every loop the unit normal of the plane it '
                             'was cut on, appended to its feature. The triplane '
                             'feature says where a loop is but not which way it was '
                             'cut, so without it an x-, a y- and a z-loop through the '
                             'same region read near-equal codes. Off by default; the '
                             '--warmup checkpoint must be trained with the same '
                             'setting (`model/warmup_cube.py --axis-condition`)')
    parser.add_argument('--distill', type=float, default=0.0, metavar='W',
                        help='weight of a teacher-student seam distillation: the '
                             'cube warm-up is read at this shape\'s own queries and '
                             'axes, and its per-pair seam distance D becomes the '
                             'ceiling the fit\'s own D is held under. Only the excess '
                             'is charged, so the term can lower D but never raise it. '
                             'D is scale-free, so the '
                             'teacher reads its own curve size (the cube scale) and '
                             'the student its own (the shape scale); the shape '
                             'target radius only converts the gap to chamfer units. '
                             'The curves themselves are '
                             'never matched -- shape and orientation are the '
                             'warm-up\'s and the fit\'s business -- so what is '
                             'trained is where z = 0 falls relative to the '
                             'neighbouring loops, which no other term can see. '
                             'Zero at step 0 because the student starts as the '
                             'teacher. 0 (default) trains on the shape alone')
    parser.add_argument('--distill-warmup', default=None, metavar='CKPT',
                        help='cube-warm-up checkpoint the distillation teacher is '
                             'read from; default is --warmup, so the teacher is the '
                             'checkpoint the fit started from')
    parser.add_argument('--code-init', choices=('blend', 'random'), default='blend',
                        help='how the embedding row of an inserted plane starts: blend '
                             '(default) writes it as the distance-weighted '
                             'interpolation of the rows on the two planes the insertion '
                             'lies between, matched by slot, so the loop starts from a '
                             'shape its neighbours already hold; random draws every '
                             'inserted row from a normal with the spread of the '
                             'inherited rows, so each new loop starts somewhere '
                             'different. The warm-up runs once, before the first fit, so '
                             'these rows are then learned in the fit itself')
    parser.add_argument('--snapshot-every', type=int, default=50,
                        help='write the fit grid every this many epochs while the run '
                             'is still going, into <out>/snapshots/, one file per '
                             'snapshot named by the epoch it was taken at. A run of '
                             '1000 epochs can then be watched taking shape instead of '
                             'only judged at its end: flip through the files and the '
                             'curves arrive on their loops. The same grid the run '
                             'writes at its end and the same red limit, so a snapshot '
                             'is comparable to that figure; 0 writes none')
    parser.add_argument('--refine-every', type=int, default=0,
                        help='refine the planes as the fit goes: every this many '
                             'epochs, bisect the widest gaps along each axis, cut the '
                             'cloud again and carry the fit over. The planes already '
                             'cut stay at the same offsets, so the loops on them are '
                             'the same loops and keep their embeddings; only the rows '
                             'of the added planes are seeded, by --code-init, and are '
                             'then fitted with the inherited ones. 0 fits the one '
                             'plane set the extraction cut')
    parser.add_argument('--add-planes', type=int, default=2,
                        help='planes added per axis at each refinement stage')
    parser.add_argument('--samples-end', type=int, default=None,
                        help='planes per axis at the last stage; the default grows by '
                             '--add-planes at every stage, so --epochs 1000 '
                             '--refine-every 100 --samples 3 --add-planes 2 ends on '
                             '21 planes per axis. Reaching this count early spends '
                             'the remaining stages on the fit at that density')
    parser.add_argument('--neurons', type=int, default=64,
                        help='width of the network every loop shares; a plane cutting a '
                             'whole silhouette is a lot of structure for one curve, and '
                             'the error there falls from 3.3 to 1.4 at 256')
    parser.add_argument('--layers', type=int, default=2,
                        help='hidden layers in the shared network, each --neurons wide; '
                             'depth adds capacity without widening the conditioning path')
    parser.add_argument('--freqs', type=int, default=1, metavar='F',
                        help='harmonics of the loop parameter the shared network is '
                             'given: sin/cos of 2 pi k z for k = 1..F. 1 (default) is '
                             'the single pair every earlier run used; it draws smooth '
                             'curves and rounds sharp corners off. More hand the '
                             'network the fast waves a corner is made of. The '
                             '--warmup checkpoint must be trained with the same '
                             'value (`model/warmup_cube.py --freqs F`)')
    parser.add_argument('--normal', dest='align', default=0.0,
                        type=float, metavar='W',
                        help='weight of the normal term, which asks the normal the '
                             'fitted curve implies -- the plane normal turned a quarter '
                             'off its tangent, tangent x plane normal -- to agree with '
                             'the outward point normal it was matched to. A contour is '
                             'where a plane cuts a surface, so that implied normal is '
                             'the surface normal, and the outward sense of the point '
                             'normal picks which way round the loop runs. Position '
                             'alone cannot tell a curve running along the surface from '
                             'one cutting across it wherever two stretches of a contour '
                             'pass close. The term is a mean of 1 - cos between those '
                             'two normals, both read in the loop\'s plane: 0 is a fit '
                             'whose normal is the one the cloud gives, 1 one whose '
                             'normal sits square to it, 2 one pointing the other way, '
                             'and the weight is in the same units as the chamfer loss. '
                             'It is held at this weight for the whole run, so it keeps '
                             'settling which way round each loop runs while the fit is '
                             'coarse and goes on holding the loop once the fit sits on '
                             'its points')
    parser.add_argument('--match-angle', type=float, default=150.0, metavar='DEG',
                        help='largest angle, in degrees, between the normal a fitted '
                             'curve implies and the outward normal of a target point '
                             'for chamfer to pair them; the pair is allowed when the '
                             'angle is no greater. Both normals are read in the slice '
                             'plane. 150 by default: it passes over a point on the far '
                             'side of a thin feature, whose normal points back at the '
                             'curve, while keeping nearly every true pair. A point '
                             'with no allowed partner is left out of the fit on that '
                             'step. 180 pairs by distance alone, as earlier runs did')
    parser.add_argument('--cycle', type=float, default=0.0, metavar='W',
                        help='weight of a cycle-consistency term, 0 (default) to '
                             'leave the inverse map untrained. The network also '
                             'carries the map back from a point on the curve and the '
                             'condition it was drawn under to the parameter z it '
                             'belongs to. The term is the mean squared error of that '
                             'reading, so a curve folding two distant parameters onto '
                             'the same point cannot be inverted and the term says so. '
                             'The point branch is detached inside forward, so the '
                             'term trains the inverse map and the condition, not the '
                             'fitted curve')
    parser.add_argument('--repel', type=float, default=0.0, metavar='W',
                        help='weight of a repulsion term on the fitted curve, 0 '
                             '(default) off. Anchors spread around the loop are '
                             'charged when a pair closer than the gap is not an '
                             'immediate parameter neighbour: two parameters far apart '
                             'on the loop should not land on the same 3D point, while '
                             'adjacent samples may. Unlike --cycle the term acts on '
                             'the points themselves, so the fit feels it; it is a '
                             'squared distance in the chamfer loss\'s units')
    parser.add_argument('--repel-gap', type=float, default=0.05, metavar='F',
                        help='the separation a non-neighbour pair is held apart by, '
                             'as a fraction of the target loop\'s own RMS radius, so '
                             'one value serves a large loop and a small one; 0.05 '
                             '(default) is a twentieth of the loop')
    parser.add_argument('--repel-samples', type=int, default=64, metavar='K',
                        help='how many anchors the dense repulsion term is evaluated '
                             'on, spread evenly in z; the full sample set can be '
                             'thousands of points per loop, so the term reads K of '
                             'them and the network is smooth in z between them')
    parser.add_argument('--code-smooth', type=float, default=0.0,
                        help='weight of a smoothness term on the embedding-table '
                             'codes: the codes of loops on neighbouring planes are '
                             'pulled together, weighted by how close their in-plane '
                             'centres are. A neighbouring cross-section of the same '
                             'limb then starts from a similar code and the fits vary '
                             'along the axis instead of each row being free. The term '
                             'is in the same units as the chamfer loss (a squared '
                             'distance summed over xyz), so it competes with the fit; '
                             '0 learns every row freely. It applies in embedding mode '
                             'only; triplane mode smooths its field with --triplane-tv')
    parser.add_argument('--code-gate', type=float, default=2.0,
                        help='a pair must have in-plane centres closer than this many '
                             'median plane spacings along their axis to enter the code '
                             'graph; larger connects more distant loops and over-smooths')
    parser.add_argument('--code-min-radius', type=float, default=0.4,
                        help='a pair must have at least this ratio of smaller to '
                             'larger loop radius; it keeps a torso and a small limb '
                             'cross-section from being forced onto one code')
    parser.add_argument('--seam-align', type=float, default=0.0, metavar='W',
                        help='weight of a phase-only seam alignment, 0 (default) to '
                             'leave every loop\'s parameterisation alone. The fit is '
                             'indifferent to where z = 0 falls on a closed curve, so '
                             'that phase is one number per loop, solved against the '
                             'code graph\'s neighbour pairs every --seam-every epochs. '
                             'Only the phases take a gradient -- the curves are read '
                             'detached -- so the fit and normal terms are the '
                             'ones the same run makes without it, and the exported '
                             'loops start where their neighbours put the seam')
    parser.add_argument('--seam-every', type=int, default=10,
                        help='solve the seam phases every this many epochs; each '
                             'solve continues from the phases the last one left')
    parser.add_argument('--seam-start', type=int, default=0,
                        help='first epoch the seam solve runs at, counted from 1; '
                             '0 solves at the first epoch and every --seam-every after')
    parser.add_argument('--seam-steps', type=int, default=100,
                        help='Adam steps per seam solve, at the trainer\'s own '
                             'seam learning rate (2e-2 turns); a solve on a few dozen '
                             'loops converges in a few tens of steps')
    parser.add_argument('--seam-lam', type=float, default=0.02,
                        help='how much a pair votes in the seam solve, in the distance '
                             'D the term reads (mean squared seam separation over the '
                             'two loops\' mean squared radius). A pair is first matched '
                             'at its best circular shift, and votes lam/(lam + D_best): '
                             'a pair that agrees once shifted is the same cross-section '
                             'with the seams in the wrong place and votes nearly in '
                             'full, two different loops vote at nearly nothing')
    parser.add_argument('--feature', choices=('embedding', 'triplane'),
                        default='triplane',
                        help='what conditions the shared network (default triplane). '
                             'embedding is the '
                             'per-plane row table: every loop has a code of its own. '
                             'triplane replaces it with three orthogonal 2D feature '
                             'grids; by default the code pools the grid nodes the '
                             'loop\'s points occupy (--feature-pool occupied_cells), '
                             'so it is a continuous function of footprint and '
                             'neighbouring loops start similar by construction')
    parser.add_argument('--triplane-res', type=int, default=32,
                        help='cells per side of each triplane grid; the grid is the '
                             'unit cube, laid over the cloud bounding box, so a cell is '
                             'roughly the longest extent divided by this. Coarser shares '
                             'more between neighbouring loops')
    parser.add_argument('--triplane-channels', type=int, default=16,
                        help='feature channels per triplane grid; the network is '
                             'conditioned on three samples concatenated, so its latent '
                             'width is three times this')
    parser.add_argument('--triplane-tv', type=float, default=0.01,
                        help='weight of a total-variation penalty on the three grids '
                             '(default 0.01, used only with --feature triplane); '
                             '0 leaves every cell free. It is in the same units as '
                             'the chamfer loss only after averaging over cells, so keep '
                             'it small (1e-3 to 1e-2) when the grid is coarse')
    parser.add_argument('--max-err', type=float, default=0.03,
                        help='a loop whose mean error is past this fraction of the '
                             'longest extent is highlighted: its curve red and heavier '
                             'in the fit figures, its title and table entry red, and a '
                             'line in bad_loops.txt. The list is a shortlist to drop '
                             'loops by, not a cut, and nothing is removed here')
    parser.add_argument('--max-align-err', type=float, default=BAD_ALIGN_ERR,
                        help='a loop whose mean 1 - cos between its predicted normal '
                             'and the paired target normals is past this is written '
                             'as a bad loop, even when its points fit. Default 0 '
                             'turns the alignment side off; a positive value turns '
                             'it on (1 side-on, 2 reversed, 0.5 a mean 60-degree '
                             'disagreement)')
    parser.add_argument('--bad-loop-err', type=float, default=BAD_LOOP_ERR,
                        help='a loop whose mean error is past this fraction of the '
                             'longest extent is written as a bad loop. Unlike the '
                             'point limit it sees a curve that is off along an arc '
                             'rather than at one point. Default 0.015; 0 turns the '
                             'mean side off')
    parser.add_argument('--large', type=float, default=1.5,
                        help='a plane whose mean error is over this multiple of the '
                             'run\'s median plane error gets its title marked "!" and '
                             'turned red, and a "large" note in planes.txt; the '
                             'absolute --max-err alone names nothing when every loop '
                             'is inside it')
    parser.add_argument('--out', default=None,
                        help='folder to write in; the default is log/<input stem>/, '
                             'and naming one is how a second experiment on the same '
                             'shape is kept beside the first')
    return parser


def code_neighbour_graph(keys, targets, gate=2.0, min_radius=0.4):
    """
    Mutual-nearest loops on adjacent planes, as a graph over loop indices.

    keys: (axis, offset, slot) per loop, from `plane_keys`
    targets: one N x 3 point set per loop, in its plane's frame
    gate: a pair enters only if its in-plane centre distance is below this many
             median plane spacings along the axis; the spacings are measured on the
             current plane set, so the gate follows the density of the run
    min_radius: smallest smaller/larger RMS radius a pair may have

    Matching is mutual nearest, so a hub loop on one plane cannot pull every loop
    of its neighbour, and the pair is undirected.  The weight falls linearly from
    1 at zero distance to 0 at the gate, which is what makes a close neighbour
    dominate the penalty while a pair at the edge only regularises weakly.  A
    plane with one loop against a plane with several picks at most one, which is
    the intended behaviour: the others are not the same cross-section.

    Returns (pairs, weights, distances), where a pair is two loop indices and a
    distance is their centre separation in the plane, both axes in world units.
    """
    keys = list(keys)
    if len(keys) < 2:
        return [], [], []
    axis = np.array([k[0] for k in keys], dtype=np.int64)
    off = np.array([k[1] for k in keys], dtype=np.float64)
    cent = np.stack([np.asarray(t, dtype=np.float64).mean(0) for t in targets])
    rad = np.array([float(np.sqrt(((np.asarray(t, dtype=np.float64) -
                                    np.asarray(t, dtype=np.float64).mean(0)) ** 2)
                                  .sum(1).mean())) for t in targets])

    pairs, weights, dists = [], [], []
    for a in range(3):
        levels = np.unique(off[axis == a])
        if len(levels) < 2:
            continue
        spacing = float(np.median(np.diff(levels)))
        if not np.isfinite(spacing) or spacing <= 0.0:
            continue
        limit = gate * spacing
        ## the two coordinates that run in the plane; the planes are axis aligned
        inplane = [j for j in range(3) if j != a]
        for left, right in zip(levels[:-1], levels[1:]):
            A = np.where((axis == a) & (off == left))[0]
            B = np.where((axis == a) & (off == right))[0]
            if not len(A) or not len(B):
                continue
            delta = cent[A][:, None, inplane] - cent[B][None, :, inplane]
            D = np.linalg.norm(delta, axis=2)
            near_b = D.argmin(1)
            near_a = D.argmin(0)
            for i, ia in enumerate(A):
                j = int(near_b[i])
                if int(near_a[j]) != i:
                    continue
                d = float(D[i, j])
                if d > limit:
                    continue
                ratio = min(rad[ia], rad[B[j]]) / max(rad[ia], rad[B[j]], 1e-12)
                if ratio < min_radius:
                    continue
                pairs.append((int(ia), int(B[j])))
                weights.append(float(1.0 - d / limit))
                dists.append(d)
    return pairs, weights, dists


def fit_loops(found, epochs=300, neurons=64, layers=2, align=0.0,
              match_angle=150.0, cycle=0.0,
              repel=0.0, repel_gap=0.05, repel_samples=64,
              max_err=0.03, max_align_err=BAD_ALIGN_ERR,
              bad_loop_err=BAD_LOOP_ERR, large=1.5, normal_arrows=20,
              plane_files=False,
              warmup='warmup/cube_warmup_random_ranges.pt',
              snapshot_every=50,
              refine_every=0, add_planes=2,
              samples_end=None, code_init='blend', code_smooth=0.0,
              code_gate=2.0, code_min_radius=0.4, feature='embedding',
              triplane_res=32, triplane_channels=16, triplane_tv=0.0,
              feature_pool='occupied_cells', axis_condition=False,
              distill=0.0, distill_warmup=None,
              seam_align=0.0, seam_every=10, seam_start=0, seam_steps=100,
              seam_lam=0.02, freeze_f_epochs=0, freqs=1):
    """Fit every extracted loop and write the predictions, figures and tables.

    found: the loop extractor that has already run, whose cloud, planes, loops
             and run folder are what is fitted and written
    warmup: a cube warm-up checkpoint from `model/warmup_cube.py`, loaded into the
             network before the first fit; None fits from a blank network.  The
             warm-up was trained on cubes and nothing else (by default of random
             size and place), so it carries no shape information: what it hands
             over is a network that draws on-cube, consistently oriented loops,
             and the triplane field it draws them from.  Only the weights and the grid are copied -- the triplane
             frame stays this run's, so the grid, which is a function of unit-cube
             position, is read on the cloud's box.  Nothing else is loaded: the
             loops, planes, targets and phases below are the shape's own.  The
             transfer needs `feature='triplane'`; in embedding mode the warm-up is
             skipped, because a row named by a cube plane does not index a shape's
             planes
    feature_pool: how a loop reads the field. 'centroid' samples the one query at
             its world centre, the code every earlier run used. 'occupied_cells'
             snaps each target point to its nearest grid node in every plane and
             averages the unique nodes, so the code is the mean of the grid cells
             the loop covers rather than one bilinear read; that gives loops with
             different extent or offset different codes even when their centres
             share a cell. The width and the network interface are unchanged. The
             warm-up has to be trained with the same pool
    plane_files: write the per-plane PLYs and PNGs under <out>/planes/, so one
             slice can be loaded and judged on its own. Off by default: the run's
             planes.txt and bad_loops.txt are always written, and the combined
             fit grid and the split loop PLYs carry the same verdict at less cost
    match_angle: degrees; chamfer pairs a predicted point only with a target point
             whose outward normal, read in the slice plane, is at most this angle
             from the normal the fitted curve implies. Pairing by distance alone can
             reach across a thin feature to a point facing the other way; 150 by
             default passes over those but keeps nearly every true pair, and a point
             with no allowed partner is left out of the fit on that step. 180 pairs
             by distance alone, as earlier runs did
    cycle: weight of a cycle-consistency term, 0 (default) to leave the inverse map
             untrained. The network also carries the map back from a point on the
             curve and the condition it was drawn under to the parameter z it belongs
             to. The term is the mean squared error of that reading, so a curve that
             folds two distant parameters onto the same point cannot be inverted and
             the term says so. The point branch is detached inside forward, so the
             term trains the inverse map and the condition, not the fitted curve
    repel, repel_gap, repel_samples: a repulsion term on the fitted curve. Anchors
             spread around the loop are charged when a pair closer than the gap is
             not an immediate parameter neighbour: two parameters far apart on the
             loop should not land on the same 3D point, while adjacent samples may.
             Unlike `cycle` the term acts on the points themselves, so the fit feels
             it. The gap is `repel_gap` times the target loop's own RMS radius and
             `repel_samples` is how many anchors the dense K x K term reads; repel is
             a squared distance, in the chamfer loss's units
    bad_loop_err: a loop whose mean error is past this fraction of the longest
             extent is written to the bad side of the split. The fixed point limit
             only sees the single worst fitted point, so a curve that is off along
             an arc while never far at one point would otherwise pass; 0 turns the
             mean side off. The three checks (point, mean, optional alignment) are
             independent, and every loop's readings go to <stem>_loop_errors.csv
    distill, distill_warmup: a teacher-student seam distillation on top of the fit.
             The checkpoint is read as a frozen teacher at this shape's own queries
             and axes, and its per-pair seam distance D is the target profile. Each
             step's student D is pulled to that profile, and the curves themselves
             are never matched, so shape and orientation stay with the fit and the
             warm-up while the seam -- the thing no fit term can see -- is held.
             Unlike the direction term its weight is not annealed. `distill_warmup`
             defaults to `warmup`, so the teacher is the checkpoint the fit started
             from. Zero leaves the fit exactly as it was
    freeze_f_epochs: freeze the shared MLP -- the f map -- for the first N epochs
             and train only the conditioning field. The warm-up's map is what keeps
             every loop closed, consistently oriented and starting from one square;
             freezing it keeps that prior while the field moves each curve onto its
             slice, instead of the first updates translating and scaling the loop by
             rewriting the map. It is released after N epochs and the fit continues
             as usual. 0 trains both from the first step
    freqs: harmonics of the loop parameter the shared MLP is given, sin/cos of
             2 pi k z for k = 1..freqs. 1 is the single pair every earlier run used;
             more let the map draw the sharp corners of a cubic slice instead of
             rounding them off. The warm-up has to be trained with the same count
    snapshot_every: a fit grid is written every this many epochs while the run is
             still going, into <out>/snapshots/, so a run can be watched taking
             shape rather than only judged once it is over.  Each one is the grid
             the run writes at its end, taken from the network as it stood on that
             epoch, and costs a predict pass over the loops rather than a second
             run.  0 writes none
    refine_every: epochs one stage lasts; 0 fits the one plane set the extractor
             cut.  Otherwise every stage bisects the widest gaps along each axis,
             so the planes already cut stay exactly where they are and the loops on
             them are the same loops: their fits are handed on to the next stage,
             and only the rows of the added planes are seeded.  A stage is its own
             `train` call, so its learning rate and direction term anneal over the
             stage
    add_planes: planes added per axis at each stage
    samples_end: planes per axis at the last stage, or None to grow by `add_planes`
             at every stage.  The epochs a stage at a density the run has already
             reached would have spent are gathered into one stage at that density,
             since nothing is re-cut for the same loops
    code_init: how the embedding row of an inserted plane starts.  'blend' writes
             it as the distance-weighted interpolation of the rows on the two
             planes the insertion lies between, matched by slot, so the loop starts
             from a shape its neighbours already hold; 'random' draws it from a
             normal with the spread of the inherited rows, so each new loop starts
             somewhere different.  Either way the rows are seeded after the inherit
             and are learned in the stage's own fit
    code_smooth, code_gate, code_min_radius: a smoothness term on the embedding
             table.  Loops on adjacent planes whose in-plane centres are mutual
             nearest, within `code_gate` median spacings and with radii within
             `code_min_radius`, are linked; their squared code distance is added
             to the loss at weight `code_smooth`.  The graph is rebuilt for every
             stage, since a stage cuts new planes.  Zero leaves every row free
    feature: 'embedding' is the row table above.  'triplane' conditions the network
             on three orthogonal 2D grids, read by default as the pool of grid
             nodes the loop's points occupy (`feature_pool='occupied_cells'`) or
             as one sample at its world centre (`'centroid'`). The code is a
             continuous function of position either way: two loops close in 3D
             read overlapping cells and start from similar features, and a plane
             set added later needs no row seed.  The graph is still built for
             diagnostics and for the same smoothness term, now applied to the
             sampled features
    triplane_res, triplane_channels, triplane_tv: cells per side, channels per
             grid, and an optional total-variation weight on the grids.  The grids
             are the unit cube, uniformly subdivided and the same for every shape;
             the cloud's bounding box, with a small margin, is only the frame that
             cube is laid over, so the fit reads the same feature map on every shape
    seam_align, seam_every, seam_start, seam_steps, seam_lam: a phase-only seam
             alignment.  The fit is indifferent to where z = 0 falls on a closed
             loop -- sliding a curve through its own parameter changes no fit,
             or normal term -- so that phase is one number per loop, solved
             against the same neighbour graph every `seam_every` epochs from
             `seam_start`, with `seam_steps` Adam steps per solve.  Only the phases
             get a gradient: the curves are detached, so the fit is exactly the one
             this run makes with `seam_align` at zero, and the exported loops start
             from a point their neighbours put the seam on.  `seam_lam` says how much
             a pair votes: the pair is matched at its best circular shift first, and
             votes `lam / (lam + D_best)`, so two loops that agree once shifted vote
             nearly in full and two different cross-sections barely at all.  0 leaves
             every phase at zero

    Each loop is fitted in its plane's own frame, which is the frame the slice
    panels draw it in.  Asked for the cloud's absolute coordinates instead, the
    one shared MLP spends its output on a constant as large as the model is far
    from the origin -- 1300 units on dancing children -- and cannot draw the
    loop at the same time: measured there, mean error 39.99 in the cloud frame
    against 2.30 in the plane frame, at the same 300 steps.

    Returns the fitted loops back in the cloud frame, in the order they came in.
    Their predicted in-plane normals are written into the loop PLYs and drawn on
    the fit panels, thinned to `normal_arrows` per loop.  A normal keeps the length
    of the fit's own tangent instead of being scaled to one, so the arrows stay in
    proportion to the curves they leave.  It is the surface normal the curve implies,
    turned the way the direction term asks the tangent for, so it faces the way the
    cloud's own normals do rather than against them.
    """
    ## every stage of the extraction is already on the object, read once here
    pc, pcn = found.points, found.normals
    p_normal, p_point = found.p_normal, found.p_point
    out, stem = found.out, found.stem

    if not found.loops:
        raise SystemExit('nothing to fit: a loop is only found once the points '
                         'are parted between centres, which needs normals')
    if max_err <= 0.0:
        raise ValueError(f'max_err must be positive, got {max_err}')
    if not 0.0 <= max_align_err <= 2.0:
        raise ValueError(f'max_align_err must be in [0, 2], got {max_align_err}')
    if bad_loop_err < 0.0:
        raise ValueError(f'bad_loop_err must be non-negative, got {bad_loop_err}')
    if snapshot_every < 0:
        raise ValueError(f'snapshot_every must be 0 or a count of epochs, '
                         f'got {snapshot_every}')
    if freqs < 1:
        raise ValueError(f'freqs must be at least 1, got {freqs}')
    if feature_pool != 'centroid' and feature != 'triplane':
        raise SystemExit('--feature-pool occupied_cells needs --feature triplane; '
                         'an embedding row is not read from the field')

    ## the cube warm-up is a file, not a stage of this run: fail on a missing one
    ## before the extraction has produced loops nothing will warm
    if warmup is not None and not os.path.isfile(warmup):
        raise SystemExit(f'--warmup: no checkpoint at {warmup}. Train one with '
                         f'`python model/warmup_cube.py`, or pass --no-warmup to fit from '
                         f'a blank network')
    ## the rehearsal is a second reading of the same checkpoint, so it is resolved
    ## once here rather than in every stage's trainer
    teacher_warmup = distill_warmup or warmup
    if distill:
        if distill < 0:
            raise SystemExit(f'--distill must be non-negative, got {distill}')
        if feature != 'triplane':
            raise SystemExit('--distill transfers through the triplane field; use '
                             '--feature triplane')
        if teacher_warmup is None or not os.path.isfile(teacher_warmup):
            raise SystemExit(f'--distill: no teacher checkpoint at {teacher_warmup}. '
                             f'Pass --distill-warmup, or train one with '
                             f'`python model/warmup_cube.py`')

    ## the limit is stated in the shape's own units, but chosen as a fraction of its
    ## longest extent, which is the only scale both a normalised and a raw copy of
    ## the same model share.  Read before the fit rather than after it, since the
    ## snapshots taken during the run are drawn against the same red limit
    limit = max_err * np.ptp(pc, axis=0).max()
    print(f'  max_err {max_err}: loops past {limit:.4f} '
          f'({max_err:.1%} of the {np.ptp(pc, axis=0).max():.3f} extent) are red')
    loop_limit = bad_loop_err * np.ptp(pc, axis=0).max()
    if bad_loop_err > 0.0:
        print(f'  bad_loop_err {bad_loop_err}: loops whose mean error is past '
              f'{loop_limit:.4f} ({bad_loop_err:.1%} of the extent) go to the '
              f'bad side')

    ## The grids are the unit cube, uniformly subdivided and the same for every
    ## shape; the cloud's box, with a small margin so a loop centre on the bounding
    ## face still falls inside, is only the frame that cube is laid over. Stretching
    ## it is a change of coordinates on a feature map, not a different field.
    triplane = None
    if feature == 'triplane':
        extent = np.ptp(pc, axis=0)
        pad = 0.02 * float(extent.max())
        lo, hi = pc.min(0) - pad, pc.max(0) + pad
        triplane = dict(centre=0.5 * (lo + hi), half=0.5 * (hi - lo),
                        resolution=triplane_res, channels=triplane_channels,
                        tv=triplane_tv)
        print(f'  triplane: {triplane_res} x {triplane_res} x 3 grids on the unit '
              f'cube, {triplane_channels} channels, mapped onto the cloud box '
              f'{np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}'
              f'{f", tv {triplane_tv:g}" if triplane_tv else ""}')

    ## ---- the plane plan ---------------------------------------------------------
    ## A fit that refines its planes as it goes: it starts on the planes the extractor
    ## cut, and every `refine_every` epochs bisects the widest gaps along each axis,
    ## so the earlier planes stay exactly where they are and the loops read off them
    ## are the same loops.  What a stage hands on is therefore a fit, not a curve to
    ## relearn: only the loops on the new planes are seeded.
    plan = refine_plan(found.samples, epochs, refine_every, add_planes, samples_end)
    offsets = np.asarray(found.offsets, dtype=np.float64)      # what has been cut
    loops, targets, normals, frame = [], [], [], []            # the current stage's
    trainer = None

    ## ---- the plane codes --------------------------------------------------------
    ## Every loop's embedding row is looked up by the plane it was cut on, not by where
    ## the loop happens to sit in a list. A stage inserts planes, which renumbers every
    ## loop after them, and a row that is handed out by position would have to be
    ## matched back up at every stage -- a guess that can only be as good as the guess
    ## of which loop continues which. Handing rows out per (plane, slot) and never
    ## moving one makes the assignment the same for the whole run by construction: a
    ## plane's row is its code, the row is allocated the first time the plane is cut,
    ## and the table of a stage is the table of the stage before plus the rows for the
    ## planes it added.
    row_of = {}

    def codes_for(stage_loops):
        """The embedding row of every loop of the current plane set, allocating the
        rows of planes (and of loops on a plane) that have not been seen before."""
        rows = []
        for key in plane_keys(p_normal, p_point, stage_loops):
            if key not in row_of:
                row_of[key] = len(row_of)
            rows.append(row_of[key])
        return np.asarray(rows, dtype=np.int64)

    def blend_added_codes(trainer, inserts, first_new):
        """Seed every inserted plane's rows from the planes it was inserted between.

        A row is written as (1 - t) * left + t * right, with t read across the gap
        and rows matched by slot.  A side without that slot is left out of the
        blend; a slot neither side has is drawn from the spread of the inherited
        rows, so no inserted row is left at the zero feature.  Insertions are
        visited in the order bisection made them, so a plane added into a gap an
        earlier insertion split already reads rows that are seeded.
        """
        keys = plane_keys(p_normal, p_point, loops)
        on_plane = {}
        for axis, offset, slot in keys:
            on_plane.setdefault(axis, {}).setdefault(offset, {})[slot] = row_of[
                (axis, offset, slot)]

        def rows_at(axis, offset):
            """The rows of the plane at `offset`: the offset is rebuilt from the
            plane's own point, so match it to the key rather than to its bits."""
            best, best_dist = None, 1e-9
            for plane_offset, slots in on_plane.get(axis, {}).items():
                dist = abs(plane_offset - offset)
                if dist <= best_dist:
                    best, best_dist = slots, dist
            return best or {}

        weight = trainer.model.embedding.weight
        sd = float(weight[:first_new].std()) if first_new else 0.0
        seeded, drawn, planes = 0, 0, 0
        with torch.no_grad():
            for axis, added in enumerate(inserts):
                planes += len(added)
                for middle, left, right in added:
                    lo = rows_at(axis, left) if left is not None else {}
                    hi = rows_at(axis, right) if right is not None else {}
                    if left is not None and right is not None and right != left:
                        t = (middle - left) / (right - left)
                    else:
                        t = 0.0
                    for slot, row in rows_at(axis, middle).items():
                        a, b = lo.get(slot), hi.get(slot)
                        if a is None and b is None:
                            if not np.isfinite(sd) or sd <= 0.0:
                                raise ValueError(
                                    f'no inherited row spread to draw the row of a '
                                    f'slot neither neighbour holds, on axis {axis} '
                                    f'at {middle}')
                            weight[row].normal_(0.0, sd)
                            drawn += 1
                            continue
                        if a is None:
                            weight[row].copy_(weight[b])
                        elif b is None:
                            weight[row].copy_(weight[a])
                        else:
                            weight[row].copy_((1.0 - t) * weight[a] + t * weight[b])
                        seeded += 1
        extra = (f', {drawn} with no slot on either side drawn at sigma {sd:.4f}'
                 if drawn else '')
        print(f'  code init: {seeded} rows on {planes} inserted planes blended from '
              f'their neighbours{extra}')

    def random_added_codes(trainer, first_new, inserts):
        """Draw each inserted plane's row from the spread of the inherited rows.

        The draw is per row and independent, so two new planes of the same loop
        count do not start on the same curve.  Its scale is the standard deviation
        of every inherited row, which keeps a seeded row the size of a learned one;
        a draw of a fixed small constant would make the size an extra choice the
        blend does not need.  Rows come from the run's own random state, so a
        seeded run is reproducible.
        """
        weight = trainer.model.embedding.weight
        with torch.no_grad():
            sd = float(weight[:first_new].std()) if first_new else 0.0
            if not np.isfinite(sd) or sd <= 0.0:
                raise ValueError('no inherited row spread to draw the new rows at; '
                                 'the first plane set has to be fitted before a '
                                 'plane is inserted')
            for row in range(first_new, trainer.num_models):
                weight[row].normal_(0.0, sd)
        planes = sum(len(added) for added in inserts)
        print(f'  code init: {trainer.num_models - first_new} rows on {planes} '
              f'inserted planes drawn at sigma {sd:.4f} from the inherited rows')

    def read_stage():
        """The loops on the extractor's current plane set, as training targets: each in
        its plane's frame, with the normals of its own points."""
        ## the plane's centre point is the frame origin; the normals ride along
        ## unchanged, since a direction does not move when the origin does
        return (found.loops,
                [pts - p_point[s] for s, _, pts, _ in found.loops],
                [n for *_, n in found.loops],
                [p_point[s] for s, _, _, _ in found.loops])

    ## ---- predict and measure, once per loop -------------------------------------
    ## The points are the curve, and the analytic tangent is what both the predicted
    ## normal and the direction error are read from, so the two are taken together
    ## and every number below comes from them: nothing evaluates the network again
    ## for a metric, and nothing is paired against a second draw of the targets.
    ##
    ## A function rather than a block, because the snapshots taken during the run
    ## draw the same pass: a picture of a fit half way through has to be measured the
    ## way the end of the run is, or the two would not be comparable
    def fitted():
        """Every loop as the trainer now stands, with its predicted normal and error."""
        local = [trainer.predict(i).cpu().numpy() for i in range(len(loops))]
        tangents = [trainer.predict_tangent(i) for i in range(len(loops))]
        ## the predicted normal is that tangent turned a quarter turn about the plane
        ## normal, in the sense the direction term asks the tangent for -- a quarter
        ## turn applied twice is a half turn, so it is tangent x axis and not axis x
        ## tangent. It is left at the length the turn gives rather than scaled to one,
        ## so a loop the fit sweeps faster draws longer arrows
        pred_normals = []
        for t, (s, *_) in zip(tangents, loops):
            axis = p_normal[s] / (np.linalg.norm(p_normal[s]) + 1e-12)
            pred_normals.append(np.cross(t, axis[None, :]))
        ## back into the cloud's own frame, or the fit cannot be laid over the shape
        preds = [p + frame[i] for i, p in enumerate(local)]

        ## ---- measure, from those same predictions -------------------------------
        ## one nearest-neighbour pass per loop serves both errors: the distances are
        ## the fit error and the indices name the target normal each predicted point
        ## is held against.  Chunked, since a whole slice against every predicted
        ## point does not fit as one cdist
        loop_errors, paired = [], []
        for p, t, n in zip(local, targets, normals):
            pred = torch.as_tensor(p, dtype=torch.float32, device=trainer.device)
            tgt = torch.as_tensor(t, dtype=torch.float32, device=trainer.device)
            dist, near = [], []
            for chunk in pred.split(2048):
                d = torch.cdist(chunk, tgt)
                dist.append(d.min(1).values)
                near.append(d.argmin(1))
            loop_errors.append(torch.cat(dist).cpu().numpy())
            paired.append(n[torch.cat(near).cpu().numpy()])

        ## 0 means every fitted curve runs the way its target points' normals say, 2
        ## that it runs against them.  Same curves and same pairing as the fit error
        ## above, over the whole target rather than a fresh sample of it, so the
        ## number does not depend on a draw.  Read per loop as well, because a curve
        ## can sit on its points and still run across them, and the good/bad split
        ## below must see that even when the direction term was not trained on
        d1 = torch.from_numpy(np.stack(tangents)).float().to(trainer.device)
        pair = torch.from_numpy(np.stack(paired)).float().to(trainer.device)
        ## the trainer's axes, read off each loop's own plane
        axes = trainer.axes
        loop_align = mini_nn.normal_penalty(
            d1, pair, axes, per_curve=True).cpu().numpy()
        align_err = float(mini_nn.normal_penalty(d1, pair, axes))

        ## ---- report the direction the fit ended on --------------------------------
        ## Chamfer distance places the curve but says nothing about which way it is
        ## walked, and a closed loop walked backwards is a stationary point of the
        ## 1 - cos normal term, so once the fit has settled on it the term cannot pull
        ## it out again: its gradient vanishes exactly at the reversal. The direction
        ## the fit ended up with is therefore reported in `align` and kept in the
        ## curves: nothing is turned round here, so the PLYs, figures and snapshots
        ## carry the traversal the network produced. A loop that disagrees with its
        ## normals is the signal the segmentation step wants; a consumer that needs
        ## the cloud's orientation must make that choice itself.
        flipped = 0
        for i, n in enumerate(pred_normals):
            ## the cosine the direction term is trained on, so the vote does not
            ## depend on how fast the fit sweeps each stretch of the loop: a mean
            ## rather than a sum, so it does not grow with the point count either.
            ## The margin is a tenth of full agreement -- less than that is a loop
            ## turned side-on or one that folds and has no one direction to take
            n_hat = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            p_hat = paired[i] / (np.linalg.norm(paired[i], axis=1, keepdims=True) + 1e-12)
            if float(np.einsum('ij,ij->i', n_hat, p_hat).mean()) < -0.1:
                flipped += 1
        return preds, pred_normals, loop_errors, loop_align, align_err, flipped

    ## a fit grid every snapshot_every epochs, so a run can be watched taking shape
    ## rather than only judged once it is over: the same figure the run ends with,
    ## drawn against the same red limit, in a file named by the epoch it was taken
    ## at, so flipping through them shows the curves arriving on their loops
    snapshots = out / 'snapshots'
    if snapshot_every:
        snapshots.mkdir(parents=True, exist_ok=True)
        print(f'  a fit grid every {snapshot_every} epochs -> {snapshots}')

    done = 0        # epochs the run has finished, which is where a snapshot's name is

    def watch(n):
        """A grid of the fit as it stands, once the run is `done + n` epochs in."""
        if not snapshot_every or (done + n) % snapshot_every:
            return
        ## the network as the epoch line above it left it, measured the way the end of
        ## the run is measured rather than by a cheaper reading of its own
        preds_, normals_, _, _, align_, _ = fitted()
        save_fit_grid(snapshots / f'{stem}_fits_e{done + n:04d}.png',
                      loops, preds_, p_normal, p_point, limit, large,
                      normals_, normal_arrows, align=align_)

    ## ---- the stages -------------------------------------------------------------
    ## A stage cuts its own plane set, builds a network for the loops on it and hands
    ## over from the stage before: the shared network, and every embedding row the two
    ## tables have in common. Because a row is a plane's code, the rows in common are
    ## the planes both stages cut, and no loop has to be matched up to find them.
    if len(plan) > 1:
        at_end = sum(1 for n, _ in plan if n == plan[-1][0])
        print(f'  planes {plan[0][0]} -> {plan[-1][0]} per axis, {add_planes} added '
              f'every {refine_every} epochs, {len(plan)} stages, {at_end} of them at '
              f'the final set')

    for stage, (planes, stage_epochs) in enumerate(plan):
        previous_trainer = trainer
        inserts = None
        last_code_graph = None
        if stage:
            ## the widest gaps of each axis, bisected: what is already cut stays cut,
            ## at the same offsets, so its slices and loops are the same ones.  The
            ## trace says which two planes each insertion lies between, which is what
            ## the seeded row is read from
            new_offsets, inserts = [], []
            for o in offsets:
                o_new, added = bisect_offsets_trace(o, planes - len(o))
                new_offsets.append(o_new)
                inserts.append(added)
            offsets = np.stack(new_offsets)
            print(f'\n  ---- stage {stage + 1}/{len(plan)}: {planes} planes per axis, '
                  f'{stage_epochs} epochs ----')
            found.restage(offsets)
            p_normal, p_point = found.p_normal, found.p_point

        loops, targets, normals, frame = read_stage()
        ## the codes of this plane set, which is the whole of what ties a row to a
        ## plane: rows already handed out are reused, and only planes cut for the first
        ## time get new ones, always at the end of the table
        keys = plane_keys(p_normal, p_point, loops)
        code_pairs, code_weights, code_dists = code_neighbour_graph(
            keys, targets, gate=code_gate, min_radius=code_min_radius)
        if code_pairs:
            print(f'  code graph: {len(code_pairs)} mutual-nearest pairs within '
                  f'{code_gate:g} spacings, radius ratio >= {code_min_radius:g}, '
                  f'median separation {np.median(code_dists):.4f}'
                  f'{"" if code_smooth else " (smoothness off)"}')
        elif code_smooth:
            print(f'  code smooth {code_smooth:g}: no neighbouring loops pass the '
                  f'gate, so the term is off for this stage')
        ## the seam distillation reads the same graph: with no pair there is no D
        ## to hold, and the term is off for this stage
        distill_weight = distill if code_pairs else 0.0
        if distill and not code_pairs:
            print('  seam distillation: no neighbouring loops pass the gate, so the '
                  'term is off for this stage')
        ## the phase solve runs on the same neighbour graph as the code term: with no
        ## pair there is nothing to hold a seam to, and every phase stays at zero
        seam_weight = seam_align if code_pairs else 0.0
        if seam_align and not code_pairs:
            print('  seam align: no neighbouring loops pass the gate, so the term is '
                  'off for this stage')
        elif seam_weight:
            print(f'  seam align: {len(code_pairs)} neighbour pairs, {seam_steps} '
                  f'Adam steps every {seam_every} epochs from epoch {seam_start}, '
                  f'lam {seam_lam:g}')
        ## only passed when on: a trainer given seam keys without a weight has been
        ## asked for phases nothing will ever solve
        seam_args = (dict(seam_align=seam_weight, seam_pairs=code_pairs,
                          seam_weights=code_weights, seam_keys=keys,
                          seam_every=seam_every, seam_start=seam_start,
                          seam_steps=seam_steps, seam_lam=seam_lam)
                     if seam_weight else {})
        if feature == 'triplane':
            ## the field is read at the loop's world centre: the target is in the
            ## plane's local frame, so the frame origin is added back here
            queries = np.stack([frame[i] + targets[i].mean(0)
                                for i in range(len(targets))])
            ## pooled features read the loop's own points, again carried into the
            ## world frame; they are the observation, not the fitted curve
            query_sets = ([frame[i] + targets[i] for i in range(len(targets))]
                          if feature_pool == 'occupied_cells' else None)
            trainer = mini_nn.Trainer(
                targets, num_models=len(targets), num_neurons=neurons,
                num_layers=layers, normals=normals, align_normal=align,
                match_angle=match_angle, cycle=cycle,
                repel=repel, repel_gap=repel_gap, repel_samples=repel_samples,
                code_weights=code_weights,
                feature_mode='triplane', triplane=triplane,
                query_points=queries, query_sets=query_sets,
                feature_pool=feature_pool,
                feature_pairs=code_pairs,
                feature_tv=triplane_tv,
                freeze_f_epochs=max(0, freeze_f_epochs - done),
                axis_condition=axis_condition, num_freqs=freqs,
                distill=distill_weight, distill_warmup=teacher_warmup,
                distill_pairs=code_pairs, distill_weights=code_weights,
                **seam_args)
            shape = tuple(trainer.model.triplane.grid.shape)
            print(f'  fitting {len(targets)} loops, M = '
                  f'{trainer.dataloader.max_points} points each, triplane {shape}, '
                  f'{trainer.model.triplane.feat_dim}-D features, '
                  f'{feature_pool}, '
                  f'{layers}x{neurons} net, {freqs} harmonic'
                  f'{"" if freqs == 1 else "s"}, align {align:g}, '
                  f'axis condition {"on" if axis_condition else "off"}'
                  f'{f", f map frozen for {freeze_f_epochs} epochs" if freeze_f_epochs else ""}'
                  f'{f", distill {distill_weight:g}" if distill_weight else ""}'
                  f'{f", cycle {cycle:g}" if cycle else ""}'
                  f'{f", repel {repel:g}" if repel else ""}')
        else:
            codes = codes_for(loops)
            code_rows = [(int(codes[i]), int(codes[j])) for i, j in code_pairs]
            trainer = mini_nn.Trainer(targets, num_models=len(row_of), codes=codes,
                                      num_neurons=neurons, num_layers=layers,
                                      normals=normals, align_normal=align,
                                      match_angle=match_angle, cycle=cycle,
                                      repel=repel, repel_gap=repel_gap,
                                      repel_samples=repel_samples,
                                      code_pairs=code_rows,
                                      code_weights=code_weights,
                                      code_smooth=code_smooth,
                                      freeze_f_epochs=max(0, freeze_f_epochs
                                                          - done),
                                      axis_condition=axis_condition,
                                      num_freqs=freqs,
                                      **seam_args)
            print(f'  fitting {len(targets)} loops, M = '
                  f'{trainer.dataloader.max_points} points each, {len(row_of)} plane '
                  f'codes, {layers}x{neurons} net, align {align:g}'
                  f'{f", f map frozen for {freeze_f_epochs} epochs" if freeze_f_epochs else ""}'
                  f'{f", cycle {cycle:g}" if cycle else ""}'
                  f'{f", repel {repel:g}" if repel else ""}')
        last_code_graph = (keys, code_pairs, code_weights, code_dists)

        ## a plane set added later takes the network over and seeds its own rows; the
        ## rows already handed out are the planes both stages cut, so no loop has to
        ## be matched up to find them
        if stage:
            trainer.inherit(previous_trainer)
            if feature == 'triplane':
                ## the field is copied whole, so an added plane already has a code:
                ## there is no row to blend or draw
                pass
            elif code_init == 'blend':
                blend_added_codes(trainer, inserts, previous_trainer.num_models)
            else:
                random_added_codes(trainer, previous_trainer.num_models, inserts)
        ## the cube warm-up, once, before the first fit.  The checkpoint is a network
        ## and a field trained on cubes and nothing else (by default random size and
        ## place), so it carries no shape: only the weights and the grid are copied,
        ## and the triplane frame stays this run's, which is where the cube is laid
        ## on the cloud's box.  A
        ## plane set added later is not warmed again; its seeded rows are its
        ## starting curves
        if not stage and warmup is not None:
            if feature != 'triplane':
                print('  warm-up: skipped -- the cube warm-up transfers through the '
                      'triplane field, and a row named by a cube plane does not index '
                      'this shape\'s planes; use --feature triplane')
            else:
                meta = mini_nn.load_warmup(warmup, trainer.model)
                report = meta.get('warmup') or {}
                warm_pool = meta.get('feature_pool', 'centroid')
                if warm_pool != feature_pool:
                    raise SystemExit(
                        f'--warmup {warmup} was trained with {warm_pool} features '
                        f'and this fit pools them as {feature_pool}; warm up with '
                        f'`model/warmup_cube.py --feature-pool {feature_pool}`')
                print(f'  warm-up {warmup}: {report.get("steps", "?")} steps on '
                      f'{report.get("loops", "?")} cube loops '
                      f'({report.get("planes", "?")} per axis), '
                      f'{warm_pool}, fit '
                      f'{report.get("fit", float("nan")):.4f}, direction '
                      f'{report.get("cos", float("nan")):+.3f} cos, seam D median '
                      f'{report.get("seam_median", float("nan")):.4f}')
        elif not stage:
            print('  warm-up off: the loops are fitted from a blank network')

        ## every epoch is offered to the watcher and it decides by the run's own count:
        ## stages are cut by the clock, and a snapshot belongs to the epoch of the whole
        ## run rather than to the epoch of the stage it was taken in.  Each stage is its
        ## own `train` call, so the learning rate and the direction term's weight
        ## anneal over the stage: new planes get a full direction term again
        trainer.train(stage_epochs, every=1, on_epoch=watch)
        done += stage_epochs

    ## ---- per-axis orientation record -------------------------------------------
    ## The epoch lines are means over every loop, so a family that runs the other way
    ## round is invisible in them. These rows split the same terms by axis, and keep
    ## every loop's winding at every epoch -- which says when a family turned over,
    ## not only that it ended up that way.
    if trainer.epoch_stats:
        terms_path = out / f'{stem}_epoch_terms.csv'
        with open(terms_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['epoch', 'axis', 'loops', 'reversed', 'winding',
                             'fold', 'fit', 'align', 'loss'])
            for row in trainer.epoch_stats:
                for name in ('x', 'y', 'z'):
                    if name not in row['axes']:
                        continue
                    s = row['axes'][name]
                    writer.writerow([
                        row['epoch'], name, s['loops'], s['reversed'],
                        f"{s['winding']:.6f}", f"{s['fold']:.6f}",
                        f"{s['fit']:.8f}",
                        '' if 'align' not in s else f"{s['align']:.6f}",
                        f"{row['loss']:.8f}"])
        wind_path = out / f'{stem}_loop_winding.npz'
        np.savez(wind_path,
                 winding=np.stack(trainer.winding_history),
                 axis=trainer.axes.abs().argmax(-1).cpu().numpy(),
                 epoch=np.arange(1, len(trainer.winding_history) + 1))
        print(f'  per-axis terms -> {terms_path}')
        print(f'  every loop winding per epoch -> {wind_path}')
        last = trainer.winding_history[-1]
        labels = trainer.axes.abs().argmax(-1).cpu().numpy()
        for a, name in enumerate('xyz'):
            mask = labels == a
            if mask.any():
                print(f'  winding {name}: {int((last[mask] < 0).sum())} of '
                      f'{int(mask.sum())} loops run clockwise seen from +{name}')

    ## ---- the run's own report, from one last pass over the same curves ----------
    preds, pred_normals, loop_errors, loop_align, align_err, flipped = fitted()

    ## a reading, not a loss: the second derivative still needs the network's graph;
    ## a circle is 39.5
    bend = trainer.bend()

    ## ---- the fitted network, so a later test can replay it --------------------
    ## The network is otherwise transient: only the predictions are written. Keeping
    ## the weights, the field and the loop metadata lets a script decode a curve at
    ## any query point without retraining, which is what an interpolation test needs.
    model_path = out / f'{stem}_model.pt'
    mini_nn.save_model(model_path, trainer,
                       keys=np.asarray(keys, dtype=np.float64),
                       plane_normal=np.stack([p_normal[s] for s, *_ in loops]),
                       plane_point=np.stack(frame),
                       samples=int(found.samples))
    print(f'  network and replay metadata -> {model_path}')

    ## ---- code smoothness, read off the table the fit ended on -------------------
    ## The curve distances above say whether each loop sits on its points; this says
    ## whether the codes of neighbouring loops actually became similar, which is the
    ## term's own objective. The all-pair median is the reference a graph pair is
    ## compared against, since code length itself has no absolute scale.
    code_report = None
    if last_code_graph is not None and last_code_graph[1]:
        keys_f, pairs_f, weights_f, dists_f = last_code_graph
        ## the fit's own code: a table row in embedding mode, the field vector
        ## sampled at the loop's query point in triplane mode. Both are one vector
        ## per loop, so the same graph distance means the same thing in either mode
        if feature == 'triplane':
            with torch.no_grad():
                table = trainer.feature_values().cpu().numpy()
            rows_f = np.arange(len(table), dtype=np.int64)
            kind = 'feature'
        else:
            table = trainer.model.embedding.weight.detach().cpu().numpy()
            rows_f = np.asarray(trainer.codes, dtype=np.int64)
            kind = 'code'
        row_a = np.array([rows_f[i] for i, _ in pairs_f], dtype=np.int64)
        row_b = np.array([rows_f[j] for _, j in pairs_f], dtype=np.int64)
        graph_d = np.linalg.norm(table[row_a] - table[row_b], axis=1)
        used = np.unique(rows_f)
        U = table[used]
        all_d = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=2)
        upper = np.triu_indices(len(U), 1)
        all_med = float(np.median(all_d[upper]))
        graph_med = float(np.median(graph_d))
        wn = np.asarray(weights_f, dtype=np.float64)
        code_report = dict(
            weight=code_smooth, pairs=len(pairs_f),
            graph_med=graph_med, all_med=all_med,
            ratio=graph_med / (all_med + 1e-12),
            weighted_msq=float(np.sum(wn * graph_d ** 2) / wn.sum()),
            sep_med=float(np.median(dists_f)))
        print(f'  {kind} (w {code_smooth:g}): {len(pairs_f)} pairs, graph distance '
              f'{graph_med:.4f} (all-pair median {all_med:.4f}, ratio '
              f'{code_report["ratio"]:.2f}), weighted mean square '
              f'{code_report["weighted_msq"]:.4f}')
        with open(out / f'{stem}_{kind}_graph.csv', 'w') as f:
            f.write('axis,src_offset,src_slot,tgt_offset,tgt_slot,inplane_dist,'
                    f'weight,{kind}_dist\n')
            for (i, j), w, d, cd in zip(pairs_f, weights_f, dists_f, graph_d):
                ka, kb = keys_f[i], keys_f[j]
                f.write(f'{ka[0]},{ka[1]:.6f},{ka[2]},{kb[1]:.6f},{kb[2]},'
                        f'{d:.6f},{w:.6f},{cd:.6f}\n')
        (out / f'{stem}_{kind}_smoothness.txt').write_text(
            f'{kind} smooth weight: {code_smooth:g}\n'
            f'pairs: {len(pairs_f)} (median in-plane separation '
            f'{code_report["sep_med"]:.6f})\n'
            f'graph {kind} distance: median {graph_med:.6f}, weighted mean square '
            f'{code_report["weighted_msq"]:.6f}\n'
            f'all-pair {kind} distance: median {all_med:.6f}\n'
            f'ratio graph/all: {code_report["ratio"]:.4f}\n')

    ## ---- the feature field itself, for later inspection -------------------------
    ## The network is transient, as it is in embedding mode, but the field is the
    ## whole point of this mode: the grids and the vector each loop sampled are what
    ## a later run, a decoder or a diagnostic would read. An npz rather than a PLY,
    ## since it is a 3 x C x R x R array and not a surface.
    if feature == 'triplane':
        field = trainer.model.triplane
        with torch.no_grad():
            feats = trainer.feature_values().cpu().numpy()
        np.savez(out / f'{stem}_triplane.npz',
                 grid=field.grid.detach().cpu().numpy(),
                 centre=field.box_centre.cpu().numpy(),
                 half=field.box_half.cpu().numpy(),
                 lo=field.lo.cpu().numpy(), hi=field.hi.cpu().numpy(),
                 resolution=field.resolution, channels=field.channels,
                 queries=trainer.queries.cpu().numpy(), features=feats,
                 feature_pool=np.asarray(feature_pool),
                 cell_counts=(trainer.cell_counts.cpu().numpy()
                              if trainer.cell_counts is not None
                              else np.zeros((0, 3), dtype=np.int64)),
                 keys=np.asarray(keys, dtype=np.float64))
        print(f'  triplane field {tuple(field.grid.shape)} and {len(feats)} sampled '
              f'features ({feature_pool}) -> {out / f"{stem}_triplane.npz"}')

    ## ---- report -----------------------------------------------------------------
    print(f'  bend {bend:.1f} (a circle is 39.5)')
    if align:
        print(f'  align {align_err:.3f} (mean 1 - cos over all loops; {flipped} of '
              f'{len(loops)} loops ran against the points\' normals)')
    if flipped:
        print(f'  those {flipped} loop{"" if flipped == 1 else "s"} keep the direction '
              f'the fit gave {"it" if flipped == 1 else "them"} in the saved files '
              f'and figures')

    ## ---- save -------------------------------------------------------------------
    err = np.concatenate(loop_errors)

    ## a loop is bad when any of three checks fails: one point past the point limit,
    ## its mean error past the mean limit, or a mean direction error past the
    ## alignment limit. The point limit misses a curve that is off along a whole arc
    ## without a single far point; the mean limit is the size-independent reading of
    ## that, and the alignment limit catches a curve that lies on its target and
    ## still runs across it. One loop is one flag
    loop_mean = np.array([float(e.mean()) for e in loop_errors])
    bad_point = np.array([e.max() > BAD_POINT_ERR for e in loop_errors])
    bad_mean = (loop_mean > loop_limit if bad_loop_err > 0.0
                else np.zeros(len(loop_errors), dtype=bool))
    bad_align = (loop_align > max_align_err if max_align_err > 0.0
                 else np.zeros(len(loop_errors), dtype=bool))
    bad_loops = bad_point | bad_mean | bad_align
    bad = np.repeat(bad_loops, [len(e) for e in loop_errors])
    loop_points = np.concatenate(preds)
    loop_normals = np.concatenate(pred_normals)

    ## good loops are coloured by the axis of the plane that cut them, so the three
    ## plane families read apart in any viewer; a bad loop stays one red, which is
    ## the check the bad file is written for. Only the two sides are written: the
    ## union is their concatenation and costs space without adding a reading
    axis = np.repeat([k[0] for k in keys], [len(e) for e in loop_errors])
    good_colors = AXIS_COLORS[axis]
    bad_colors = np.tile(np.array([255, 0, 0, 255], np.uint8), (len(bad), 1))
    ## the split PLYs lose which loops came in together, so the flag of every loop
    ## is kept here in the checkpoint's own order; a character per loop is enough
    (out / f'{stem}_loop_split.txt').write_text(
        ''.join('b' if b else 'g' for b in bad_loops) + '\n')
    ## one row per loop in checkpoint order, with every error and every flag, so a
    ## different threshold can be tried on a finished run without fitting it again
    with open(out / f'{stem}_loop_errors.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['axis', 'offset', 'slot', 'points', 'mean_err', 'max_err',
                         'align_err', 'bad_point', 'bad_mean', 'bad_align', 'bad'])
        for key, e, mean, align, bp, bm, ba, b in zip(
                keys, loop_errors, loop_mean, loop_align, bad_point, bad_mean,
                bad_align, bad_loops):
            writer.writerow([key[0], f'{key[1]:.4f}', key[2], len(e),
                             f'{mean:.6f}', f'{float(e.max()):.6f}',
                             f'{float(align):.6f}', int(bp), int(bm), int(ba),
                             int(b)])
    print(f'  per-loop errors -> {out / f"{stem}_loop_errors.csv"}')
    ## a side that holds no loop is left out rather than written empty: trimesh has
    ## no vertices to take a normals range over, and an empty ply says nothing anyway
    for name, keep, cols in (('good', ~bad, good_colors),
                             ('bad', bad, bad_colors)):
        if keep.any():
            write_points(out / f'{stem}_{name}_loops.ply', loop_points[keep],
                         cols[keep], loop_normals[keep])
        else:
            print(f'  no {name} loops: {out / f"{stem}_{name}_loops.ply"} not written')
    mean_reason = (f'{np.count_nonzero(bad_mean)} past the mean limit '
                   f'{loop_limit:.3f}' if bad_loop_err > 0.0 else 'mean limit off')
    print(f'  loop mean error {loop_mean.min():.3f}-{loop_mean.max():.3f} '
          f'({mean_reason})')
    if max_align_err > 0.0:
        print(f'  alignment per loop {loop_align.min():.3f}-'
              f'{loop_align.max():.3f} (mean 1 - cos; bad past '
              f'{max_align_err:.3f})')
        align_reason = f'{np.count_nonzero(bad_align)} past the alignment limit'
    else:
        print(f'  alignment per loop {loop_align.min():.3f}-'
              f'{loop_align.max():.3f} (mean 1 - cos; not used for the split)')
        align_reason = 'alignment limit off'
    reasons = (f'{np.count_nonzero(bad_point)} past the point limit '
               f'{BAD_POINT_ERR:.2f}, {mean_reason}, {align_reason}')
    n_bad = int(np.count_nonzero(bad_loops))
    print(f'  {len(preds)} fitted loops, error {err.min():.3f}-{err.max():.3f} '
          f'({n_bad} bad loops all red: {reasons}; the rest coloured by plane '
          f'axis x/y/z; predicted normals in nx/ny/nz)')
    print(f'  good and bad loops saved separately, where either holds any '
          f'-> {out / f"{stem}_good_loops.ply"}, {out / f"{stem}_bad_loops.ply"}')

    ## the alignment penalty is one number over the whole run, so it goes into the
    ## figure's own line rather than into any one panel
    save_fit_grid(out / f'{stem}_fits.png', loops, preds, p_normal, p_point,
                  limit, large, pred_normals, normal_arrows, align=align_err)
    ## the same results again one plane at a time, so a single slice can be
    ## loaded and judged without the rest of the run beside it
    save_plane_files(out, pc, pcn, found.slices, found.centres, loops, preds,
                     p_normal, p_point, limit, large, pred_normals, normal_arrows,
                     write_files=plane_files)
    ## the slice and field artefacts were written for the planes the run started on,
    ## so they are written again for the planes it ended on: the extraction figures
    ## and the fit grid have to show the same planes, or the folder contradicts itself
    if len(plan) > 1:
        print(f'  extraction artefacts re-written for the {plan[-1][0]} planes per '
              f'axis the run ended on')
        found.resave()
    return preds


def main(argv=None):
    """Extract the loops, then fit them when ``--train`` is given."""
    args = build_parser().parse_args(argv)
    arrows = None if args.arrows == 'all' else int(args.arrows)
    normal_arrows = (None if args.normal_arrows == 'all'
                     else int(args.normal_arrows))
    ## checked here rather than in the fit, so a mistyped switch fails before the
    ## extraction spends a minute producing loops nothing will be fitted to
    if args.train and args.warmup is not None and not os.path.isfile(args.warmup):
        raise SystemExit(f'--warmup: no checkpoint at {args.warmup}. Train one with '
                         f'`python model/warmup_cube.py`, or pass --no-warmup to fit from '
                         f'a blank network')
    if args.snapshot_every < 0:
        raise SystemExit(f'--snapshot-every must be 0 or a count of epochs, got '
                         f'{args.snapshot_every}')
    if args.freqs < 1:
        raise SystemExit(f'--freqs must be at least 1, got {args.freqs}')
    if args.refine_every < 0:
        raise SystemExit(f'--refine-every must be 0 or a count of epochs, got '
                         f'{args.refine_every}')
    if args.refine_every and args.add_planes < 1:
        raise SystemExit(f'--add-planes must be at least 1 to refine the planes, got '
                         f'{args.add_planes}')
    if args.samples_end is not None and args.samples_end < args.samples:
        raise SystemExit(f'--samples-end must not be under --samples: '
                         f'{args.samples_end} < {args.samples}')

    ## one object holds the whole extraction, so the fit reads stages off it
    ## instead of a seven-way tuple threaded through both modules
    found = LoopExtractor(args.path, samples=args.samples, margin=args.margin,
                          band=args.band, max_align=args.max_align,
                          out=args.out)
    found.extract(figures=not args.no_image, n_arrows=arrows,
                  field=args.field, assign=args.assign)

    if args.train:
        fit_loops(found, epochs=args.epochs, neurons=args.neurons,
                  layers=args.layers,
                  align=args.align, match_angle=args.match_angle,
                  cycle=args.cycle, repel=args.repel, repel_gap=args.repel_gap,
                  repel_samples=args.repel_samples,
                  max_err=args.max_err, max_align_err=args.max_align_err,
                  bad_loop_err=args.bad_loop_err,
                  large=args.large,
                  normal_arrows=normal_arrows,
                  plane_files=args.plane_files,
                  warmup=args.warmup,
                  snapshot_every=args.snapshot_every,
                  refine_every=args.refine_every, add_planes=args.add_planes,
                  samples_end=args.samples_end, code_init=args.code_init,
                  code_smooth=args.code_smooth, code_gate=args.code_gate,
                  code_min_radius=args.code_min_radius, feature=args.feature,
                  triplane_res=args.triplane_res,
                  triplane_channels=args.triplane_channels,
                  triplane_tv=args.triplane_tv,
                  feature_pool=args.feature_pool,
                  axis_condition=args.axis_condition,
                  freeze_f_epochs=args.freeze_f_epochs, freqs=args.freqs,
                  distill=args.distill, distill_warmup=args.distill_warmup,
                  seam_align=args.seam_align, seam_every=args.seam_every,
                  seam_start=args.seam_start, seam_steps=args.seam_steps,
                  seam_lam=args.seam_lam)
    return found


if __name__ == '__main__':
    main()
