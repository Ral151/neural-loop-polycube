"""The fitting run behind `main.py extract` and `main.py fit`.

Loops are extracted with `utils.loop_extraction.LoopExtractor`; `fit` then trains
one shared network on all of them (`model.trainer.Trainer`) and writes the
checkpoint, figures and good / bad loop PLYs.
"""

import argparse
import csv
import os

import numpy as np
import torch

from model.losses import normal_penalty
from model.trainer import Trainer, save_model
from model.warmup import load_warmup
from utils.loop_extraction import DEFAULT_PLY, LoopExtractor
from utils.ply_io import write_points
from utils.slicing import bisect_offsets_trace, plane_keys
from utils.visualization import AXIS_COLORS, save_fit_grid, save_plane_files


# a loop with any fitted point farther than this from its target (plane frame) is bad
BAD_POINT_ERR = 0.15


def refine_plan(samples, epochs, every=0, add=2, end=None):
    """Plan the stages as [(planes per axis, epochs)], the epochs summing to `epochs`.

    Starts at `samples` planes and adds `add` per axis (by bisection) every `every`
    epochs, up to `end`; once the count stops growing the remaining epochs form one
    final stage. every <= 0 gives a single stage.
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
        # same plane count as the previous stage: extend it instead
        if plan and plan[-1][0] == planes:
            continue
        plan.append([planes, every])
    # the last stage takes whatever epochs remain
    plan[-1][1] = epochs - every * (len(plan) - 1)
    return [tuple(stage) for stage in plan]


def add_extraction_args(parser):
    """The options that place the planes and cut the loops: `extract` and `fit`."""
    parser.add_argument('--path', default=DEFAULT_PLY)
    parser.add_argument('--samples', type=int, default=5, help='planes sampled per axis')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='inset from each bounding box face, as a fraction of the axis extent')
    parser.add_argument('--no-image', action='store_true',
                        help='skip the projected slice grid png')
    parser.add_argument('--arrows', default='80',
                        help='normal arrows per panel (0 for none, all for every normal)')
    parser.add_argument('--field', action='store_true',
                        help='draw the implicit field of each loop behind the slice '
                             'points, and on its own in a second png')
    parser.add_argument('--assign', action='store_true',
                        help='colour each slice point by the loop centre it belongs to')
    parser.add_argument('--band', type=float, default=0.0075,
                        help='half-width of the slab counted as on a plane, as a '
                             'fraction of the longest extent')
    parser.add_argument('--max-align', type=float, default=0.95,
                        help='drop a point from a plane when its normal is within this '
                             "cosine of the plane's normal (its surface lies flat in "
                             'the slice); 1.0 keeps every point')
    parser.add_argument('--out', default=None,
                        help='output folder (default log/<input stem>/)')
    return parser


def add_fit_args(parser):
    """The options of the network fit, read only by `fit`."""
    parser.add_argument('--normal-arrows', default='20',
                        help='arrows drawn from each fitted curve along its '
                             'predicted normal, per loop (0 for none, all for every point)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='training epochs (one batch of every loop each)')
    parser.add_argument('--freeze-f-epochs', type=int, default=0, metavar='N',
                        help='freeze the shared MLP for the first N epochs and train '
                             "only the triplane field, keeping the warm-up's map while "
                             'the loops move onto their slices; 0 trains both from the '
                             'start')
    parser.add_argument('--warmup', default='warmup/cube_warmup_axis.pt',
                        metavar='CKPT',
                        help='cube warm-up checkpoint loaded before the fit (from '
                             '`main.py warmup`); its network and grid sizes must match '
                             "this run's")
    parser.add_argument('--no-warmup', dest='warmup', action='store_const', const=None,
                        help='fit from a blank network (warm-up ablation)')
    parser.add_argument('--no-axis-condition', dest='axis_condition',
                        action='store_false',
                        help='condition on the feature alone, as the runs before the '
                             'axis input did; needs a warm-up trained with '
                             '`main.py warmup --no-axis-condition`, or --no-warmup')
    parser.add_argument('--distill', type=float, default=0.0, metavar='W',
                        help="weight of the seam distillation: each neighbour pair's "
                             "seam distance D is kept at or below the warm-up teacher's "
                             'D; 0 (default) disables')
    parser.add_argument('--distill-warmup', default=None, metavar='CKPT',
                        help='checkpoint of the distillation teacher (default: '
                             '--warmup)')
    parser.add_argument('--code-init', choices=('blend', 'random'), default='blend',
                        help='embedding mode: how rows of inserted planes start; blend '
                             "interpolates the neighbouring planes' rows, random draws "
                             'from their spread')
    parser.add_argument('--snapshot-every', type=int, default=50,
                        help='write the fit grid every N epochs into <out>/snapshots/; '
                             '0 writes none')
    parser.add_argument('--save-states', default='', metavar='E1,E2,...',
                        help='save the network after these epochs (from 1) as '
                             '<out>/states/<stem>_state_eNNNN.pt, for `main.py clusters '
                             '--state`; empty saves none')
    parser.add_argument('--refine-every', type=int, default=0,
                        help='every N epochs bisect the widest plane gaps per axis, '
                             're-cut the cloud and continue the fit on the larger plane '
                             'set; 0 keeps one plane set')
    parser.add_argument('--add-planes', type=int, default=2,
                        help='planes added per axis at each refinement stage')
    parser.add_argument('--samples-end', type=int, default=None,
                        help='planes per axis at the last stage (default: grow by '
                             '--add-planes every stage)')
    parser.add_argument('--neurons', type=int, default=64,
                        help='width of the shared MLP')
    parser.add_argument('--layers', type=int, default=2,
                        help='hidden layers of the shared MLP')
    parser.add_argument('--normal', dest='align', default=0.0,
                        type=float, metavar='W',
                        help="weight of the normal term (1 - cos between the curve's "
                             'implied normal and the matched point normal; also fixes '
                             'the loop direction), cosine-annealed to 0 over each stage')
    parser.add_argument('--code-smooth', type=float, default=0.0,
                        help='weight of a term keeping the codes / features of '
                             'neighbouring loops close, weighted by their in-plane '
                             'centre distance; 0 disables')
    parser.add_argument('--code-gate', type=float, default=2.0,
                        help='max in-plane centre distance of a neighbour pair, in '
                             'median plane spacings')
    parser.add_argument('--code-min-radius', type=float, default=0.4,
                        help='min smaller / larger loop radius ratio of a neighbour '
                             'pair')
    parser.add_argument('--seam-align', type=float, default=0.0, metavar='W',
                        help='weight of the per-loop seam phase solve on the neighbour '
                             'pairs (moves only where z = 0 falls, not the curves); 0 '
                             'disables')
    parser.add_argument('--seam-every', type=int, default=10,
                        help='solve the seam phases every N epochs')
    parser.add_argument('--seam-start', type=int, default=0,
                        help='first epoch of the seam solve (from 1); 0 starts at once')
    parser.add_argument('--seam-steps', type=int, default=100,
                        help='Adam steps per seam solve')
    parser.add_argument('--seam-lam', type=float, default=0.02,
                        help='seam solve vote scale: a pair votes lam / (lam + D at its '
                             'best circular shift), so unrelated loops barely count')
    parser.add_argument('--feature', choices=('embedding', 'triplane'),
                        default='triplane',
                        help='what conditions the MLP: triplane (feature sampled at '
                             "each loop's centre, default) or embedding (one learned "
                             'row per loop)')
    parser.add_argument('--triplane-res', type=int, default=32,
                        help='cells per side of each triplane grid')
    parser.add_argument('--triplane-channels', type=int, default=16,
                        help='channels per triplane grid (the feature is 3x this long)')
    parser.add_argument('--triplane-tv', type=float, default=0.01,
                        help='weight of the total-variation smoothness on the triplane '
                             'grids')
    parser.add_argument('--max-err', type=float, default=0.03,
                        help='highlight a loop (red in figures, listed in '
                             'bad_loops.txt) whose mean error exceeds this fraction of '
                             'the longest extent')
    parser.add_argument('--large', type=float, default=1.5,
                        help='mark a plane whose mean error exceeds this multiple of '
                             'the median plane error')
    return parser


def build_parser(train=True):
    """The command line of `main.py extract` (train=False) and `main.py fit`."""
    parser = argparse.ArgumentParser(
        prog=f'main.py {"fit" if train else "extract"}',
        description=('Extract cross-section loops from a point cloud and fit them '
                     'with the shared network.') if train else
                    'Extract cross-section loops from a point cloud.')
    add_extraction_args(parser)
    if train:
        add_fit_args(parser)
    return parser


def code_neighbour_graph(keys, targets, gate=2.0, min_radius=0.4):
    """Neighbour graph of loops on adjacent planes: mutual-nearest in-plane centres.

    keys: (axis, offset, slot) per loop; targets: N x 3 points per loop (plane frame).
    A pair needs centre distance < gate * median plane spacing and radius ratio
    >= min_radius; its weight falls linearly from 1 (same centre) to 0 (at the gate).
    Returns (pairs, weights, in-plane distances).
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
        # the two in-plane coordinates (planes are axis aligned)
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
              max_err=0.03, large=1.5, normal_arrows=20,
              warmup='warmup/cube_warmup_axis.pt',
              snapshot_every=50, save_states=(),
              refine_every=0, add_planes=2,
              samples_end=None, code_init='blend', code_smooth=0.0,
              code_gate=2.0, code_min_radius=0.4, feature='triplane',
              triplane_res=32, triplane_channels=16, triplane_tv=0.01,
              axis_condition=True,
              distill=0.0, distill_warmup=None,
              seam_align=0.0, seam_every=10, seam_start=0, seam_steps=100,
              seam_lam=0.02, freeze_f_epochs=0):
    """Fit every extracted loop and write the checkpoint, figures and loop PLYs.

    found: a `LoopExtractor` that has already run (cloud, planes, loops, out folder).
    The other arguments mirror the `main.py fit` options (see `add_fit_args`).
    Each loop is fitted in its plane's own frame; fitting in absolute coordinates
    wastes the MLP on a large constant offset. With refine_every > 0 the fit runs in
    stages (see `refine_plan`), each its own `train` call, handing the network on.

    Returns the fitted loops in the cloud frame. Loops whose implied normals run
    against the cloud's are reversed for the saved files and figures.
    """
    # extraction results
    pc, pcn = found.points, found.normals
    p_normal, p_point = found.p_normal, found.p_point
    out, stem = found.out, found.stem

    if not found.loops:
        raise SystemExit('nothing to fit: a loop is only found once the points '
                         'are parted between centres, which needs normals')
    if max_err <= 0.0:
        raise ValueError(f'max_err must be positive, got {max_err}')
    if snapshot_every < 0:
        raise ValueError(f'snapshot_every must be 0 or a count of epochs, '
                         f'got {snapshot_every}')

    # fail early on a missing warm-up checkpoint
    if warmup is not None and not os.path.isfile(warmup):
        raise SystemExit(f'--warmup: no checkpoint at {warmup}. Train one with '
                         f'`python main.py warmup`, or pass --no-warmup to fit from '
                         f'a blank network')
    # the distillation teacher defaults to the fit's own warm-up
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
                             f'`python main.py warmup`')

    # red limit in shape units, as a fraction of the longest extent
    limit = max_err * np.ptp(pc, axis=0).max()
    print(f'  max_err {max_err}: loops past {limit:.4f} '
          f'({max_err:.1%} of the {np.ptp(pc, axis=0).max():.3f} extent) are red')

    # triplane box = the cloud bounding box plus a 2% margin
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

    # ---- refinement plan: stages of growing plane sets ----
    plan = refine_plan(found.samples, epochs, refine_every, add_planes, samples_end)
    offsets = np.asarray(found.offsets, dtype=np.float64)      # what has been cut
    loops, targets, normals, frame = [], [], [], []            # the current stage's
    trainer = None

    # ---- embedding mode: one row per (axis, offset, slot), allocated the first
    # time that loop is cut and never moved, so rows survive plane insertion ----
    row_of = {}

    def codes_for(stage_loops):
        """Embedding row of every loop, allocating rows for planes/slots not seen before."""
        rows = []
        for key in plane_keys(p_normal, p_point, stage_loops):
            if key not in row_of:
                row_of[key] = len(row_of)
            rows.append(row_of[key])
        return np.asarray(rows, dtype=np.int64)

    def blend_added_codes(trainer, inserts, first_new):
        """Seed inserted planes' rows as (1 - t) * left + t * right of the neighbouring
        planes' rows, matched by slot; a slot neither side has is drawn randomly."""
        keys = plane_keys(p_normal, p_point, loops)
        on_plane = {}
        for axis, offset, slot in keys:
            on_plane.setdefault(axis, {}).setdefault(offset, {})[slot] = row_of[
                (axis, offset, slot)]

        def rows_at(axis, offset):
            """Slot -> row map of the plane at `offset`, matched with a float tolerance."""
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
        """Draw each inserted plane's row from a normal with the inherited rows' spread."""
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
        """Current loops as targets: (loops, points in plane frame, normals, frame origins)."""
        return (found.loops,
                [pts - p_point[s] for s, _, pts, _ in found.loops],
                [n for *_, n in found.loops],
                [p_point[s] for s, _, _, _ in found.loops])

    # ---- decode and measure every loop (shared with the snapshots) ----
    def fitted():
        """Decode every loop: (preds, normals, per-point errors, align error, flips)."""
        local = [trainer.predict(i).cpu().numpy() for i in range(len(loops))]
        tangents = [trainer.predict_tangent(i) for i in range(len(loops))]
        # implied normal = tangent x axis, left unnormalised (arrow length ~ speed)
        pred_normals = []
        for t, (s, *_) in zip(tangents, loops):
            axis = p_normal[s] / (np.linalg.norm(p_normal[s]) + 1e-12)
            pred_normals.append(np.cross(t, axis[None, :]))
        # back into the cloud frame
        preds = [p + frame[i] for i, p in enumerate(local)]

        # one chunked nearest-neighbour pass per loop: the distances are the fit
        # error, the indices pick the paired target normals
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

        # mean 1 - cos over all loops (0 = runs with the normals, 2 = against them)
        align_err = float('nan')
        if trainer.axes is not None:
            d1 = torch.from_numpy(np.stack(tangents)).float().to(trainer.device)
            pair = torch.from_numpy(np.stack(paired)).float().to(trainer.device)
            align_err = float(normal_penalty(d1, pair, trainer.axes))

        # a loop walked backwards is a stationary point of the normal term, so flip
        # every loop whose implied normals mostly oppose the paired ones
        flipped = 0
        for i, (t, n) in enumerate(zip(tangents, pred_normals)):
            # mean-cosine vote, with a margin of 0.1
            n_hat = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            p_hat = paired[i] / (np.linalg.norm(paired[i], axis=1, keepdims=True) + 1e-12)
            if float(np.einsum('ij,ij->i', n_hat, p_hat).mean()) < -0.1:
                flipped += 1
                preds[i] = preds[i][::-1]
                pred_normals[i] = -n[::-1]
                loop_errors[i] = loop_errors[i][::-1]
                tangents[i] = -t[::-1]
        return preds, pred_normals, loop_errors, align_err, flipped

    # fit-grid snapshots during the run
    snapshots = out / 'snapshots'
    if snapshot_every:
        snapshots.mkdir(parents=True, exist_ok=True)
        print(f'  a fit grid every {snapshot_every} epochs -> {snapshots}')

    done = 0        # epochs finished so far, over the whole run

    save_states = sorted({int(e) for e in save_states})
    if save_states:
        (out / 'states').mkdir(parents=True, exist_ok=True)
        print(f'  network state after epoch{"s" if len(save_states) > 1 else ""} '
              f'{", ".join(map(str, save_states))} -> {out / "states"}')

    def watch(n):
        """Epoch hook: save the requested network states and fit-grid snapshots."""
        epoch = done + n
        if epoch in save_states:
            torch.save({k: v.detach().cpu().clone()
                        for k, v in trainer.model.state_dict().items()},
                       out / 'states' / f'{stem}_state_e{epoch:04d}.pt')
        if not snapshot_every or epoch % snapshot_every:
            return
        preds_, normals_, _, align_, _ = fitted()
        save_fit_grid(snapshots / f'{stem}_fits_e{done + n:04d}.png',
                      loops, preds_, p_normal, p_point, limit, large,
                      normals_, normal_arrows, align=align_)

    # ---- stages: each cuts its plane set, builds a trainer and inherits the last ----
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
            # bisect the widest gaps; the existing planes keep their offsets
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
        # neighbour graph of this plane set (smoothness, distillation, seam solve)
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
        # the distillation and the seam solve both need neighbour pairs
        distill_weight = distill if code_pairs else 0.0
        if distill and not code_pairs:
            print('  seam distillation: no neighbouring loops pass the gate, so the '
                  'term is off for this stage')
        seam_weight = seam_align if code_pairs else 0.0
        if seam_align and not code_pairs:
            print('  seam align: no neighbouring loops pass the gate, so the term is '
                  'off for this stage')
        elif seam_weight:
            print(f'  seam align: {len(code_pairs)} neighbour pairs, {seam_steps} '
                  f'Adam steps every {seam_every} epochs from epoch {seam_start}, '
                  f'lam {seam_lam:g}')
        # seam options only when the solve is on
        seam_args = (dict(seam_align=seam_weight, seam_pairs=code_pairs,
                          seam_weights=code_weights, seam_keys=keys,
                          seam_every=seam_every, seam_start=seam_start,
                          seam_steps=seam_steps, seam_lam=seam_lam)
                     if seam_weight else {})
        if feature == 'triplane':
            # query = the loop's world centre (frame origin + local centroid)
            queries = np.stack([frame[i] + targets[i].mean(0)
                                for i in range(len(targets))])
            trainer = Trainer(
                targets, num_models=len(targets), num_neurons=neurons,
                num_layers=layers, normals=normals, align=align,
                code_weights=code_weights, code_smooth=code_smooth,
                feature_mode='triplane', triplane=triplane,
                query_points=queries, feature_pairs=code_pairs,
                feature_tv=triplane_tv, axis_condition=axis_condition,
                freeze_f_epochs=max(0, freeze_f_epochs - done),
                distill=distill_weight, distill_warmup=teacher_warmup,
                distill_pairs=code_pairs, distill_weights=code_weights,
                **seam_args)
            shape = tuple(trainer.model.triplane.grid.shape)
            print(f'  fitting {len(targets)} loops, M = '
                  f'{trainer.dataloader.max_points} points each, triplane {shape}, '
                  f'{trainer.model.triplane.feat_dim}-D features, '
                  f'{layers}x{neurons} net, align {align:g}, '
                  f'axis condition {"on" if axis_condition else "off"}'
                  f'{f", f map frozen for {freeze_f_epochs} epochs" if freeze_f_epochs else ""}'
                  f'{f", distill {distill_weight:g}" if distill_weight else ""}')
        else:
            codes = codes_for(loops)
            code_rows = [(int(codes[i]), int(codes[j])) for i, j in code_pairs]
            trainer = Trainer(targets, num_models=len(row_of), codes=codes,
                              num_neurons=neurons, num_layers=layers,
                              normals=normals, align=align,
                              code_pairs=code_rows,
                              code_weights=code_weights,
                              code_smooth=code_smooth,
                              axis_condition=axis_condition,
                              freeze_f_epochs=max(0, freeze_f_epochs - done),
                              **seam_args)
            print(f'  fitting {len(targets)} loops, M = '
                  f'{trainer.dataloader.max_points} points each, {len(row_of)} plane '
                  f'codes, {layers}x{neurons} net, align {align:g}, '
                  f'axis condition {"on" if axis_condition else "off"}'
                  f'{f", f map frozen for {freeze_f_epochs} epochs" if freeze_f_epochs else ""}')
        last_code_graph = (keys, code_pairs, code_weights, code_dists)

        # later stages take the network over from the previous one
        if stage:
            trainer.inherit(previous_trainer)
            if feature == 'triplane':
                # triplane: the copied field already covers the added planes
                pass
            elif code_init == 'blend':
                blend_added_codes(trainer, inserts, previous_trainer.num_models)
            else:
                random_added_codes(trainer, previous_trainer.num_models, inserts)
        # cube warm-up, before the first stage only
        if not stage and warmup is not None:
            if feature != 'triplane':
                print('  warm-up: skipped -- the cube warm-up transfers through the '
                      'triplane field, and a row named by a cube plane does not index '
                      'this shape\'s planes; use --feature triplane')
            else:
                meta = load_warmup(warmup, trainer.model)
                report = meta.get('warmup') or {}
                print(f'  warm-up {warmup}: {report.get("steps", "?")} steps on '
                      f'{report.get("loops", "?")} cube loops '
                      f'({report.get("planes", "?")} per axis), fit '
                      f'{report.get("fit", float("nan")):.4f}, direction '
                      f'{report.get("cos", float("nan")):+.3f} cos, seam D median '
                      f'{report.get("seam_median", float("nan")):.4f}')
        elif not stage:
            print('  warm-up off: the loops are fitted from a blank network')

        # train this stage; `watch` counts epochs over the whole run
        trainer.train(stage_epochs, every=1, on_epoch=watch)
        done += stage_epochs

    # ---- per-axis orientation record: epoch-terms CSV and per-loop winding ----
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

    # ---- final decode ----
    preds, pred_normals, loop_errors, align_err, flipped = fitted()

    # bend reading (a circle is 39.5)
    bend = trainer.bend()

    # ---- checkpoint, so curves can be decoded later without retraining ----
    model_path = out / f'{stem}_model.pt'
    save_model(model_path, trainer,
               keys=np.asarray(keys, dtype=np.float64),
               plane_normal=np.stack([p_normal[s] for s, *_ in loops]),
               plane_point=np.stack(frame),
               samples=int(found.samples))
    print(f'  network and replay metadata -> {model_path}')

    # ---- code smoothness report: neighbour-pair distance vs the all-pair median ----
    code_report = None
    if last_code_graph is not None and last_code_graph[1]:
        keys_f, pairs_f, weights_f, dists_f = last_code_graph
        # one code per loop: the sampled feature (triplane) or the table row (embedding)
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

    # ---- the triplane grid and each loop's sampled feature ----
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
                 keys=np.asarray(keys, dtype=np.float64))
        print(f'  triplane field {tuple(field.grid.shape)} and {len(feats)} sampled '
              f'features -> {out / f"{stem}_triplane.npz"}')

    # ---- report ----
    print(f'  bend {bend:.1f} (a circle is 39.5)')
    if align:
        print(f'  align {align_err:.3f} (mean 1 - cos over all loops; {flipped} of '
              f'{len(loops)} loops ran against the points\' normals)')
    if flipped:
        print(f'  {flipped} loop{"" if flipped == 1 else "s"} ran against the points\' '
              f'normals and {"was" if flipped == 1 else "were"} turned round for the '
              f'saved files and figures')

    # ---- save ----
    err = np.concatenate(loop_errors)

    # one point past BAD_POINT_ERR makes the whole loop bad
    bad_loops = np.array([e.max() > BAD_POINT_ERR for e in loop_errors])
    bad = np.repeat(bad_loops, [len(e) for e in loop_errors])
    loop_points = np.concatenate(preds)
    loop_normals = np.concatenate(pred_normals)

    # good loops coloured by slice axis, bad loops red
    axis = np.repeat([k[0] for k in keys], [len(e) for e in loop_errors])
    good_colors = AXIS_COLORS[axis]
    bad_colors = np.tile(np.array([255, 0, 0, 255], np.uint8), (len(bad), 1))
    # per-loop good/bad flags ('g'/'b') in checkpoint order
    (out / f'{stem}_loop_split.txt').write_text(
        ''.join('b' if b else 'g' for b in bad_loops) + '\n')
    # an empty side is skipped rather than written as an empty PLY
    for name, keep, cols in (('good', ~bad, good_colors),
                             ('bad', bad, bad_colors)):
        if keep.any():
            write_points(out / f'{stem}_{name}_loops.ply', loop_points[keep],
                         cols[keep], loop_normals[keep])
        else:
            print(f'  no {name} loops: {out / f"{stem}_{name}_loops.ply"} not written')
    print(f'  {len(preds)} fitted loops, error {err.min():.3f}-{err.max():.3f} '
          f'({np.count_nonzero(bad_loops)} bad loops all red, the rest coloured '
          f'by plane axis x/y/z; predicted normals in nx/ny/nz)')
    print(f'  good and bad loops saved separately, where either holds any '
          f'-> {out / f"{stem}_good_loops.ply"}, {out / f"{stem}_bad_loops.ply"}')

    save_fit_grid(out / f'{stem}_fits.png', loops, preds, p_normal, p_point,
                  limit, large, pred_normals, normal_arrows, align=align_err)
    # one figure per plane as well
    save_plane_files(out, pc, pcn, found.slices, found.centres, loops, preds,
                     p_normal, p_point, limit, large, pred_normals, normal_arrows)
    # after refinement, rewrite the extraction figures for the final planes
    if len(plan) > 1:
        print(f'  extraction artefacts re-written for the {plan[-1][0]} planes per '
              f'axis the run ended on')
        found.resave()
    return preds


def check_fit_args(args):
    """Validate the fit options before extraction starts; parses --save-states."""
    if args.warmup is not None and not os.path.isfile(args.warmup):
        raise SystemExit(f'--warmup: no checkpoint at {args.warmup}. Train one with '
                         f'`python main.py warmup`, or pass --no-warmup to fit from '
                         f'a blank network')
    if args.snapshot_every < 0:
        raise SystemExit(f'--snapshot-every must be 0 or a count of epochs, got '
                         f'{args.snapshot_every}')
    try:
        args.save_states = [int(e) for e in args.save_states.split(',') if e.strip()]
    except ValueError:
        raise SystemExit(f'--save-states takes epochs separated by commas, got '
                         f'{args.save_states!r}') from None
    if any(e < 1 or e > args.epochs for e in args.save_states):
        raise SystemExit(f'--save-states epochs must lie in 1..{args.epochs}, got '
                         f'{args.save_states}')
    if args.refine_every < 0:
        raise SystemExit(f'--refine-every must be 0 or a count of epochs, got '
                         f'{args.refine_every}')
    if args.refine_every and args.add_planes < 1:
        raise SystemExit(f'--add-planes must be at least 1 to refine the planes, got '
                         f'{args.add_planes}')
    if args.samples_end is not None and args.samples_end < args.samples:
        raise SystemExit(f'--samples-end must not be under --samples: '
                         f'{args.samples_end} < {args.samples}')


def main(argv=None, train=True):
    """`main.py extract` (train=False) cuts the loops; `main.py fit` also fits them."""
    args = build_parser(train).parse_args(argv)
    arrows = None if args.arrows == 'all' else int(args.arrows)
    if train:
        check_fit_args(args)

    found = LoopExtractor(args.path, samples=args.samples, margin=args.margin,
                          band=args.band, max_align=args.max_align,
                          out=args.out)
    found.extract(figures=not args.no_image, n_arrows=arrows,
                  field=args.field, assign=args.assign)

    if train:
        normal_arrows = (None if args.normal_arrows == 'all'
                         else int(args.normal_arrows))
        fit_loops(found, epochs=args.epochs, neurons=args.neurons,
                  layers=args.layers,
                  align=args.align, max_err=args.max_err,
                  large=args.large, normal_arrows=normal_arrows,
                  warmup=args.warmup,
                  snapshot_every=args.snapshot_every, save_states=args.save_states,
                  refine_every=args.refine_every, add_planes=args.add_planes,
                  samples_end=args.samples_end, code_init=args.code_init,
                  code_smooth=args.code_smooth, code_gate=args.code_gate,
                  code_min_radius=args.code_min_radius, feature=args.feature,
                  triplane_res=args.triplane_res,
                  triplane_channels=args.triplane_channels,
                  triplane_tv=args.triplane_tv,
                  axis_condition=args.axis_condition,
                  freeze_f_epochs=args.freeze_f_epochs,
                  distill=args.distill, distill_warmup=args.distill_warmup,
                  seam_align=args.seam_align, seam_every=args.seam_every,
                  seam_start=args.seam_start, seam_steps=args.seam_steps,
                  seam_lam=args.seam_lam)
    return found
