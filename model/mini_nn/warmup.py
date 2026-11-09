import numpy as np
import torch

from .cubes import cube_loops, random_cube_loops
from .curves import UNIT_CUBE_HALF, curve_tangent, normal_penalty
from .triplane import cell_pool_structure


def warm_on_cube(model, steps=300, lr=3e-2, align=1.0, seam=1.0, planes=4,
                 samples=512, feature_pool='centroid', cubes=0,
                 cube_half=(0.2, 0.45), resample_every=1, seed=0,
                 on_step=None, on_step_every=1):
    """
    Train `model` on cubes and nothing else.

    A single cube's loops are `cube_loops`: every cross-section is a square, in its
    own plane's frame, walked counter-clockwise seen from the plane normal, so the
    target is exact and needs no correspondence search. With `cubes` > 0 the batch
    is `random_cube_loops`: cubes of random size, placed at random inside the grid's
    domain and redrawn every `resample_every` steps, so the field is trained across
    its interior at many scales rather than only where one fixed cube's loops fall,
    and the map sees a range of feature codes rather than twelve. Each step takes
    three terms:

    - fit: |f(z) - p(z)| against the cube's cross-sections, so the curves lie on the
      cube;
    - direction: the normal the curve implies, tangent x plane normal, against the
      cube's outward face normal, so every loop along an axis circulates the same
      way;
    - seam: the seam distance D between loops adjacent along an axis -- their mean
      squared separation over their own mean squared radius, both centred -- taken
      with the phase at zero and differentiated through the network. The point is
      that the network itself puts z = 0 where its neighbours put it, so the
      placement is carried by the weights rather than by a phase buffer.

    The triplane frame is set to the unit cube for the run and put back afterwards.
    The grids are a function of unit-cube position, so a field warmed here reads the
    same way once the frame is moved onto a shape's box; the frame belongs to the
    caller, not to the warm-up.

    steps: optimisation steps, every loop in every step
    lr: starting rate; a MultiStepLR takes it down by a decade at half the run and
            again at three quarters, so the last quarter trains at 1% of `lr`
    align: weight of the direction term, 0 to train position and seam only
    seam: weight of the seam term, 0 to leave z = 0 unconstrained
    planes: even planes per axis on the cube, which sets the neighbouring pairs;
            none cuts the centre (see `cube_loops`)
    samples: points round every loop, and so the parameter grid
    feature_pool: 'centroid' reads the field at each cube loop's centre, as every
            earlier warm-up did. 'occupied_cells' pools the grid nodes the loop's
            own points occupy, which is the condition a pooled shape fit reads, so
            the map learns from the same statistic it will be handed later
    cubes: cubes in every batch. 0 is the single unit cube at the origin, the
            fixed warm-up earlier checkpoints came from, and is this function's
            default; `warmup_cube.py` defaults to 8. N > 0 draws N cubes
            with a half-side from `cube_half` and a centre uniform inside the
            domain, so the loops land in the grid's interior and every batch sees
            new codes
    cube_half: (min, max) half-side of a random cube in the unit-cube units the
            triplane covers; both below UNIT_CUBE_HALF so the whole cube fits
            anywhere in the domain. This function defaults to (0.2, 0.45);
            `warmup_cube.py` defaults to (0.05, 0.45)
    resample_every: steps between fresh random batches; 1 redraws every step
    seed: seed for the random cube draw, so a run is reproducible
    on_step, on_step_every: optional callback(step, out, readings) after every
            `on_step_every` steps, where `out` is the K x samples x 3 prediction the
            step ended on and `readings` holds `fit`, the per-loop direction `cos`
            (None without the direction term), the median seam `gap` (None without
            the seam term) and the batch `cube` the prediction was drawn on.
            Drawing and file writing live in the caller: training takes no plotting
            dependency, and a caller that wants no grids passes nothing and pays
            nothing

    Returns a dict with the three reported readings: `fit` and `fit_max` (distance
    to the cube's cross-sections, same units as the fit), `cos` (mean direction
    cosine against the outward normals) and `seam_median` / `seam_max` (D between
    neighbouring loops at z = 0).
    """
    if getattr(model, 'feature_mode', None) != 'triplane':
        raise ValueError('the cube warm-up conditions the network on the triplane '
                         'field; embedding rows name loop planes, not cube ones')
    if steps < 1:
        raise ValueError(f'steps must be at least 1, got {steps}')
    if lr <= 0:
        raise ValueError(f'lr must be positive, got {lr}')
    for name, weight in (('align', align), ('seam', seam)):
        if weight < 0:
            raise ValueError(f'{name} must be non-negative, got {weight}')
    if planes < 2 or planes % 2:
        raise ValueError(f'the cube needs an even number of planes per axis, none '
                         f'at the centre, got {planes}')
    if samples < 4:
        raise ValueError(f'every loop needs at least four samples, got {samples}')
    if feature_pool not in ('centroid', 'occupied_cells'):
        raise ValueError(f"feature_pool must be 'centroid' or 'occupied_cells', "
                         f"got {feature_pool!r}")
    if on_step_every < 1:
        raise ValueError(f'on_step_every must be at least 1, got {on_step_every}')
    if cubes < 0:
        raise ValueError(f'cubes must be 0 or a count, got {cubes}')
    if not 0.0 < cube_half[0] <= cube_half[1] < UNIT_CUBE_HALF:
        raise ValueError(f'cube_half must sit inside (0, {UNIT_CUBE_HALF}), got '
                         f'{cube_half}')
    if resample_every < 1:
        raise ValueError(f'resample_every must be at least 1, got {resample_every}')

    device = next(model.parameters()).device
    triplane = model.triplane
    rng = np.random.default_rng(seed) if cubes else None

    def draw():
        """The batch of loops for the next stretch of steps."""
        if not cubes:
            return cube_loops(planes, samples)
        return random_cube_loops(planes, samples, count=cubes,
                                 half_range=cube_half, rng=rng)

    def tensors(cube):
        """A drawn batch on the device, with the readings that depend on it.

        The seam denominator is the pair's size on the cube, not the predicted
        pair's own: the predicted size is the fit's business, and letting the seam
        term see it in the gradient gives it a cheaper way down than placing the
        seam -- grow one curve towards the other. Against a fixed target scale, D
        falls only when the two curves are read at the same place, which is the
        property being trained.
        """
        targets = torch.from_numpy(cube['targets']).float().to(device)
        normals = torch.from_numpy(cube['normals']).float().to(device)
        axes = torch.from_numpy(cube['axes']).float().to(device)
        queries = torch.from_numpy(cube['queries']).float().to(device)
        pairs = torch.as_tensor(cube['pairs'], device=device)
        plane = targets - targets.mean(1, keepdim=True)
        pair_scale = 0.5 * ((plane[pairs[:, 0]] ** 2).sum(-1).mean(1)
                            + (plane[pairs[:, 1]] ** 2).sum(-1).mean(1))
        cells = masks = None
        if feature_pool == 'occupied_cells':
            ## the loop's own points in world coordinates: the target is plane-local,
            ## so the plane's centre is added back, the same split the real fit uses
            world = cube['targets'] + cube['queries'][:, None, :]
            cells, masks = cell_pool_structure(triplane, world)
        return dict(targets=targets, normals=normals, axes=axes, queries=queries,
                    pairs=pairs, pair_scale=pair_scale, cells=cells, masks=masks,
                    cube=cube)

    cube = draw()
    batch = tensors(cube)
    count = batch['targets'].shape[0]

    def seam_gap(out, pairs, pair_scale):
        """D between the paired loops, each centred on its own points."""
        v = out - out.mean(1, keepdim=True)
        vi, vj = v[pairs[:, 0]], v[pairs[:, 1]]
        return ((vi - vj) ** 2).sum(-1).mean(1) / pair_scale.clamp_min(1e-12)

    saved_centre = triplane.box_centre.detach().clone()
    saved_half = triplane.box_half.detach().clone()
    coords = torch.linspace(0, 1, samples + 1, device=device)[:-1, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ## a decade off at half and at three quarters of the run, so the fit settles
    ## before each cut and the last quarter holds a floor of 1% of `lr`. Constructing
    ## the scheduler steps it once, so no rate moves before step 1 runs, and milestone
    ## m lowers the rate that step m + 1 trains with
    milestones = sorted({max(2, steps // 2), max(2, 3 * steps // 4)})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones,
                                                     gamma=0.1)
    every = max(1, steps // 12)

    def features(queries, cells, masks):
        if feature_pool == 'occupied_cells':
            return triplane.pool_cells(cells, masks)
        return triplane(queries)

    try:
        with torch.no_grad():
            triplane.box_centre.zero_()
            triplane.box_half.fill_(UNIT_CUBE_HALF)
        z = coords[None].expand(count, -1, -1).clone().requires_grad_(True)

        for step in range(1, steps + 1):
            ## a fresh size and place for every stretch of random steps; the count is
            ## fixed, so the parameter leaf z is drawn once above
            if cubes and step > 1 and (step - 1) % resample_every == 0:
                cube = draw()
                batch = tensors(cube)
            targets, normals, axes = (batch['targets'], batch['normals'],
                                      batch['axes'])
            queries, pairs = batch['queries'], batch['pairs']
            pair_scale = batch['pair_scale']
            cells, masks = batch['cells'], batch['masks']
            optimizer.zero_grad()
            ## the rate this step's update will use, read before the scheduler's
            ## step at the bottom moves it
            rate = optimizer.param_groups[0]['lr']
            ## read again every step: the field's sample is a graph input and the
            ## step before freed it
            feat = features(queries, cells, masks)
            out = model(z, None, feat)[0]
            fit = (out - targets).norm(dim=-1).mean()
            loss = fit
            direction = None
            if align:
                direction = normal_penalty(curve_tangent(out, z), normals, axes)
                loss = loss + align * direction
            gap = None
            if seam:
                gap = seam_gap(out, pairs, pair_scale)
                loss = loss + seam * gap.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ## the state this step ends on, when asked for: reading it again shows the
            ## weights the update just wrote, so a run of grids shows the curves
            ## arriving rather than only where they stopped. The direction reading
            ## costs one derivative, the same one the loss takes
            if on_step is not None and step % on_step_every == 0:
                snap = model(z, None, features(queries, cells, masks))[0]
                readings = dict(
                    fit=float((snap - targets).norm(dim=-1).mean()),
                    cos=(None if not align else
                         (1.0 - normal_penalty(curve_tangent(snap, z), normals,
                                               axes, per_curve=True)).detach()),
                    gap=None if not seam else float(
                        seam_gap(snap, pairs, pair_scale).median()),
                    cube=cube)
                on_step(step, snap.detach(), readings)
            if step % every == 0 or step == steps:
                line = f'  cube {step}/{steps}: fit {fit.item():.4f}'
                if direction is not None:
                    line += f', direction {1.0 - direction.item():+.3f} cos'
                if gap is not None:
                    line += f', seam D {float(gap.median()):.4f}'
                line += f', lr {rate:.2e}'
                print(line)

        ## the three readings the task asks for, on the state the run ended on
        targets, normals, axes = (batch['targets'], batch['normals'], batch['axes'])
        queries, pairs = batch['queries'], batch['pairs']
        pair_scale, cells, masks = batch['pair_scale'], batch['cells'], batch['masks']
        feat = features(queries, cells, masks)
        zz = coords[None].expand(count, -1, -1).clone().requires_grad_(True)
        out = model(zz, None, feat)[0]
        with torch.no_grad():
            per = (out - targets).norm(dim=-1).mean(1)
            gap = seam_gap(out, pairs, pair_scale)
        cos = 1.0 - float(normal_penalty(curve_tangent(out, zz), normals, axes))
        report = dict(steps=steps, planes=planes, samples=samples, loops=count,
                      feature_pool=feature_pool, cubes=cubes,
                      cube_half=list(cube_half),
                      resample_every=resample_every if cubes else None,
                      fit=float(per.mean()), fit_max=float(per.max()), cos=cos,
                      seam_median=float(gap.median()), seam_max=float(gap.max()))
        layout = (f'{count} loops over {cubes} random cubes ({planes} per axis)'
                  if cubes else f'{count} loops ({planes} per axis)')
        redraw = ('' if not cubes else
                  f', redrawn every {resample_every} step'
                  f'{"s" if resample_every != 1 else ""}')
        print(f'  cube warm-up {steps} steps, {layout}{redraw}: fit '
              f'{report["fit"]:.4f} mean, {report["fit_max"]:.4f} max; direction '
              f'{cos:+.3f} cos; seam D median {report["seam_median"]:.4f}, max '
              f'{report["seam_max"]:.4f}')
        return report
    finally:
        with torch.no_grad():
            triplane.box_centre.copy_(saved_centre)
            triplane.box_half.copy_(saved_half)


def save_warmup(path, model, feature_pool='centroid', **meta):
    """Write a cube-warmed network and field, so a shape's fit can load it as its
    initialisation.

    The checkpoint holds the shared MLP and the triplane grid, and the cube frame
    the grid was warmed in. Nothing shape-specific is written: there are no rows and
    no phases, because the warm-up had no shape to name.
    """
    triplane = getattr(model, 'triplane', None)
    if triplane is None:
        raise ValueError('a cube warm-up checkpoint needs a triplane model')
    payload = dict(
        kind='cube_warmup',
        feature_mode='triplane',
        num_neurons=int(model.model[0].out_features),
        num_layers=int(model.num_layers),
        pe=bool(model.pe),
        embedding_dim=int(model.embedding_dim),
        feature_pool=feature_pool,
        num_models=1,
        codes=np.zeros(1, dtype=np.int64),
        queries=None,
        triplane=dict(centre=triplane.box_centre.detach().cpu().numpy(),
                      half=triplane.box_half.detach().cpu().numpy(),
                      resolution=triplane.resolution, channels=triplane.channels,
                      tv=triplane.tv),
        phases=None,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
    )
    payload.update(meta)
    torch.save(payload, path)
    return path


def load_warmup(path, model):
    """Copy a cube warm-up checkpoint into `model`, keeping the model's own triplane
    frame.

    The frame says where the unit cube is laid on this shape's box, and the grid is a
    function of unit-cube position, so the frame belongs to the fit rather than to
    the warm-up: copying it would put the warmed cells where this shape never reads.
    The shapes are checked first, so a checkpoint warmed on a different network or
    grid is refused rather than silently loaded in part.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('kind') != 'cube_warmup':
        raise ValueError(f'{path} is not a cube warm-up checkpoint')
    if getattr(model, 'feature_mode', None) != 'triplane':
        raise ValueError('a cube warm-up transfers through the triplane field, and '
                         'this model conditions on embedding rows')
    source = ckpt['state_dict']
    own = model.state_dict()
    shared = {name: value for name, value in source.items()
              if not name.endswith(('box_centre', 'box_half'))}
    bad = [name for name, value in shared.items()
           if name not in own or own[name].shape != value.shape]
    if bad:
        shown = ', '.join(f'{name} {tuple(shared[name].shape)} vs '
                          f'{tuple(own[name].shape) if name in own else "missing"}'
                          for name in bad[:4])
        raise ValueError(f'{path} does not fit this network or grid: {shown}'
                         f'{" ..." if len(bad) > 4 else ""}. Warm up with the same '
                         f'neurons, layers, resolution and channels')
    model.load_state_dict(shared, strict=False)
    return ckpt
