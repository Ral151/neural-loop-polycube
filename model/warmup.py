"""Train the network on the unit cube, and save/load that warm-up checkpoint."""

import numpy as np
import torch

from model.cube import UNIT_CUBE_HALF, cube_loops
from model.losses import cosine_anneal, curve_tangent, normal_penalty


def warm_on_cube(model, steps=300, lr=3e-2, align=1.0, seam=1.0, planes=4,
                 samples=512):
    """Train `model` on the cube's square cross-sections (see `cube_loops`).

    Loss per step = fit (distance to the exact square)
                  + align * direction (implied normal vs the face normal)
                  + seam * D (z = 0 of neighbouring loops at the same place).
    The triplane box is set to the unit cube during the run and restored after.
    lr is cosine-annealed to 5%. Returns the final fit / cos / seam readings.
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

    device = next(model.parameters()).device
    cube = cube_loops(planes, samples)
    targets = torch.from_numpy(cube['targets']).float().to(device)
    normals = torch.from_numpy(cube['normals']).float().to(device)
    axes = torch.from_numpy(cube['axes']).float().to(device)
    queries = torch.from_numpy(cube['queries']).float().to(device)
    pairs = torch.as_tensor(cube['pairs'], device=device)
    count = targets.shape[0]

    # D is normalised by the target squares' size, not the predicted curves', so it
    # can only drop by aligning the seams, not by resizing the curves
    plane = targets - targets.mean(1, keepdim=True)
    pair_scale = 0.5 * ((plane[pairs[:, 0]] ** 2).sum(-1).mean(1)
                        + (plane[pairs[:, 1]] ** 2).sum(-1).mean(1))

    def seam_gap(out):
        """D between the paired loops, each centred on its own points."""
        v = out - out.mean(1, keepdim=True)
        vi, vj = v[pairs[:, 0]], v[pairs[:, 1]]
        return ((vi - vj) ** 2).sum(-1).mean(1) / pair_scale.clamp_min(1e-12)

    triplane = model.triplane
    saved_centre = triplane.box_centre.detach().clone()
    saved_half = triplane.box_half.detach().clone()
    coords = torch.linspace(0, 1, samples + 1, device=device)[:-1, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    every = max(1, steps // 12)
    try:
        with torch.no_grad():
            triplane.box_centre.zero_()
            triplane.box_half.fill_(UNIT_CUBE_HALF)
        z = coords[None].expand(count, -1, -1).clone().requires_grad_(True)
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            # resample each step: the previous backward freed the graph
            feat = triplane(queries)
            out = model(z, None, feat, axes)
            fit = (out - targets).norm(dim=-1).mean()
            loss = fit
            direction = None
            if align:
                direction = normal_penalty(curve_tangent(out, z), normals, axes)
                loss = loss + align * direction
            gap = None
            if seam:
                gap = seam_gap(out)
                loss = loss + seam * gap.mean()
            loss.backward()
            optimizer.step()
            for group in optimizer.param_groups:
                group['lr'] = lr * cosine_anneal(step - 1, steps, eps=0.05)
            if step % every == 0 or step == steps:
                line = f'  cube {step}/{steps}: fit {fit.item():.4f}'
                if direction is not None:
                    line += f', direction {1.0 - direction.item():+.3f} cos'
                if gap is not None:
                    line += f', seam D {float(gap.median()):.4f}'
                print(line)

        # final readings on the trained state
        feat = triplane(queries)
        zz = coords[None].expand(count, -1, -1).clone().requires_grad_(True)
        out = model(zz, None, feat, axes)
        with torch.no_grad():
            per = (out - targets).norm(dim=-1).mean(1)
            gap = seam_gap(out)
        cos = 1.0 - float(normal_penalty(curve_tangent(out, zz), normals, axes))
        report = dict(steps=steps, planes=planes, samples=samples, loops=count,
                      fit=float(per.mean()), fit_max=float(per.max()), cos=cos,
                      seam_median=float(gap.median()), seam_max=float(gap.max()))
        print(f'  cube warm-up {steps} steps, {count} loops ({planes} per axis): fit '
              f'{report["fit"]:.4f} mean, {report["fit_max"]:.4f} max; direction '
              f'{cos:+.3f} cos; seam D median {report["seam_median"]:.4f}, max '
              f'{report["seam_max"]:.4f}')
        return report
    finally:
        with torch.no_grad():
            triplane.box_centre.copy_(saved_centre)
            triplane.box_half.copy_(saved_half)


def save_warmup(path, model, **meta):
    """Save the warmed MLP and triplane grid (no shape-specific data) to `path`."""
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
        axis_condition=bool(getattr(model, 'axis_condition', False)),
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
    """Load a warm-up checkpoint into `model`, keeping the model's own triplane box.

    Refuses a checkpoint whose network or grid sizes do not match.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('kind') != 'cube_warmup':
        raise ValueError(f'{path} is not a cube warm-up checkpoint')
    if getattr(model, 'feature_mode', None) != 'triplane':
        raise ValueError('a cube warm-up transfers through the triplane field, and '
                         'this model conditions on embedding rows')
    if bool(ckpt.get('axis_condition', False)) != bool(model.axis_condition):
        raise ValueError(
            f'{path} was warmed {"with" if ckpt.get("axis_condition") else "without"} '
            f'axis conditioning and this network is '
            f'{"axis-conditioned" if model.axis_condition else "not"}; build the '
            f'model to match, or retrain the warm-up')
    source = ckpt['state_dict']
    own = model.state_dict()
    # everything except the box, which belongs to this shape's fit
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
