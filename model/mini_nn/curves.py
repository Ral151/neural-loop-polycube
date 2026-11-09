import numpy as np
import torch


## the tangent of a predicted curve, component by component, with respect to z
def curve_tangent(out, z):
    """
    d out/dz, B x N x 3.

    z is one scalar per point, so the derivative there is a 3-vector and one backward
    pass cannot hold it: differentiating the sum of all three components returns the
    rate at which they rise together, which is a number with no direction in it and
    reads zero on a curve that turns without climbing. Each component is therefore
    summed and differentiated on its own.

    create_graph keeps the tangent inside the graph. Without it the tangent is a
    detached number and the second derivative cannot be taken from it, so the curve
    could be pulled but not straightened.

    out: B x N x 3 predictions
    z: B x N x 1 parameter coords, a leaf of the graph that produced out
    """
    t = []
    for k in range(out.shape[-1]):
        ## one component summed leaves a scalar per point, and the gradient of that
        ## w.r.t. the single input value at the point is that component's own rate
        t.append(torch.autograd.grad(out[..., k].sum(), z, create_graph=True)[0])
    return torch.cat(t, -1)


## the rate the tangent of a predicted curve changes, with respect to z
def curve_curvature(d1, z):
    """
    d2 out/dz2, B x N x 3, taken from a tangent that is still inside the graph.

    Separate from curve_tangent so a loss that only needs the direction of the curve
    does not pay for a second set of backward passes.

    d1: B x N x 3 tangent, as curve_tangent returns it
    z: B x N x 1 parameter coords, a leaf of the graph that produced out
    """
    d2 = []
    for k in range(d1.shape[-1]):
        d2.append(torch.autograd.grad(d1[..., k].sum(), z, create_graph=True)[0])
    return torch.cat(d2, -1)


## how much a predicted curve bends, relative to how fast it sweeps round
def bend_penalty(d1, d2, eps=1e-12):
    """
    Mean squared second derivative of each curve over its own mean squared speed,
    averaged over the batch.

    Both derivatives scale with the curve, so their ratio does not: a circle reads
    4*pi^2 = 39.5 whatever its radius and whatever number of points describes it, and
    a curve that zig-zags or doubles back over itself reads far above that. The
    number is therefore comparable across shapes and loop sizes. It is a reading,
    not a loss term: the fit does not optimise it (the run prints it at the end).

    The derivative is taken with respect to the parameter, not the arc length, so
    the same curve parameterised unevenly reads higher too -- which is what a wiggly
    fit usually is: points bunch up where the curve turns, and a fit that chases
    them has to speed up and slow down.

    d1, d2: B x N x 3, as curve_tangent and curve_curvature return them
    """
    speed = (d1 ** 2).sum(-1).mean(1)
    bend = (d2 ** 2).sum(-1).mean(1)
    ## the speed is the scale the bend is read against, not something to lower:
    ## differentiating through it would let the penalty be paid by shrinking the curve
    ## instead of straightening it, and a shrinking loop is the one way the fit loses
    ## the shape altogether
    return (bend / (speed.detach() + eps)).mean()


## the unit normal a fitted curve implies, B x N x 3
def curve_normal(d1, axis, eps=1e-12):
    """
    The normal of a fitted curve, tangent x plane normal.

    A contour lies in the surface, so its tangent is square to both the surface
    normal and the slice's plane normal, which leaves one in-plane direction a
    quarter turn from the tangent. That is the normal the figures draw and the
    pipeline exports. The network predicts points, not normals, so this is how
    the predicted normal is read off the curve -- by both the direction term and
    the chamfer's pairing.

    d1: B x N x 3 tangent, as curve_tangent returns it
    axis: B x 3 plane normals, one per curve
    """
    a = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    t = d1 / (d1.norm(dim=-1, keepdim=True) + eps)
    n_pred = torch.cross(t, a[:, None, :].expand_as(t), dim=-1)
    return n_pred / (n_pred.norm(dim=-1, keepdim=True) + eps)


## the part of a normal that lies in the slice plane, B x ... x 3, with its length
def in_plane_normal(n, axis, eps=1e-12):
    """
    Project normals into the slice plane, unit, and return how much of each one
    was left to project.

    The predicted curve normal lies in the plane by construction, and a cloud
    normal leans out of it -- on laurent_hand only 0.69 of its length is in-plane
    -- so the leaning part is not something a curve can ever match. The returned
    length is that in-plane fraction, which says how much of the given normal is
    a direction in the loop rather than a tilt out of it; near zero the in-plane
    direction is noise and should not decide anything.

    n: B x ... x 3 normals
    axis: B x 3 plane normals
    """
    a = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    v = n - (n * a[:, None, :]).sum(-1, keepdim=True) * a[:, None, :]
    size = v.norm(dim=-1, keepdim=True)
    return v / (size + eps), size.squeeze(-1)


## the target normals as the pairing reads them: unit, and in the slice plane
def project_normals(n, axis, min_inplane=0.2, eps=1e-12):
    """Carry target normals into the slice plane, unit, before the fit uses them.

    A predicted curve normal lies in the slice plane by construction, while a
    cloud normal leans out of it -- on laurent_hand only 0.69 of its length is
    in-plane -- and the leaning part is not something a curve can ever match.
    Projecting once, here, leaves both the chamfer's pairing and the direction
    term reading the same in-plane vector.

    A normal with less than `min_inplane` of its length left in the plane has no
    reliable direction there; its projected remainder is noise, so it is zeroed.
    A zero normal reads cos 0 against everything, which lets it pass the pairing
    test without deciding any pair.

    n: B x ... x 3 normals
    axis: B x 3 plane normals
    min_inplane: smallest in-plane fraction that still counts as a direction
    """
    unit, inplane = in_plane_normal(n, axis, eps)
    if min_inplane <= 0.0:
        return unit
    return torch.where((inplane < min_inplane)[..., None], torch.zeros_like(unit),
                       unit)


## how far a fitted curve's implied normal sits from the one the cloud gives
def normal_cosine(d1, n_target, axis, eps=1e-12):
    """
    Cos(predicted normal, given normal) at every point, B x N.

    The same construction as normal_penalty, kept separate because the per-loop
    direction is read off it: the orientation gauge and the report both need one
    cosine per loop rather than one mean over the batch.
    """
    n_pred = curve_normal(d1, axis, eps)
    n_given, _ = in_plane_normal(n_target, axis, eps)
    return (n_pred * n_given).sum(-1)


def normal_penalty(d1, n_target, axis, eps=1e-12, per_curve=False, mask=None):
    """
    Mean 1 - cos(predicted normal, given normal) over the paired points, averaged over
    the batch -- or over the points of each curve with per_curve, as a B vector --
    where the predicted normal is the one the fit is drawn with:

        predicted normal = tangent x plane normal

    That is the loss written as its definition rather than as an equivalent turn. The
    network predicts points, not normals, so the normal of a fitted curve is read off
    it: a contour lies in the surface, so its tangent is square to both the surface
    normal and the slice's plane normal, which leaves one in-plane direction a
    quarter turn from the tangent -- tangent x plane normal, the same turn the figures
    draw and the pipeline exports.

    Both normals are compared in the loop's own plane, because that is where the
    predicted one lies: the given normal is projected into the plane before the angle
    is taken, so what is measured is the direction of the surface normal round the
    loop rather than how far the cloud's normal leans out of the slice.

    The term is signed: 0 when the curve implies the normals it was given, 1 when its
    normal sits square to them (the curve cutting straight into the surface), and 2
    when it runs the other way round, which reads as the normal pointing inward.
    Squaring the dot instead would throw the sense away and leave the curve free to run
    either way, which is exactly the wiggling and doubling back the term is there to
    stop.

    d1: B x N x 3 tangent, as curve_tangent returns it
    n_target: B x N x 3 outward normals of the target points, one per point, as
              chamfer paired them to the predicted points
    axis: B x 3 plane normals, one per curve; the loop's own plane, read off its
          points, is what picks tangent x axis out of the circle of directions square
          to the tangent
    per_curve: average the points of each curve separately and return B, instead of
          one number over the batch
    mask: B x N bool, the points whose pairing the chamfer accepted. Points outside
          it were left unmatched, so their gathered normal is an arbitrary nearest
          one and must not be charged to the curve; None counts every point
    """
    ## the cosine of the two normals, so the penalty is 0 only when the curve implies
    ## exactly the normal it was handed
    cos = normal_cosine(d1, n_target, axis, eps)
    if mask is None:
        return (1.0 - cos).mean(-1) if per_curve else (1.0 - cos).mean()
    w = mask.to(cos.dtype)
    if per_curve:
        return ((1.0 - cos) * w).sum(-1) / w.sum(-1).clamp_min(1.0)
    ## one mean over the matched points of the batch, so a curve with fewer matches
    ## does not count for less than one with more
    return ((1.0 - cos) * w).sum() / w.sum().clamp_min(1.0)


## how many times a closed curve turns about a point, and how much turning it spends
def curve_winding(points, axis, centre=None, eps=1e-12):
    """
    Net turn of each closed curve about `centre`, in turns, and the total turn it
    spends getting there. B each.

    The angle of each point is read in the plane's own frame -- u the world axis
    least aligned with the plane normal, v = axis x u -- and the increments are
    wrapped into (-pi, pi] before they are summed, including the seam from the last
    point back to the first. A simple loop reads +1 counter-clockwise seen from
    +axis, -1 clockwise, and a curve that runs out and back reads |1| with the
    out-and-back cancelling. That is the sense a per-point cosine cannot report: on
    a retrace the forward and backward stretches pair to the same target points and
    average to zero, while the net winding still says which way the loop closes.

    `foldover` is the same sum with the increments taken in absolute value. A loop
    that turns once reads 1, a retrace adds the angle it walks back, so a curve that
    covers its own path reads above 1. It is a reading, not a gauge: a very
    non-convex loop can also read above 1.

    points: B x N x 3, the curve in closed order
    axis: B x 3 plane normals; the basis is chosen the same way `in_plane_u` chooses
          the warm-up square's x, so the cube and a fit agree on the sense
    centre: B x 3 reference point; default the curve's own mean
    """
    if centre is None:
        centre = points.mean(1)
    axis = axis.to(points.dtype)
    centre = centre.to(points.dtype)
    a = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    world = torch.eye(3, device=points.device, dtype=points.dtype)
    ## ties broken by the lower index, as in_plane_u does: a plane normal exactly on
    ## a world axis leaves two axes equally least aligned, and an arbitrary pick would
    ## let a floating point wobble decide the frame
    pick = (a.abs() <= a.abs().min(dim=-1, keepdim=True).values + 1e-9)
    u = world[pick.float().argmax(-1)]
    u = u - (u * a).sum(-1, keepdim=True) * a
    u = u / (u.norm(dim=-1, keepdim=True) + eps)
    v = torch.cross(a, u, dim=-1)
    rel = points - centre[:, None, :]
    theta = torch.atan2((rel * v[:, None, :]).sum(-1),
                        (rel * u[:, None, :]).sum(-1))
    two_pi = 2.0 * np.pi
    turns = theta[:, 1:] - theta[:, :-1]
    turns = torch.remainder(turns + np.pi, two_pi) - np.pi
    seam = torch.remainder(theta[:, :1] - theta[:, -1:] + np.pi, two_pi) - np.pi
    turns = torch.cat([turns, seam], dim=1)
    return turns.sum(1) / two_pi, turns.abs().sum(1) / two_pi


def winding_number(points, axis, centre=None):
    """`curve_winding` for one curve in numpy, returning (winding, foldover)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    u = in_plane_u(a)
    u = u - (u @ a) * a
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    c = pts.mean(0) if centre is None else np.asarray(centre, dtype=np.float64)
    rel = pts - c
    theta = np.arctan2(rel @ v, rel @ u)
    two_pi = 2.0 * np.pi
    turns = (np.diff(theta) + np.pi) % two_pi - np.pi
    seam = (theta[0] - theta[-1] + np.pi) % two_pi - np.pi
    turns = np.append(turns, seam)
    return turns.sum() / two_pi, np.abs(turns).sum() / two_pi


## The universal cube the warm-up target is cut from: side 1, centred at the origin,
## the same for every shape and every loop.  It is a definition, not a measurement --
## nothing reads a cloud, or a box around one, to size it.  A shape larger than the
## cube is warmed on the cube anyway; the fit is what puts the curve on the shape.
UNIT_CUBE_HALF = 0.5


def in_plane_u(axis):
    """The square's x for a plane of the given normal: the world axis least aligned
    with it, ties broken by the lower index.

    A normal that lies exactly on a world axis is why this exists: two world axes
    are then equally least-aligned, and choosing by the smaller floating point
    component would let a 1e-4 wobble in the normal decide the frame. The lower
    index is a convention, so every loop cut by the same plane gets the same
    square and a phase is read against one origin.

    axis: unit normal of the plane the square lies in
    """
    mag = np.abs(np.asarray(axis, dtype=np.float64).reshape(3))
    return np.eye(3)[int(np.flatnonzero(mag <= mag.min() + 1e-9)[0])]
