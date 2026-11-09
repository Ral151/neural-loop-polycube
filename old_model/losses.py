"""Loss terms and curve measurements shared by the warm-up and the fit."""

import numpy as np
import torch
import torch.nn as nn


class CD_Loss(nn.Module):
    """Two-sided Chamfer distance between predicted and target point sets."""

    def __init__(self):
        super(CD_Loss, self).__init__()

    def forward(self, z, target):
        """Return (loss, near) for z: B x N x 3 and target: B x M x 3.

        near (B x N) is the target index each predicted point was matched to, so
        per-point terms can reuse the same pairing.
        """
        # nearest neighbours from detached copies; only the indices are needed
        dm = torch.cdist(z.detach(), target.detach())
        near = torch.argmin(dm, dim=2)
        near_target = target.gather(1, near[..., None].expand(-1, -1, 3))
        near_z = z.gather(1, torch.argmin(dm, dim=1)[..., None].expand(-1, -1, 3))
        loss_1 = ((z - near_target) ** 2).sum(-1).mean()
        loss_2 = ((target - near_z) ** 2).sum(-1).mean()
        return loss_1 + loss_2, near


def curve_tangent(out, z):
    """Tangent d out/dz (B x N x 3) of predicted curves, kept in the graph.

    Each xyz component is differentiated separately; summing them first would
    lose the direction.
    """
    t = []
    for k in range(out.shape[-1]):
        t.append(torch.autograd.grad(out[..., k].sum(), z, create_graph=True)[0])
    return torch.cat(t, -1)


def curve_curvature(d1, z):
    """Second derivative d2 out/dz2 (B x N x 3) from a tangent still in the graph."""
    d2 = []
    for k in range(d1.shape[-1]):
        d2.append(torch.autograd.grad(d1[..., k].sum(), z, create_graph=True)[0])
    return torch.cat(d2, -1)


def bend_penalty(d1, d2, eps=1e-12):
    """Scale-free bending: mean |d2|^2 over mean |d1|^2 per curve, batch-averaged.

    A circle reads 4*pi^2 ~ 39.5; wiggly curves read higher. Only reported, not
    optimised.
    """
    speed = (d1 ** 2).sum(-1).mean(1)
    bend = (d2 ** 2).sum(-1).mean(1)
    # detach the speed so the value cannot be lowered by shrinking the curve
    return (bend / (speed.detach() + eps)).mean()


def normal_cosine(d1, n_target, axis, eps=1e-12):
    """Per-point cosine (B x N) between the curve's implied normal and the given one.

    Implied normal = tangent x plane normal; the given normal is first projected
    into the slice plane, since the implied one always lies in it.
    """
    a = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    t = d1 / (d1.norm(dim=-1, keepdim=True) + eps)
    n_pred = torch.cross(t, a[:, None, :].expand_as(t), dim=-1)
    n_pred = n_pred / (n_pred.norm(dim=-1, keepdim=True) + eps)
    n_given = n_target - (n_target * a[:, None, :]).sum(-1, keepdim=True) * a[:, None, :]
    n_given = n_given / (n_given.norm(dim=-1, keepdim=True) + eps)
    return (n_pred * n_given).sum(-1)


def normal_penalty(d1, n_target, axis, eps=1e-12, per_curve=False):
    """Mean 1 - cos(implied normal, given normal): 0 aligned, 2 reversed.

    The sign matters, so it also fixes which way round the loop runs.
    d1: B x N x 3 tangent; n_target: B x N x 3 normals of the Chamfer-paired
    targets; axis: B x 3 plane normals. per_curve returns one value per curve.
    """
    cos = normal_cosine(d1, n_target, axis, eps)
    return (1.0 - cos).mean(-1) if per_curve else (1.0 - cos).mean()


def curve_winding(points, axis, centre=None, eps=1e-12):
    """Winding of closed curves about `centre` (default their mean): (net, foldover).

    net is +1 for counter-clockwise seen from +axis, -1 clockwise. foldover sums
    the absolute turning: 1 for a simple loop, above 1 when the curve doubles
    back (or is very non-convex). points: B x N x 3 in loop order; axis: B x 3.
    """
    if centre is None:
        centre = points.mean(1)
    axis = axis.to(points.dtype)
    centre = centre.to(points.dtype)
    a = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    world = torch.eye(3, device=points.device, dtype=points.dtype)
    # in-plane frame: u = world axis least aligned with the normal (ties -> lower
    # index, as in `in_plane_u`), v = axis x u
    pick = (a.abs() <= a.abs().min(dim=-1, keepdim=True).values + 1e-9)
    u = world[pick.float().argmax(-1)]
    u = u - (u * a).sum(-1, keepdim=True) * a
    u = u / (u.norm(dim=-1, keepdim=True) + eps)
    v = torch.cross(a, u, dim=-1)
    rel = points - centre[:, None, :]
    theta = torch.atan2((rel * v[:, None, :]).sum(-1),
                        (rel * u[:, None, :]).sum(-1))
    two_pi = 2.0 * np.pi
    # angle steps wrapped into (-pi, pi], including the closing step last -> first
    turns = theta[:, 1:] - theta[:, :-1]
    turns = torch.remainder(turns + np.pi, two_pi) - np.pi
    seam = torch.remainder(theta[:, :1] - theta[:, -1:] + np.pi, two_pi) - np.pi
    turns = torch.cat([turns, seam], dim=1)
    return turns.sum(1) / two_pi, turns.abs().sum(1) / two_pi


def cosine_anneal(t, total, eps=0):
    """Weight at step t of `total`: 1 at the first step, `eps` at the last, half-cosine."""
    return eps + (1.0 - eps) * 0.5 * (1.0 + np.cos(np.pi * t / max(total - 1, 1)))
