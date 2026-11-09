import math

import torch
import torch.nn as nn


class CD_Loss(nn.Module):
    def __init__(self, max_angle=150.0):
        super(CD_Loss, self).__init__()
        self.max_angle = float(max_angle)
    
    def forward(self, z, target, target_normals=None, pred_normals=None,
                max_angle=None, eps=1e-12):
        """
        Chamfer distance over the pairs the normal test accepted, and the pairing
        it used.

        z: B x N x 3
        target: B x M x 3
        target_normals: B x M x 3 outward normals of the target points, already
                      carried into the slice plane by the caller (see
                      `project_normals`), or None. This loss only compares the
                      directions it is handed; it does not know about a plane
        pred_normals: B x N x 3 normals implied by the predicted curve, as
                      curve_normal returns them, or None
        max_angle: degrees; a pair is allowed when the angle between the two
              normals is no greater than this. None takes the value the loss was
              built with. 180 turns the test off

        Pairing by distance alone will match a stretch of contour to a point on
        the far side of a thin feature, where the surface faces the other way
        and the outward normal points back at the curve. With both normal sets
        given, a pair is only allowed when their normals are at most
        `max_angle` apart, so such a point is passed over for one on the same
        side. A point with no allowed partner is left unmatched: it contributes
        nothing to either direction of the chamfer on this step, rather than
        being pulled to a point whose normal says it belongs on the other side.

        Returns (loss, near, match_meta). near is B x N: the target point
        each predicted point was matched to. Any per-point term has to use that same
        pairing, or the two halves of the loss pull at different correspondences --
        and asking for it here costs nothing, since the distance matrix is already
        there. near is only meaningful where match_meta['near_ok'] is true;
        match_meta also gives the reverse mask and the matched and rerouted fractions
        of each direction. match_meta is None when no normals were given.
        """
        ## save the gpu memory: only the nearest-neighbour indices come back
        dm = torch.cdist(z.detach(), target.detach())
        near_plain = torch.argmin(dm, dim=2)
        back_plain = torch.argmin(dm, dim=1)
        match_meta = None
        if target_normals is not None and pred_normals is not None:
            angle = self.max_angle if max_angle is None else float(max_angle)
            if not 0.0 < angle <= 180.0:
                raise ValueError(f'max_angle must be in (0, 180], got {angle}')
            ## the caller has already carried the target normals into the slice
            ## plane; normalise what it handed over and compare directions
            n_given = target_normals / (target_normals.norm(dim=-1, keepdim=True)
                                        + eps)
            n_pred = pred_normals / (pred_normals.norm(dim=-1, keepdim=True) + eps)
            cos = torch.einsum('bnc,bmc->bnm', n_pred, n_given)
            if angle >= 180.0:
                ## the test is off; every pair is by distance alone
                ok = torch.ones_like(cos, dtype=torch.bool)
            else:
                ## no greater than max_angle, written on the cosine: cos is
                ## decreasing, so the allowed side is cos >= cos(max_angle). A target
                ## normal the caller zeroed for lack of an in-plane direction reads
                ## cos 0 and so is allowed here
                ok = cos >= math.cos(math.radians(angle))
            near_ok = ok.any(-1)                       # B x N, predicted points matched
            back_ok = ok.any(1)                        # B x M, target points matched
            ## a point with no allowed partner keeps the plain nearest as its index,
            ## but its mask is false and the loss below leaves it out
            near = torch.where(near_ok, torch.argmin(
                dm.masked_fill(~ok, float('inf')), dim=2), near_plain)
            back = torch.where(back_ok, torch.argmin(
                dm.masked_fill(~ok, float('inf')), dim=1), back_plain)
            match_meta = dict(near_ok=near_ok, back_ok=back_ok,
                              matched=float(near_ok.float().mean()),
                              matched_back=float(back_ok.float().mean()),
                              rerouted=float(((near != near_plain) & near_ok)
                                             .float().mean()),
                              rerouted_back=float(((back != back_plain) & back_ok)
                                                  .float().mean()))
        else:
            near, back = near_plain, back_plain
        near_target = target.gather(1, near[..., None].expand(-1, -1, 3))
        near_z = z.gather(1, back[..., None].expand(-1, -1, 3))
        ## squared distances: a mean over the xyz components would cancel out. The
        ## mean is over the points that were matched, so a point left out of the
        ## pairing is not charged to an arbitrary neighbour
        d1 = ((z - near_target) ** 2).sum(-1)
        d2 = ((target - near_z) ** 2).sum(-1)
        if match_meta is None:
            loss_1 = d1.mean()
            loss_2 = d2.mean()
        else:
            w1 = match_meta['near_ok'].to(d1.dtype)
            w2 = match_meta['back_ok'].to(d2.dtype)
            loss_1 = (d1 * w1).sum() / w1.sum().clamp_min(1.0)
            loss_2 = (d2 * w2).sum() / w2.sum().clamp_min(1.0)
        return loss_1 + loss_2, near, match_meta
