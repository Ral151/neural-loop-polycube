#!/usr/bin/env python3
"""Mesh-free cubic stylization of a point cloud (Liu & Jacobson 2019).

The paper needs mesh connectivity for its local cells.  We replace each cell
with a symmetric kNN neighbourhood, which is the "local map around a point"
assumption: every neighbourhood is treated as a small sheet patch whose rest
shape and current shape are compared through one rigid motion.

Energy (same two terms as the paper):

    E(X, R) = sum_e || (x_i - x_j) - R_i (p_i - p_j) ||^2
              + lam * sum_i || R_i n_i ||_1

``R_i`` is a per-vertex rotation, ``n_i`` the rest normal.  The first term is
ARAP (content preservation); the second is the cubeness term.  Because
``R_i n_i`` is a unit vector, its L1 norm ranges from 1 (axis aligned) to
sqrt(3) (diagonal), so the minimiser aligns every normal with +/-x, +/-y or
+/-z -- flat panels and sharp creases.

Local step (per vertex): the ARAP-optimal rotation R* is the Procrustes fit of
the neighbourhood.  We then move R along the shortest geodesic that takes
``R* n_i`` to one of the six signed axes, choosing the axis and fraction t that
minimise the exact two-term cost.  That is a 1-D trust region in SO(3); no
step-size tuning, and it is exact for the path it explores.

Global step: differentiate the ARAP term, giving a sparse graph-Laplacian
system ``L X = B(R)``, factorised once (L is constant) and solved per
iteration.

The cloud is rescaled so the mean kNN edge is 1 before solving, which makes
``lam`` comparable to the paper's dimensionless weight; the result is mapped
back before export.

Example:
    python plc_utils/cubic_stylize_points.py \
        --points normalized_shapes/rocker-arm_points.ply \
        --out logs/cubic/rocker --lam 3 --iters 60
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

import _paths  # noqa: E402
from utils.utils import _load_point_file                          # noqa: E402


def build_graph(points: np.ndarray, k: int):
    """Symmetric kNN graph: edge list, rest edge vectors, degree."""
    n = len(points)
    _, nbr = cKDTree(points).query(points, k=k + 1, workers=-1)
    nbr = np.asarray(nbr[:, 1:], dtype=np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = nbr.reshape(-1)
    adj = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n, n)).tocsr()
    adj = adj.maximum(adj.T)
    adj = adj.tocoo()
    upper = adj.row < adj.col
    r, c = adj.row[upper], adj.col[upper]
    deg = np.bincount(np.concatenate([r, c]), minlength=n).astype(np.float64)
    lap = sparse.coo_matrix(
        (np.concatenate([-np.ones(len(r)), -np.ones(len(r)), deg]),
         (np.concatenate([r, c, np.arange(n)]),
          np.concatenate([c, r, np.arange(n)]))),
        shape=(n, n)).tocsr()
    return r, c, lap


def skew(u: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(u[:, 0])
    return torch.stack([
        torch.stack([z, -u[:, 2], u[:, 1]], dim=-1),
        torch.stack([u[:, 2], z, -u[:, 0]], dim=-1),
        torch.stack([-u[:, 1], u[:, 0], z], dim=-1),
    ], dim=1)


def rodrigues(u: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Batched rotation exp(phi [u]_x); u is unit (N, 3), phi is (N,)."""
    k = skew(u)
    eye = torch.eye(3, device=u.device, dtype=u.dtype).expand(len(u), 3, 3)
    s = torch.sin(phi)[:, None, None]
    co = torch.cos(phi)[:, None, None]
    return (co * eye + s * k
            + (1.0 - co) * u.unsqueeze(2) * u.unsqueeze(1))


def procrustes(cov: torch.Tensor) -> torch.Tensor:
    """argmax_R tr(R C) over SO(3), batched."""
    u, _, vh = torch.linalg.svd(cov)
    v = vh.transpose(1, 2)
    d = torch.det(v @ u.transpose(1, 2))
    keep = torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1)
    return v @ torch.diag_embed(keep) @ u.transpose(1, 2)


def pca_normals(points, k, ref_normals, device):
    """Normals of the *deformed* geometry from local PCA, sign-aligned."""
    t = torch.as_tensor(points, dtype=torch.float32, device=device)
    _, nbr = cKDTree(points).query(points, k=k + 1, workers=-1)
    nbr = torch.as_tensor(nbr[:, 1:], dtype=torch.long, device=device)
    diff = t[nbr] - t[:, None, :]
    cov = diff.transpose(1, 2) @ diff
    _, vec = torch.linalg.eigh(cov)
    n = vec[..., 0]
    ref = torch.as_tensor(ref_normals, dtype=torch.float32, device=device)
    n = n * torch.where((n * ref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return n.cpu().numpy()


def local_step(cov, n0, edges, d0, lam, axis_steps, device):
    """ARAP Procrustes + L1 axis alignment along the shortest geodesic."""
    n = cov.shape[0]
    r_star = procrustes(cov)
    m_star = (r_star @ n0.unsqueeze(-1)).squeeze(-1)

    # cost is -2 tr(R C) + lam |R n0|_1
    def cost_of(rot, m):
        tr = (rot * cov.transpose(1, 2)).sum((-1, -2))
        return -2.0 * tr + lam * m.abs().sum(-1)

    best_cost = cost_of(r_star, m_star)
    best_r = r_star
    best_m = m_star

    axes = torch.tensor(
        [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
         [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]],
        device=device)
    ts = torch.linspace(0.0, 1.0, axis_steps, device=device)

    for a in axes:
        a = a.expand(n, 3)
        cross = torch.cross(m_star, a, dim=-1)
        s = cross.norm(dim=-1)
        co = (m_star * a).sum(-1).clamp(-1.0, 1.0)
        theta = torch.atan2(s, co)
        u = cross / s.clamp_min(1e-9).unsqueeze(-1)
        u = torch.where((s > 1e-7).unsqueeze(-1), u,
                        torch.zeros_like(u))
        for t in ts:
            phi = t * theta
            rot = rodrigues(u, phi) @ r_star
            m = (rodrigues(u, phi) @ m_star.unsqueeze(-1)).squeeze(-1)
            cost = cost_of(rot, m)
            better = cost < best_cost
            best_cost = torch.where(better, cost, best_cost)
            best_r = torch.where(better[:, None, None], rot, best_r)
            best_m = torch.where(better[:, None], m, best_m)
    return best_r, best_m, best_cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--lam', type=float, default=6.0,
                    help='cubeness weight (cloud rescaled to unit edge)')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--axis-steps', type=int, default=11)
    ap.add_argument('--normal-mode', choices=('pca', 'rotated'),
                    default='pca',
                    help='export normals from the deformed geometry (pca) '
                         'or the rotated rest normals (rotated)')
    ap.add_argument('--checkpoint-every', type=int, default=0)
    ap.add_argument('--views', type=int, default=6)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')

    pts, normals = _load_point_file(args.points)
    if normals is None:
        raise SystemExit('the point cloud needs normals')
    pts = np.asarray(pts, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True),
                          1e-12)
    print(f'{len(pts):,} points, normals present', flush=True)

    r_idx, c_idx, lap = build_graph(pts, args.k)
    edge_len = np.linalg.norm(pts[c_idx] - pts[r_idx], axis=1)
    spacing = float(edge_len.mean())
    p0 = pts / spacing
    d0 = p0[c_idx] - p0[r_idx]
    print(f'knn {args.k}, mean edge {spacing:.5f} (working units), '
          f'{len(r_idx):,} undirected edges', flush=True)

    eps = 1e-8
    lap_p = (lap + eps * sparse.eye(len(pts), format='csr')).tocsc()
    t0 = time.time()
    lu = splu(lap_p)
    print(f'Laplacian factorised in {time.time() - t0:.1f}s', flush=True)

    dev_r = torch.as_tensor(r_idx, dtype=torch.long, device=device)
    dev_c = torch.as_tensor(c_idx, dtype=torch.long, device=device)
    d0_t = torch.as_tensor(d0, dtype=torch.float32, device=device)
    n0 = torch.as_tensor(normals, dtype=torch.float32, device=device)
    x = torch.as_tensor(p0, dtype=torch.float32, device=device)
    n_pts = len(p0)
    eye3 = torch.eye(3, device=device)

    hist = []
    for it in range(1, args.iters + 1):
        d = x[dev_c] - x[dev_r]
        # A vertex's covariance sums every incident edge.  Edges are stored
        # once (r < c); from the c endpoint both vectors are negated, so the
        # outer product d0 d^T is the same and must be added to both ends.
        outer = d0_t.unsqueeze(2) * d.unsqueeze(1)
        cov = torch.zeros(n_pts, 3, 3, device=device)
        cov.index_add_(0, dev_r, outer)
        cov.index_add_(0, dev_c, outer)

        r_rot, m_rot, _ = local_step(cov, n0, (dev_r, dev_c), d0,
                                     args.lam, args.axis_steps, device)

        # global step: (Lx)_i = -sum_j 0.5 (R_i + R_j) d0_ij, with
        # d0_ij = p_j - p_i.  Every undirected edge (r, c) contributes
        # -m to row r and +m to row c (d0_flipped = -d0), so that R = I
        # reproduces x = p exactly and the right-hand side stays
        # divergence-free.
        m_edge = 0.5 * ((r_rot[dev_r] + r_rot[dev_c])
                        @ d0_t.unsqueeze(-1)).squeeze(-1)
        b = torch.zeros(n_pts, 3, device=device)
        b.index_add_(0, dev_r, -m_edge)
        b.index_add_(0, dev_c, m_edge)
        b = b - b.mean(0, keepdim=True)
        x_new = lu.solve(b.detach().cpu().numpy())
        x_new = torch.as_tensor(np.ascontiguousarray(x_new),
                                dtype=torch.float32, device=device)
        x_new = x_new - x_new.mean(0, keepdim=True) + x.mean(0, keepdim=True)
        x = x_new

        d = x[dev_c] - x[dev_r]
        arap = (d - (r_rot[dev_r] @ d0_t.unsqueeze(-1)).squeeze(-1))
        arap = float((arap ** 2).sum(-1).mean())
        l1 = float(m_rot.abs().sum(-1).mean())
        diag = float((x.max(0).values - x.min(0).values).norm())
        hist.append({'iter': it, 'arap': arap, 'l1': l1, 'diag': diag})
        if it == 1 or it % 5 == 0 or it == args.iters:
            print(f'it {it:3d}/{args.iters}  arap {arap:.4f}  '
                  f'mean |n|1 {l1:.4f}  bbox {diag:.3f}  '
                  f'[{time.time() - t0:.0f}s]', flush=True)
        if (args.checkpoint_every > 0
                and it % args.checkpoint_every == 0):
            np.save(os.path.join(args.out, f'points_it{it}.npy'),
                    (x * spacing).detach().cpu().numpy())

    x_out = (x * spacing).detach().cpu().numpy()
    nrm_out = m_rot.detach().cpu().numpy()
    nrm_out /= np.maximum(np.linalg.norm(nrm_out, axis=1, keepdims=True),
                          1e-12)
    if args.normal_mode == 'pca':
        nrm_out = pca_normals(x_out, args.k, normals, device)
        print('exported normals re-estimated from the deformed geometry',
              flush=True)

    rel = r_rot[dev_r].transpose(1, 2) @ r_rot[dev_c]
    cos = (rel.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0
    rot_edge_deg = float(torch.acos(cos.clamp(-1.0, 1.0)).mean()
                         * 180.0 / np.pi)
    print(f'rotation coherence across edges: '
          f'{rot_edge_deg:.1f} deg mean', flush=True)

    stem = Path(args.points).stem
    ply_path = os.path.join(args.out, f'{stem}_cubic_lam{args.lam:g}.ply')
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x_out)
        pcd.normals = o3d.utility.Vector3dVector(nrm_out)
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
        print(f'wrote {ply_path}', flush=True)
    except ImportError:
        np.save(os.path.join(args.out, f'{stem}_cubic_lam{args.lam:g}.npy'),
                np.concatenate([x_out, nrm_out], axis=1))

    metrics = {
        'points': os.path.abspath(args.points),
        'k': args.k, 'lam': args.lam, 'iters': args.iters,
        'mean_edge': spacing,
        'rest_mean_l1': float(np.abs(normals).sum(1).mean()),
        'final_mean_l1': float(np.abs(nrm_out).sum(1).mean()),
        'axis_frac': float((np.abs(nrm_out).max(1) > 0.9).mean()),
        'disp_mean': float(np.linalg.norm(x_out - pts, axis=1).mean()),
        'disp_p95': float(np.percentile(
            np.linalg.norm(x_out - pts, axis=1), 95)),
        'rest_diag': float(np.linalg.norm(pts.max(0) - pts.min(0))),
        'final_diag': float(np.linalg.norm(x_out.max(0) - x_out.min(0))),
        'rot_edge_deg': rot_edge_deg,
        'history': hist,
        'wall_seconds': time.time() - t0,
    }
    with open(os.path.join(args.out, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=1)

    # before/after views from the same cameras
    try:
        from utils.render_targets import (_label_uint8, _normal_shading,
                                          make_orbit_views,
                                          prepare_point_cloud,
                                          render_pointcloud)
        import imageio.v2 as imageio
        rest_t = torch.as_tensor(pts, dtype=torch.float32, device=device)
        views, _, _, _ = make_orbit_views(rest_t, count=args.views,
                                          device=device)
        r0 = render_pointcloud(
            prepare_point_cloud(pts, normals, device=device), views,
            args.size, args.size)
        r1 = render_pointcloud(
            prepare_point_cloud(x_out, nrm_out, device=device), views,
            args.size, args.size)
        idx = np.linspace(0, len(views) - 1,
                          min(4, len(views))).astype(int)
        rows = []
        for label, render in (('rest', r0), ('cubic', r1)):
            tiles = [_label_uint8(
                _normal_shading(render.normal[i], render.alpha[i]),
                f'{label} v{i}') for i in idx]
            rows.append(np.hstack(tiles))
        imageio.imwrite(os.path.join(args.out, 'before_after.png'),
                        np.vstack(rows))
        print('wrote before_after.png', flush=True)
    except Exception as exc:                              # pragma: no cover
        print(f'[warn] preview skipped: {exc}', flush=True)

    print(json.dumps({k: v for k, v in metrics.items()
                      if k != 'history'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
