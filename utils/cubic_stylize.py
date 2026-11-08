"""Mesh-free cubic stylization of a point cloud (after Liu & Jacobson 2019).

Mesh cells are replaced by a symmetric kNN graph. 

    python main.py stylize --points Datasets/Genus0/rocker-arm.ply \
        --out log/cubic/rocker-arm --lam 3 --iters 60
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

from utils.ply_io import load_points, write_cloud


def build_graph(points: np.ndarray, k: int):
    """Symmetric kNN graph: (edge rows, edge cols) with row < col, and its graph Laplacian."""
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
    """Batched skew-symmetric (cross-product) matrices of N x 3 vectors."""
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
    """Unit normals of the deformed points by local PCA, signs matched to ref_normals."""
    t = torch.as_tensor(points, dtype=torch.float32, device=device)
    _, nbr = cKDTree(points).query(points, k=k + 1, workers=-1)
    nbr = torch.as_tensor(nbr[:, 1:], dtype=torch.long, device=device)
    diff = t[nbr] - t[:, None, :]
    cov = diff.transpose(1, 2) @ diff
    # chunked: cuSOLVER batched eigh fails on very large batches (torch 2.9.1 / CUDA 12.8)
    vec = torch.cat([torch.linalg.eigh(part)[1] for part in cov.split(16384)])
    n = vec[..., 0]
    ref = torch.as_tensor(ref_normals, dtype=torch.float32, device=device)
    n = n * torch.where((n * ref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return n.cpu().numpy()


def local_step(cov, n0, lam, axis_steps, device):
    """Local step: per point, the rotation minimising -2 tr(R C) + lam |R n0|_1.

    Starts from the Procrustes (ARAP) rotation and tries `axis_steps` fractions of
    the geodesic that turns the normal onto each of the six signed axes.
    Returns (rotations, rotated normals, cost).
    """
    n = cov.shape[0]
    r_star = procrustes(cov)
    m_star = (r_star @ n0.unsqueeze(-1)).squeeze(-1)

    # cost is -2 tr(R C) + lam |R n0|_1
    def cost_of(rot, m):
        """Local cost of rotations rot with rotated normals m."""
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


def save_preview(path, rest, rest_n, cubic, cubic_n, views, max_points=20000):
    """Save a figure: rest (top) and stylized (bottom) clouds from the same cameras,
    coloured by normal so flat axis-aligned panels read as flat colour."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keep = np.random.default_rng(0).permutation(len(rest))[:max_points]
    lo = np.minimum(rest.min(0), cubic.min(0))
    hi = np.maximum(rest.max(0), cubic.max(0))
    views = max(1, views)
    fig = plt.figure(figsize=(3.2 * views, 6.6))
    for row, (name, pts, nrm) in enumerate((('rest', rest, rest_n),
                                            ('cubic', cubic, cubic_n))):
        colour = np.clip(0.5 + 0.5 * nrm[keep], 0.0, 1.0)
        for v in range(views):
            ax = fig.add_subplot(2, views, row * views + v + 1, projection='3d')
            ax.scatter(*pts[keep].T, c=colour, s=0.3, lw=0, depthshade=False)
            ax.view_init(elev=20, azim=-60 + 360.0 * v / views)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo)
            ax.set_axis_off()
            ax.set_title(f'{name} v{v}', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    """Stylize a point cloud and write the result, its metrics and a preview."""
    ap = argparse.ArgumentParser(prog='main.py stylize',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--points', required=True,
                    help='input point cloud (.ply with nx/ny/nz)')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--k', type=int, default=8, help='neighbours per point in the kNN graph')
    ap.add_argument('--lam', type=float, default=6.0,
                    help='cubeness weight (cloud rescaled to unit edge)')
    ap.add_argument('--iters', type=int, default=60, help='local / global iterations')
    ap.add_argument('--axis-steps', type=int, default=11,
                    help='fractions tried along each geodesic in the local step')
    ap.add_argument('--normal-mode', choices=('pca', 'rotated'),
                    default='pca',
                    help='export normals from the deformed geometry (pca) '
                         'or the rotated rest normals (rotated)')
    ap.add_argument('--checkpoint-every', type=int, default=0,
                    help='save the points as .npy every N iterations; 0 never')
    ap.add_argument('--views', type=int, default=4,
                    help='camera views in before_after.png, 0 for no preview')
    ap.add_argument('--device', default='cuda', help='torch device (falls back to cpu)')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')

    pts, normals = load_points(args.points)
    if normals is None:
        raise SystemExit('the point cloud needs normals')
    pts = np.asarray(pts, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True),
                          1e-12)
    print(f'{len(pts):,} points, normals present', flush=True)

    # kNN graph, then scale to unit mean edge length
    r_idx, c_idx, lap = build_graph(pts, args.k)
    edge_len = np.linalg.norm(pts[c_idx] - pts[r_idx], axis=1)
    spacing = float(edge_len.mean())
    p0 = pts / spacing
    d0 = p0[c_idx] - p0[r_idx]
    print(f'knn {args.k}, mean edge {spacing:.5f} (working units), '
          f'{len(r_idx):,} undirected edges', flush=True)

    # the Laplacian is constant: factorise it once (tiny ridge for invertibility)
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

    hist = []
    for it in range(1, args.iters + 1):
        d = x[dev_c] - x[dev_r]
        # per-point covariance over incident edges; each edge is stored once
        # (r < c) and contributes the same outer product to both endpoints
        outer = d0_t.unsqueeze(2) * d.unsqueeze(1)
        cov = torch.zeros(n_pts, 3, 3, device=device)
        cov.index_add_(0, dev_r, outer)
        cov.index_add_(0, dev_c, outer)

        r_rot, m_rot, _ = local_step(cov, n0, args.lam, args.axis_steps, device)

        # global step: solve L x = b, each edge adding -/+ 0.5 (R_r + R_c) d0 to
        # its endpoints (R = I reproduces the rest points exactly)
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

        # readings: ARAP residual, mean L1 of the normals, bounding-box diagonal
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

    # mean rotation difference between neighbours (smoothness of the rotations)
    rel = r_rot[dev_r].transpose(1, 2) @ r_rot[dev_c]
    cos = (rel.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0
    rot_edge_deg = float(torch.acos(cos.clamp(-1.0, 1.0)).mean()
                         * 180.0 / np.pi)
    print(f'rotation coherence across edges: '
          f'{rot_edge_deg:.1f} deg mean', flush=True)

    stem = Path(args.points).stem
    ply_path = os.path.join(args.out, f'{stem}_cubic_lam{args.lam:g}.ply')
    write_cloud(ply_path, x_out, nrm_out)
    print(f'wrote {ply_path}', flush=True)

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
    if args.views:
        save_preview(os.path.join(args.out, 'before_after.png'), pts, normals,
                     x_out, nrm_out, args.views)
        print('wrote before_after.png', flush=True)

    print(json.dumps({k: v for k, v in metrics.items()
                      if k != 'history'}, indent=1), flush=True)

