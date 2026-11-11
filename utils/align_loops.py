"""Synchronized loop seams and tube coordinates.

Standalone implementation of the `todo.md` plan ("A smooth switch with a
closed-form solution", "Synchronized loops define the tube coordinates").  It
does not import `loop_extraction.py` or `plc_main.py`, and nothing calls it
yet; it is a self-contained module for the next stage of the loop work.

Conventions
-----------
- `z` is the loop parameter in turns: `z = k / K` for `k = 0..K-1`.
- `v` runs along the tube, `v_i` is the coordinate of loop `i`.
- `x_i(z)` is loop `i` after removing the out-of-plane component and centring
  at its field centre, sampled on the common `z` grid.
- Pair sign: `x_i(z) ~ x_j(z + delta_ij)`, so the seam offsets satisfy
  `alpha_j - alpha_i = delta_ij`.

The pair residual is normalized by the pair's mean squared radius, which makes
`lambda` dimensionless and independent of the shape's scale:

    sigma_ij^2 = (mean_k ||x_i(z_k)||^2 + mean_k ||x_j(z_k)||^2) / 2
    D_ij(d)    = mean_k ||x_i(z_k) - x_j(z_k + d)||^2 / sigma_ij^2

For fixed `D`, `E(s) = s^2 D + lambda (1 - s)^2` gives

    s* = lambda / (lambda + D),      E* = lambda D / (lambda + D)

with a positive second derivative, so `s*` is the unique minimiser and no
alternating optimisation is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # sparse solve is optional; dense least squares is the fallback
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr as _lsqr
except Exception:  # pragma: no cover - exercised only without scipy
    csr_matrix = None
    _lsqr = None


# --------------------------------------------------------------------------- #
# closed-form switch
# --------------------------------------------------------------------------- #
def switch(D: float, lam: float) -> tuple[float, float]:
    """Return `(s*, E*)` for the switchable energy at a fixed residual `D`.

    `D` is a squared alignment error, `lam > 0` is the switch penalty.
    """
    if lam <= 0.0:
        raise ValueError(f"lambda must be positive, got {lam}")
    D = float(max(D, 0.0))
    s = lam / (lam + D)
    return s, lam * D / (lam + D)


def wrap_turns(x):
    """Wrap offsets in turns to the representative nearest zero."""
    return np.asarray(x) - np.round(np.asarray(x))


# --------------------------------------------------------------------------- #
# pair residual and its closed-form minimiser
# --------------------------------------------------------------------------- #
def _circular_corr(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """`corr[d] = mean_k xi(k) . xj(k + d)` for every integer shift `d`."""
    n = len(xi)
    out = np.zeros(n)
    for dim in range(xi.shape[1]):
        out += np.fft.irfft(
            np.conj(np.fft.rfft(xi[:, dim])) * np.fft.rfft(xj[:, dim]), n
        )
    return out / n


def circular_shift(x: np.ndarray, delta: float) -> np.ndarray:
    """Shift a periodic, uniformly sampled curve by `delta` turns."""
    n = len(x)
    freqs = np.fft.rfftfreq(n)  # cycles per sample
    spec = np.fft.rfft(x, axis=0) * np.exp(2j * np.pi * freqs * delta * n)[:, None]
    return np.fft.irfft(spec, n=n, axis=0)


def pair_residual(xi: np.ndarray, xj: np.ndarray, n_grid: int | None = None):
    """Minimise the normalized squared alignment error over a circular shift.

    Parameters
    ----------
    xi, xj : (K, 2) or (K, 3)
        Loops sampled on the same `z` grid, already centred and projected onto
        the shared plane.
    n_grid : int, optional
        Number of shifts in the coarse search (default `K`).

    Returns
    -------
    delta : float
        Shift in turns with `x_i(z) ~ x_j(z + delta)`, in `[0, 1)`.
    D : float
        Normalized residual at `delta`.
    sigma2 : float
        Mean squared radius used for the normalization.
    """
    xi = np.asarray(xi, float)
    xj = np.asarray(xj, float)
    if xi.shape != xj.shape or xi.ndim != 2:
        raise ValueError(f"loops must have the same (K, D) shape, got "
                         f"{xi.shape} and {xj.shape}")
    K = len(xi)
    if K < 8:
        raise ValueError(f"need at least 8 samples per loop, got {K}")

    si = float(np.mean(np.sum(xi * xi, axis=1)))
    sj = float(np.mean(np.sum(xj * xj, axis=1)))
    sigma2 = 0.5 * (si + sj)
    if sigma2 <= 1e-12:
        raise ValueError("degenerate pair: both loops have zero radius")

    corr = _circular_corr(xi, xj)
    D_all = (si + sj - 2.0 * corr) / sigma2

    if n_grid is None:
        n_grid = K
    n_grid = max(4, min(int(n_grid), K))
    grid = np.round(np.arange(n_grid) * K / n_grid).astype(int) % K
    k0 = int(grid[np.argmin(D_all[grid])])

    # parabolic refinement on the full integer grid, then evaluate exactly
    dm, d0, dp = D_all[(k0 - 1) % K], D_all[k0], D_all[(k0 + 1) % K]
    denom = dm - 2.0 * d0 + dp
    frac = 0.5 * (dm - dp) / denom if denom > 1e-15 else 0.0
    frac = float(np.clip(frac, -0.5, 0.5))
    delta = ((k0 + frac) / K) % 1.0

    resid = xi - circular_shift(xj, delta)
    D = float(np.mean(np.sum(resid * resid, axis=1)) / sigma2)
    return delta, max(D, 0.0), sigma2


# --------------------------------------------------------------------------- #
# pairs, offsets, coordinates, loft
# --------------------------------------------------------------------------- #
@dataclass
class PairFit:
    i: int
    j: int
    delta: float          # turns, x_i(z) ~ x_j(z + delta)
    D: float              # normalized squared alignment error
    s: float              # closed-form switch
    E: float              # switch energy
    keep: bool            # s >= min_switch


@dataclass
class SyncResult:
    alpha: np.ndarray
    pairs: list[PairFit]
    lam: float
    min_switch: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def kept(self) -> list[PairFit]:
        return [p for p in self.pairs if p.keep]


def candidate_pairs(centers, offsets, slots=None, neighbors: int = 2):
    """Loop pairs on adjacent plane offsets of the same axis.

    Each loop is compared with its `neighbors` nearest centres on each adjacent
    plane, plus the loop with the same slot when slots are given.  The switch,
    not this rule, decides which pairs survive.
    """
    centers = np.asarray(centers, float)
    offsets = np.asarray(offsets, float)
    if len(centers) != len(offsets):
        raise ValueError("centers and offsets must have the same length")
    uniq = np.unique(offsets)
    groups = {u: np.flatnonzero(offsets == u) for u in uniq}
    pairs = set()
    for u in uniq:
        k = int(np.searchsorted(uniq, u))
        adjacent = []
        if k > 0:
            adjacent.append(uniq[k - 1])
        if k + 1 < len(uniq):
            adjacent.append(uniq[k + 1])
        for un in adjacent:
            js = groups[un]
            for i in groups[u]:
                d = np.linalg.norm(centers[js] - centers[i], axis=1)
                for j in js[np.argsort(d)[:max(1, neighbors)]]:
                    pairs.add((min(int(i), int(j)), max(int(i), int(j))))
                if slots is not None:
                    for j in js[slots[js] == slots[i]]:
                        pairs.add((min(int(i), int(j)), max(int(i), int(j))))
    return sorted(pairs)


def solve_offsets(n_loops: int, pairs: list[PairFit], mu: float | None = None):
    """Least-squares seam offsets from the kept pairs.

    Minimises `sum w_ij (alpha_j - alpha_i - delta_ij)^2 + mu sum alpha_i^2`
    with `w_ij = s_ij^2`, which matches the `s^2 D` term.  The prior `mu` fixes
    the free global gauge; isolated loops stay at zero.
    """
    kept = [p for p in pairs if p.keep]
    alpha = np.zeros(n_loops)
    if not kept:
        return alpha, {'mu': mu, 'edges': 0, 'residual_rms': 0.0}

    w = np.array([p.s ** 2 for p in kept], float)
    if mu is None:
        mu = 1e-3 * float(np.mean(w))
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for p, wi in zip(kept, w):
        sw = np.sqrt(wi)
        rows += [r, r]
        cols += [p.i, p.j]
        vals += [-sw, sw]
        rhs.append(sw * p.delta)
        r += 1
    sp = np.sqrt(mu)
    for i in range(n_loops):
        rows.append(r)
        cols.append(i)
        vals.append(sp)
        rhs.append(0.0)
        r += 1

    if csr_matrix is not None:
        A = csr_matrix((vals, (rows, cols)), shape=(r, n_loops))
        alpha = _lsqr(A, np.asarray(rhs), atol=1e-12, btol=1e-12,
                      iter_lim=10 * n_loops)[0]
    else:  # pragma: no cover
        A = np.zeros((r, n_loops))
        A[rows, cols] = vals
        alpha = np.linalg.lstsq(A, np.asarray(rhs), rcond=None)[0]

    resid = np.array([wrap_turns(alpha[p.j] - alpha[p.i] - p.delta) for p in kept])
    return alpha, {
        'mu': mu,
        'edges': len(kept),
        'residual_rms': float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 0.0,
    }


def loop_coordinates(centers):
    """Cumulative distance between loop centres: `v_0 = 0`."""
    centers = np.asarray(centers, float)
    if len(centers) < 2:
        return np.zeros(len(centers))
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def loft(xi, xj, t):
    """First reconstruction: linear interpolation between two synchronized loops."""
    t = np.asarray(t, float)
    xi = np.asarray(xi, float)
    xj = np.asarray(xj, float)
    if t.ndim == 0:
        return (1.0 - t) * xi + t * xj
    return (1.0 - t)[:, None] * xi + t[:, None] * xj


def sample_tube(loops, v, queries):
    """Loft the tube at longitudinal coordinates `queries`.

    `loops[i]` must already be synchronized (`H_i`) and sampled on the shared
    `z` grid; `v` must be increasing.
    """
    loops = [np.asarray(x, float) for x in loops]
    v = np.asarray(v, float)
    if len(loops) != len(v):
        raise ValueError("loops and v must have the same length")
    if np.any(np.diff(v) <= 0):
        raise ValueError("v must be strictly increasing")
    queries = np.atleast_1d(np.asarray(queries, float))
    out = []
    for q in queries:
        if q < v[0] or q > v[-1]:
            raise ValueError(f"query {q} outside [{v[0]}, {v[-1]}]")
        k = int(np.clip(np.searchsorted(v, q) - 1, 0, len(v) - 2))
        t = (q - v[k]) / (v[k + 1] - v[k])
        out.append(loft(loops[k], loops[k + 1], t))
    return np.stack(out)


def synchronize(loops, centers, offsets, slots=None, lam: float = 0.02,
                neighbors: int = 2, min_switch: float = 0.3, mu=None):
    """Fit pair shifts, apply the closed-form switch, solve seam offsets.

    `loops` are the centred, projected cross-sections; `centers` are their field
    centres; `offsets` are the plane offsets (only adjacent offsets of the same
    axis are compared, so call this per axis).  Vanishing pairs raise
    `ValueError` from `pair_residual` only when both radii are zero; they are
    otherwise reported through `D`.
    """
    loops = [np.asarray(x, float) for x in loops]
    if len(loops) != len(centers):
        raise ValueError("loops and centers must have the same length")
    fits = []
    for i, j in candidate_pairs(centers, offsets, slots, neighbors):
        delta, D, _ = pair_residual(loops[i], loops[j])
        s, E = switch(D, lam)
        fits.append(PairFit(i, j, delta, D, s, E, s >= min_switch))
    alpha, diag = solve_offsets(len(loops), fits, mu)
    diag['dropped'] = sum(not p.keep for p in fits)
    return SyncResult(alpha, fits, lam, min_switch, diag)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _loop(K: int, delta: float = 0.0) -> np.ndarray:
    z = np.arange(K) / K
    z = z - delta
    x = np.cos(2 * np.pi * z) + 0.35 * np.cos(6 * np.pi * z + 0.4)
    y = 0.8 * np.sin(2 * np.pi * z) + 0.2 * np.sin(4 * np.pi * z)
    return np.stack([x, y], axis=1)


def self_test(verbose: bool = True) -> bool:
    K = 256
    ok = True

    # 1. exact circular shift is recovered and switched on
    delta_true = 0.137
    xi = _loop(K, 0.0)
    xj = _loop(K, delta_true)
    xi_c = xi - xi.mean(0)
    xj_c = xj - xj.mean(0)
    delta, D, _ = pair_residual(xi_c, xj_c)
    s, E = switch(D, 0.02)
    err = abs(wrap_turns(delta - delta_true))
    good = err < 2.0 / K and s > 0.99
    ok &= good
    if verbose:
        print(f"[1] shifted pair   delta {delta:.5f} (true {delta_true:.5f}, "
              f"err {err*K:.2f} samples)  D {D:.2e}  s {s:.4f}  {'ok' if good else 'FAIL'}")

    # 2. unrelated shape is switched off
    rng = np.random.default_rng(0)
    z = np.arange(K) / K
    xr = np.zeros((K, 2))
    for m in range(1, 5):
        amp = rng.uniform(0.3, 1.0, 2)
        phase = rng.uniform(0, 2 * np.pi, 2)
        xr += amp * np.stack([np.cos(2 * np.pi * m * z + phase[0]),
                              np.sin(2 * np.pi * m * z + phase[1])], 1)
    xr -= xr.mean(0)
    _, Dr, _ = pair_residual(xi_c, xr)
    sr, _ = switch(Dr, 0.02)
    good = sr < 0.3 and Dr > 0.5
    ok &= good
    if verbose:
        print(f"[2] bad pair       D {Dr:.3f}  s {sr:.4f}  {'ok' if good else 'FAIL'}")

    # 3. three loops on consecutive planes recover the relative offsets
    d01, d12 = 0.10, 0.15
    loops = [_loop(K, 0.0) - _loop(K, 0.0).mean(0),
             _loop(K, d01) - _loop(K, d01).mean(0),
             _loop(K, d01 + d12) - _loop(K, d01 + d12).mean(0)]
    centers = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], float)
    res = synchronize(loops, centers, [0.0, 1.0, 2.0], lam=0.02, mu=1e-6)
    e01 = abs(wrap_turns((res.alpha[1] - res.alpha[0]) - d01))
    e12 = abs(wrap_turns((res.alpha[2] - res.alpha[1]) - d12))
    good = e01 < 3.0 / K and e12 < 3.0 / K and len(res.kept) == 2
    ok &= good
    if verbose:
        print(f"[3] seam offsets   alpha {np.round(res.alpha, 4)}  "
              f"errors {e01*K:.2f}/{e12*K:.2f} samples  "
              f"kept {len(res.kept)}/{len(res.pairs)}  rms "
              f"{res.diagnostics['residual_rms']:.5f}  {'ok' if good else 'FAIL'}")

    # 4. longitudinal coordinates and loft
    v = loop_coordinates(np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], float))
    good = np.allclose(v, [0, 3, 7])
    mid = loft(loops[0], loops[1], 0.5)
    good &= np.allclose(mid, 0.5 * (loops[0] + loops[1]))
    tube = sample_tube(loops, np.array([0.0, 1.0, 2.0]), [0.5])
    good &= tube.shape == (1, K, 2) and np.allclose(tube[0], mid)
    ok &= good
    if verbose:
        print(f"[4] v {v}  loft/sample shape {tube.shape}  {'ok' if good else 'FAIL'}")

    if verbose:
        print('self-test:', 'PASS' if ok else 'FAIL')
    return bool(ok)


if __name__ == '__main__':
    raise SystemExit(0 if self_test() else 1)
