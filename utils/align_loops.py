"""Seam synchronisation of loop chains (step 3 of `utils.propagation`)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


try:  # sparse solve is optional; dense least squares is the fallback
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr as _lsqr
except Exception:  # pragma: no cover - exercised only without scipy
    csr_matrix = None
    _lsqr = None



# closed-form switch
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



# pair residual and its closed-form minimiser
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



# pairs, offsets, coordinates, loft
@dataclass
class PairFit:
    """Fitted shift, residual and switch of one loop pair."""
    i: int
    j: int
    delta: float          # turns, x_i(z) ~ x_j(z + delta)
    D: float              # normalized squared alignment error
    s: float              # closed-form switch
    E: float              # switch energy
    keep: bool            # s >= min_switch


@dataclass
class SyncResult:
    """Seam offsets alpha per loop, the pair fits and solve diagnostics."""
    alpha: np.ndarray
    pairs: list[PairFit]
    lam: float
    min_switch: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def kept(self) -> list[PairFit]:
        """Pairs whose switch kept them."""
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


def synchronize(loops, centers, offsets, slots=None, lam: float = 0.02,
                neighbors: int = 2, min_switch: float = 0.3, mu=None):
    """Fit pair shifts, apply the closed-form switch and solve the seam offsets.

    loops: centred, projected cross-sections on a common z grid; centers: their
    centres; offsets: plane offsets (only adjacent offsets are compared, so call
    this per axis). Pairs with s < min_switch are dropped from the solve.
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
