"""Over-dispersed Poisson bootstrap (England & Verrall, 2002) and one-year risk.

Ultimate view: the full distribution of outstanding claims until run-off.
One-year view (Solvency II): the distribution of the claims development result
(CDR) over the next calendar year, obtained by simulating the next diagonal and
re-estimating the chain-ladder ("actuary in the box", Ohlsson & Lauzeningks, 2009).
"""
import numpy as np

from .chain_ladder import (chain_ladder, complete_triangle, development_factors,
                           latest_diagonal, to_cumulative, to_incremental)


def fitted_incremental(cum, f):
    """Fitted values of the ODP model: back-cast each row from its latest diagonal."""
    n = cum.shape[0]
    fitted_cum = np.full_like(cum, np.nan)
    for i in range(n):
        last = n - 1 - i
        fitted_cum[i, last] = cum[i, last]
        for k in range(last - 1, -1, -1):
            fitted_cum[i, k] = fitted_cum[i, k + 1] / f[k]
    return to_incremental(fitted_cum)


def pearson_residuals(cum):
    """Bias-adjusted Pearson residuals and the ODP scale parameter phi."""
    n = cum.shape[0]
    f = development_factors(cum)
    inc = to_incremental(cum)
    m = fitted_incremental(cum, f)
    r = (inc - m) / np.sqrt(np.abs(m))
    n_obs = n * (n + 1) // 2
    n_par = 2 * n - 1
    phi = np.nansum(r ** 2) / (n_obs - n_par)
    r_adj = r * np.sqrt(n_obs / (n_obs - n_par))
    # The two corner cells have zero residual by construction: exclude them from resampling
    pool = r_adj[~np.isnan(r_adj)]
    pool = pool[np.abs(pool) > 1e-9]
    return {"fitted": m, "residuals": r, "residuals_adj": r_adj, "pool": pool, "phi": phi}


def _process_draw(rng, mean, phi):
    """Gamma draw with mean `mean` and variance phi * |mean| (sign kept for negatives)."""
    mean = np.asarray(mean, dtype=float)
    out = np.zeros_like(mean)
    pos = np.abs(mean) > 0
    shape = np.abs(mean[pos]) / phi
    out[pos] = np.sign(mean[pos]) * rng.gamma(shape, phi)
    return out


def _pseudo_triangle(rng, fitted, pool, mask):
    pseudo = fitted.copy()
    draws = rng.choice(pool, size=mask.sum(), replace=True)
    pseudo[mask] = fitted[mask] + draws * np.sqrt(np.abs(fitted[mask]))
    return to_cumulative(pseudo)


def bootstrap_ultimate(cum, n_sims=10_000, seed=42, process=True):
    """Predictive distribution of total and per-origin reserves (ultimate view).

    process=False keeps parameter uncertainty only (no gamma process noise).
    """
    rng = np.random.default_rng(seed)
    n = cum.shape[0]
    pr = pearson_residuals(cum)
    mask = ~np.isnan(pr["fitted"])
    reserves = np.empty((n_sims, n))
    for s in range(n_sims):
        pc = _pseudo_triangle(rng, pr["fitted"], pr["pool"], mask)
        f_star = development_factors(pc)
        full = complete_triangle(pc, f_star)         # chain-ladder on the pseudo data itself
        future_inc = to_incremental(full)
        future_inc[mask] = 0.0                       # keep only the lower triangle
        draw = _process_draw(rng, future_inc, pr["phi"]) if process else future_inc
        reserves[s] = draw.sum(axis=1)
    return {"reserves": reserves, "total": reserves.sum(axis=1), "phi": pr["phi"]}


def bootstrap_one_year(cum, n_sims=10_000, seed=7):
    """Distribution of the one-year claims development result (CDR).

    CDR = R0 - (payments next year + R1), where R0 is today's chain-ladder reserve
    and R1 the chain-ladder reserve re-estimated one year later.
    """
    rng = np.random.default_rng(seed)
    n = cum.shape[0]
    pr = pearson_residuals(cum)
    mask = ~np.isnan(pr["fitted"])
    r0 = chain_ladder(cum)["reserve"].sum()
    latest = latest_diagonal(cum)
    cdr = np.empty(n_sims)
    for s in range(n_sims):
        pc = _pseudo_triangle(rng, pr["fitted"], pr["pool"], mask)
        f_star = development_factors(pc)
        # Next calendar year: one more development period for origins 1..n-1
        new = np.full((n, n + 1), np.nan)
        new[:, :n] = cum
        paid_next = 0.0
        for i in range(1, n):
            k = n - 1 - i                            # column of the latest diagonal
            inc_mean = latest[i] * (f_star[k] - 1)
            inc = _process_draw(rng, np.array([inc_mean]), pr["phi"])[0]
            new[i, k + 1] = latest[i] + inc
            paid_next += inc
        tri1 = new[:, :n]
        f1 = np.empty(n - 1)
        for k in range(n - 1):
            rows = np.array([i for i in range(n) if not np.isnan(tri1[i, k + 1]) and not np.isnan(tri1[i, k])])
            f1[k] = tri1[rows, k + 1].sum() / tri1[rows, k].sum()
        r1 = 0.0
        for i in range(1, n):
            last = n - i                             # new latest column for origin i
            if last >= n - 1:
                continue
            r1 += tri1[i, last] * (np.prod(f1[last:]) - 1)
        cdr[s] = r0 - (paid_next + r1)
    return {"cdr": cdr, "r0": r0}
