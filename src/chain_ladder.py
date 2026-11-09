"""Deterministic chain-ladder and Mack (1993) standard errors, from scratch.

Conventions: C is an n x n cumulative triangle (numpy array), rows are origin
years, columns are development periods, NaN below the latest diagonal.
"""
import numpy as np


def to_incremental(cum):
    inc = cum.copy()
    inc[:, 1:] = cum[:, 1:] - cum[:, :-1]
    return inc


def to_cumulative(inc):
    return np.nancumsum(inc, axis=1) + np.where(np.isnan(inc), np.nan, 0)


def latest_diagonal(cum):
    n = cum.shape[0]
    return np.array([cum[i, n - 1 - i] for i in range(n)])


def development_factors(cum):
    """Volume-weighted age-to-age factors f_k = sum_i C[i,k+1] / sum_i C[i,k]."""
    n = cum.shape[0]
    f = np.empty(n - 1)
    for k in range(n - 1):
        rows = np.arange(n - 1 - k)
        f[k] = cum[rows, k + 1].sum() / cum[rows, k].sum()
    return f


def complete_triangle(cum, f):
    """Project the lower triangle with the development factors."""
    n = cum.shape[0]
    full = cum.copy()
    for i in range(n):
        for k in range(n - i, n):
            full[i, k] = full[i, k - 1] * f[k - 1]
    return full


def chain_ladder(cum):
    f = development_factors(cum)
    full = complete_triangle(cum, f)
    latest = latest_diagonal(cum)
    ultimate = full[:, -1]
    return {"factors": f, "full": full, "latest": latest,
            "ultimate": ultimate, "reserve": ultimate - latest}


def mack_sigma2(cum, f):
    """Mack variance parameters sigma_k^2, with Mack's extrapolation for the last one."""
    n = cum.shape[0]
    s2 = np.empty(n - 1)
    for k in range(n - 2):
        rows = np.arange(n - 1 - k)
        ratios = cum[rows, k + 1] / cum[rows, k]
        s2[k] = (cum[rows, k] * (ratios - f[k]) ** 2).sum() / (len(rows) - 1)
    s2[n - 2] = min(s2[n - 3] ** 2 / s2[n - 4], s2[n - 4], s2[n - 3])
    return s2


def mack(cum):
    """Mack (1993) mean squared error of prediction, by origin year and in total.

    Returns standard errors split into process and parameter components.
    """
    n = cum.shape[0]
    res = chain_ladder(cum)
    f, full = res["factors"], res["full"]
    s2 = mack_sigma2(cum, f)
    col_sums = np.array([cum[np.arange(n - 1 - k), k].sum() for k in range(n - 1)])

    process = np.zeros(n)
    parameter = np.zeros(n)
    for i in range(1, n):
        ult = full[i, -1]
        for k in range(n - 1 - i, n - 1):
            process[i] += ult ** 2 * s2[k] / f[k] ** 2 / full[i, k]
            parameter[i] += ult ** 2 * s2[k] / f[k] ** 2 / col_sums[k]
    msep = process + parameter

    # Total: sum of origin MSEPs plus the covariance from shared factor estimates
    cross = 0.0
    for i in range(1, n):
        for j in range(i + 1, n):
            terms = sum(2 * s2[k] / f[k] ** 2 / col_sums[k] for k in range(n - 1 - i, n - 1))
            cross += full[i, -1] * full[j, -1] * terms
    total_msep = msep.sum() + cross
    total_parameter = parameter.sum() + cross

    return {**res, "sigma2": s2,
            "se": np.sqrt(msep), "se_process": np.sqrt(process), "se_parameter": np.sqrt(parameter),
            "total_se": np.sqrt(total_msep),
            "total_se_process": np.sqrt(process.sum()),
            "total_se_parameter": np.sqrt(total_parameter)}
