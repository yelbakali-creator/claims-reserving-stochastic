"""Stochastic claims reserving study: chain-ladder, Mack, ODP bootstrap, one-year risk."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.bootstrap import bootstrap_one_year, bootstrap_ultimate, pearson_residuals
from src.chain_ladder import mack

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY = "#1F3864", "#C0392B", "#7F8C8D"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
N_SIMS = 10_000


def risk_measures(x, loss_is_positive=True):
    """Mean, SD, VaR and TVaR. For losses, the tail is the upper tail."""
    x = np.asarray(x)
    loss = x if loss_is_positive else -x
    var995 = np.quantile(loss, 0.995)
    return {"mean": x.mean(), "sd": x.std(ddof=1), "cv": x.std(ddof=1) / abs(x.mean()) if x.mean() else np.nan,
            "p75": np.quantile(loss, 0.75), "var_99_5": var995, "tvar_99_5": loss[loss >= var995].mean()}


def main():
    tri = pd.read_csv(ROOT / "data" / "taylor_ashe_cumulative.csv", index_col=0)
    C = tri.values
    origins = tri.index.values

    # 1. Chain-ladder and Mack
    m = mack(C)

    # 2. ODP bootstrap: ultimate view, parameter-only run, one-year view
    ult = bootstrap_ultimate(C, N_SIMS, seed=42)
    par = bootstrap_ultimate(C, N_SIMS, seed=42, process=False)
    one = bootstrap_one_year(C, N_SIMS, seed=7)

    # 3. Summary by origin year
    q = lambda a, p: np.quantile(a, p, axis=0)
    by_origin = pd.DataFrame({
        "origin": origins, "latest": m["latest"], "ultimate": m["ultimate"], "cl_reserve": m["reserve"],
        "mack_se": m["se"], "mack_cv": np.where(m["reserve"] > 0, m["se"] / np.maximum(m["reserve"], 1), np.nan),
        "boot_mean": ult["reserves"].mean(axis=0), "boot_sd": ult["reserves"].std(axis=0, ddof=1),
        "boot_p2_5": q(ult["reserves"], 0.025), "boot_p97_5": q(ult["reserves"], 0.975),
    })
    by_origin.to_csv(TABLES / "reserves_by_origin.csv", index=False)
    pd.DataFrame({"development_period": np.arange(1, len(m["factors"]) + 1),
                  "factor": m["factors"], "mack_sigma2": m["sigma2"]}).to_csv(TABLES / "development_factors.csv", index=False)

    # 4. Validation against published results and an independent library
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import chainladder as cl
        genins = cl.load_sample("genins")
        dev = cl.Development(sigma_interpolation="mack").fit_transform(genins)
        ref_mack = cl.MackChainladder().fit(dev)
        ref_boot = cl.BootstrapODPSample(n_sims=N_SIMS, random_state=42).fit(genins)
        ref_par = cl.Chainladder().fit(ref_boot.resampled_triangles_).ibnr_.sum("origin").values.flatten()
    validation = pd.DataFrame([
        ["Chain-ladder total reserve", m["reserve"].sum(), float(ref_mack.ibnr_.sum()), 18_680_856],
        ["Mack total standard error", m["total_se"], float(ref_mack.total_mack_std_err_.values.flatten()[0]), 2_447_095],
        ["Bootstrap SD, parameter only", par["total"].std(ddof=1), ref_par.std(ddof=1), np.nan],
        ["ODP scale parameter phi", ult["phi"], np.nan, 52_601],
    ], columns=["quantity", "this_project", "chainladder_library", "published_mack_1993_ev_2002"])
    validation.to_csv(TABLES / "validation.csv", index=False)

    # 5. Risk measures and a Solvency II reading of the one-year view
    r0 = m["reserve"].sum()
    rm_ult = risk_measures(ult["total"])
    rm_one = risk_measures(one["cdr"], loss_is_positive=False)
    sigma_one = rm_one["sd"] / r0
    reserve_risk_sim = rm_one["var_99_5"] + rm_one["mean"]          # 99.5% loss in excess of the expected CDR
    lognormal_rho = np.exp(norm.ppf(0.995) * np.sqrt(np.log(1 + sigma_one ** 2))) / np.sqrt(1 + sigma_one ** 2) - 1
    risk = pd.DataFrame({
        "measure": ["Best estimate reserve R0", "Mean", "Standard deviation", "Coefficient of variation",
                    "75th percentile", "VaR 99.5%", "TVaR 99.5%"],
        "ultimate_view_total_reserve": [r0, rm_ult["mean"], rm_ult["sd"], rm_ult["cv"], rm_ult["p75"],
                                        rm_ult["var_99_5"], rm_ult["tvar_99_5"]],
        "one_year_view_loss_on_cdr": [r0, -rm_one["mean"], rm_one["sd"], np.nan, rm_one["p75"],
                                      rm_one["var_99_5"], rm_one["tvar_99_5"]],
    })
    risk.to_csv(TABLES / "risk_measures.csv", index=False)
    solvency = pd.DataFrame({
        "item": ["One-year sigma (SD of CDR / R0)", "Simulated 99.5% reserve risk / R0",
                 "Standard formula approximation 3 x sigma", "Lognormal rho(sigma)"],
        "value": [sigma_one, reserve_risk_sim / r0, 3 * sigma_one, lognormal_rho],
    })
    solvency.to_csv(TABLES / "solvency_ii_comparison.csv", index=False)

    # 6. Charts
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = ult["total"] / 1e6
    ax.hist(x, bins=80, color=NAVY, alpha=0.85, density=True)
    s = np.sqrt(np.log(1 + (x.std() / x.mean()) ** 2)); mu = np.log(x.mean()) - s ** 2 / 2
    grid = np.linspace(x.min(), x.max(), 300)
    ax.plot(grid, np.exp(-(np.log(grid) - mu) ** 2 / (2 * s ** 2)) / (grid * s * np.sqrt(2 * np.pi)),
            color=GREY, lw=1.5, label="Lognormal, same mean and SD")
    for val, lab, col in [(r0 / 1e6, "Chain-ladder best estimate", RED),
                          (rm_ult["var_99_5"] / 1e6, "VaR 99.5%", "black")]:
        ax.axvline(val, color=col, ls="--", lw=1.2, label=f"{lab}: {val:.1f}m")
    ax.set_xlabel("Total outstanding claims (millions)")
    ax.set_yticks([])
    ax.set_title("Predictive distribution of reserves (ODP bootstrap, 10,000 simulations)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "reserve_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(one["cdr"] / 1e6, bins=80, color=NAVY, alpha=0.85)
    q005 = np.quantile(one["cdr"], 0.005) / 1e6
    ax.axvline(q005, color=RED, ls="--", lw=1.2, label=f"0.5% quantile: {q005:.1f}m (1-in-200 year)")
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("One-year claims development result (millions; negative = loss)")
    ax.set_yticks([])
    ax.set_title("Solvency II one-year view: distribution of the CDR")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "one_year_cdr.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    idx = np.arange(1, len(origins))
    b = by_origin.iloc[1:]
    ax.bar(idx - 0.2, b["cl_reserve"] / 1e6, width=0.4, color=NAVY, yerr=1.96 * b["mack_se"] / 1e6,
           capsize=3, label="Chain-ladder reserve, Mack +/- 1.96 SE")
    ax.errorbar(idx + 0.2, b["boot_mean"] / 1e6,
                yerr=[(b["boot_mean"] - b["boot_p2_5"]) / 1e6, (b["boot_p97_5"] - b["boot_mean"]) / 1e6],
                fmt="o", color=RED, capsize=3, label="Bootstrap mean, 95% predictive interval")
    ax.set_xticks(idx, origins[1:])
    ax.set_ylabel("Reserve (millions)")
    ax.set_title("Reserves and uncertainty by origin year")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "reserves_by_origin.png", dpi=150); plt.close(fig)

    pr = pearson_residuals(C)
    r = pr["residuals_adj"]
    n = C.shape[0]
    rows, cols = np.where(~np.isnan(r))
    vals = r[rows, cols]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, xs, lab in zip(axes, [rows + 1, cols + 1, rows + cols + 1],
                           ["Origin year index", "Development period", "Calendar period"]):
        ax.scatter(xs, vals, color=NAVY, s=14, alpha=0.7)
        means = pd.Series(vals).groupby(xs).mean()
        ax.plot(means.index, means.values, color=RED, lw=1.2)
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xlabel(lab)
    axes[0].set_ylabel("Adjusted Pearson residual")
    fig.suptitle("Residual diagnostics by origin, development and calendar period")
    fig.tight_layout(); fig.savefig(FIGS / "residual_diagnostics.png", dpi=150); plt.close(fig)

    # 7. Console summary
    pd.set_option("display.width", 160)
    print(f"Chain-ladder reserve: {r0:,.0f} | Mack SE: {m['total_se']:,.0f} ({m['total_se'] / r0:.1%})")
    print(f"Bootstrap ultimate: mean {rm_ult['mean']:,.0f}, SD {rm_ult['sd']:,.0f} ({rm_ult['cv']:.1%}), "
          f"VaR 99.5% {rm_ult['var_99_5']:,.0f}")
    print(f"One-year CDR: SD {rm_one['sd']:,.0f} ({sigma_one:.1%} of R0), 99.5% loss {rm_one['var_99_5']:,.0f}")
    print("\nValidation\n" + validation.round(0).to_string(index=False))
    print("\nSolvency II comparison\n" + solvency.round(4).to_string(index=False))
    print("\nReserves by origin\n" + by_origin.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
