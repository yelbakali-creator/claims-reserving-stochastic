"""Validation tests: chain-ladder and Mack against Mack (1993), ODP scale parameter against England & Verrall (2002)."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.bootstrap import bootstrap_ultimate, pearson_residuals
from src.chain_ladder import mack

C = pd.read_csv(Path(__file__).parents[1] / "data" / "taylor_ashe_cumulative.csv", index_col=0).values


def test_chain_ladder_reserve_matches_mack_1993():
    assert round(mack(C)["reserve"].sum()) == 18_680_856


def test_mack_standard_error_matches_mack_1993():
    m = mack(C)
    assert round(m["total_se"]) == 2_447_095
    # Published standard errors by origin year (Mack, 1993)
    published = [75_535, 121_699, 133_549, 261_406, 411_010, 558_317, 875_328, 971_258, 1_363_155]
    assert np.allclose(np.round(m["se"][1:]), published)


def test_odp_scale_parameter_matches_england_verrall():
    assert round(pearson_residuals(C)["phi"]) == 52_601


def test_bootstrap_mean_close_to_chain_ladder():
    res = bootstrap_ultimate(C, n_sims=2_000, seed=1)
    assert abs(res["total"].mean() / 18_680_856 - 1) < 0.03
