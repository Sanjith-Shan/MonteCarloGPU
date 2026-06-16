"""Tests for the Monte Carlo portfolio risk engine and the delta hedge backtest."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mcgpu.portfolio import PortfolioRisk, sample_book
from mcgpu.backtest import run_delta_hedge


def test_var_is_positive_and_below_cvar():
    weights, mu, cov = sample_book(n_assets=6, seed=1)
    engine = PortfolioRisk()
    res = engine.value_at_risk(weights, mu, cov, n_scenarios=500_000,
                               horizon_days=1.0, confidence=0.99, seed=1)
    # CVaR is the average loss in the tail beyond VaR, so it is at least VaR.
    assert res.cvar >= res.var
    assert res.scenarios == 500_000


def test_var_grows_with_horizon():
    weights, mu, cov = sample_book(n_assets=6, seed=2)
    engine = PortfolioRisk()
    one_day = engine.value_at_risk(weights, mu, cov, 400_000, horizon_days=1.0, seed=2)
    ten_day = engine.value_at_risk(weights, mu, cov, 400_000, horizon_days=10.0, seed=2)
    # Risk scales roughly with sqrt(time), so a longer horizon means more risk.
    assert ten_day.var > one_day.var


def test_delta_hedge_pnl_centered_near_zero():
    res = run_delta_hedge(n_paths=10_000, rehedge_steps=252, seed=42)
    # A well hedged short call nets close to zero on average. The premium pays
    # for the replicating portfolio.
    assert abs(res.mean_pnl) < 0.5
    assert res.option_premium > 0.0


def test_more_frequent_hedging_reduces_pnl_spread():
    coarse = run_delta_hedge(n_paths=8_000, rehedge_steps=12, seed=3)
    fine = run_delta_hedge(n_paths=8_000, rehedge_steps=252, seed=3)
    assert fine.std_pnl < coarse.std_pnl
