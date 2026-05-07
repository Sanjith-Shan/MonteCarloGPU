"""Unit tests for the analytical Black Scholes pricer."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mcgpu.black_scholes import bs_price, bs_greeks, norm_cdf


def test_norm_cdf_known_values():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-9
    assert abs(norm_cdf(-100.0)) < 1e-9
    assert abs(norm_cdf(100.0) - 1.0) < 1e-9


def test_atm_call_reference():
    # Textbook value for S=K=100, r=5%, vol=20%, T=1.
    price = bs_price(100, 100, 0.05, 0.20, 1.0, call=True)
    assert abs(price - 10.4506) < 1e-3


def test_put_call_parity():
    S, K, r, T = 100, 105, 0.05, 1.0
    c = bs_price(S, K, r, 0.25, T, call=True)
    p = bs_price(S, K, r, 0.25, T, call=False)
    parity = S - K * math.exp(-r * T)
    assert abs((c - p) - parity) < 1e-6


def test_call_delta_in_unit_interval():
    g = bs_greeks(100, 100, 0.05, 0.20, 1.0, call=True)
    assert 0.0 <= g.delta <= 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0


def test_put_delta_negative():
    g = bs_greeks(100, 100, 0.05, 0.20, 1.0, call=False)
    assert -1.0 <= g.delta <= 0.0
