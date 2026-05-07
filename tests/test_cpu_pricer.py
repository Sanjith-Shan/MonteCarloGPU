"""Validate the NumPy CPU Monte Carlo pricers against analytical references."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mcgpu.black_scholes import bs_price
from mcgpu.cpu_pricer import (price_european_cpu, price_asian_cpu,
                              price_barrier_cpu)

REF = dict(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)


def test_european_within_three_std_errors():
    bs = bs_price(REF["S0"], REF["K"], REF["r"], REF["sigma"], REF["T"])
    price, se = price_european_cpu(**REF, n_paths=400_000, n_steps=100, seed=1)
    assert abs(price - bs) < 3.0 * se + 0.02


def test_european_put_call_parity():
    import math
    c, _ = price_european_cpu(**REF, n_paths=400_000, n_steps=50, seed=7, call=True)
    p, _ = price_european_cpu(**REF, n_paths=400_000, n_steps=50, seed=7, call=False)
    parity = REF["S0"] - REF["K"] * math.exp(-REF["r"] * REF["T"])
    assert abs((c - p) - parity) < 0.1


def test_asian_cheaper_than_european():
    euro, _ = price_european_cpu(**REF, n_paths=300_000, n_steps=100, seed=3)
    asian, _ = price_asian_cpu(**REF, n_paths=300_000, n_steps=100, seed=3)
    assert 0.0 < asian < euro


def test_barrier_cheaper_than_european():
    euro, _ = price_european_cpu(**REF, n_paths=300_000, n_steps=100, seed=3)
    barrier, _ = price_barrier_cpu(**REF, B=120.0, n_paths=300_000, n_steps=100, seed=3)
    assert 0.0 <= barrier < euro


def test_standard_error_shrinks_with_paths():
    _, se_small = price_european_cpu(**REF, n_paths=50_000, n_steps=50, seed=5)
    _, se_large = price_european_cpu(**REF, n_paths=800_000, n_steps=50, seed=5)
    assert se_large < se_small
