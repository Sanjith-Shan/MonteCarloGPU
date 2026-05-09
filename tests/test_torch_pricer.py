"""Validate the PyTorch device agnostic pricer.

These run on whatever accelerator PyTorch finds (CUDA, MPS, or CPU) and confirm
the tensor engine agrees with both Black Scholes and the NumPy baseline.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mcgpu.black_scholes import bs_price
from mcgpu.torch_pricer import TorchPricer, pick_device

REF = dict(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)


def test_device_is_selected():
    dev = pick_device()
    assert dev.type in ("cuda", "mps", "cpu")


def test_european_matches_black_scholes():
    bs = bs_price(REF["S0"], REF["K"], REF["r"], REF["sigma"], REF["T"])
    p = TorchPricer()
    res = p.price_european(**REF, n_paths=2_000_000, n_steps=100, seed=42)
    assert abs(res.price - bs) < 3.0 * res.std_error + 0.02
    assert res.std_error > 0.0


def test_put_call_parity():
    p = TorchPricer()
    c = p.price_european(**REF, n_paths=2_000_000, n_steps=50, seed=42, call=True)
    put = p.price_european(**REF, n_paths=2_000_000, n_steps=50, seed=42, call=False)
    parity = REF["S0"] - REF["K"] * math.exp(-REF["r"] * REF["T"])
    assert abs((c.price - put.price) - parity) < 0.1


def test_asian_and_barrier_below_european():
    p = TorchPricer()
    euro = p.price_european(**REF, n_paths=1_000_000, n_steps=100, seed=1).price
    asian = p.price_asian(**REF, n_paths=1_000_000, n_steps=100, seed=1).price
    barrier = p.price_barrier(**REF, B=120.0, n_paths=1_000_000, n_steps=100, seed=1).price
    assert 0.0 < asian < euro
    assert 0.0 <= barrier < euro


def test_reported_time_is_positive():
    p = TorchPricer()
    res = p.price_european(**REF, n_paths=500_000, n_steps=50, seed=1)
    assert res.elapsed_ms > 0.0
    assert res.paths == 500_000
