"""Regression test on the Monte Carlo convergence rate.

The standard error of a Monte Carlo estimate should scale as 1/sqrt(N). Fitting
log(std_error) against log(N) should give a slope near -0.5. This locks in the
statistical property the whole engine relies on.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mcgpu.torch_pricer import TorchPricer


def _ols_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def test_standard_error_scales_as_inverse_sqrt_n():
    p = TorchPricer()
    counts = [100_000, 400_000, 1_600_000, 6_400_000]
    log_n, log_se = [], []
    for n in counts:
        res = p.price_european(100, 100, 0.05, 0.20, 1.0, n, 100, seed=42)
        log_n.append(math.log(n))
        log_se.append(math.log(res.std_error))
    slope = _ols_slope(log_n, log_se)
    assert abs(slope + 0.5) < 0.08
