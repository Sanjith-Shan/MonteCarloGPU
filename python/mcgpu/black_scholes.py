"""Analytical Black Scholes pricing and Greeks.

These closed form solutions are the ground truth for validating the Monte Carlo
estimates. Only European options have a closed form, so Asian and barrier
options are validated by convergence and cross checks instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, r: float, sigma: float, T: float,
             call: bool = True) -> float:
    """Black Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    disc = math.exp(-r * T)
    if call:
        return S * norm_cdf(d1) - K * disc * norm_cdf(d2)
    return K * disc * norm_cdf(-d2) - S * norm_cdf(-d1)


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float


def bs_greeks(S: float, K: float, r: float, sigma: float, T: float,
              call: bool = True) -> Greeks:
    """Analytical Greeks for a European option. Theta is quoted per year."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    disc = math.exp(-r * T)
    sqrtT = math.sqrt(T)
    delta = norm_cdf(d1) if call else norm_cdf(d1) - 1.0
    gamma = norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * norm_pdf(d1) * sqrtT
    term1 = -(S * norm_pdf(d1) * sigma) / (2.0 * sqrtT)
    if call:
        theta = term1 - r * K * disc * norm_cdf(d2)
    else:
        theta = term1 + r * K * disc * norm_cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta)
