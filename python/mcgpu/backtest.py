"""Delta hedging backtest.

A classic sanity check for an option pricing and Greeks engine. We sell a
European call, then rehedge the delta on a schedule using simulated underlying
paths. If the pricing and Greeks are correct the hedged profit and loss should
be centered near zero with a spread that shrinks as we hedge more frequently.
This ties the pricer to a realistic trading workflow, which is the kind of
backtesting the role calls out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_price, bs_greeks


@dataclass
class BacktestResult:
    mean_pnl: float
    std_pnl: float
    n_paths: int
    rehedge_steps: int
    option_premium: float


def run_delta_hedge(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
                    rehedge_steps=252, n_paths=20_000, seed=42) -> BacktestResult:
    """Simulate selling one call and delta hedging it to expiry.

    Returns the distribution of hedged P and L across paths. A well hedged short
    call should net close to zero on average because the premium collected pays
    for the replicating stock position.
    """
    rng = np.random.default_rng(seed)
    dt = T / rehedge_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    premium = bs_price(S0, K, r, sigma, T, call=True)

    z = rng.standard_normal((n_paths, rehedge_steps))
    log_incr = drift + diffusion * z
    log_paths = math.log(S0) + np.cumsum(log_incr, axis=1)
    paths = np.exp(log_paths)
    # Prepend the initial spot as column zero.
    paths = np.concatenate([np.full((n_paths, 1), S0), paths], axis=1)

    cash = np.full(n_paths, premium)   # collect the premium up front
    shares = np.zeros(n_paths)

    for step in range(rehedge_steps):
        t_remaining = T - step * dt
        S = paths[:, step]
        # Vectorized Black Scholes delta at each path node.
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * t_remaining) / (
            sigma * np.sqrt(t_remaining))
        delta = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / math.sqrt(2.0)))
        # Grow cash at the risk free rate, then trade to the new hedge.
        cash *= math.exp(r * dt)
        trade = delta - shares
        cash -= trade * S
        shares = delta

    ST = paths[:, -1]
    cash *= math.exp(r * dt)
    # Unwind stock and settle the short call.
    cash += shares * ST
    cash -= np.maximum(ST - K, 0.0)

    return BacktestResult(
        mean_pnl=float(cash.mean()), std_pnl=float(cash.std()),
        n_paths=n_paths, rehedge_steps=rehedge_steps, option_premium=premium,
    )
