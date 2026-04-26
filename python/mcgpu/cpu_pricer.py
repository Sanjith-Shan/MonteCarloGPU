"""NumPy CPU Monte Carlo pricers.

This is the honest CPU baseline for the speedup benchmark. It is fully
vectorized so we compare against optimized CPU code rather than a strawman
Python loop. Memory grows with n_paths times n_steps, so very large path counts
are simulated in chunks to keep the working set bounded.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Cap the number of path floats held in memory at once. Chunking keeps the
# baseline usable at 100M paths without exhausting host RAM.
_CHUNK_ELEMENTS = 50_000_000


def _chunk_sizes(n_paths: int, n_steps: int) -> list[int]:
    per_path = max(n_steps, 1)
    chunk_paths = max(1, _CHUNK_ELEMENTS // per_path)
    sizes = []
    remaining = n_paths
    while remaining > 0:
        take = min(chunk_paths, remaining)
        sizes.append(take)
        remaining -= take
    return sizes


def price_european_cpu(S0, K, r, sigma, T, n_paths, n_steps=252, seed=42,
                       call=True) -> Tuple[float, float]:
    """Price a European option. Returns (price, standard_error)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    payoff_sum = 0.0
    payoff_sq_sum = 0.0
    for m in _chunk_sizes(n_paths, n_steps):
        z = rng.standard_normal((m, n_steps))
        log_paths = math.log(S0) + np.cumsum(drift + diffusion * z, axis=1)
        ST = np.exp(log_paths[:, -1])
        payoff = np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)
        payoff_sum += payoff.sum()
        payoff_sq_sum += np.square(payoff).sum()

    disc = math.exp(-r * T)
    mean = payoff_sum / n_paths
    var = payoff_sq_sum / n_paths - mean * mean
    price = disc * mean
    stderr = disc * math.sqrt(max(var, 0.0) / n_paths)
    return price, stderr


def price_asian_cpu(S0, K, r, sigma, T, n_paths, n_steps=252, seed=42
                    ) -> Tuple[float, float]:
    """Arithmetic average Asian call. Returns (price, standard_error)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    payoff_sum = 0.0
    payoff_sq_sum = 0.0
    for m in _chunk_sizes(n_paths, n_steps):
        z = rng.standard_normal((m, n_steps))
        log_paths = math.log(S0) + np.cumsum(drift + diffusion * z, axis=1)
        avg = np.exp(log_paths).mean(axis=1)
        payoff = np.maximum(avg - K, 0.0)
        payoff_sum += payoff.sum()
        payoff_sq_sum += np.square(payoff).sum()

    disc = math.exp(-r * T)
    mean = payoff_sum / n_paths
    var = payoff_sq_sum / n_paths - mean * mean
    return disc * mean, disc * math.sqrt(max(var, 0.0) / n_paths)


def price_barrier_cpu(S0, K, B, r, sigma, T, n_paths, n_steps=252, seed=42
                      ) -> Tuple[float, float]:
    """Up-and-out barrier call. Returns (price, standard_error)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    payoff_sum = 0.0
    payoff_sq_sum = 0.0
    for m in _chunk_sizes(n_paths, n_steps):
        z = rng.standard_normal((m, n_steps))
        log_paths = math.log(S0) + np.cumsum(drift + diffusion * z, axis=1)
        paths = np.exp(log_paths)
        knocked = (paths >= B).any(axis=1)
        ST = paths[:, -1]
        payoff = np.where(knocked, 0.0, np.maximum(ST - K, 0.0))
        payoff_sum += payoff.sum()
        payoff_sq_sum += np.square(payoff).sum()

    disc = math.exp(-r * T)
    mean = payoff_sum / n_paths
    var = payoff_sq_sum / n_paths - mean * mean
    return disc * mean, disc * math.sqrt(max(var, 0.0) / n_paths)
