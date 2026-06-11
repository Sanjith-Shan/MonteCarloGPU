"""Monte Carlo portfolio risk on the GPU.

The same path generation engine that prices a single option also drives
portfolio level risk. Here we simulate correlated asset returns with a
Cholesky factor of the covariance matrix and report Value at Risk and
Conditional VaR of a linear portfolio. This is the risk management side of the
same workload that NVIDIA GPUs run for financial services desks.

Everything is batched and device agnostic through PyTorch so a book of
thousands of positions and millions of scenarios fits the same accelerator
path as the option pricer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .torch_pricer import pick_device, _sync


@dataclass
class RiskResult:
    var: float          # Value at Risk (positive number, a loss)
    cvar: float         # Conditional VaR / expected shortfall beyond VaR
    mean_pnl: float     # expected profit and loss over the horizon
    vol_pnl: float      # standard deviation of P and L
    confidence: float   # e.g. 0.99
    horizon_days: float
    scenarios: int
    elapsed_ms: float
    device: str


class PortfolioRisk:
    """Correlated multi asset Monte Carlo VaR engine."""

    def __init__(self, device: Optional[str] = None, batch: int = 2_000_000):
        self.device = pick_device(device)
        self.batch = batch

    def value_at_risk(self,
                      weights: np.ndarray,      # dollar exposure per asset
                      mu: np.ndarray,           # annualized drift per asset
                      cov: np.ndarray,          # annualized covariance matrix
                      n_scenarios: int,
                      horizon_days: float = 1.0,
                      confidence: float = 0.99,
                      seed: int = 42) -> RiskResult:
        n_assets = len(weights)
        dt = horizon_days / 252.0

        w = torch.tensor(weights, device=self.device, dtype=torch.float32)
        drift = torch.tensor((mu - 0.5 * np.diag(cov)) * dt, device=self.device,
                             dtype=torch.float32)
        # Cholesky factor of the covariance scaled to the horizon. Correlated
        # shocks come from L @ z where z is standard normal.
        L = np.linalg.cholesky(cov * dt).astype(np.float32)
        L_t = torch.tensor(L, device=self.device)

        gen = torch.Generator(device=self.device).manual_seed(int(seed))

        _sync(self.device)
        import time
        t0 = time.perf_counter()

        pnl_chunks = []
        total = 0
        while total < n_scenarios:
            m = min(self.batch, n_scenarios - total)
            z = torch.randn((m, n_assets), generator=gen, device=self.device,
                            dtype=torch.float32)
            shocks = drift + z @ L_t.T                 # (m, n_assets) log returns
            simple_returns = torch.expm1(shocks)       # arithmetic returns
            pnl = simple_returns @ w                    # (m,) portfolio P and L
            pnl_chunks.append(pnl.to("cpu"))
            total += m

        _sync(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pnl = torch.cat(pnl_chunks).numpy()
        loss_quantile = np.quantile(pnl, 1.0 - confidence)
        var = -loss_quantile
        tail = pnl[pnl <= loss_quantile]
        cvar = -tail.mean() if tail.size > 0 else var

        return RiskResult(
            var=float(var), cvar=float(cvar), mean_pnl=float(pnl.mean()),
            vol_pnl=float(pnl.std()), confidence=confidence,
            horizon_days=horizon_days, scenarios=n_scenarios,
            elapsed_ms=elapsed_ms, device=str(self.device),
        )


def sample_book(n_assets: int = 8, seed: int = 7):
    """Build a small synthetic but plausible book for demos and tests.

    Returns (weights, mu, cov). Exposures are in dollars. The covariance is a
    correlation matrix with a common market factor scaled by per asset vols.
    """
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.15, 0.45, n_assets)
    # One factor correlation structure so the matrix is positive definite.
    beta = rng.uniform(0.3, 0.9, n_assets)
    corr = np.outer(beta, beta)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(vols, vols)
    mu = rng.uniform(0.02, 0.12, n_assets)
    weights = rng.uniform(-1_000_000, 2_000_000, n_assets)
    return weights, mu, cov
