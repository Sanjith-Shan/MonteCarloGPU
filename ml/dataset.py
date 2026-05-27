"""Training data for the neural surrogate pricer.

We sample option parameters uniformly across realistic ranges and label each
one with the exact Black Scholes call price. Because the label is analytical the
dataset is cheap to generate and noise free, so any error the model shows is
genuine approximation error rather than Monte Carlo noise. Prices are computed
per unit strike (K = 1) so the network learns a scale invariant pricing map that
generalizes across strikes.

Features per option (5):
    moneyness  S / K
    rate       r
    vol        sigma
    maturity   T
    fwd_var    sigma * sqrt(T)     a helpful engineered feature (total vol)

Target (1):
    call price / K
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Reuse the analytical pricer from the main package.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from mcgpu.black_scholes import bs_price  # noqa: E402

FEATURE_NAMES = ["moneyness", "rate", "vol", "maturity", "fwd_vol"]


@dataclass
class Dataset:
    X: np.ndarray  # (n, 5) features
    y: np.ndarray  # (n, 1) targets, call price per unit strike


def _features(moneyness, rate, vol, maturity):
    fwd_vol = vol * np.sqrt(maturity)
    return np.stack([moneyness, rate, vol, maturity, fwd_vol], axis=1)


def generate(cfg) -> Dataset:
    """Generate a labeled dataset from a config dict (the `data` block)."""
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_samples"]

    m = rng.uniform(*cfg["moneyness_range"], n)   # S / K, with K = 1 so S = m
    r = rng.uniform(*cfg["rate_range"], n)
    sig = rng.uniform(*cfg["vol_range"], n)
    T = rng.uniform(*cfg["maturity_range"], n)

    # Vectorized Black Scholes call price per unit strike.
    sqrtT = np.sqrt(T)
    d1 = (np.log(m) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    Ncdf = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    price = m * Ncdf(d1) - np.exp(-r * T) * Ncdf(d2)

    X = _features(m, r, sig, T).astype(np.float32)
    y = price.reshape(-1, 1).astype(np.float32)
    return Dataset(X=X, y=y)


def split(ds: Dataset, val_fraction, test_fraction, seed=42):
    """Deterministic train/val/test split."""
    rng = np.random.default_rng(seed)
    n = len(ds.X)
    idx = rng.permutation(n)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def take(i):
        return Dataset(X=ds.X[i], y=ds.y[i])

    return take(train_idx), take(val_idx), take(test_idx)


def standardizer(X: np.ndarray):
    """Return (mean, std) for feature standardization. Fit on train only."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def price_reference(moneyness, rate, vol, maturity) -> float:
    """Single option Black Scholes price per unit strike, for spot checks."""
    return bs_price(moneyness, 1.0, rate, vol, maturity, call=True)
