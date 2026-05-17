#!/usr/bin/env python3
"""Generate an option price surface.

Sweeps a grid of spot price and volatility and prices a European call at each
node with the GPU Monte Carlo engine. The result is a 2D surface that feeds the
price heatmap chart and doubles as the training grid for the neural surrogate
pricer in ml/. Saved to results/surface.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mcgpu.config import REFERENCE
from mcgpu.torch_pricer import TorchPricer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(n_spot=25, n_vol=25, paths=500_000):
    ref = REFERENCE
    pricer = TorchPricer()
    spots = np.linspace(60.0, 140.0, n_spot)
    vols = np.linspace(0.10, 0.50, n_vol)
    print(f"Pricing a {n_spot}x{n_vol} surface on {pricer.device} "
          f"({n_spot * n_vol} nodes, {paths:,} paths each)")

    grid = np.zeros((n_vol, n_spot))
    for i, vol in enumerate(vols):
        for j, spot in enumerate(spots):
            res = pricer.price_european(spot, ref.strike, ref.rate, vol,
                                        ref.maturity, paths, ref.n_steps, ref.seed)
            grid[i, j] = res.price
        print(f"  vol {vol:.3f} row done")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "surface.json"), "w") as f:
        json.dump({
            "spots": spots.tolist(),
            "vols": vols.tolist(),
            "strike": ref.strike,
            "rate": ref.rate,
            "maturity": ref.maturity,
            "prices": grid.tolist(),
        }, f)
    print("Wrote results/surface.json")


if __name__ == "__main__":
    main()
