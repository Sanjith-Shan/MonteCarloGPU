#!/usr/bin/env python3
"""Portfolio risk and delta hedge backtest demo.

Runs the two risk workloads on top of the same Monte Carlo engine that prices a
single option. First it computes Value at Risk and Conditional VaR for a small
correlated book. Then it backtests delta hedging a short call to expiry. Writes
results/risk.json.

Usage:
    python3 python/risk_demo.py
    python3 python/risk_demo.py --scenarios 2000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mcgpu.portfolio import PortfolioRisk, sample_book
from mcgpu.backtest import run_delta_hedge

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=int, default=2_000_000)
    ap.add_argument("--assets", type=int, default=8)
    args = ap.parse_args()

    weights, mu, cov = sample_book(n_assets=args.assets, seed=7)
    gross = float(abs(weights).sum())

    engine = PortfolioRisk()
    print(f"Monte Carlo Value at Risk on {engine.device}")
    print(f"  book of {args.assets} assets, gross exposure ${gross:,.0f}")
    risk = engine.value_at_risk(weights, mu, cov, args.scenarios,
                                horizon_days=1.0, confidence=0.99, seed=7)
    print(f"  scenarios      {risk.scenarios:,}")
    print(f"  1 day 99% VaR  ${risk.var:,.0f}")
    print(f"  1 day 99% CVaR ${risk.cvar:,.0f}")
    print(f"  mean P and L   ${risk.mean_pnl:,.0f}")
    print(f"  P and L vol    ${risk.vol_pnl:,.0f}")
    print(f"  time           {risk.elapsed_ms:.1f} ms")

    print("\nDelta hedge backtest, short one call, hedged to expiry")
    for steps in (12, 52, 252):
        bt = run_delta_hedge(rehedge_steps=steps, n_paths=20_000, seed=42)
        print(f"  rehedge {steps:>3}x  mean P and L {bt.mean_pnl:+.4f}  "
              f"std {bt.std_pnl:.4f}  premium {bt.option_premium:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "var": {
            "scenarios": risk.scenarios, "var_99": risk.var, "cvar_99": risk.cvar,
            "mean_pnl": risk.mean_pnl, "vol_pnl": risk.vol_pnl,
            "gross_exposure": gross, "elapsed_ms": risk.elapsed_ms,
            "device": risk.device,
        },
        "delta_hedge": {
            str(s): vars(run_delta_hedge(rehedge_steps=s, n_paths=20_000, seed=42))
            for s in (12, 52, 252)
        },
    }
    with open(os.path.join(RESULTS_DIR, "risk.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/risk.json")


if __name__ == "__main__":
    main()
