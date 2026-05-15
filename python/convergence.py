#!/usr/bin/env python3
"""Monte Carlo convergence analysis.

Demonstrates the O(1/sqrt(N)) error scaling that underpins Monte Carlo pricing.
For a sweep of path counts we record the price estimate and its standard error,
fit the slope of log(std_error) against log(N), and confirm it lands near the
theoretical -0.5. Output goes to results/convergence.json and a Markdown table
at results/convergence.md.
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mcgpu.black_scholes import bs_price
from mcgpu.config import REFERENCE
from mcgpu.torch_pricer import TorchPricer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SWEEP = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]


def ols_slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def main():
    ref = REFERENCE
    pricer = TorchPricer()
    analytical = bs_price(ref.spot, ref.strike, ref.rate, ref.vol, ref.maturity)
    print(f"Convergence study on {pricer.device}. BS reference = {analytical:.6f}\n")

    rows = []
    for n in SWEEP:
        res = pricer.price_european(ref.spot, ref.strike, ref.rate, ref.vol,
                                    ref.maturity, n, ref.n_steps, ref.seed)
        rows.append({
            "paths": n,
            "price": res.price,
            "std_error": res.std_error,
            "abs_error": abs(res.price - analytical),
        })
        print(f"  N={n:>10}  price={res.price:9.5f}  "
              f"std_err={res.std_error:9.6f}  abs_err={abs(res.price-analytical):9.6f}")

    log_n = [math.log(r["paths"]) for r in rows]
    log_se = [math.log(r["std_error"]) for r in rows]
    slope = ols_slope(log_n, log_se)
    print(f"\n  Fitted log-log slope = {slope:.4f}  (theory = -0.5000)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "convergence.json"), "w") as f:
        json.dump({"bs_price": analytical, "slope": slope, "rows": rows}, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "convergence.md"), "w") as f:
        f.write("# Monte Carlo Convergence\n\n")
        f.write(f"Black Scholes reference price: {analytical:.6f}\n\n")
        f.write(f"Fitted log-log slope of standard error vs paths: "
                f"{slope:.4f} (theory -0.5000)\n\n")
        f.write("| Paths | MC Price | Std Error | Abs Error |\n")
        f.write("|-------|----------|-----------|-----------|\n")
        for r in rows:
            f.write(f"| {r['paths']:,} | {r['price']:.5f} | "
                    f"{r['std_error']:.6f} | {r['abs_error']:.6f} |\n")
    print("\nWrote results/convergence.json and results/convergence.md")


if __name__ == "__main__":
    main()
