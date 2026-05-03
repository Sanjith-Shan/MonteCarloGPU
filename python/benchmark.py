#!/usr/bin/env python3
"""Full CPU vs GPU benchmark suite.

Runs four studies and writes the combined results to results/benchmark.json.

  1. Scaling      CPU (NumPy) vs GPU (PyTorch) across a path count sweep.
  2. Accuracy     Monte Carlo vs Black Scholes analytical, absolute error.
  3. Option types European, Asian, and barrier at a fixed path count.
  4. Greeks       Finite difference Greeks vs analytical for a European call.

The GPU path uses PyTorch so it runs on CUDA, Apple MPS, or CPU. On a machine
with no accelerator the GPU and CPU numbers converge, which the report notes.

Usage:
    python3 python/benchmark.py               # full sweep
    python3 python/benchmark.py --quick       # small sweep for a smoke test
    python3 python/benchmark.py --max-paths 10000000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# Allow running as `python3 python/benchmark.py` from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mcgpu.black_scholes import bs_price, bs_greeks
from mcgpu.cpu_pricer import price_european_cpu
from mcgpu.config import REFERENCE, DEFAULT_PATH_SWEEP
from mcgpu.torch_pricer import TorchPricer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def mc_greeks_torch(pricer: TorchPricer, ref, n_paths):
    """Central difference Greeks with common random numbers. Reusing one seed
    across the bumped runs correlates the paths so most variance cancels."""
    S, K, r, sig, T = ref.spot, ref.strike, ref.rate, ref.vol, ref.maturity
    seed, steps = ref.seed, ref.n_steps

    def price(spot=S, vol=sig, mat=T):
        return pricer.price_european(spot, K, r, vol, mat, n_paths, steps, seed).price

    hS, hV, hT = 0.01 * S, 0.001, 1.0 / 365.0
    base = price()
    up, down = price(spot=S + hS), price(spot=S - hS)
    vu, vd = price(vol=sig + hV), price(vol=sig - hV)
    tf = price(mat=T - hT)
    return {
        "delta": (up - down) / (2 * hS),
        "gamma": (up - 2 * base + down) / (hS * hS),
        "vega": (vu - vd) / (2 * hV),
        "theta": (tf - base) / hT,
    }


def bench_scaling(pricer, ref, sweep, run_cpu=True):
    print("\n[1/4] Scaling benchmark (CPU NumPy vs GPU PyTorch)")
    print(f"  {'paths':>12}  {'cpu_ms':>10}  {'gpu_ms':>10}  {'speedup':>8}  {'gpu_price':>10}")
    rows = []
    for n in sweep:
        gpu = pricer.price_european(ref.spot, ref.strike, ref.rate, ref.vol,
                                    ref.maturity, n, ref.n_steps, ref.seed)
        cpu_ms = None
        if run_cpu:
            t0 = time.perf_counter()
            cpu_price, _ = price_european_cpu(ref.spot, ref.strike, ref.rate,
                                              ref.vol, ref.maturity, n,
                                              ref.n_steps, ref.seed)
            cpu_ms = (time.perf_counter() - t0) * 1000.0
        speedup = (cpu_ms / gpu.elapsed_ms) if cpu_ms else None
        rows.append({
            "paths": n,
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu.elapsed_ms,
            "speedup": speedup,
            "gpu_price": gpu.price,
            "gpu_std_error": gpu.std_error,
        })
        sp = f"{speedup:7.1f}x" if speedup else "     n/a"
        cm = f"{cpu_ms:10.1f}" if cpu_ms else "       n/a"
        print(f"  {n:>12}  {cm}  {gpu.elapsed_ms:10.1f}  {sp}  {gpu.price:10.4f}")
    return rows


def bench_accuracy(pricer, ref, sweep):
    print("\n[2/4] Accuracy benchmark (Monte Carlo vs Black Scholes)")
    analytical = bs_price(ref.spot, ref.strike, ref.rate, ref.vol, ref.maturity)
    print(f"  Black Scholes reference = {analytical:.6f}")
    print(f"  {'paths':>12}  {'mc_price':>10}  {'abs_error':>10}  {'std_error':>10}")
    rows = []
    for n in sweep:
        res = pricer.price_european(ref.spot, ref.strike, ref.rate, ref.vol,
                                    ref.maturity, n, ref.n_steps, ref.seed)
        abs_err = abs(res.price - analytical)
        rows.append({"paths": n, "mc_price": res.price, "bs_price": analytical,
                     "abs_error": abs_err, "std_error": res.std_error})
        print(f"  {n:>12}  {res.price:10.4f}  {abs_err:10.5f}  {res.std_error:10.5f}")
    return {"bs_price": analytical, "rows": rows}


def bench_option_types(pricer, ref, n_paths):
    print(f"\n[3/4] Option type comparison at {n_paths:,} paths")
    euro = pricer.price_european(ref.spot, ref.strike, ref.rate, ref.vol,
                                 ref.maturity, n_paths, ref.n_steps, ref.seed)
    asian = pricer.price_asian(ref.spot, ref.strike, ref.rate, ref.vol,
                               ref.maturity, n_paths, ref.n_steps, ref.seed)
    barrier = pricer.price_barrier(ref.spot, ref.strike, ref.barrier, ref.rate,
                                   ref.vol, ref.maturity, n_paths, ref.n_steps,
                                   ref.seed)
    out = {}
    for name, res in [("european", euro), ("asian", asian), ("barrier", barrier)]:
        out[name] = {"price": res.price, "std_error": res.std_error,
                     "gpu_ms": res.elapsed_ms}
        print(f"  {name:>10}: price {res.price:8.4f}  time {res.elapsed_ms:8.1f} ms")
    return out


def bench_greeks(pricer, ref, n_paths):
    print(f"\n[4/4] Greeks benchmark at {n_paths:,} paths")
    mc = mc_greeks_torch(pricer, ref, n_paths)
    an = bs_greeks(ref.spot, ref.strike, ref.rate, ref.vol, ref.maturity)
    analytical = {"delta": an.delta, "gamma": an.gamma, "vega": an.vega, "theta": an.theta}
    print(f"  {'greek':>7}  {'monte_carlo':>12}  {'analytical':>12}")
    for g in ("delta", "gamma", "vega", "theta"):
        print(f"  {g:>7}  {mc[g]:12.4f}  {analytical[g]:12.4f}")
    return {"monte_carlo": mc, "analytical": analytical}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small sweep smoke test")
    ap.add_argument("--max-paths", type=int, default=None)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU baseline")
    ap.add_argument("--device", default=None, help="force cuda|mps|cpu")
    args = ap.parse_args()

    sweep = [10_000, 100_000, 1_000_000] if args.quick else list(DEFAULT_PATH_SWEEP)
    if args.max_paths:
        sweep = [n for n in sweep if n <= args.max_paths]

    pricer = TorchPricer(device=args.device)
    print(f"MonteCarloGPU benchmark  |  device = {pricer.device}  |  "
          f"host = {platform.processor() or platform.machine()}")

    ref = REFERENCE
    # Warm up the accelerator so the first timed run is not penalized.
    pricer.price_european(ref.spot, ref.strike, ref.rate, ref.vol, ref.maturity,
                          200_000, ref.n_steps, ref.seed)

    accuracy_n = min(sweep[-1], 10_000_000)
    results = {
        "meta": {
            "device": str(pricer.device),
            "host": platform.platform(),
            "python": platform.python_version(),
            "reference": vars(ref),
            "sweep": sweep,
        },
        "scaling": bench_scaling(pricer, ref, sweep, run_cpu=not args.no_cpu),
        "accuracy": bench_accuracy(pricer, ref, sweep),
        "option_types": bench_option_types(pricer, ref, accuracy_n),
        "greeks": bench_greeks(pricer, ref, accuracy_n),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
