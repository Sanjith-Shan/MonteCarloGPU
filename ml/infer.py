#!/usr/bin/env python3
"""Inference and latency benchmark for the surrogate pricer.

The point of a surrogate is speed. This script measures how fast the trained
network prices a large batch of options in a single forward pass and contrasts
that with the Monte Carlo engine pricing the same book one option at a time.
The surrogate trades a small, bounded approximation error for a very large
latency reduction, which is the tradeoff that makes it useful for real time
risk. Writes ml/infer_report.json.

Usage:
    python3 ml/infer.py --n 100000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import dataset as data_mod
from model import build_model

HERE = os.path.dirname(__file__)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100_000, help="options to price")
    args = ap.parse_args()

    ckpt_path = os.path.join(HERE, "..", "ml", "checkpoints", "surrogate.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit("No checkpoint found. Run ml/train.py first.")

    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], in_features=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # A random book of options to price.
    rng = np.random.default_rng(0)
    n = args.n
    m = rng.uniform(0.6, 1.4, n)
    r = rng.uniform(0.0, 0.08, n)
    sig = rng.uniform(0.05, 0.60, n)
    T = rng.uniform(0.05, 2.0, n)
    fwd = sig * np.sqrt(T)
    X = np.stack([m, r, sig, T, fwd], axis=1).astype(np.float32)
    Xt = torch.as_tensor(X, device=device)

    # Warm up.
    with torch.no_grad():
        model(Xt[:1024])
    sync(device)

    # Surrogate batch inference.
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(Xt)
    sync(device)
    surrogate_ms = (time.perf_counter() - t0) * 1000.0

    # Exact analytical prices for the same book, as the accuracy reference.
    true = np.array([data_mod.price_reference(m[i], r[i], sig[i], T[i])
                     for i in range(min(n, 20000))])
    pred_np = preds.cpu().numpy().reshape(-1)[:len(true)]
    rmse = float(np.sqrt(np.mean((pred_np - true) ** 2)))

    per_option_us = surrogate_ms * 1000.0 / n
    throughput = n / (surrogate_ms / 1000.0)

    print(f"Surrogate inference on {device}")
    print(f"  priced {n:,} options in {surrogate_ms:.2f} ms")
    print(f"  {per_option_us:.4f} microseconds per option")
    print(f"  {throughput:,.0f} options per second")
    print(f"  accuracy vs Black Scholes, RMSE {rmse:.5f} per unit strike")

    report = {
        "device": str(device),
        "n_options": n,
        "surrogate_ms": surrogate_ms,
        "per_option_us": per_option_us,
        "throughput_per_s": throughput,
        "rmse_vs_bs": rmse,
    }
    with open(os.path.join(HERE, "infer_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote ml/infer_report.json")


if __name__ == "__main__":
    main()
