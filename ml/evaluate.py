#!/usr/bin/env python3
"""Evaluate the trained surrogate pricer on a held out test set.

Regenerates the same deterministic split as training, loads the best
checkpoint, and reports error metrics against the exact Black Scholes labels.
Metrics are broken out by moneyness bucket because approximation error is
usually largest deep out of the money where the price is nearly zero. Writes
ml/eval_report.json.

Usage:
    python3 ml/evaluate.py
"""

from __future__ import annotations

import json
import os

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


def predict(model, X, device, batch=16384):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ckpt_path = os.path.join(HERE, "..", "ml", "checkpoints", "surrogate.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit("No checkpoint found. Run ml/train.py first.")

    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    ds = data_mod.generate(cfg["data"])
    _, _, test = data_mod.split(ds, cfg["data"]["val_fraction"],
                                cfg["data"]["test_fraction"], cfg["data"]["seed"])

    model = build_model(cfg["model"], in_features=test.X.shape[1]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    pred = predict(model, test.X, device).reshape(-1)
    true = test.y.reshape(-1)
    err = pred - true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(np.abs(err)))
    # Relative error where the price is not tiny, to avoid dividing by zero deep
    # out of the money.
    mask = true > 0.01
    mape = float(np.mean(np.abs(err[mask] / true[mask]))) * 100.0

    print(f"Surrogate pricer test metrics on {len(true):,} options ({device})")
    print(f"  MAE  {mae:.5f} per unit strike")
    print(f"  RMSE {rmse:.5f} per unit strike")
    print(f"  Max  {max_err:.5f}")
    print(f"  MAPE {mape:.3f}% (price > 0.01)")

    # Bucket by moneyness (feature 0).
    moneyness = test.X[:, 0]
    buckets = [(0.0, 0.9, "OTM"), (0.9, 1.1, "ATM"), (1.1, 10.0, "ITM")]
    by_bucket = {}
    print("\n  by moneyness bucket:")
    for lo, hi, name in buckets:
        m = (moneyness >= lo) & (moneyness < hi)
        if m.sum() == 0:
            continue
        b_rmse = float(np.sqrt(np.mean(err[m] ** 2)))
        by_bucket[name] = {"count": int(m.sum()), "rmse": b_rmse}
        print(f"    {name:>3}  n={int(m.sum()):>6}  rmse {b_rmse:.5f}")

    report = {
        "device": str(device),
        "n_test": len(true),
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_err,
        "mape_pct": mape,
        "by_moneyness": by_bucket,
    }
    with open(os.path.join(HERE, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote ml/eval_report.json")


if __name__ == "__main__":
    main()
